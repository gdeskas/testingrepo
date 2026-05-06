# Databricks notebook source

# MAGIC %md
# MAGIC # Model Selection Evaluation — All 6 Prompts
# MAGIC
# MAGIC Compares multiple LLM/VLM candidates against the golden dataset across all
# MAGIC pipeline prompts. Reads `PROMPT`, `RESPONSE_FORMAT`, `TEMPERATURE` from the
# MAGIC `current_prompt` table — only the `MODEL_ENDPOINT` is overridden per candidate.
# MAGIC
# MAGIC **For each (prompt_type, candidate_model, document):**
# MAGIC 1. Run the prompt `N_RUNS` times → measure **consistency**
# MAGIC 2. Compare each run to the golden dataset → measure **accuracy** via:
# MAGIC    - Categorical fields → normalised exact match
# MAGIC    - Numeric fields    → ±1% tolerance
# MAGIC    - Date fields       → parsed comparison
# MAGIC    - Free-text fields  → LLM-as-Judge (judge = strongest candidate)
# MAGIC 3. Capture latency and error rate per call
# MAGIC
# MAGIC **Outputs:** Delta tables in the `eval` schema + MLflow runs +
# MAGIC scorecard charts ranking candidates per prompt and overall.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "uc_comm_afl_dev", "Catalog")
dbutils.widgets.text("slr_schema", "brkrflw_slr", "Silver Schema")
dbutils.widgets.text("eval_schema", "brkrflw-lkh-eval", "Evaluation Schema")
dbutils.widgets.text("n_runs", "3", "Consistency runs per (model, doc)")
dbutils.widgets.text("max_docs", "20", "Max documents per prompt")

catalog     = dbutils.widgets.get("catalog")
slr_schema  = dbutils.widgets.get("slr_schema")
eval_schema = dbutils.widgets.get("eval_schema")
N_RUNS      = int(dbutils.widgets.get("n_runs"))
MAX_DOCS    = int(dbutils.widgets.get("max_docs"))

# COMMAND ----------

# DBTITLE 1,Imports
import json
import time
import re
import uuid
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from pyspark.sql import functions as F

EVAL_RUN_ID = f"modelsel_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
print(f"Evaluation run ID: {EVAL_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model Candidates and Judges
# MAGIC
# MAGIC Two separate candidate pools and two fixed judges:
# MAGIC - **Text prompts** → candidates exclude the judge (GPT-OSS-120B)
# MAGIC - **Vision prompt** (`categorise_image`) → Gemma3-12B and Llama4-Maverick,
# MAGIC   judged by Claude-Sonnet-4
# MAGIC
# MAGIC Holding the judge fixed and excluding it from the candidate pool removes
# MAGIC self-judging bias entirely.

# COMMAND ----------

# DBTITLE 1,Define candidate endpoints and judges
# ── Text candidates (judge GPT-OSS-120B is NOT in this list) ─────────────
TEXT_CANDIDATES = {
    # label                  : Databricks model serving endpoint
    "GPT-OSS-20B"            : "databricks-gpt-oss-20b",
    "Llama3-70B-Instruct"    : "databricks-meta-llama-3-70b-instruct",
    "Llama4-Maverick"        : "databricks-llama-4-maverick",
    "Claude-Sonnet-4"        : "databricks-claude-sonnet-4",
}

# ── Vision candidates (judge Claude-Sonnet-4 is NOT in this list) ────────
VISION_CANDIDATES = {
    "Gemma3-12B"             : "databricks-gemma-3-12b",
    "Llama4-Maverick"        : "databricks-llama-4-maverick",
}

# ── Fixed judges ─────────────────────────────────────────────────────────
TEXT_JUDGE_LABEL    = "GPT-OSS-120B"
TEXT_JUDGE_ENDPOINT = "databricks-gpt-oss-120b"

VISION_JUDGE_LABEL    = "Claude-Sonnet-4"
VISION_JUDGE_ENDPOINT = "databricks-claude-sonnet-4"

print(f"Text candidates   ({len(TEXT_CANDIDATES)}): {list(TEXT_CANDIDATES)}")
print(f"Vision candidates ({len(VISION_CANDIDATES)}): {list(VISION_CANDIDATES)}")
print(f"Text judge:                                  {TEXT_JUDGE_LABEL} → {TEXT_JUDGE_ENDPOINT}")
print(f"Vision judge:                                {VISION_JUDGE_LABEL} → {VISION_JUDGE_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prompt Catalogue
# MAGIC
# MAGIC Maps each `PROMPT_TYPE` (matching the `current_prompt` table) to:
# MAGIC - the source table holding evaluation documents,
# MAGIC - the input column,
# MAGIC - the document ID column(s),
# MAGIC - the golden-dataset table,
# MAGIC - and the field-type classification used by the accuracy router.

# COMMAND ----------

# DBTITLE 1,Field-type classification per prompt
# Field types determine which comparator is used:
#   "categorical" → normalised exact match (deterministic)
#   "numeric"     → ±1% tolerance         (deterministic)
#   "date"        → multi-format parse    (deterministic)
#   "free_text"   → LLM-as-Judge          (uses the judge model)

PROMPT_CATALOGUE = {
    # ─────────────────────────────────────────────────────────────
    "categorise_image": {
        "candidates"      : VISION_CANDIDATES,
        "judge_endpoint"  : VISION_JUDGE_ENDPOINT,
        "judge_label"     : VISION_JUDGE_LABEL,
        "source_table"    : "eval_snap_files_loaded",
        "source_filter"   : "FILE_EXT IN ('png','jpg','jpeg','bmp','gif')",
        "id_columns"      : ["FILE_ID"],
        "input_column"    : "CONTENT",          # binary image
        "is_vision"       : True,
        "golden_table"    : "golden_image_category",
        "field_types"     : {
            "category"    : "categorical",
            "description" : "free_text",
            "extracted_text": "free_text",
            "bank_name"   : "free_text",
            "statement_period_start": "date",
            "statement_period_end"  : "date",
        },
    },
    # ─────────────────────────────────────────────────────────────
    "categorise_other_doc_type": {
        "candidates"      : TEXT_CANDIDATES,
        "judge_endpoint"  : TEXT_JUDGE_ENDPOINT,
        "judge_label"     : TEXT_JUDGE_LABEL,
        "source_table"    : "eval_snap_text_extract",
        "source_filter"   : "TEXT IS NOT NULL",
        "id_columns"      : ["FILE_ID"],
        "input_column"    : "TEXT",
        "is_vision"       : False,
        "golden_table"    : "golden_doc_category",
        "field_types"     : {
            "category"    : "categorical",
            "description" : "free_text",
            "received_dt" : "date",
            "bank_name"   : "free_text",
            "statement_period_start": "date",
            "statement_period_end"  : "date",
        },
    },
    # ─────────────────────────────────────────────────────────────
    "classify_asset": {
        "candidates"      : TEXT_CANDIDATES,
        "judge_endpoint"  : TEXT_JUDGE_ENDPOINT,
        "judge_label"     : TEXT_JUDGE_LABEL,
        "source_table"    : "eval_snap_doc_category",
        "source_filter"   : "CATEGORY = 'asset_image'",
        "id_columns"      : ["FILE_ID"],
        "input_column"    : "CONTEXT_PARSED",
        "is_vision"       : False,
        "golden_table"    : "golden_asset",
        "field_types"     : {
            "asset_type"        : "categorical",
            "asset_subtype"     : "categorical",
            "asset_description" : "free_text",
            "make"              : "free_text",
            "model"             : "free_text",
            "year"              : "numeric",
        },
    },
    # ─────────────────────────────────────────────────────────────
    "extract_account_proposal_information": {
        "candidates"      : TEXT_CANDIDATES,
        "judge_endpoint"  : TEXT_JUDGE_ENDPOINT,
        "judge_label"     : TEXT_JUDGE_LABEL,
        "source_table"    : "eval_snap_doc_category",
        "source_filter"   : "CATEGORY = 'proposal'",
        "id_columns"      : ["PROPOSAL_ID", "FILE_ID"],
        "input_column"    : "CONTEXT_PARSED",
        "is_vision"       : False,
        "golden_table"    : "golden_proposal",
        "field_types"     : {
            "loan_date"                 : "date",
            "broker"                    : "free_text",
            "proposer"                  : "free_text",
            "proposer_address"          : "free_text",
            "proposer_trading_address"  : "free_text",
            "proposer_registered_address": "free_text",
            "proposer_email"            : "free_text",
            "proposer_phone"            : "free_text",
            "proposer_website"          : "free_text",
            "sic_code"                  : "categorical",
            "crn"                       : "categorical",
            "proposer_year_established" : "numeric",
            "legal_form"                : "categorical",
            "loan_amount"               : "numeric",
            "currency"                  : "categorical",
            "finance_type"              : "categorical",
            "regulated"                 : "categorical",
            "rate_type"                 : "categorical",
            "deal_type"                 : "categorical",
            "term_type"                 : "categorical",
            "loan_period"               : "numeric",
            "payment_frequency"         : "categorical",
            "vat_number"                : "categorical",
            "initial_payment"           : "numeric",
            "vat_payment"               : "numeric",
            "vat_deferral"              : "categorical",
            "monthly_payment"           : "numeric",
            "balloon_payment"           : "numeric",
        },
    },
    # ─────────────────────────────────────────────────────────────
    "extract_corporate_party_identify_role": {
        "candidates"      : TEXT_CANDIDATES,
        "judge_endpoint"  : TEXT_JUDGE_ENDPOINT,
        "judge_label"     : TEXT_JUDGE_LABEL,
        "source_table"    : "eval_snap_doc_category",
        "source_filter"   : "CATEGORY IN ('proposal','email_body_with_proposal_details')",
        "id_columns"      : ["PROPOSAL_ID", "FILE_ID"],
        "input_column"    : "CONTEXT_PARSED",
        "is_vision"       : False,
        "golden_table"    : "golden_corporate_roles",
        "field_types"     : {
            "company_name"       : "free_text",
            "role"               : "categorical",
            "registered_address" : "free_text",
            "trading_address"    : "free_text",
            "email"              : "free_text",
            "phone"              : "free_text",
            "crn"                : "categorical",
        },
    },
    # ─────────────────────────────────────────────────────────────
    "extract_person_party_identify_role": {
        "candidates"      : TEXT_CANDIDATES,
        "judge_endpoint"  : TEXT_JUDGE_ENDPOINT,
        "judge_label"     : TEXT_JUDGE_LABEL,
        "source_table"    : "eval_snap_doc_category",
        "source_filter"   : "CATEGORY IN ('proposal','email_body_with_proposal_details','id')",
        "id_columns"      : ["PROPOSAL_ID", "FILE_ID"],
        "input_column"    : "CONTEXT_PARSED",
        "is_vision"       : False,
        "golden_table"    : "golden_person_roles",
        "field_types"     : {
            "first_name"     : "free_text",
            "last_name"      : "free_text",
            "role"           : "categorical",
            "date_of_birth"  : "date",
            "email"          : "free_text",
            "phone"          : "free_text",
            "address"        : "free_text",
        },
    },
}

print(f"Prompt catalogue loaded for {len(PROMPT_CATALOGUE)} prompts.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Prompt Configurations from `current_prompt`
# MAGIC
# MAGIC We pull `PROMPT`, `RESPONSE_FORMAT`, `TEMPERATURE` from the table.
# MAGIC `MODEL_ENDPOINT` is **ignored** here — it gets overridden per candidate.

# COMMAND ----------

# DBTITLE 1,Load prompt configs
prompt_table = f"`{catalog}`.`{slr_schema}`.current_prompt"
prompt_configs = {}

for prompt_type in PROMPT_CATALOGUE.keys():
    rows = spark.sql(f"""
        SELECT PROMPT_ID, PROMPT_TYPE, PROMPT, RESPONSE_FORMAT, TEMPERATURE, MODEL_ENDPOINT
        FROM {prompt_table}
        WHERE PROMPT_TYPE = '{prompt_type}'
          AND CURRENT = TRUE
    """).collect()

    if not rows:
        print(f"  ⚠ Skipping {prompt_type}: no CURRENT row in prompt table")
        continue

    row = rows[0]
    resp_fmt_raw = row["RESPONSE_FORMAT"]
    resp_fmt_str = resp_fmt_raw if isinstance(resp_fmt_raw, str) else json.dumps(resp_fmt_raw)

    prompt_configs[prompt_type] = {
        "prompt_id"        : row["PROMPT_ID"],
        "prompt_text"      : row["PROMPT"],
        "response_format"  : resp_fmt_str,
        "temperature"      : float(row["TEMPERATURE"]),
        "production_endpoint": row["MODEL_ENDPOINT"],
    }
    print(f"  ✓ {prompt_type:<45s} prod={row['MODEL_ENDPOINT']}  T={row['TEMPERATURE']}")

print(f"\nLoaded {len(prompt_configs)} prompt configs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Evaluation Documents
# MAGIC
# MAGIC Reads from the `eval_snap_*` tables produced by `eval_00_pipeline_orchestrator`.
# MAGIC If any source table is missing, the cell below will tell you to run `eval_00` first.

# COMMAND ----------

# DBTITLE 1,Pre-flight — check eval_snap_* tables exist
missing_tables = []
for prompt_type, cfg in PROMPT_CATALOGUE.items():
    if prompt_type not in prompt_configs:
        continue
    src = f"`{catalog}`.`{slr_schema}`.{cfg['source_table']}"
    try:
        spark.table(src).limit(1).collect()
    except Exception as e:
        missing_tables.append((prompt_type, cfg['source_table'], str(e)[:120]))

if missing_tables:
    print("ERROR — the following snapshot tables are missing or unreadable:")
    for pt, t, err in missing_tables:
        print(f"  {pt:<45s} → {t}  ({err})")
    raise RuntimeError(
        "Snapshot tables are missing. Run `eval_00_pipeline_orchestrator` first "
        "to materialise the eval_snap_* tables, then re-run this notebook."
    )
print("✓ All required snapshot tables exist.")

# COMMAND ----------

# DBTITLE 1,Load documents per prompt
eval_docs = {}

for prompt_type, cfg in PROMPT_CATALOGUE.items():
    if prompt_type not in prompt_configs:
        continue
    src_table = f"`{catalog}`.`{slr_schema}`.{cfg['source_table']}"
    select_cols = ", ".join(cfg["id_columns"] + [cfg["input_column"]])
    query = f"""
        SELECT {select_cols}
        FROM {src_table}
        WHERE {cfg['source_filter']}
        LIMIT {MAX_DOCS}
    """
    docs = spark.sql(query).collect()
    eval_docs[prompt_type] = [r.asDict() for r in docs]
    print(f"  {prompt_type:<45s} {len(eval_docs[prompt_type]):>3d} docs")

print(f"\nTotal docs across prompts: {sum(len(v) for v in eval_docs.values())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Comparator Functions
# MAGIC
# MAGIC Field-type-aware accuracy scoring. Returns `(score, reasoning)` where
# MAGIC `score ∈ {0.0, 0.5, 1.0}` and `reasoning` is a short human-readable string.

# COMMAND ----------

# DBTITLE 1,Deterministic comparators
def _is_null(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a"))

def _norm(v):
    if _is_null(v):
        return None
    return re.sub(r"\s+", " ", str(v).strip().lower())

def compare_categorical(extracted, expected):
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    return (1.0, "Match") if _norm(extracted) == _norm(expected) \
        else (0.0, f"Mismatch: '{extracted}' vs '{expected}'")

def compare_numeric(extracted, expected, tolerance=0.01):
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    def _num(v):
        try:
            return float(re.sub(r"[^\d.\-]", "", str(v)))
        except Exception:
            return None
    e, x = _num(extracted), _num(expected)
    if e is None or x is None:
        return 0.0, f"Cannot parse: '{extracted}' vs '{expected}'"
    if x == 0:
        return (1.0, "Both zero") if e == 0 else (0.0, f"Expected 0, got {e}")
    diff = abs(e - x) / abs(x)
    if diff <= tolerance:
        return 1.0, f"Within ±{tolerance:.0%} ({diff:.2%})"
    if diff <= tolerance * 5:
        return 0.5, f"Close ({diff:.2%})"
    return 0.0, f"Outside tolerance ({diff:.2%})"

def compare_date(extracted, expected):
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y",
            "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%d.%m.%Y",
            "%Y%m%d", "%d/%m/%y", "%d-%m-%y"]
    def _p(v):
        for fmt in fmts:
            try:
                return datetime.strptime(str(v).strip(), fmt)
            except ValueError:
                continue
        return None
    e, x = _p(extracted), _p(expected)
    if e is None or x is None:
        return (1.0, "Unparseable but string-equal") if _norm(extracted) == _norm(expected) \
            else (0.0, f"Cannot parse: '{extracted}' vs '{expected}'")
    if e == x:
        return 1.0, "Match"
    return (0.5, f"Off by {abs((e-x).days)} day(s)") if abs((e-x).days) <= 1 \
        else (0.0, f"Differ by {abs((e-x).days)} days")

print("Deterministic comparators ready.")

# COMMAND ----------

# DBTITLE 1,LLM-as-Judge comparator (free-text fields, batched per document)
JUDGE_PROMPT_TEMPLATE = """You are an accuracy judge for a UK loan document extraction system.
Compare each EXTRACTED value against the EXPECTED ground-truth value.

UK conventions to apply:
- Ltd = Limited, PLC = Public Limited Company, St = Street, Rd = Road
- Phone: ignore formatting (+44 vs 0)
- Email: case-insensitive
- CRN/VAT: ignore spaces
- Both null → 1.0; one null → 0.0

Score each field:
- 1.0 = semantically identical (abbreviations, formatting, equivalent info)
- 0.5 = partially correct (core info present, missing/extra detail)
- 0.0 = incorrect or unrelated

Fields to compare:
{fields_json}

Return ONLY a JSON object with key "results" containing an array of
{{"field": str, "score": float, "reasoning": str}} — one entry per field.
"""

JUDGE_RESPONSE_FORMAT = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "judge_scores",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field"    : {"type": "string"},
                            "score"    : {"type": "number"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["field", "score", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        },
    },
})

def judge_free_text_batch(free_text_pairs, judge_endpoint, judge_label):
    """Judge all free-text fields for one document in a single LLM call.
    `free_text_pairs` = [{"field": str, "extracted": Any, "expected": Any}, ...]
    Returns: {field_name: (score, reasoning), ...}
    """
    if not free_text_pairs:
        return {}

    fields_json = json.dumps(free_text_pairs, default=str)
    prompt = JUDGE_PROMPT_TEMPLATE.format(fields_json=fields_json)

    safe_prompt = prompt.replace("'", "''")
    safe_resp_fmt = JUDGE_RESPONSE_FORMAT.replace("'", "''")

    try:
        resp = spark.sql(f"""
            SELECT ai_query(
                '{judge_endpoint}',
                '{safe_prompt}',
                responseFormat => '{safe_resp_fmt}'
            ) AS resp
        """).collect()[0]["resp"]
        parsed = json.loads(resp) if isinstance(resp, str) else resp
        out = {}
        for r in parsed.get("results", []):
            fname = r.get("field", "")
            score = max(0.0, min(1.0, float(r.get("score", 0.0))))
            out[fname] = (score, str(r.get("reasoning", "")))
        # Fill any missing
        for pair in free_text_pairs:
            out.setdefault(pair["field"], (0.0, "Field not returned by judge"))
        return out
    except Exception as e:
        return {p["field"]: (0.0, f"Judge failed: {e}") for p in free_text_pairs}

print(f"LLM-as-Judge ready — text judge: {TEXT_JUDGE_ENDPOINT}, vision judge: {VISION_JUDGE_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Field router
def score_document(extracted_dict, expected_dict, field_types, judge_endpoint, judge_label):
    """Score every field of one document. Routes by type and batches free-text
    fields into a single judge call."""
    deterministic = []
    free_text_pairs = []

    for field, ftype in field_types.items():
        ext = extracted_dict.get(field) if extracted_dict else None
        exp = expected_dict.get(field) if expected_dict else None

        if ftype == "categorical":
            s, r = compare_categorical(ext, exp)
            deterministic.append((field, ftype, s, r, ext, exp))
        elif ftype == "numeric":
            s, r = compare_numeric(ext, exp)
            deterministic.append((field, ftype, s, r, ext, exp))
        elif ftype == "date":
            s, r = compare_date(ext, exp)
            deterministic.append((field, ftype, s, r, ext, exp))
        elif ftype == "free_text":
            free_text_pairs.append({"field": field, "extracted": ext, "expected": exp})
        else:
            deterministic.append((field, ftype, 0.0, f"Unknown type {ftype}", ext, exp))

    judge_results = judge_free_text_batch(free_text_pairs, judge_endpoint, judge_label)
    rows = []
    for field, ftype, score, reasoning, ext, exp in deterministic:
        rows.append({
            "field": field, "field_type": ftype, "score": score,
            "reasoning": reasoning, "extracted": str(ext), "expected": str(exp),
            "scored_by": "deterministic",
        })
    for pair in free_text_pairs:
        s, r = judge_results.get(pair["field"], (0.0, "no result"))
        rows.append({
            "field": pair["field"], "field_type": "free_text", "score": s,
            "reasoning": r, "extracted": str(pair["extracted"]),
            "expected": str(pair["expected"]), "scored_by": judge_label,
        })
    return rows

print("Field router ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extraction Runner — calls a candidate model with the configured prompt

# COMMAND ----------

# DBTITLE 1,Run the extraction prompt with a given endpoint
def run_extraction(prompt_type, endpoint, document, prompt_cfg, is_vision=False):
    """Call `ai_query` once with a candidate endpoint. Returns
    (parsed_dict_or_None, latency_seconds, error_or_None)."""
    cfg = PROMPT_CATALOGUE[prompt_type]
    input_col = cfg["input_column"]
    raw_input = document.get(input_col)
    if raw_input is None:
        return None, 0.0, "no_input"

    safe_input    = str(raw_input).replace("'", "''")
    safe_prompt   = prompt_cfg["prompt_text"].replace("'", "''")
    safe_resp_fmt = prompt_cfg["response_format"].replace("'", "''")
    temp          = prompt_cfg["temperature"]

    if is_vision:
        # Vision call — pass the image bytes; ai_query supports image inputs
        # via a base64-encoded data URI in many Databricks endpoints.
        import base64
        b64 = base64.b64encode(raw_input).decode("utf-8") if isinstance(raw_input, (bytes, bytearray)) \
              else raw_input
        sql = f"""
            SELECT ai_query(
                '{endpoint}',
                CONCAT('{safe_prompt}', '\n\n[IMAGE_BASE64]', '{b64}'),
                responseFormat => '{safe_resp_fmt}',
                modelParameters => named_struct('temperature', {temp})
            ) AS resp
        """
    else:
        sql = f"""
            SELECT ai_query(
                '{endpoint}',
                CONCAT('{safe_prompt}', '\n\n', '{safe_input}'),
                responseFormat => '{safe_resp_fmt}',
                modelParameters => named_struct('temperature', {temp})
            ) AS resp
        """

    t0 = time.time()
    try:
        resp = spark.sql(sql).collect()[0]["resp"]
        latency = time.time() - t0
        parsed = json.loads(resp) if isinstance(resp, str) else resp
        return parsed, latency, None
    except Exception as e:
        return None, time.time() - t0, str(e)[:200]

print("run_extraction() ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Main Evaluation Loop
# MAGIC
# MAGIC For each (prompt_type, candidate, doc, run_idx) we record:
# MAGIC - the raw extraction
# MAGIC - latency + error
# MAGIC - field-level scores (judge or deterministic)
# MAGIC
# MAGIC Consistency is computed across the `N_RUNS` per (model, doc).

# COMMAND ----------

# DBTITLE 1,Helpers: load golden and compute consistency
def load_golden(prompt_type):
    cfg = PROMPT_CATALOGUE[prompt_type]
    table = f"`{catalog}`.`{eval_schema}`.{cfg['golden_table']}"
    try:
        rows = spark.sql(f"SELECT * FROM {table}").collect()
        # Index by id_columns (joined as tuple)
        id_cols = cfg["id_columns"]
        return {tuple(r[c] for c in id_cols): r.asDict() for r in rows}
    except Exception as e:
        print(f"  ⚠ No golden table for {prompt_type} ({e}). Accuracy will be 0.")
        return {}

def exact_match_rate(values):
    """Fraction of values equal to the mode."""
    vals = [_norm(v) for v in values]
    if not vals:
        return 0.0
    mode_val, mode_count = Counter(vals).most_common(1)[0]
    return mode_count / len(vals)

# COMMAND ----------

# DBTITLE 1,Run the full evaluation
all_run_rows  = []   # one row per (prompt, model, doc, run, field)
all_call_rows = []   # one row per (prompt, model, doc, run) — latency, error

for prompt_type, cfg in PROMPT_CATALOGUE.items():
    if prompt_type not in prompt_configs:
        continue

    p_cfg     = prompt_configs[prompt_type]
    docs      = eval_docs.get(prompt_type, [])
    candidates = cfg["candidates"]
    judge_endpoint = cfg["judge_endpoint"]
    judge_label    = cfg["judge_label"]
    golden    = load_golden(prompt_type)
    id_cols   = cfg["id_columns"]
    is_vision = cfg.get("is_vision", False)

    print(f"\n=== {prompt_type} | {len(docs)} docs × {len(candidates)} models × {N_RUNS} runs ===")
    print(f"    judge: {judge_label}")

    with mlflow.start_run(run_name=f"{EVAL_RUN_ID}_{prompt_type}", nested=True):
        mlflow.log_params({
            "prompt_type": prompt_type,
            "prompt_id":   p_cfg["prompt_id"],
            "n_docs":      len(docs),
            "n_runs":      N_RUNS,
            "judge":       judge_label,
        })

        for model_label, endpoint in candidates.items():
            t_model_start = time.time()
            model_call_count = 0
            model_err_count  = 0

            for doc in docs:
                doc_id = tuple(doc[c] for c in id_cols)
                expected = golden.get(doc_id, {})

                # Collect N_RUNS extractions for consistency
                runs = []
                for run_idx in range(N_RUNS):
                    parsed, latency, err = run_extraction(
                        prompt_type, endpoint, doc, p_cfg, is_vision=is_vision
                    )
                    model_call_count += 1
                    if err:
                        model_err_count += 1

                    all_call_rows.append({
                        "eval_run_id": EVAL_RUN_ID,
                        "prompt_type": prompt_type,
                        "model_label": model_label,
                        "endpoint":    endpoint,
                        "doc_id":      "|".join(str(x) for x in doc_id),
                        "run_idx":     run_idx,
                        "latency_s":   latency,
                        "error":       err or "",
                    })
                    runs.append(parsed or {})

                    # Score this run vs golden (skip if no golden)
                    if expected:
                        field_rows = score_document(
                            parsed or {}, expected, cfg["field_types"],
                            judge_endpoint, judge_label,
                        )
                        for fr in field_rows:
                            all_run_rows.append({
                                "eval_run_id": EVAL_RUN_ID,
                                "prompt_type": prompt_type,
                                "model_label": model_label,
                                "doc_id":      "|".join(str(x) for x in doc_id),
                                "run_idx":     run_idx,
                                **fr,
                            })

                # Consistency per field for this (model, doc)
                for field in cfg["field_types"].keys():
                    vals = [r.get(field) for r in runs]
                    cons = exact_match_rate(vals)
                    all_run_rows.append({
                        "eval_run_id": EVAL_RUN_ID,
                        "prompt_type": prompt_type,
                        "model_label": model_label,
                        "doc_id":      "|".join(str(x) for x in doc_id),
                        "run_idx":     -1,  # marker for consistency aggregate
                        "field":       field,
                        "field_type":  cfg["field_types"][field],
                        "score":       cons,
                        "reasoning":   f"consistency over {N_RUNS} runs",
                        "extracted":   "",
                        "expected":    "",
                        "scored_by":   "consistency",
                    })

            elapsed = time.time() - t_model_start
            err_rate = model_err_count / max(1, model_call_count)
            print(f"  {model_label:<22s}  calls={model_call_count}  "
                  f"errors={model_err_count} ({err_rate:.1%})  "
                  f"elapsed={elapsed:.0f}s")

            mlflow.log_metrics({
                f"{model_label}_call_count": model_call_count,
                f"{model_label}_error_rate": err_rate,
                f"{model_label}_elapsed_s":  elapsed,
            })

print(f"\nDone. Field-level rows: {len(all_run_rows)}  Call rows: {len(all_call_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Persist Results to Delta

# COMMAND ----------

# DBTITLE 1,Write detailed and summary tables
runs_df = pd.DataFrame(all_run_rows)
calls_df = pd.DataFrame(all_call_rows)

# Detailed field-level results
runs_table = f"`{catalog}`.`{eval_schema}`.eval_model_selection_runs"
spark.createDataFrame(runs_df.astype(str)) \
     .withColumn("eval_ts", F.current_timestamp()) \
     .write.format("delta").mode("append").option("mergeSchema", "true") \
     .saveAsTable(runs_table)

calls_table = f"`{catalog}`.`{eval_schema}`.eval_model_selection_calls"
spark.createDataFrame(calls_df.astype(str)) \
     .withColumn("eval_ts", F.current_timestamp()) \
     .write.format("delta").mode("append").option("mergeSchema", "true") \
     .saveAsTable(calls_table)

print(f"Wrote {len(runs_df)} run rows → {runs_table}")
print(f"Wrote {len(calls_df)} call rows → {calls_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Leaderboard
# MAGIC
# MAGIC One headline number per (prompt_type, model): mean accuracy on the
# MAGIC accuracy runs (ignoring the consistency aggregate rows), plus mean
# MAGIC consistency, mean latency, and error rate.

# COMMAND ----------

# DBTITLE 1,Build the leaderboard
acc_df = runs_df[(runs_df["scored_by"] != "consistency") &
                 (runs_df["run_idx"].astype(str) != "-1")].copy()
acc_df["score"] = acc_df["score"].astype(float)

cons_df = runs_df[runs_df["scored_by"] == "consistency"].copy()
cons_df["score"] = cons_df["score"].astype(float)

calls_df["latency_s"] = calls_df["latency_s"].astype(float)
calls_df["is_error"]  = (calls_df["error"].astype(str) != "").astype(int)

accuracy = acc_df.groupby(["prompt_type", "model_label"])["score"].mean().rename("accuracy")
consistency = cons_df.groupby(["prompt_type", "model_label"])["score"].mean().rename("consistency")
latency = calls_df.groupby(["prompt_type", "model_label"])["latency_s"].mean().rename("avg_latency_s")
error_rate = calls_df.groupby(["prompt_type", "model_label"])["is_error"].mean().rename("error_rate")

leaderboard = pd.concat([accuracy, consistency, latency, error_rate], axis=1).reset_index()
leaderboard["combined_score"] = (
    0.6 * leaderboard["accuracy"].fillna(0) +
    0.4 * leaderboard["consistency"].fillna(0)
)
leaderboard = leaderboard.sort_values(
    ["prompt_type", "combined_score"], ascending=[True, False]
)
display(leaderboard)

leaderboard_table = f"`{catalog}`.`{eval_schema}`.eval_model_selection_leaderboard"
spark.createDataFrame(leaderboard.astype(str)) \
     .withColumn("eval_run_id", F.lit(EVAL_RUN_ID)) \
     .withColumn("eval_ts", F.current_timestamp()) \
     .write.format("delta").mode("append").option("mergeSchema", "true") \
     .saveAsTable(leaderboard_table)

print(f"Wrote leaderboard → {leaderboard_table}")

# COMMAND ----------

# DBTITLE 1,Scorecard chart per prompt
prompts = leaderboard["prompt_type"].unique()
fig, axes = plt.subplots(len(prompts), 1, figsize=(10, 3.5 * len(prompts)))
if len(prompts) == 1:
    axes = [axes]

for ax, prompt in zip(axes, prompts):
    sub = leaderboard[leaderboard["prompt_type"] == prompt].set_index("model_label")
    sub[["accuracy", "consistency"]].plot(kind="barh", ax=ax)
    ax.set_title(prompt)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.legend(loc="lower right")

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# DBTITLE 1,Final summary
print(f"\n{'='*70}")
print(f"MODEL SELECTION EVALUATION — {EVAL_RUN_ID}")
print(f"{'='*70}")
print(f"Prompts evaluated:    {len(prompt_configs)}")
print(f"Text candidates:      {len(TEXT_CANDIDATES)}  ({list(TEXT_CANDIDATES)})")
print(f"Vision candidates:    {len(VISION_CANDIDATES)}  ({list(VISION_CANDIDATES)})")
print(f"Runs per (model,doc): {N_RUNS}")
print(f"Total LLM calls:      {len(calls_df)}")
print(f"Text judge:           {TEXT_JUDGE_LABEL}")
print(f"Vision judge:         {VISION_JUDGE_LABEL}")
print(f"Tables written:")
print(f"  - {runs_table}")
print(f"  - {calls_table}")
print(f"  - {leaderboard_table}")
