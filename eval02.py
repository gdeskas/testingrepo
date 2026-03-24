# Databricks notebook source

# MAGIC %md
# MAGIC # Accuracy Evaluation — Reference Model Comparison
# MAGIC
# MAGIC Compares production model extractions (GPT-OSS-20b) against a stronger
# MAGIC reference model (GPT-OSS-120b) to measure accuracy **without a golden dataset**.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. Read production outputs from snapshot tables (already extracted by 20b)
# MAGIC 2. Read the same source documents from snapshot tables
# MAGIC 3. Run the 120b model on the same documents using the same prompts
# MAGIC 4. Compare field by field: production vs reference
# MAGIC 5. Score using deterministic comparison (categorical/numeric/date) + LLM judge (free-text)
# MAGIC
# MAGIC The 120b model acts as both the reference extractor and the judge for free-text fields.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "comm-afl-dev", "Catalog")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "Silver Schema")

catalog = dbutils.widgets.get("catalog")
slr_schema = dbutils.widgets.get("slr_schema")

# COMMAND ----------

# DBTITLE 1,Imports
import json
import time
import re
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Model configuration
PRODUCTION_MODEL = "gdproposal-load-gpt-oss-20b"   # already ran in pipeline
REFERENCE_MODEL  = "proposal-load-gpt-oss-120b"     # stronger model for reference extraction
JUDGE_MODEL      = REFERENCE_MODEL                   # 120b also judges free-text comparisons

print(f"Production model:  {PRODUCTION_MODEL}")
print(f"Reference model:   {REFERENCE_MODEL}")
print(f"Judge model:       {JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Catalogue
# MAGIC
# MAGIC Same structure as eval_01. Each prompt defines its fields, field types,
# MAGIC source table for documents, and which snapshot table has the production outputs.

# COMMAND ----------

# DBTITLE 1,Prompt catalogue with field types and production output tables
PROMPT_CATALOGUE = {
    # ── 1. Document categorisation ──────────────────────────────────────────
    "categorise_doc": {
        "prompt_type": "text_categorisation",
        "doc_source_table": "eval_snap_doc_category",       # where to read source documents
        "doc_source_filter": "CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10",
        "doc_text_column": "CONTEXT_PARSED",
        "doc_id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "category", "image_description", "document_type",
            "bank_statement_period", "bank_statement_bank_name",
            "text", "reasoning",
        ],
        "field_types": {
            "category": "categorical",
            "image_description": "free_text",
            "document_type": "categorical",
            "bank_statement_period": "free_text",
            "bank_statement_bank_name": "free_text",
            "text": "free_text",
            "reasoning": "free_text",
        },
        "critical_fields": {"category"},
    },

    # ── 2. Asset classification ─────────────────────────────────────────────
    "classify_asset": {
        "prompt_type": "asset_extraction",
        "doc_source_table": "eval_snap_doc_category",
        "doc_source_filter": """
            lower(CATEGORY) IN ('asset_image', 'proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "doc_text_column": "CONTEXT_PARSED",
        "doc_id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "asset_type", "asset_description", "manufacturer",
            "model", "year", "registration", "reasoning",
        ],
        "field_types": {
            "asset_type": "categorical",
            "asset_description": "free_text",
            "manufacturer": "free_text",
            "model": "free_text",
            "year": "categorical",
            "registration": "free_text",
            "reasoning": "free_text",
        },
        "critical_fields": {"asset_type"},
    },

    # ── 3. Proposal extraction ──────────────────────────────────────────────
    "extract_proposal": {
        "prompt_type": "account_extraction",
        "doc_source_table": "eval_snap_doc_category",
        "doc_source_filter": """
            lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "doc_text_column": "CONTEXT_PARSED",
        "doc_id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "loan_date", "broker", "proposer", "proposer_address",
            "proposer_trading_address", "proposer_registered_address",
            "proposer_email", "proposer_phone", "proposer_website",
            "sic_code", "crn", "proposer_year_established", "legal_form",
            "loan_amount", "currency", "finance_type", "regulated",
            "rate_type", "deal_type", "term_type", "loan_period",
            "payment_frequency", "vat_number", "initial_payment",
            "vat_payment", "vat_deferral", "monthly_payment",
            "balloon_payment", "confidence_score",
        ],
        "field_types": {
            "loan_date": "date",
            "broker": "free_text", "proposer": "free_text",
            "proposer_address": "free_text", "proposer_trading_address": "free_text",
            "proposer_registered_address": "free_text",
            "proposer_email": "free_text", "proposer_phone": "free_text",
            "proposer_website": "free_text",
            "sic_code": "categorical", "crn": "free_text",
            "proposer_year_established": "categorical", "legal_form": "categorical",
            "loan_amount": "numeric", "currency": "categorical",
            "finance_type": "categorical", "regulated": "categorical",
            "rate_type": "categorical", "deal_type": "categorical",
            "term_type": "categorical", "loan_period": "numeric",
            "payment_frequency": "categorical", "vat_number": "free_text",
            "initial_payment": "numeric", "vat_payment": "numeric",
            "vat_deferral": "numeric", "monthly_payment": "numeric",
            "balloon_payment": "numeric", "confidence_score": "numeric",
        },
        "critical_fields": {
            "loan_amount", "finance_type", "loan_period", "deal_type",
            "rate_type", "monthly_payment", "balloon_payment", "initial_payment",
        },
    },

    # ── 4. Corporate party extraction ───────────────────────────────────────
    "extract_corporate_party": {
        "prompt_type": "corporate_extraction",
        "doc_source_table": "eval_snap_doc_category",
        "doc_source_filter": """
            lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "doc_text_column": "CONTEXT_PARSED",
        "doc_id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "company_role", "company_name", "company_number",
            "street_name", "city_name", "postal_code", "country_name",
            "trading_address", "registered_address",
            "mob", "email", "reasoning",
        ],
        "field_types": {
            "company_role": "categorical", "company_name": "free_text",
            "company_number": "free_text", "street_name": "free_text",
            "city_name": "free_text", "postal_code": "free_text",
            "country_name": "categorical", "trading_address": "free_text",
            "registered_address": "free_text",
            "mob": "free_text", "email": "free_text", "reasoning": "free_text",
        },
        "critical_fields": {"company_name", "company_role"},
    },

    # ── 5. Person party extraction ──────────────────────────────────────────
    "extract_person_party": {
        "prompt_type": "person_extraction",
        "doc_source_table": "eval_snap_doc_category",
        "doc_source_filter": """
            lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "doc_text_column": "CONTEXT_PARSED",
        "doc_id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "role", "salutation", "first_name", "last_name",
            "company_name", "gender", "date_of_birth",
            "country_code", "mob", "email", "job_title",
            "street_address", "city_name", "postal_code", "country_name",
            "reasoning",
        ],
        "field_types": {
            "role": "categorical", "salutation": "categorical",
            "first_name": "free_text", "last_name": "free_text",
            "company_name": "free_text", "gender": "categorical",
            "date_of_birth": "date", "country_code": "categorical",
            "mob": "free_text", "email": "free_text", "job_title": "free_text",
            "street_address": "free_text", "city_name": "free_text",
            "postal_code": "free_text", "country_name": "categorical",
            "reasoning": "free_text",
        },
        "critical_fields": {"first_name", "last_name", "role"},
    },
}

CRITICAL_THRESHOLD = 0.90
SUPPORTING_THRESHOLD = 0.75

print(f"Prompt catalogue: {len(PROMPT_CATALOGUE)} prompts")
for name, cfg in PROMPT_CATALOGUE.items():
    n_free = sum(1 for f in cfg["output_fields"] if cfg["field_types"].get(f) == "free_text")
    print(f"  {name:30s}  {len(cfg['output_fields'])} fields ({n_free} LLM-judged)")

# COMMAND ----------

# DBTITLE 1,Select prompts to evaluate
PROMPTS_TO_EVALUATE = list(PROMPT_CATALOGUE.keys())
# PROMPTS_TO_EVALUATE = ["extract_proposal"]  # uncomment for single prompt
print(f"Will evaluate: {PROMPTS_TO_EVALUATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Prompt Configs from Unity Catalog

# COMMAND ----------

# DBTITLE 1,Load prompt configs (same prompts, will swap endpoint for reference model)
prompt_table = f"`{catalog}`.`{slr_schema}`.current_prompt"

prompt_configs = {}
for prompt_name in PROMPTS_TO_EVALUATE:
    cfg = PROMPT_CATALOGUE[prompt_name]
    pt = cfg["prompt_type"]

    rows = spark.sql(f"""
        SELECT PROMPT_ID, PROMPT_TYPE, MODEL_ENDPOINT, PROMPT,
               RESPONSE_FORMAT, TEMPERATURE
        FROM {prompt_table}
        WHERE PROMPT_TYPE = '{pt}' AND CURRENT = true
    """).collect()

    if not rows:
        print(f"  WARNING: No current prompt for '{pt}' — skipping {prompt_name}")
        continue

    row = rows[0]
    resp_fmt = row["RESPONSE_FORMAT"]
    if isinstance(resp_fmt, str):
        resp_fmt = json.loads(resp_fmt)

    prompt_configs[prompt_name] = {
        "prompt_id": row["PROMPT_ID"],
        "production_endpoint": row["MODEL_ENDPOINT"],
        "prompt_text": row["PROMPT"],
        "response_format": resp_fmt,
        "temperature": float(row["TEMPERATURE"]),
    }
    print(f"  Loaded: {prompt_name:30s}  prod={row['MODEL_ENDPOINT']}  ref={REFERENCE_MODEL}")

print(f"\n{len(prompt_configs)}/{len(PROMPTS_TO_EVALUATE)} prompt configs loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deterministic Comparison Functions

# COMMAND ----------

# DBTITLE 1,Comparison functions
def _norm(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _is_null(val):
    return val is None or _norm(val) in ("null", "none", "", "n/a", "na")


def compare_categorical(extracted, reference):
    if _is_null(extracted) and _is_null(reference):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(reference):
        return 0.0, f"One null: '{extracted}' vs '{reference}'"
    if _norm(extracted) == _norm(reference):
        return 1.0, "Exact match"
    return 0.0, f"Mismatch: '{extracted}' vs '{reference}'"


def compare_numeric(extracted, reference, tolerance=0.01):
    if _is_null(extracted) and _is_null(reference):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(reference):
        return 0.0, f"One null: '{extracted}' vs '{reference}'"
    try:
        ext_num = float(re.sub(r"[£$€,\s]", "", str(extracted)))
        ref_num = float(re.sub(r"[£$€,\s]", "", str(reference)))
    except (ValueError, TypeError):
        if _norm(extracted) == _norm(reference):
            return 1.0, "Non-numeric but string-equal"
        return 0.0, f"Cannot parse: '{extracted}' vs '{reference}'"
    if ref_num == 0:
        return (1.0, "Both zero") if ext_num == 0 else (0.0, f"Expected 0, got {ext_num}")
    pct_diff = abs(ext_num - ref_num) / abs(ref_num)
    if pct_diff <= tolerance:
        return 1.0, f"Within ±{tolerance:.0%} ({pct_diff:.2%})"
    if pct_diff <= tolerance * 5:
        return 0.5, f"Close ({pct_diff:.2%})"
    return 0.0, f"Outside tolerance ({pct_diff:.2%})"


def compare_date(extracted, reference):
    if _is_null(extracted) and _is_null(reference):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(reference):
        return 0.0, f"One null: '{extracted}' vs '{reference}'"
    DATE_FMTS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y",
                 "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%d.%m.%Y",
                 "%Y%m%d", "%d/%m/%y", "%d-%m-%y"]
    def _parse(val):
        for fmt in DATE_FMTS:
            try: return datetime.strptime(str(val).strip(), fmt)
            except ValueError: continue
        return None
    ed, rd = _parse(extracted), _parse(reference)
    if ed is None or rd is None:
        return (1.0, "String-equal") if _norm(extracted) == _norm(reference) else (0.0, "Cannot parse")
    if ed == rd:
        return 1.0, "Dates match"
    diff = abs((ed - rd).days)
    return (0.5, f"Off by {diff} day(s)") if diff <= 1 else (0.0, f"Differ by {diff} days")


print("Deterministic comparison functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Judge (120b)

# COMMAND ----------

# DBTITLE 1,LLM judge function
JUDGE_PROMPT = """You are an accuracy judge for a UK loan document extraction system.
Compare a PRODUCTION extraction against a REFERENCE extraction for a specific field.

Field: {field_name} ({field_context})

PRODUCTION value: {production}
REFERENCE value:  {reference}

Score:
- 1.0 = Semantically identical (abbreviations, formatting differences only)
- 0.5 = Partially correct (core info present, missing/extra detail)
- 0.0 = Incorrect or unrelated

UK conventions: Ltd=Limited, St=Street, +44=0. Email: case-insensitive. CRN/VAT: ignore spaces.
Both null = 1.0. One null with other having value = 0.0.

Respond ONLY with JSON: {{"score": <float>, "reasoning": "<one sentence>"}}"""

FIELD_CONTEXT = {
    "broker": "Broker company", "proposer": "Loan applicant company",
    "proposer_address": "UK address", "company_name": "Corporate entity",
    "first_name": "Person first name", "last_name": "Person surname",
    "street_name": "UK street", "city_name": "UK city",
    "postal_code": "UK postcode", "email": "Email address",
    "mob": "Mobile number", "crn": "Company Registration Number",
    "vat_number": "VAT number", "asset_description": "Asset description",
    "manufacturer": "Asset manufacturer", "registration": "Vehicle registration",
    "image_description": "Document image description",
    "bank_statement_period": "Statement period", "bank_statement_bank_name": "Bank name",
    "text": "Extracted text", "job_title": "Professional role",
    "trading_address": "Trading address", "registered_address": "Registered address",
    "proposer_trading_address": "Trading address", "proposer_registered_address": "Registered address",
    "proposer_email": "Email", "proposer_phone": "Phone", "proposer_website": "Website",
    "company_number": "Company number", "street_address": "Street address",
    "model": "Asset model", "reasoning": "LLM reasoning",
}


def judge_free_text(field_name: str, production: str, reference: str) -> tuple:
    if _is_null(production) and _is_null(reference):
        return 1.0, "Both null"
    if _is_null(production) or _is_null(reference):
        return 0.0, f"One null: '{production}' vs '{reference}'"
    if _norm(production) == _norm(reference):
        return 1.0, "Exact match after normalisation"

    context = FIELD_CONTEXT.get(field_name, field_name)
    prompt = JUDGE_PROMPT.format(
        field_name=field_name, field_context=context,
        production=str(production), reference=str(reference),
    )
    try:
        prompt_df = spark.createDataFrame([(prompt,)], ["judge_prompt"])
        resp_fmt = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "judge_response", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}},
                    "required": ["score", "reasoning"], "additionalProperties": False,
                },
            },
        }).replace("'", "\\'")

        result_df = prompt_df.selectExpr(f"""
            ai_query('{JUDGE_MODEL}', judge_prompt,
                     responseFormat => '{resp_fmt}') AS resp
        """)
        raw = result_df.collect()[0]["resp"]
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        return score, str(parsed.get("reasoning", ""))
    except Exception as e:
        if _norm(production) == _norm(reference):
            return 1.0, f"Judge failed, fallback match: {e}"
        return 0.0, f"Judge failed: {e}"


print(f"LLM judge defined — using {JUDGE_MODEL}")

# COMMAND ----------

# DBTITLE 1,Unified field scorer
def score_field(field_name: str, production_val, reference_val, field_type: str) -> dict:
    if field_type == "categorical":
        score, reasoning = compare_categorical(production_val, reference_val)
        method = "deterministic_categorical"
    elif field_type == "numeric":
        score, reasoning = compare_numeric(production_val, reference_val)
        method = "deterministic_numeric"
    elif field_type == "date":
        score, reasoning = compare_date(production_val, reference_val)
        method = "deterministic_date"
    else:
        score, reasoning = judge_free_text(field_name, production_val, reference_val)
        method = "llm_judge"
    return {
        "field": field_name, "production_value": str(production_val),
        "reference_value": str(reference_val), "score": score,
        "reasoning": reasoning, "comparison_method": method,
        "is_correct": score >= 0.5, "field_type": field_type,
    }

print("score_field() defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference Model Extraction Function

# COMMAND ----------

# DBTITLE 1,Reference model extractor (120b)
def extract_with_reference_model(text_content: str, prompt_name: str) -> dict:
    """
    Run the 120b reference model on a document using the same prompt
    as production. Returns parsed dict of extracted fields.
    """
    pcfg = prompt_configs[prompt_name]
    ccfg = PROMPT_CATALOGUE[prompt_name]
    output_fields = ccfg["output_fields"]

    try:
        text_truncated = text_content[:8000] if text_content else ""
        prompt_df = spark.createDataFrame([(text_truncated,)], ["doc_text"])

        resp_fmt_str = json.dumps(pcfg["response_format"]).replace("'", "\\'")
        prompt_escaped = pcfg["prompt_text"].replace("'", "\\'")

        result_df = prompt_df.selectExpr(f"""
            ai_query(
                '{REFERENCE_MODEL}',
                CONCAT('{prompt_escaped}', '\\n\\nDocument text:\\n', doc_text),
                responseFormat => '{resp_fmt_str}'
            ) AS llm_response
        """)

        raw = result_df.collect()[0]["llm_response"]
        if isinstance(raw, str):
            parsed = json.loads(raw)
        elif isinstance(raw, dict):
            parsed = raw
        else:
            parsed = json.loads(str(raw))

        return {field: parsed.get(field, None) for field in output_fields}

    except Exception as e:
        print(f"    REF ERROR: {e}")
        return {field: None for field in output_fields}


print(f"extract_with_reference_model() defined — endpoint: {REFERENCE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Source Documents and Production Outputs

# COMMAND ----------

# DBTITLE 1,Load documents with production outputs for all prompts
all_eval_data = {}

for prompt_name in prompt_configs:
    cfg = PROMPT_CATALOGUE[prompt_name]
    full_table = f"`{catalog}`.`{slr_schema}`.{cfg['doc_source_table']}"

    try:
        base_df = spark.table(full_table)
        filtered_df = base_df.filter(cfg["doc_source_filter"])
        count = filtered_df.count()

        # Collect ID columns + text column + any output fields already in the table
        select_cols = cfg["doc_id_columns"] + [cfg["doc_text_column"]]

        # Also try to grab production output fields if they exist in the snapshot
        available_cols = [c.name for c in filtered_df.schema.fields]
        prod_fields_available = [f for f in cfg["output_fields"] if f.upper() in [c.upper() for c in available_cols]]

        for f in prod_fields_available:
            # Find the actual column name (case-insensitive match)
            actual = next(c for c in available_cols if c.upper() == f.upper())
            if actual not in select_cols:
                select_cols.append(actual)

        docs = [row.asDict() for row in filtered_df.select(*select_cols).collect()]
        all_eval_data[prompt_name] = docs
        print(f"  {prompt_name:30s} → {count} docs (prod fields in table: {len(prod_fields_available)})")

    except Exception as e:
        print(f"  {prompt_name:30s} → ERROR: {e}")
        all_eval_data[prompt_name] = []

total_docs = sum(len(v) for v in all_eval_data.values())
print(f"\nTotal: {total_docs} document-prompt pairs")
print(f"Reference model calls needed: {total_docs} (one 120b call per document per prompt)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation
# MAGIC
# MAGIC For each prompt × each document:
# MAGIC 1. Extract production values (from snapshot table or re-read)
# MAGIC 2. Run 120b reference model on the same document text
# MAGIC 3. Compare field by field

# COMMAND ----------

# DBTITLE 1,Main evaluation loop
eval_run_id = f"ref_accuracy_{datetime.now():%Y%m%d_%H%M%S}"
all_scores = []
llm_judge_calls = 0
deterministic_calls = 0
ref_extract_calls = 0

print(f"Eval run: {eval_run_id}")
print(f"Reference model: {REFERENCE_MODEL}")
print("=" * 70)

for prompt_name in prompt_configs:
    cfg = PROMPT_CATALOGUE[prompt_name]
    docs = all_eval_data.get(prompt_name, [])
    output_fields = cfg["output_fields"]
    text_col = cfg["doc_text_column"]

    if not docs:
        print(f"\n[{prompt_name}] No documents — skipping.")
        continue

    print(f"\n{'='*70}")
    print(f"[{prompt_name}] {len(docs)} docs — extracting with 120b + comparing")
    print(f"{'='*70}")

    t_start = time.time()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.get("DOCUMENT_ID", doc.get("PROPOSAL_ID", f"doc_{doc_idx}"))
        text_content = doc.get(text_col, "")

        if doc_idx % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  [{doc_idx+1}/{len(docs)}] ({elapsed:.0f}s elapsed)")

        # ── Get production values (from snapshot table columns) ──────────
        production = {}
        for field in output_fields:
            # Try exact case, then uppercase
            val = doc.get(field, doc.get(field.upper(), None))
            production[field] = val

        # ── Get reference values (fresh 120b extraction) ─────────────────
        reference = extract_with_reference_model(text_content, prompt_name)
        ref_extract_calls += 1

        # ── Compare field by field ───────────────────────────────────────
        for field in output_fields:
            prod_val = production.get(field)
            ref_val = reference.get(field)
            field_type = cfg["field_types"].get(field, "free_text")

            result = score_field(field, prod_val, ref_val, field_type)
            result["eval_run_id"] = eval_run_id
            result["prompt_name"] = prompt_name
            result["document_id"] = doc_id
            result["proposal_id"] = doc.get("PROPOSAL_ID", "")
            result["is_critical"] = field in cfg.get("critical_fields", set())

            if result["comparison_method"] == "llm_judge":
                llm_judge_calls += 1
            else:
                deterministic_calls += 1

            all_scores.append(result)

    elapsed = time.time() - t_start
    prompt_scores = [s for s in all_scores if s["prompt_name"] == prompt_name]
    mean_score = np.mean([s["score"] for s in prompt_scores]) if prompt_scores else 0
    print(f"  Done: {len(prompt_scores)} field scores, mean={mean_score:.2%} ({elapsed:.0f}s)")

print(f"\nTotal: {len(all_scores)} field scores")
print(f"  Reference extractions: {ref_extract_calls}")
print(f"  LLM judge calls:       {llm_judge_calls}")
print(f"  Deterministic:          {deterministic_calls}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# DBTITLE 1,Build results
if not all_scores:
    print("No scores generated.")
    dbutils.notebook.exit("No scores")

scores_df = pd.DataFrame(all_scores)
print(f"Overall agreement with reference model: {scores_df['score'].mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Per-prompt agreement
prompt_accuracy = (
    scores_df.groupby("prompt_name")
    .agg(mean_score=("score", "mean"), pct_correct=("is_correct", "mean"),
         fields_scored=("score", "count"),
         llm_judged=("comparison_method", lambda x: (x == "llm_judge").sum()))
    .sort_values("mean_score").reset_index()
)

print("PER-PROMPT AGREEMENT (production 20b vs reference 120b)")
print("=" * 80)
for _, row in prompt_accuracy.iterrows():
    print(f"  {row['prompt_name']:30s}  agreement={row['mean_score']:.2%}  "
          f"n={row['fields_scored']:.0f}  (LLM judge: {row['llm_judged']:.0f})")

display(spark.createDataFrame(prompt_accuracy))

# COMMAND ----------

# DBTITLE 1,Per-field agreement
field_accuracy = (
    scores_df.groupby(["prompt_name", "field"])
    .agg(mean_score=("score", "mean"), pct_correct=("is_correct", "mean"),
         count=("score", "count"), is_critical=("is_critical", "first"),
         field_type=("field_type", "first"))
    .sort_values("mean_score").reset_index()
)

field_accuracy["threshold"] = field_accuracy["is_critical"].map(
    {True: CRITICAL_THRESHOLD, False: SUPPORTING_THRESHOLD})
field_accuracy["meets_threshold"] = field_accuracy["mean_score"] >= field_accuracy["threshold"]

failing = field_accuracy[~field_accuracy["meets_threshold"]]
print(f"Fields meeting threshold: {field_accuracy['meets_threshold'].sum()}/{len(field_accuracy)}")
if len(failing) > 0:
    print(f"\nFIELDS BELOW THRESHOLD ({len(failing)}):")
    for _, row in failing.iterrows():
        crit = "CRITICAL" if row["is_critical"] else "        "
        print(f"  {crit} {row['prompt_name']:25s} / {row['field']:25s}  {row['mean_score']:.2%}")

display(spark.createDataFrame(field_accuracy.head(30)))

# COMMAND ----------

# DBTITLE 1,Lowest-scoring comparisons with reasoning
worst = scores_df[scores_df["score"] < 1.0].sort_values("score")
if len(worst) > 0:
    cols = ["prompt_name", "document_id", "field", "production_value",
            "reference_value", "score", "reasoning", "comparison_method"]
    display(spark.createDataFrame(worst.head(30)[cols]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualisations

# COMMAND ----------

# DBTITLE 1,Per-prompt field agreement charts
prompts_with_scores = prompt_accuracy["prompt_name"].tolist()
n = len(prompts_with_scores)

if n > 0:
    cols_per_row = min(n, 3)
    fig, axes = plt.subplots(1, cols_per_row, figsize=(6 * cols_per_row, 8), squeeze=False)

    for i, pname in enumerate(prompts_with_scores[:cols_per_row]):
        ax = axes[0][i]
        subset = field_accuracy[field_accuracy["prompt_name"] == pname].sort_values("mean_score")
        colors = ["#E24B4A" if not m else "#639922" for m in subset["meets_threshold"]]
        ax.barh(subset["field"], subset["mean_score"], color=colors)
        ax.axvline(x=CRITICAL_THRESHOLD, color="#185FA5", linestyle="--", linewidth=1)
        ax.axvline(x=SUPPORTING_THRESHOLD, color="#854F0B", linestyle=":", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_title(pname, fontsize=11)
    plt.tight_layout()
    display(fig)
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow & Delta

# COMMAND ----------

# DBTITLE 1,Log to MLflow
mlflow.set_experiment("/Shared/eval_accuracy_ref_model")

with mlflow.start_run(run_name=f"ref_{eval_run_id}"):
    mlflow.log_metrics({
        "overall_agreement": round(scores_df["score"].mean(), 4),
        "overall_pct_correct": round(scores_df["is_correct"].mean(), 4),
        "total_fields_scored": len(scores_df),
        "ref_extract_calls": ref_extract_calls,
        "llm_judge_calls": llm_judge_calls,
        "deterministic_calls": deterministic_calls,
        "fields_passing": int(field_accuracy["meets_threshold"].sum()),
        "fields_failing": int((~field_accuracy["meets_threshold"]).sum()),
    })
    mlflow.log_params({
        "eval_run_id": eval_run_id,
        "production_model": PRODUCTION_MODEL,
        "reference_model": REFERENCE_MODEL,
        "judge_model": JUDGE_MODEL,
        "prompts": ",".join(prompts_with_scores),
    })
    for name, df in [("prompt_accuracy", prompt_accuracy),
                     ("field_accuracy", field_accuracy)]:
        path = f"/tmp/{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)
    print(f"Logged to MLflow: {eval_run_id}")

# COMMAND ----------

# DBTITLE 1,Write to Delta
scores_sdf = spark.createDataFrame(scores_df.astype(str))
scores_sdf = scores_sdf.withColumn("eval_ts", F.current_timestamp())

scores_table = f"`{catalog}`.`{slr_schema}`.eval_accuracy_scores"
scores_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true").saveAsTable(scores_table)
print(f"Wrote {len(scores_df)} scores to {scores_table}")

field_acc_sdf = spark.createDataFrame(field_accuracy.astype(str))
field_acc_sdf = (field_acc_sdf
    .withColumn("eval_run_id", F.lit(eval_run_id))
    .withColumn("production_model", F.lit(PRODUCTION_MODEL))
    .withColumn("reference_model", F.lit(REFERENCE_MODEL))
    .withColumn("eval_ts", F.current_timestamp()))

summary_table = f"`{catalog}`.`{slr_schema}`.eval_accuracy_summary"
field_acc_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true").saveAsTable(summary_table)
print(f"Wrote summary to {summary_table}")

# COMMAND ----------

# DBTITLE 1,Final summary
print(f"\n{'='*70}")
print("REFERENCE MODEL ACCURACY EVALUATION — COMPLETE")
print(f"{'='*70}")
print(f"Eval run:              {eval_run_id}")
print(f"Production model:      {PRODUCTION_MODEL}")
print(f"Reference model:       {REFERENCE_MODEL}")
print(f"Prompts evaluated:     {len(prompts_with_scores)}")
print(f"Fields scored:         {len(scores_df)}")
print(f"  Ref extractions:     {ref_extract_calls}")
print(f"  LLM judge calls:     {llm_judge_calls}")
print(f"  Deterministic:       {deterministic_calls}")
print(f"Overall agreement:     {scores_df['score'].mean():.2%}")
print(f"Fields passing:        {field_accuracy['meets_threshold'].sum()}/{len(field_accuracy)}")

if len(failing) > 0:
    print(f"\nBELOW THRESHOLD ({len(failing)}):")
    for _, row in failing.iterrows():
        crit = " [CRITICAL]" if row["is_critical"] else ""
        print(f"  {row['prompt_name']:25s} / {row['field']:25s}  "
              f"{row['mean_score']:.2%} (need {row['threshold']:.0%}){crit}")
else:
    print("\nAll fields meet thresholds!")

print(f"\nDelta: {scores_table}")
print(f"       {summary_table}")
print(f"MLflow: /Shared/eval_accuracy_ref_model")
