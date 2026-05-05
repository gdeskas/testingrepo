# Databricks notebook source
# MAGIC %md
# MAGIC # Eval 04 — Model Selection via LLM-as-Judge
# MAGIC
# MAGIC **Purpose:** Compare multiple LLM model endpoints across all 5 prompt types
# MAGIC using consistency (N-run agreement) and accuracy (LLM-as-judge scoring).
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. Define a list of model serving endpoints to evaluate
# MAGIC 2. For each model × each prompt × each document: call `ai_query` N times
# MAGIC 3. Compute per-field consistency metrics per model
# MAGIC 4. Run LLM-as-judge accuracy scoring (using a separate judge model)
# MAGIC 5. Aggregate results into a comparison scorecard
# MAGIC 6. Log everything to MLflow + Delta
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `eval_00` has been run — `eval_snap_*` tables are populated with documents
# MAGIC - `current_prompt` table has prompts with `CURRENT = true`
# MAGIC - All candidate model serving endpoints are provisioned in Databricks
# MAGIC - Judge model endpoint is accessible

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "comm-afl-dev", "Catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz", "Bronze Schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "Silver Schema")
dbutils.widgets.text("eval_schema", "brkrflw-lkh-eval", "Evaluation Schema")
dbutils.widgets.text("num_runs", "3", "Runs per document (consistency)")
dbutils.widgets.text("max_docs_per_prompt", "10", "Max docs per prompt (for speed)")

catalog = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")
eval_schema = dbutils.widgets.get("eval_schema")
NUM_RUNS = int(dbutils.widgets.get("num_runs"))
MAX_DOCS = int(dbutils.widgets.get("max_docs_per_prompt"))

print(f"Catalog:       {catalog}")
print(f"Silver:        {slr_schema}")
print(f"Eval:          {eval_schema}")
print(f"Runs/doc:      {NUM_RUNS}")
print(f"Max docs:      {MAX_DOCS}")

# COMMAND ----------

# DBTITLE 1,Imports
import json
import time
import re
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from pyspark.sql import functions as F

sns.set_theme(style="whitegrid", palette="muted")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model Candidates
# MAGIC
# MAGIC Define the model serving endpoints to compare. Add or remove endpoints as needed.
# MAGIC Each entry maps a short label (used in charts and tables) to the Databricks
# MAGIC model serving endpoint name.

# COMMAND ----------

# DBTITLE 1,Model candidate definitions
MODEL_CANDIDATES = {
    "GPT-OSS-20B":    "proposal-load-gpt-oss-20b",       # current production
    "GPT-OSS-120B":   "proposal-load-gpt-oss-120b",      # larger variant
    "Llama3-70B":     "proposal-load-llama3-70b",         # Meta Llama 3 70B
    "Llama4-Scout":   "proposal-load-llama4-scout",       # Meta Llama 4 Scout
    "Qwen-Instruct":  "proposal-load-qwen-instruct",      # Alibaba Qwen
}

# Judge model — used for accuracy scoring (should be a strong, separate model)
JUDGE_MODEL = "proposal-load-gpt-oss-120b"

print(f"Model candidates: {len(MODEL_CANDIDATES)}")
for label, endpoint in MODEL_CANDIDATES.items():
    print(f"  {label:<20s} → {endpoint}")
print(f"\nJudge model: {JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prompt Catalogue
# MAGIC
# MAGIC Same structure as eval_01/eval_02. Each entry defines which `current_prompt` row
# MAGIC to load, which fields to evaluate, and where to find source documents.

# COMMAND ----------

# DBTITLE 1,Prompt catalogue definition
PROMPT_CATALOGUE = {
    # ── Classification prompts ──────────────────────────────────────────────
    "categorise_doc": {
        "prompt_type": "document_categorisation",
        "task_type": "classification",
        "source_table": "eval_snap_doc_category",
        "source_filter": "CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10",
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "category", "text_description", "document_type",
            "bank_statement_period", "bank_statement_bank_name",
            "text", "reasoning",
        ],
        "field_types": {
            "category": "categorical",
            "text_description": "free_text",
            "document_type": "categorical",
            "bank_statement_period": "free_text",
            "bank_statement_bank_name": "free_text",
            "text": "free_text",
            "reasoning": "free_text",
        },
        "primary_field": "category",
        "critical_fields": {"category"},
    },

    "classify_asset": {
        "prompt_type": "asset_classification",
        "task_type": "classification",
        "source_table": "eval_snap_doc_category",
        "source_filter": "CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10",
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "asset_type", "make", "model", "manufacturer",
            "year_of_manufacture", "price_exc_vat", "price_inc_vat",
            "vat", "short_description", "description", "reasoning",
        ],
        "field_types": {
            "asset_type": "categorical",
            "make": "free_text",
            "model": "free_text",
            "manufacturer": "free_text",
            "year_of_manufacture": "categorical",
            "price_exc_vat": "numeric",
            "price_inc_vat": "numeric",
            "vat": "numeric",
            "short_description": "free_text",
            "description": "free_text",
            "reasoning": "free_text",
        },
        "primary_field": "asset_type",
        "critical_fields": {"asset_type"},
    },

    # ── Extraction prompts ──────────────────────────────────────────────────
    "extract_proposal": {
        "prompt_type": "extract_account_proposal_information",
        "task_type": "extraction",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "loan_date", "broker", "proposer", "proposer_address",
            "proposer_trading_address", "proposer_registered_address",
            "proposer_email", "proposer_phone", "proposer_website",
            "sic_code", "crn", "proposer_year_established", "legal_form",
            "loan_amount", "currency", "finance_type", "regulated",
            "rate_type", "deal_type", "term_type", "loan_period",
            "payment_frequency", "vat_number", "initial_payment",
            "vat_on_asset", "vat_deferral", "monthly_payment",
            "balloon_payment", "confidence_score", "reasoning",
        ],
        "field_types": {
            "loan_date": "date", "broker": "free_text", "proposer": "free_text",
            "proposer_address": "free_text", "proposer_trading_address": "free_text",
            "proposer_registered_address": "free_text",
            "proposer_email": "free_text", "proposer_phone": "free_text",
            "proposer_website": "free_text",
            "sic_code": "free_text", "crn": "free_text",
            "proposer_year_established": "categorical", "legal_form": "categorical",
            "loan_amount": "numeric", "currency": "categorical",
            "finance_type": "categorical", "regulated": "categorical",
            "rate_type": "categorical", "deal_type": "categorical",
            "term_type": "categorical", "loan_period": "numeric",
            "payment_frequency": "categorical", "vat_number": "free_text",
            "initial_payment": "numeric", "vat_on_asset": "numeric",
            "vat_deferral": "free_text", "monthly_payment": "numeric",
            "balloon_payment": "numeric", "confidence_score": "numeric",
            "reasoning": "free_text",
        },
        "primary_field": "proposer",
        "critical_fields": {"loan_amount", "finance_type", "proposer", "broker"},
    },

    "extract_corporate_party": {
        "prompt_type": "corporate_extraction",
        "task_type": "extraction",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "company_role", "company_name", "company_number",
            "street_address", "city_name", "postal_code", "country_name",
            "trading_street_address", "trading_city_name",
            "trading_postal_code", "trading_country_name",
            "registered_street_address", "registered_city_name",
            "registered_postal_code", "registered_country_name",
            "legal_form", "mob", "email", "reasoning",
        ],
        "field_types": {
            "company_role": "categorical", "company_name": "free_text",
            "company_number": "free_text",
            "street_address": "free_text", "city_name": "free_text",
            "postal_code": "free_text", "country_name": "categorical",
            "trading_street_address": "free_text", "trading_city_name": "free_text",
            "trading_postal_code": "free_text", "trading_country_name": "categorical",
            "registered_street_address": "free_text", "registered_city_name": "free_text",
            "registered_postal_code": "free_text", "registered_country_name": "categorical",
            "legal_form": "categorical", "mob": "free_text", "email": "free_text",
            "reasoning": "free_text",
        },
        "primary_field": "company_name",
        "critical_fields": {"company_name", "company_role"},
    },

    "extract_person_party": {
        "prompt_type": "person_extraction",
        "task_type": "extraction",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10
        """,
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "role", "first_name", "last_name",
            "company_name", "street_address", "city_name",
            "postal_code", "country_name",
            "mob", "email", "job_title", "date_of_birth", "reasoning",
        ],
        "field_types": {
            "role": "categorical", "first_name": "free_text",
            "last_name": "free_text", "company_name": "free_text",
            "street_address": "free_text", "city_name": "free_text",
            "postal_code": "free_text", "country_name": "categorical",
            "mob": "free_text", "email": "free_text",
            "job_title": "free_text", "date_of_birth": "date",
            "reasoning": "free_text",
        },
        "primary_field": "last_name",
        "critical_fields": {"first_name", "last_name", "role"},
    },
}

# Select which prompts to evaluate (can be a subset for faster iteration)
PROMPTS_TO_EVALUATE = list(PROMPT_CATALOGUE.keys())
# PROMPTS_TO_EVALUATE = ["extract_proposal"]  # uncomment to run just one

print(f"Prompts to evaluate: {PROMPTS_TO_EVALUATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Prompts from Unity Catalog
# MAGIC
# MAGIC Read prompt text, response format, and temperature from the `current_prompt` table.
# MAGIC The model endpoint is **overridden** per candidate — we don't use the table's endpoint.

# COMMAND ----------

# DBTITLE 1,Load prompt configurations
prompt_table = f"{catalog}.{slr_schema}.current_prompt"
prompt_configs = {}

for prompt_name in PROMPTS_TO_EVALUATE:
    ccfg = PROMPT_CATALOGUE[prompt_name]
    prompt_type = ccfg["prompt_type"]

    row = spark.sql(f"""
        SELECT PROMPT_ID, PROMPT_TYPE, MODEL_ENDPOINT, PROMPT, RESPONSE_FORMAT, TEMPERATURE
        FROM {prompt_table}
        WHERE PROMPT_TYPE = '{prompt_type}'
          AND CURRENT = true
    """).collect()

    assert len(row) > 0, f"No current prompt found for PROMPT_TYPE='{prompt_type}'"
    row = row[0]

    resp_fmt = json.loads(row["RESPONSE_FORMAT"]) if isinstance(row["RESPONSE_FORMAT"], str) else row["RESPONSE_FORMAT"]

    prompt_configs[prompt_name] = {
        "prompt_id": row["PROMPT_ID"],
        "prompt_text": row["PROMPT"],
        "response_format": resp_fmt,
        "temperature": float(row["TEMPERATURE"]),
        "production_endpoint": row["MODEL_ENDPOINT"],
    }
    print(f"  Loaded: {prompt_name} (prompt_id={row['PROMPT_ID']}, prod_endpoint={row['MODEL_ENDPOINT']})")

print(f"\nLoaded {len(prompt_configs)} prompt configurations.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Evaluation Documents
# MAGIC
# MAGIC Read documents from the `eval_snap_*` tables (populated by eval_00).
# MAGIC Limit to `MAX_DOCS` per prompt to keep runtime manageable.

# COMMAND ----------

# DBTITLE 1,Load documents per prompt
eval_docs = {}

for prompt_name in PROMPTS_TO_EVALUATE:
    ccfg = PROMPT_CATALOGUE[prompt_name]
    table = f"{catalog}.{slr_schema}.{ccfg['source_table']}"
    text_col = ccfg["text_column"]
    id_cols = ccfg["id_columns"]
    filt = ccfg["source_filter"]

    select_cols = ", ".join(id_cols + [text_col])
    query = f"SELECT {select_cols} FROM {table} WHERE {filt} LIMIT {MAX_DOCS}"

    docs = spark.sql(query).collect()
    eval_docs[prompt_name] = [r.asDict() for r in docs]
    print(f"  {prompt_name}: {len(eval_docs[prompt_name])} documents loaded")

print(f"\nTotal documents across all prompts: {sum(len(v) for v in eval_docs.values())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Utility Functions

# COMMAND ----------

# DBTITLE 1,Normalisation and consistency metrics
def _norm(val):
    """Normalise a value for fair comparison."""
    if val is None:
        return "__NULL__"
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match_rate(values: list) -> float:
    """Fraction of values that equal the mode (most common value)."""
    normed = [_norm(v) for v in values]
    if not normed:
        return 0.0
    mode_count = Counter(normed).most_common(1)[0][1]
    return mode_count / len(normed)


def modal_value(values: list):
    """Return the most common value (unnormalised)."""
    normed_to_raw = {}
    for v in values:
        n = _norm(v)
        if n not in normed_to_raw:
            normed_to_raw[n] = v
    normed = [_norm(v) for v in values]
    if not normed:
        return None
    mode = Counter(normed).most_common(1)[0][0]
    return normed_to_raw.get(mode)


def null_rate(values: list) -> float:
    """Fraction of values that are None/null."""
    nulls = sum(1 for v in values if v is None or str(v).strip().lower() in ("none", "", "null"))
    return nulls / len(values) if values else 0.0

print("Utility functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. LLM Call Function
# MAGIC
# MAGIC Calls `ai_query` for a single document using a specified model endpoint.
# MAGIC Uses the DataFrame pattern (not string interpolation) to avoid SQL escaping issues.

# COMMAND ----------

# DBTITLE 1,call_model_for_eval function
def call_model_for_eval(text_content: str, prompt_name: str, model_endpoint: str) -> dict:
    """
    Call a specific model endpoint for a single document.
    Returns dict of {field: value} or {field: None} on failure.
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
                '{model_endpoint}',
                CONCAT('{prompt_escaped}', '\\n\\nDocument text:\\n', doc_text),
                responseFormat => '{resp_fmt_str}',
                modelParameters => named_struct('temperature', {pcfg['temperature']})
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
        print(f"    ERROR [{model_endpoint}]: {e}")
        return {field: None for field in output_fields}

print("call_model_for_eval() defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. LLM-as-Judge Function
# MAGIC
# MAGIC Uses a separate judge model to score each extracted field against the source
# MAGIC document. Batches all fields per document into a single judge call.

# COMMAND ----------

# DBTITLE 1,Judge prompt template and function
JUDGE_PROMPT_TEMPLATE = """You are a quality reviewer for a UK loan document extraction system.

You are given:
1. A SOURCE DOCUMENT (the original text from a loan proposal)
2. A list of EXTRACTED FIELDS with their values (what the AI extracted)

Your task: Read the source document carefully and score EACH extracted field.

Scoring per field:
- 1.0 = CORRECT — the extracted value accurately reflects what is in the document
- 0.5 = PARTIALLY CORRECT — the core information is right but minor issues (formatting, abbreviations, missing detail)
- 0.0 = INCORRECT — the value is wrong, fabricated, or not found in the document

Rules:
- If a field genuinely has no value in the document and the extraction is null/None, score 1.0
- If a field has a value in the document but the extraction is null/None, score 0.0
- UK addresses: abbreviations like St/Street, Rd/Road are acceptable (1.0)
- Company names: Ltd/Limited, PLC/Public Limited Company are equivalent (1.0)
- Phone numbers: formatting differences (+44 vs 0) are acceptable (1.0)
- Monetary values: minor rounding differences are acceptable (1.0), wrong amounts are 0.0
- Read the ENTIRE document before scoring — relevant information may appear anywhere

EXTRACTED FIELDS TO JUDGE:
{fields_json}

SOURCE DOCUMENT:
{document_text}

Respond ONLY with a JSON array, one object per field, in the SAME ORDER as above:
[{{"field": "<field_name>", "score": <float>, "reasoning": "<one sentence>"}}]"""


def judge_document_batch(fields_with_values: list, document_text: str) -> list:
    """
    Judge all extracted fields for a single document in ONE LLM call.
    Returns list of dicts with {field, score, reasoning}.
    """
    try:
        doc_truncated = document_text[:6000] if document_text else ""

        fields_json = json.dumps(
            [{"field": f["field"], "extracted_value": str(f["value"]) if f["value"] else "null"}
             for f in fields_with_values],
            indent=2,
        )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            fields_json=fields_json,
            document_text=doc_truncated,
        )

        prompt_df = spark.createDataFrame([(prompt,)], ["judge_prompt"])

        result_df = prompt_df.selectExpr(f"""
            ai_query(
                '{JUDGE_MODEL}',
                judge_prompt,
                modelParameters => named_struct('temperature', 0.0)
            ) AS judge_response
        """)

        raw = result_df.collect()[0]["judge_response"]

        if isinstance(raw, str):
            # Try to parse the JSON array
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = re.sub(r"^```(?:json)?\n?", "", raw_clean)
                raw_clean = re.sub(r"\n?```$", "", raw_clean)
            parsed = json.loads(raw_clean)
        else:
            parsed = raw

        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "results" in parsed:
            return parsed["results"]
        else:
            return [{"field": f["field"], "score": 0.0, "reasoning": "Parse error"} for f in fields_with_values]

    except Exception as e:
        print(f"    JUDGE ERROR: {e}")
        return [{"field": f["field"], "score": 0.0, "reasoning": f"Judge error: {e}"} for f in fields_with_values]

print("Judge function defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Main Evaluation Loop
# MAGIC
# MAGIC For each model candidate × each prompt × each document:
# MAGIC 1. Run `ai_query` N times → compute consistency
# MAGIC 2. Take the modal value → run LLM-as-judge for accuracy
# MAGIC
# MAGIC Results accumulate in `all_consistency_results` and `all_accuracy_results`.

# COMMAND ----------

# DBTITLE 1,Run model selection evaluation
eval_run_id = f"model_sel_{datetime.now():%Y%m%d_%H%M%S}"
all_consistency_results = []
all_accuracy_results = []
model_timings = {}

total_models = len(MODEL_CANDIDATES)
total_prompts = len(PROMPTS_TO_EVALUATE)

print(f"Eval run: {eval_run_id}")
print(f"Models:   {total_models}")
print(f"Prompts:  {total_prompts}")
print(f"Runs/doc: {NUM_RUNS}")
print("=" * 80)

for model_idx, (model_label, model_endpoint) in enumerate(MODEL_CANDIDATES.items()):
    model_t0 = time.time()
    model_calls = 0
    model_errors = 0

    print(f"\n{'#' * 80}")
    print(f"MODEL [{model_idx+1}/{total_models}]: {model_label} ({model_endpoint})")
    print(f"{'#' * 80}")

    for prompt_name in PROMPTS_TO_EVALUATE:
        ccfg = PROMPT_CATALOGUE[prompt_name]
        docs = eval_docs[prompt_name]
        output_fields = ccfg["output_fields"]
        text_col = ccfg["text_column"]

        # Skip reasoning field from consistency/accuracy (it varies by design)
        eval_fields = [f for f in output_fields if f != "reasoning"]

        print(f"\n  ── {prompt_name} ({len(docs)} docs, {len(eval_fields)} fields) ──")

        for doc_idx, doc in enumerate(docs):
            doc_id = doc.get("DOCUMENT_ID", f"doc_{doc_idx}")
            file_name = doc.get("FILE_NAME", "unknown")
            text_content = doc.get(text_col, "")

            if doc_idx % 5 == 0:
                print(f"    [{doc_idx+1}/{len(docs)}] {file_name[:50]}...")

            # ── Step 1: N runs for consistency ──────────────────────────────
            run_results = []
            for run_i in range(NUM_RUNS):
                t0 = time.time()
                result = call_model_for_eval(text_content, prompt_name, model_endpoint)
                elapsed = time.time() - t0
                result["_run"] = run_i
                result["_elapsed_s"] = round(elapsed, 2)
                run_results.append(result)
                model_calls += 1

                # Count errors (all fields None)
                if all(result.get(f) is None for f in eval_fields):
                    model_errors += 1

            # ── Step 2: Compute per-field consistency ───────────────────────
            for field in eval_fields:
                values = [r.get(field) for r in run_results]
                emr = exact_match_rate(values)
                mv = modal_value(values)
                nr = null_rate(values)

                all_consistency_results.append({
                    "eval_run_id": eval_run_id,
                    "model_label": model_label,
                    "model_endpoint": model_endpoint,
                    "prompt_name": prompt_name,
                    "task_type": ccfg["task_type"],
                    "document_id": doc_id,
                    "file_name": file_name,
                    "field": field,
                    "field_type": ccfg["field_types"].get(field, "free_text"),
                    "is_critical": field in ccfg.get("critical_fields", set()),
                    "exact_match_rate": emr,
                    "modal_value": str(mv) if mv is not None else None,
                    "null_rate": nr,
                    "num_runs": NUM_RUNS,
                    "all_values": str([str(v)[:80] for v in values]),
                })

            # ── Step 3: Judge accuracy on modal values ──────────────────────
            fields_for_judge = []
            for field in eval_fields:
                values = [r.get(field) for r in run_results]
                mv = modal_value(values)
                fields_for_judge.append({"field": field, "value": mv})

            judge_results = judge_document_batch(fields_for_judge, text_content)

            for jr in judge_results:
                field = jr.get("field", "")
                all_accuracy_results.append({
                    "eval_run_id": eval_run_id,
                    "model_label": model_label,
                    "model_endpoint": model_endpoint,
                    "prompt_name": prompt_name,
                    "task_type": ccfg["task_type"],
                    "document_id": doc_id,
                    "file_name": file_name,
                    "field": field,
                    "field_type": ccfg["field_types"].get(field, "free_text"),
                    "is_critical": field in ccfg.get("critical_fields", set()),
                    "modal_value": str(dict((f["field"], f["value"]) for f in fields_for_judge).get(field, "")),
                    "judge_score": float(jr.get("score", 0.0)),
                    "judge_reasoning": jr.get("reasoning", ""),
                })

    model_elapsed = time.time() - model_t0
    model_timings[model_label] = {
        "elapsed_s": round(model_elapsed, 1),
        "total_calls": model_calls,
        "error_calls": model_errors,
        "error_rate": round(model_errors / max(model_calls, 1), 3),
    }
    print(f"\n  ✓ {model_label} complete: {model_calls} calls, {model_errors} errors, {model_elapsed:.0f}s")

print(f"\n{'=' * 80}")
print("All models complete.")
for label, t in model_timings.items():
    print(f"  {label:<20s}: {t['total_calls']} calls, {t['error_calls']} errors ({t['error_rate']:.1%}), {t['elapsed_s']:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Build DataFrames

# COMMAND ----------

# DBTITLE 1,Create pandas DataFrames from results
consistency_df = pd.DataFrame(all_consistency_results)
accuracy_df = pd.DataFrame(all_accuracy_results)

print(f"Consistency results: {len(consistency_df)} rows")
print(f"Accuracy results:    {len(accuracy_df)} rows")
print(f"\nModels:  {consistency_df['model_label'].nunique()}")
print(f"Prompts: {consistency_df['prompt_name'].nunique()}")
print(f"Docs:    {consistency_df['document_id'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model Comparison Scorecard
# MAGIC
# MAGIC Aggregate consistency and accuracy across all prompts per model.

# COMMAND ----------

# DBTITLE 1,Scorecard — overall model comparison
# ── Consistency scorecard ──
cons_scorecard = (
    consistency_df
    .groupby("model_label")
    .agg(
        mean_consistency=("exact_match_rate", "mean"),
        median_consistency=("exact_match_rate", "median"),
        fields_above_85=("exact_match_rate", lambda x: (x >= 0.85).mean()),
        mean_null_rate=("null_rate", "mean"),
    )
    .round(3)
)

# ── Accuracy scorecard ──
acc_scorecard = (
    accuracy_df
    .groupby("model_label")
    .agg(
        mean_accuracy=("judge_score", "mean"),
        median_accuracy=("judge_score", "median"),
        fields_perfect=("judge_score", lambda x: (x == 1.0).mean()),
        fields_zero=("judge_score", lambda x: (x == 0.0).mean()),
    )
    .round(3)
)

# ── Combined scorecard ──
scorecard = cons_scorecard.join(acc_scorecard, how="outer")

# Add timing info
timing_df = pd.DataFrame(model_timings).T
timing_df.index.name = "model_label"
scorecard = scorecard.join(timing_df[["elapsed_s", "error_rate"]], how="left")

# Composite score: weighted average of consistency + accuracy
scorecard["composite_score"] = (
    0.4 * scorecard["mean_consistency"] +
    0.5 * scorecard["mean_accuracy"] +
    0.1 * (1 - scorecard["mean_null_rate"])
).round(3)

scorecard = scorecard.sort_values("composite_score", ascending=False)

print("=" * 90)
print("MODEL SELECTION SCORECARD")
print("=" * 90)
print(scorecard.to_string())
print()
print("Composite = 0.4 × consistency + 0.5 × accuracy + 0.1 × (1 − null_rate)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Per-Prompt Breakdown

# COMMAND ----------

# DBTITLE 1,Scorecard per model × prompt
cons_by_prompt = (
    consistency_df
    .groupby(["model_label", "prompt_name"])
    .agg(
        mean_consistency=("exact_match_rate", "mean"),
        mean_null_rate=("null_rate", "mean"),
    )
    .round(3)
)

acc_by_prompt = (
    accuracy_df
    .groupby(["model_label", "prompt_name"])
    .agg(
        mean_accuracy=("judge_score", "mean"),
    )
    .round(3)
)

prompt_breakdown = cons_by_prompt.join(acc_by_prompt, how="outer").reset_index()

print("Per-prompt breakdown:")
for pn in PROMPTS_TO_EVALUATE:
    subset = prompt_breakdown[prompt_breakdown["prompt_name"] == pn].sort_values("mean_accuracy", ascending=False)
    print(f"\n  ── {pn} ──")
    print(subset[["model_label", "mean_consistency", "mean_null_rate", "mean_accuracy"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Critical Fields Analysis
# MAGIC
# MAGIC Focus on fields marked as `critical` in the prompt catalogue — these are the
# MAGIC fields that matter most for downstream Salesforce integration.

# COMMAND ----------

# DBTITLE 1,Critical fields comparison
critical_cons = consistency_df[consistency_df["is_critical"]].copy()
critical_acc = accuracy_df[accuracy_df["is_critical"]].copy()

if len(critical_cons) > 0:
    crit_scorecard = (
        critical_cons
        .groupby("model_label")
        .agg(critical_consistency=("exact_match_rate", "mean"))
        .round(3)
    )

    crit_acc = (
        critical_acc
        .groupby("model_label")
        .agg(critical_accuracy=("judge_score", "mean"))
        .round(3)
    )

    crit_combined = crit_scorecard.join(crit_acc, how="outer").sort_values("critical_accuracy", ascending=False)

    print("CRITICAL FIELDS SCORECARD")
    print("=" * 60)
    print(crit_combined.to_string())
    print()

    # Show per-field detail for critical fields
    crit_detail = (
        critical_acc
        .groupby(["model_label", "prompt_name", "field"])
        .agg(mean_score=("judge_score", "mean"))
        .round(3)
        .reset_index()
        .sort_values(["prompt_name", "field", "mean_score"], ascending=[True, True, False])
    )
    print("\nCritical field detail:")
    print(crit_detail.to_string(index=False))
else:
    print("No critical fields found in results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Visualisations

# COMMAND ----------

# DBTITLE 1,Heatmap — consistency by model × prompt
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Consistency heatmap
pivot_cons = prompt_breakdown.pivot(index="model_label", columns="prompt_name", values="mean_consistency")
sns.heatmap(pivot_cons, annot=True, fmt=".2f", cmap="YlGn", vmin=0.5, vmax=1.0, ax=axes[0])
axes[0].set_title("Mean Consistency by Model × Prompt", fontsize=13, fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# Accuracy heatmap
pivot_acc = prompt_breakdown.pivot(index="model_label", columns="prompt_name", values="mean_accuracy")
sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="YlGn", vmin=0.5, vmax=1.0, ax=axes[1])
axes[1].set_title("Mean Accuracy (Judge Score) by Model × Prompt", fontsize=13, fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Bar chart — composite scores
fig, ax = plt.subplots(figsize=(10, 5))

sc = scorecard.reset_index().sort_values("composite_score", ascending=True)
colors = sns.color_palette("viridis", len(sc))
bars = ax.barh(sc["model_label"], sc["composite_score"], color=colors)

# Add value labels
for bar, val in zip(bars, sc["composite_score"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontweight="bold")

ax.set_xlabel("Composite Score (0.4×consistency + 0.5×accuracy + 0.1×(1−null_rate))")
ax.set_title("Model Selection — Composite Score Ranking", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.1)
ax.axvline(x=0.85, color="red", linestyle="--", alpha=0.5, label="Target (0.85)")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Scatter — consistency vs accuracy per model
fig, ax = plt.subplots(figsize=(8, 6))

for model_label in scorecard.index:
    model_cons = consistency_df[consistency_df["model_label"] == model_label]
    model_acc = accuracy_df[accuracy_df["model_label"] == model_label]

    # Merge on document+field to get paired values
    merged = (
        model_cons[["document_id", "field", "prompt_name", "exact_match_rate"]]
        .merge(
            model_acc[["document_id", "field", "prompt_name", "judge_score"]],
            on=["document_id", "field", "prompt_name"],
            how="inner",
        )
    )

    if len(merged) > 0:
        ax.scatter(
            merged["exact_match_rate"].mean(),
            merged["judge_score"].mean(),
            s=150, label=model_label, zorder=3,
        )

ax.set_xlabel("Mean Consistency (Exact Match Rate)", fontsize=12)
ax.set_ylabel("Mean Accuracy (Judge Score)", fontsize=12)
ax.set_title("Consistency vs Accuracy by Model", fontsize=14, fontweight="bold")
ax.set_xlim(0.4, 1.05)
ax.set_ylim(0.4, 1.05)
ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.3)
ax.axvline(x=0.85, color="red", linestyle="--", alpha=0.3)
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Latency comparison
fig, ax = plt.subplots(figsize=(10, 5))

timing_data = pd.DataFrame(model_timings).T.reset_index()
timing_data.columns = ["model_label", "elapsed_s", "total_calls", "error_calls", "error_rate"]
timing_data["avg_call_s"] = timing_data["elapsed_s"] / timing_data["total_calls"]
timing_data = timing_data.sort_values("avg_call_s", ascending=True)

bars = ax.barh(timing_data["model_label"], timing_data["avg_call_s"], color=sns.color_palette("muted", len(timing_data)))
for bar, err in zip(bars, timing_data["error_rate"]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"err: {err:.1%}", va="center", fontsize=9, color="red" if err > 0.05 else "grey")

ax.set_xlabel("Average seconds per ai_query call")
ax.set_title("Model Latency Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Persist Results to Delta

# COMMAND ----------

# DBTITLE 1,Write consistency results to Delta
cons_spark_df = spark.createDataFrame(consistency_df)
cons_table = f"{catalog}.{eval_schema}.model_selection_consistency"

cons_spark_df.write.format("delta").mode("append").saveAsTable(cons_table)
print(f"Wrote {len(consistency_df)} consistency rows to {cons_table}")

# COMMAND ----------

# DBTITLE 1,Write accuracy results to Delta
acc_spark_df = spark.createDataFrame(accuracy_df)
acc_table = f"{catalog}.{eval_schema}.model_selection_accuracy"

acc_spark_df.write.format("delta").mode("append").saveAsTable(acc_table)
print(f"Wrote {len(accuracy_df)} accuracy rows to {acc_table}")

# COMMAND ----------

# DBTITLE 1,Write scorecard to Delta
scorecard_out = scorecard.reset_index()
scorecard_out["eval_run_id"] = eval_run_id
scorecard_out["eval_timestamp"] = datetime.now().isoformat()
scorecard_out["num_runs"] = NUM_RUNS
scorecard_out["max_docs"] = MAX_DOCS

sc_spark_df = spark.createDataFrame(scorecard_out)
sc_table = f"{catalog}.{eval_schema}.model_selection_scorecard"

sc_spark_df.write.format("delta").mode("append").saveAsTable(sc_table)
print(f"Wrote scorecard to {sc_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Log to MLflow

# COMMAND ----------

# DBTITLE 1,MLflow experiment logging
experiment_name = f"/Shared/eval_model_selection"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=eval_run_id) as parent_run:
    # Log top-level params
    mlflow.log_param("eval_run_id", eval_run_id)
    mlflow.log_param("num_runs", NUM_RUNS)
    mlflow.log_param("max_docs", MAX_DOCS)
    mlflow.log_param("num_models", len(MODEL_CANDIDATES))
    mlflow.log_param("prompts_evaluated", ",".join(PROMPTS_TO_EVALUATE))
    mlflow.log_param("judge_model", JUDGE_MODEL)

    # Log overall winner
    winner = scorecard.index[0]
    mlflow.log_param("recommended_model", winner)
    mlflow.log_metric("winner_composite_score", scorecard.loc[winner, "composite_score"])

    # Child run per model
    for model_label in scorecard.index:
        with mlflow.start_run(run_name=f"{eval_run_id}__{model_label}", nested=True):
            row = scorecard.loc[model_label]
            mlflow.log_param("model_label", model_label)
            mlflow.log_param("model_endpoint", MODEL_CANDIDATES[model_label])

            mlflow.log_metric("mean_consistency", row["mean_consistency"])
            mlflow.log_metric("median_consistency", row["median_consistency"])
            mlflow.log_metric("fields_above_85pct", row["fields_above_85"])
            mlflow.log_metric("mean_null_rate", row["mean_null_rate"])
            mlflow.log_metric("mean_accuracy", row["mean_accuracy"])
            mlflow.log_metric("median_accuracy", row["median_accuracy"])
            mlflow.log_metric("fields_perfect", row["fields_perfect"])
            mlflow.log_metric("fields_zero", row["fields_zero"])
            mlflow.log_metric("composite_score", row["composite_score"])
            mlflow.log_metric("elapsed_s", row["elapsed_s"])
            mlflow.log_metric("error_rate", row["error_rate"])

            # Per-prompt metrics
            for pn in PROMPTS_TO_EVALUATE:
                pn_data = prompt_breakdown[
                    (prompt_breakdown["model_label"] == model_label) &
                    (prompt_breakdown["prompt_name"] == pn)
                ]
                if len(pn_data) > 0:
                    pn_row = pn_data.iloc[0]
                    mlflow.log_metric(f"{pn}__consistency", pn_row["mean_consistency"])
                    mlflow.log_metric(f"{pn}__accuracy", pn_row["mean_accuracy"])

    print(f"MLflow experiment: {experiment_name}")
    print(f"Parent run: {parent_run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Recommendation Summary

# COMMAND ----------

# DBTITLE 1,Final recommendation
print("=" * 80)
print("MODEL SELECTION RECOMMENDATION")
print("=" * 80)
print()

winner = scorecard.index[0]
runner_up = scorecard.index[1] if len(scorecard) > 1 else None

print(f"  RECOMMENDED MODEL: {winner}")
print(f"    Endpoint:     {MODEL_CANDIDATES[winner]}")
print(f"    Composite:    {scorecard.loc[winner, 'composite_score']:.3f}")
print(f"    Consistency:  {scorecard.loc[winner, 'mean_consistency']:.3f}")
print(f"    Accuracy:     {scorecard.loc[winner, 'mean_accuracy']:.3f}")
print(f"    Null rate:    {scorecard.loc[winner, 'mean_null_rate']:.3f}")
print(f"    Error rate:   {scorecard.loc[winner, 'error_rate']:.1%}")
print(f"    Runtime:      {scorecard.loc[winner, 'elapsed_s']:.0f}s")

if runner_up:
    gap = scorecard.loc[winner, "composite_score"] - scorecard.loc[runner_up, "composite_score"]
    print(f"\n  RUNNER UP: {runner_up}")
    print(f"    Composite:    {scorecard.loc[runner_up, 'composite_score']:.3f}")
    print(f"    Gap to winner: {gap:.3f}")

    if gap < 0.02:
        print(f"\n  ⚠ The gap between {winner} and {runner_up} is very small ({gap:.3f}).")
        print("    Consider also evaluating latency, cost, and error rates before deciding.")

# Check if any critical fields fall below threshold for the winner
if len(critical_acc) > 0:
    winner_critical = critical_acc[critical_acc["model_label"] == winner]
    weak_critical = winner_critical.groupby("field")["judge_score"].mean()
    weak = weak_critical[weak_critical < 0.85]
    if len(weak) > 0:
        print(f"\n  ⚠ Warning — {winner} has weak critical fields:")
        for field, score in weak.items():
            print(f"    {field}: {score:.3f}")

print()
print(f"Full results persisted to {catalog}.{eval_schema}.model_selection_*")
print(f"MLflow experiment: {experiment_name}")
