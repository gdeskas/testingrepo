# Databricks notebook source

# MAGIC %md
# MAGIC # Unified Accuracy Evaluation — LLM-as-Judge
# MAGIC
# MAGIC Evaluates accuracy of **all 5 LLM/VLM prompts** against a golden dataset.
# MAGIC No manual `is_correct` flagging — every field is scored automatically.
# MAGIC
# MAGIC | # | Prompt | Comparison strategy |
# MAGIC |---|--------|-------------------|
# MAGIC | 1 | `categorise_image` | Categorical (exact match) |
# MAGIC | 2 | `categorise_other_doc_type` | Categorical (exact match) |
# MAGIC | 3 | `classify_asset` | Categorical + LLM judge for descriptions |
# MAGIC | 4 | `extract_account_proposal_information` | Mixed: categorical, numeric (±1%), date, LLM judge |
# MAGIC | 5 | `extract_corporate_party_identify_role` | LLM judge for names/addresses, categorical for roles |
# MAGIC | 6 | `extract_person_party_identify_role` | LLM judge for names/contacts, categorical for roles |
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. Read production outputs from Silver Delta tables (all proposals, one read)
# MAGIC 2. Load golden dataset expected values from the evaluation schema
# MAGIC 3. For each field, route to the right scorer: deterministic or LLM judge
# MAGIC 4. Write scores to Delta + MLflow — no manual review needed

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

# MAGIC %md
# MAGIC ## Prompt Catalogue with Field Type Classification
# MAGIC
# MAGIC Each prompt's fields are classified as `categorical`, `numeric`, `date`, or
# MAGIC `free_text`. This determines which comparison function is used:
# MAGIC - **categorical** → normalised exact match (no LLM needed)
# MAGIC - **numeric** → tolerance-based ±1% (no LLM needed)
# MAGIC - **date** → multi-format parse + compare (no LLM needed)
# MAGIC - **free_text** → LLM-as-judge via `ai_query`

# COMMAND ----------

# DBTITLE 1,Prompt catalogue with field types
PROMPT_CATALOGUE = {
    # ── 1. Image categorisation (VLM) ───────────────────────────────────────
    "categorise_image": {
        "prompt_type": "image_categorisation",
        "source_table": "eval_snap_img_category",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
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

    # ── 2. Document categorisation ──────────────────────────────────────────
    "categorise_doc": {
        "prompt_type": "document_categorisation",
        "source_table": "eval_snap_doc_category",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
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

    # ── 3. Asset classification ─────────────────────────────────────────────
    "classify_asset": {
        "prompt_type": "asset_classification",
        "source_table": "eval_snap_doc_category",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
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

    # ── 4. Proposal extraction ──────────────────────────────────────────────
    "extract_proposal": {
        "prompt_type": "extract_account_proposal_information",
        "source_table": "eval_snap_proposals_extract",
        "id_columns": ["PROPOSAL_ID"],
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
            "broker": "free_text",
            "proposer": "free_text",
            "proposer_address": "free_text",
            "proposer_trading_address": "free_text",
            "proposer_registered_address": "free_text",
            "proposer_email": "free_text",
            "proposer_phone": "free_text",
            "proposer_website": "free_text",
            "sic_code": "categorical",
            "crn": "free_text",
            "proposer_year_established": "categorical",
            "legal_form": "categorical",
            "loan_amount": "numeric",
            "currency": "categorical",
            "finance_type": "categorical",
            "regulated": "categorical",
            "rate_type": "categorical",
            "deal_type": "categorical",
            "term_type": "categorical",
            "loan_period": "numeric",
            "payment_frequency": "categorical",
            "vat_number": "free_text",
            "initial_payment": "numeric",
            "vat_payment": "numeric",
            "vat_deferral": "numeric",
            "monthly_payment": "numeric",
            "balloon_payment": "numeric",
            "confidence_score": "numeric",
        },
        "critical_fields": {
            "loan_amount", "finance_type", "loan_period", "deal_type",
            "rate_type", "monthly_payment", "balloon_payment", "initial_payment",
        },
    },

    # ── 5. Corporate party extraction ───────────────────────────────────────
    "extract_corporate_party": {
        "prompt_type": "corporate_party_extraction",
        "source_table": "eval_snap_doc_category",  # adjust if you have a dedicated output table
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "company_role", "company_name", "company_number",
            "street_name", "city_name", "postal_code", "country_name",
            "trading_address", "registered_address",
            "mob", "email", "reasoning",
        ],
        "field_types": {
            "company_role": "categorical",
            "company_name": "free_text",
            "company_number": "free_text",
            "street_name": "free_text",
            "city_name": "free_text",
            "postal_code": "free_text",
            "country_name": "categorical",
            "trading_address": "free_text",
            "registered_address": "free_text",
            "mob": "free_text",
            "email": "free_text",
            "reasoning": "free_text",
        },
        "critical_fields": {"company_name", "company_role"},
    },

    # ── 6. Person party extraction ──────────────────────────────────────────
    "extract_person_party": {
        "prompt_type": "person_party_extraction",
        "source_table": "eval_snap_doc_category",  # adjust if you have a dedicated output table
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID"],
        "output_fields": [
            "role", "salutation", "first_name", "last_name",
            "company_name", "gender", "date_of_birth",
            "country_code", "mob", "email", "job_title",
            "street_address", "city_name", "postal_code", "country_name",
            "reasoning",
        ],
        "field_types": {
            "role": "categorical",
            "salutation": "categorical",
            "first_name": "free_text",
            "last_name": "free_text",
            "company_name": "free_text",
            "gender": "categorical",
            "date_of_birth": "date",
            "country_code": "categorical",
            "mob": "free_text",
            "email": "free_text",
            "job_title": "free_text",
            "street_address": "free_text",
            "city_name": "free_text",
            "postal_code": "free_text",
            "country_name": "categorical",
            "reasoning": "free_text",
        },
        "critical_fields": {"first_name", "last_name", "role"},
    },
}

CRITICAL_THRESHOLD = 0.90
SUPPORTING_THRESHOLD = 0.75

print(f"Prompt catalogue: {len(PROMPT_CATALOGUE)} prompts")
for name, cfg in PROMPT_CATALOGUE.items():
    n_fields = len(cfg["output_fields"])
    n_free = sum(1 for f in cfg["output_fields"] if cfg["field_types"].get(f) == "free_text")
    n_crit = len(cfg.get("critical_fields", set()))
    print(f"  {name:30s}  {n_fields} fields ({n_free} LLM-judged, {n_crit} critical)")

# COMMAND ----------

# DBTITLE 1,Select prompts to evaluate
PROMPTS_TO_EVALUATE = list(PROMPT_CATALOGUE.keys())
# PROMPTS_TO_EVALUATE = ["extract_proposal"]  # uncomment for single prompt

print(f"Will evaluate: {PROMPTS_TO_EVALUATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deterministic Comparison Functions

# COMMAND ----------

# DBTITLE 1,Comparison functions (categorical, numeric, date)
def _norm(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _is_null(val):
    return val is None or _norm(val) in ("null", "none", "", "n/a", "na")


def compare_categorical(extracted, expected):
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    if _norm(extracted) == _norm(expected):
        return 1.0, "Exact match"
    return 0.0, f"Mismatch: '{extracted}' vs '{expected}'"


def compare_numeric(extracted, expected, tolerance=0.01):
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    try:
        ext_num = float(re.sub(r"[£$€,\s]", "", str(extracted)))
        exp_num = float(re.sub(r"[£$€,\s]", "", str(expected)))
    except (ValueError, TypeError):
        if _norm(extracted) == _norm(expected):
            return 1.0, "Non-numeric but string-equal"
        return 0.0, f"Cannot parse: '{extracted}' vs '{expected}'"
    if exp_num == 0:
        return (1.0, "Both zero") if ext_num == 0 else (0.0, f"Expected 0, got {ext_num}")
    pct_diff = abs(ext_num - exp_num) / abs(exp_num)
    if pct_diff <= tolerance:
        return 1.0, f"Within ±{tolerance:.0%} ({pct_diff:.2%})"
    if pct_diff <= tolerance * 5:
        return 0.5, f"Close ({pct_diff:.2%})"
    return 0.0, f"Outside tolerance ({pct_diff:.2%}): {ext_num} vs {exp_num}"


def compare_date(extracted, expected):
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y",
        "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%d.%m.%Y",
        "%Y%m%d", "%d/%m/%y", "%d-%m-%y",
    ]
    def _parse(val):
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(str(val).strip(), fmt)
            except ValueError:
                continue
        return None
    ext_d, exp_d = _parse(extracted), _parse(expected)
    if ext_d is None or exp_d is None:
        if _norm(extracted) == _norm(expected):
            return 1.0, "Unparseable but string-equal"
        return 0.0, f"Cannot parse: '{extracted}' vs '{expected}'"
    if ext_d == exp_d:
        return 1.0, "Dates match"
    diff = abs((ext_d - exp_d).days)
    if diff <= 1:
        return 0.5, f"Off by {diff} day(s)"
    return 0.0, f"Differ by {diff} days"


print("Deterministic comparison functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM-as-Judge

# COMMAND ----------

# DBTITLE 1,Load judge model config
prompt_table = f"`{catalog}`.`{slr_schema}`.current_prompt"

# Use the proposal extraction model as the judge (or override below)
judge_row = spark.sql(f"""
    SELECT MODEL_ENDPOINT FROM {prompt_table}
    WHERE PROMPT_TYPE = 'extract_account_proposal_information' AND CURRENT = true
    LIMIT 1
""").collect()

JUDGE_MODEL = judge_row[0]["MODEL_ENDPOINT"] if judge_row else "databricks-gpt-oss-20b"
print(f"Judge model: {JUDGE_MODEL}")

# COMMAND ----------

# DBTITLE 1,LLM judge prompt and function
JUDGE_PROMPT = """You are an accuracy judge for a UK loan document extraction system.
Compare an EXTRACTED value against the EXPECTED ground-truth value.

Field name: {field_name}
Field context: {field_context}

EXTRACTED: {extracted}
EXPECTED:  {expected}

Scoring:
- 1.0 = Semantically identical (abbreviations, formatting, equivalent info)
- 0.5 = Partially correct (core info present, missing/extra detail)
- 0.0 = Incorrect or unrelated

UK conventions: Ltd=Limited, PLC=Public Limited Company, St=Street, Rd=Road.
Phone: ignore formatting (+44 vs 0). Email: case-insensitive. CRN/VAT: ignore spaces.
Both null = 1.0. One null, other has value = 0.0.

Respond ONLY with JSON: {{"score": <float>, "reasoning": "<one sentence>"}}"""

FIELD_CONTEXT = {
    "broker": "Broker company name", "proposer": "Loan applicant company",
    "proposer_address": "UK business address", "proposer_trading_address": "UK trading address",
    "proposer_registered_address": "UK registered address",
    "proposer_email": "Email address", "proposer_phone": "UK phone number",
    "proposer_website": "Website URL", "crn": "UK Company Registration Number",
    "vat_number": "UK VAT number", "company_name": "Corporate entity name",
    "company_number": "Company registration number",
    "first_name": "Person's first name", "last_name": "Person's surname",
    "street_name": "UK street address", "street_address": "UK street address",
    "city_name": "UK city/town", "postal_code": "UK postcode",
    "trading_address": "Full trading address", "registered_address": "Full registered address",
    "mob": "Mobile phone number", "email": "Email address",
    "job_title": "Professional title/role",
    "image_description": "Description of document image",
    "asset_description": "Description of financed asset",
    "manufacturer": "Asset manufacturer name", "model": "Asset model name",
    "registration": "Vehicle/asset registration",
    "bank_statement_period": "Bank statement date period",
    "bank_statement_bank_name": "Bank name on statement",
    "text": "Text extracted from document",
}


def judge_free_text(field_name: str, extracted: str, expected: str) -> tuple:
    """LLM-as-judge for free-text fields via ai_query."""
    if _is_null(extracted) and _is_null(expected):
        return 1.0, "Both null"
    if _is_null(extracted) or _is_null(expected):
        return 0.0, f"One null: '{extracted}' vs '{expected}'"
    if _norm(extracted) == _norm(expected):
        return 1.0, "Exact match after normalisation"

    context = FIELD_CONTEXT.get(field_name, f"Field: {field_name}")
    prompt = JUDGE_PROMPT.format(
        field_name=field_name, field_context=context,
        extracted=str(extracted), expected=str(expected),
    )

    try:
        escaped = prompt.replace("'", "\\'")
        resp_fmt = json.dumps({
            "type": "json_schema",
            "json_schema": {
                "name": "judge_response", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False,
                },
            },
        }).replace("'", "\\'")

        result_df = spark.sql(f"""
            SELECT ai_query('{JUDGE_MODEL}', '{escaped}',
                            responseFormat => '{resp_fmt}') AS resp
        """)
        raw = result_df.collect()[0]["resp"]
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        return score, str(parsed.get("reasoning", ""))
    except Exception as e:
        # Fallback to string comparison
        if _norm(extracted) == _norm(expected):
            return 1.0, f"Judge failed, fallback match: {e}"
        return 0.0, f"Judge failed: {e}"


print("LLM judge function defined.")

# COMMAND ----------

# DBTITLE 1,Unified field scorer
def score_field(field_name: str, extracted, expected, field_type: str) -> dict:
    """Route to appropriate comparison and return score dict."""
    if field_type == "categorical":
        score, reasoning = compare_categorical(extracted, expected)
        method = "deterministic_categorical"
    elif field_type == "numeric":
        score, reasoning = compare_numeric(extracted, expected)
        method = "deterministic_numeric"
    elif field_type == "date":
        score, reasoning = compare_date(extracted, expected)
        method = "deterministic_date"
    else:
        score, reasoning = judge_free_text(field_name, extracted, expected)
        method = "llm_judge"

    return {
        "field": field_name,
        "extracted_value": str(extracted) if extracted is not None else None,
        "expected_value": str(expected) if expected is not None else None,
        "score": score,
        "reasoning": reasoning,
        "comparison_method": method,
        "is_correct": score >= 0.5,
        "field_type": field_type,
    }


print("score_field() defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Golden Dataset

# COMMAND ----------

# DBTITLE 1,Load golden dataset (flat table: sample_id, prompt_type, field, expected_value)
golden_table = f"`{catalog}`.`{slr_schema}`.golden_dataset"

try:
    golden_raw = spark.sql(f"""
        SELECT sample_id, prompt_type, field, expected_value
        FROM {golden_table}
        WHERE expected_value IS NOT NULL
    """).toPandas()

    # Build nested lookup: {prompt_type: {sample_id: {field: value}}}
    golden_lookup = {}
    for _, row in golden_raw.iterrows():
        pt = row["prompt_type"]
        sid = row["sample_id"]
        if pt not in golden_lookup:
            golden_lookup[pt] = {}
        if sid not in golden_lookup[pt]:
            golden_lookup[pt][sid] = {}
        golden_lookup[pt][sid][row["field"]] = row["expected_value"]

    total_samples = sum(len(v) for v in golden_lookup.values())
    print(f"Golden dataset loaded: {total_samples} samples across {len(golden_lookup)} prompt types")
    for pt, samples in golden_lookup.items():
        print(f"  {pt:45s} {len(samples)} samples")

except Exception as e:
    print(f"Golden dataset table not found or error: {e}")
    print("Define expected values manually in the next cell.")
    golden_lookup = {}

# COMMAND ----------

# DBTITLE 1,Manual golden dataset override (optional)
# Uncomment and populate to define expected values without a Delta table.
# Structure: {prompt_type: {sample_id: {field: expected_value}}}

# golden_lookup["extract_account_proposal_information"] = {
#     "proposal_abc123": {
#         "broker": "Smith & Partners Finance",
#         "proposer": "Acme Manufacturing Ltd",
#         "loan_amount": "250000.00",
#         "finance_type": "Hire Purchase",
#     },
# }

if not golden_lookup:
    print("WARNING: No golden dataset loaded. Accuracy evaluation requires expected values.")
    print("Either create the golden_dataset table or populate the dict above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Production Outputs from Silver Tables

# COMMAND ----------

# DBTITLE 1,Load extracted outputs for all prompts
production_outputs = {}

for prompt_name in PROMPTS_TO_EVALUATE:
    cfg = PROMPT_CATALOGUE[prompt_name]
    table_name = cfg["source_table"]
    full_table = f"`{catalog}`.`{slr_schema}`.{table_name}"
    id_cols = cfg["id_columns"]
    output_fields = cfg["output_fields"]

    try:
        # Select ID columns + output fields that exist in the table
        available_cols = [c.name for c in spark.table(full_table).schema.fields]

        # Only select output fields that actually exist in the table
        select_fields = []
        for f in output_fields:
            # Handle case where field might be nested in a PARSED_RESPONSE struct
            if f in available_cols:
                select_fields.append(f)
            elif f"PARSED_RESPONSE.{f}" in available_cols or "PARSED_RESPONSE" in available_cols:
                select_fields.append(f"PARSED_RESPONSE.{f} AS {f}")

        all_select = id_cols + select_fields
        select_str = ", ".join(all_select)

        df = spark.sql(f"SELECT {select_str} FROM {full_table}")
        rows = [row.asDict() for row in df.collect()]

        production_outputs[prompt_name] = rows
        print(f"  {prompt_name:30s} → {len(rows)} rows from {table_name}")

    except Exception as e:
        print(f"  {prompt_name:30s} → ERROR: {e}")
        production_outputs[prompt_name] = []

print(f"\nTotal: {sum(len(v) for v in production_outputs.values())} output rows loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Accuracy Evaluation

# COMMAND ----------

# DBTITLE 1,Score all fields across all prompts
eval_run_id = f"accuracy_{datetime.now():%Y%m%d_%H%M%S}"
all_scores = []
llm_judge_calls = 0
deterministic_calls = 0

print(f"Eval run: {eval_run_id}")
print("=" * 70)

for prompt_name in PROMPTS_TO_EVALUATE:
    cfg = PROMPT_CATALOGUE[prompt_name]
    pt = cfg["prompt_type"]
    outputs = production_outputs.get(prompt_name, [])
    expected_by_sample = golden_lookup.get(pt, {})

    if not outputs:
        print(f"\n[{prompt_name}] No outputs — skipping.")
        continue

    if not expected_by_sample:
        print(f"\n[{prompt_name}] No golden dataset for prompt_type='{pt}' — skipping.")
        continue

    matched = 0
    print(f"\n[{prompt_name}] {len(outputs)} outputs, {len(expected_by_sample)} golden samples")

    for row in outputs:
        # Try to match by PROPOSAL_ID (or first ID column)
        row_id = row.get("PROPOSAL_ID", row.get("DOCUMENT_ID", ""))
        expected = expected_by_sample.get(row_id, {})

        if not expected:
            # Try partial matching
            for gid, gvals in expected_by_sample.items():
                if gid in str(row_id) or str(row_id) in gid:
                    expected = gvals
                    break

        if not expected:
            continue

        matched += 1

        for field in cfg["output_fields"]:
            extracted_val = row.get(field)
            expected_val = expected.get(field)

            # Skip fields not in golden dataset
            if expected_val is None and field not in expected:
                continue

            field_type = cfg["field_types"].get(field, "free_text")
            result = score_field(field, extracted_val, expected_val, field_type)
            result["eval_run_id"] = eval_run_id
            result["prompt_name"] = prompt_name
            result["sample_id"] = row_id
            result["is_critical"] = field in cfg.get("critical_fields", set())

            if result["comparison_method"] == "llm_judge":
                llm_judge_calls += 1
            else:
                deterministic_calls += 1

            all_scores.append(result)

    print(f"  Matched {matched} samples, scored {sum(1 for s in all_scores if s['prompt_name'] == prompt_name)} fields")

print(f"\nTotal: {len(all_scores)} field scores ({llm_judge_calls} LLM judge, {deterministic_calls} deterministic)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results Analysis

# COMMAND ----------

# DBTITLE 1,Build results
if not all_scores:
    print("No scores generated. Check golden dataset sample IDs match production PROPOSAL_IDs.")
    dbutils.notebook.exit("No scores")

scores_df = pd.DataFrame(all_scores)
print(f"Total scores: {len(scores_df)}")
print(f"Overall accuracy: {scores_df['score'].mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Per-prompt accuracy
prompt_accuracy = (
    scores_df.groupby("prompt_name")
    .agg(
        mean_score=("score", "mean"),
        pct_correct=("is_correct", "mean"),
        fields_scored=("score", "count"),
        llm_judged=("comparison_method", lambda x: (x == "llm_judge").sum()),
    )
    .sort_values("mean_score")
    .reset_index()
)

print("\nPER-PROMPT ACCURACY")
print("=" * 80)
for _, row in prompt_accuracy.iterrows():
    print(f"  {row['prompt_name']:30s}  score={row['mean_score']:.2%}  "
          f"correct={row['pct_correct']:.2%}  n={row['fields_scored']:.0f}  "
          f"(LLM: {row['llm_judged']:.0f})")

display(spark.createDataFrame(prompt_accuracy))

# COMMAND ----------

# DBTITLE 1,Per-field accuracy across all prompts
field_accuracy = (
    scores_df.groupby(["prompt_name", "field"])
    .agg(
        mean_score=("score", "mean"),
        pct_correct=("is_correct", "mean"),
        count=("score", "count"),
        is_critical=("is_critical", "first"),
        field_type=("field_type", "first"),
        comparison_method=("comparison_method", "first"),
    )
    .sort_values("mean_score")
    .reset_index()
)

# Add threshold
field_accuracy["threshold"] = field_accuracy["is_critical"].map(
    {True: CRITICAL_THRESHOLD, False: SUPPORTING_THRESHOLD}
)
field_accuracy["meets_threshold"] = field_accuracy["mean_score"] >= field_accuracy["threshold"]

failing = field_accuracy[~field_accuracy["meets_threshold"]]
passing = field_accuracy[field_accuracy["meets_threshold"]]
print(f"\nFields meeting threshold: {len(passing)}/{len(field_accuracy)}")
if len(failing) > 0:
    print(f"\nFAILING ({len(failing)}):")
    for _, row in failing.iterrows():
        crit = "CRITICAL" if row["is_critical"] else "        "
        print(f"  {crit} {row['prompt_name']:25s} / {row['field']:30s}  {row['mean_score']:.2%}")

display(spark.createDataFrame(field_accuracy.head(30)))

# COMMAND ----------

# DBTITLE 1,Worst-scoring judgements with reasoning
worst = scores_df[scores_df["score"] < 1.0].sort_values("score")
if len(worst) > 0:
    sample_cols = ["prompt_name", "sample_id", "field", "extracted_value",
                   "expected_value", "score", "reasoning", "comparison_method"]
    display(spark.createDataFrame(worst.head(30)[sample_cols]))

# COMMAND ----------

# DBTITLE 1,Accuracy by comparison method
method_acc = (
    scores_df.groupby("comparison_method")
    .agg(mean_score=("score", "mean"), count=("score", "count"))
    .reset_index()
)
display(spark.createDataFrame(method_acc))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualisations

# COMMAND ----------

# DBTITLE 1,Per-prompt field accuracy charts
prompts_with_scores = prompt_accuracy["prompt_name"].tolist()
n = len(prompts_with_scores)

if n > 0:
    fig, axes = plt.subplots(1, min(n, 3), figsize=(6 * min(n, 3), 8), squeeze=False)

    for i, pname in enumerate(prompts_with_scores[:3]):
        ax = axes[0][i]
        subset = field_accuracy[field_accuracy["prompt_name"] == pname].sort_values("mean_score")
        colors = []
        for _, row in subset.iterrows():
            if not row["meets_threshold"]:
                colors.append("#E24B4A")
            elif row["is_critical"]:
                colors.append("#378ADD")
            else:
                colors.append("#639922")

        ax.barh(subset["field"], subset["mean_score"], color=colors)
        ax.axvline(x=CRITICAL_THRESHOLD, color="#185FA5", linestyle="--", linewidth=1, label=f"Critical ({CRITICAL_THRESHOLD:.0%})")
        ax.axvline(x=SUPPORTING_THRESHOLD, color="#854F0B", linestyle=":", linewidth=1, label=f"Supporting ({SUPPORTING_THRESHOLD:.0%})")
        ax.set_xlim(0, 1.05)
        ax.set_title(pname, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    display(fig)
    plt.close()

    # If more than 3 prompts, show remaining
    if n > 3:
        fig2, axes2 = plt.subplots(1, n - 3, figsize=(6 * (n - 3), 8), squeeze=False)
        for i, pname in enumerate(prompts_with_scores[3:]):
            ax = axes2[0][i]
            subset = field_accuracy[field_accuracy["prompt_name"] == pname].sort_values("mean_score")
            colors = ["#E24B4A" if not m else "#639922" for m in subset["meets_threshold"]]
            ax.barh(subset["field"], subset["mean_score"], color=colors)
            ax.axvline(x=CRITICAL_THRESHOLD, color="#185FA5", linestyle="--", linewidth=1)
            ax.axvline(x=SUPPORTING_THRESHOLD, color="#854F0B", linestyle=":", linewidth=1)
            ax.set_xlim(0, 1.05)
            ax.set_title(pname, fontsize=11)
        plt.tight_layout()
        display(fig2)
        plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Logging & Delta Persistence

# COMMAND ----------

# DBTITLE 1,Log to MLflow
mlflow.set_experiment("/Shared/eval_accuracy_unified")

with mlflow.start_run(run_name=f"accuracy_{eval_run_id}"):
    mlflow.log_metrics({
        "overall_mean_score": round(scores_df["score"].mean(), 4),
        "overall_pct_correct": round(scores_df["is_correct"].mean(), 4),
        "total_fields_scored": len(scores_df),
        "llm_judge_calls": llm_judge_calls,
        "deterministic_calls": deterministic_calls,
        "prompts_evaluated": len(prompts_with_scores),
        "fields_passing": int(field_accuracy["meets_threshold"].sum()),
        "fields_failing": int((~field_accuracy["meets_threshold"]).sum()),
    })

    mlflow.log_params({
        "eval_run_id": eval_run_id,
        "judge_model": JUDGE_MODEL,
        "critical_threshold": CRITICAL_THRESHOLD,
        "supporting_threshold": SUPPORTING_THRESHOLD,
        "prompts": ",".join(prompts_with_scores),
    })

    # Per-prompt metrics
    for _, row in prompt_accuracy.iterrows():
        mlflow.log_metric(f"prompt_{row['prompt_name']}_score", round(row["mean_score"], 4))

    # Artifacts
    for name, df in [("prompt_accuracy", prompt_accuracy),
                     ("field_accuracy", field_accuracy),
                     ("scores_sample", scores_df.head(500))]:
        path = f"/tmp/{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)

    print(f"Logged to MLflow: {eval_run_id}")

# COMMAND ----------

# DBTITLE 1,Write scores to Delta
scores_sdf = spark.createDataFrame(scores_df.astype(str))
scores_sdf = scores_sdf.withColumn("eval_ts", F.current_timestamp())

scores_table = f"`{catalog}`.`{slr_schema}`.eval_accuracy_scores"
scores_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(scores_table)

print(f"Wrote {len(scores_df)} scores to {scores_table}")

# COMMAND ----------

# DBTITLE 1,Write field summary to Delta
field_acc_sdf = spark.createDataFrame(field_accuracy.astype(str))
field_acc_sdf = (
    field_acc_sdf
    .withColumn("eval_run_id", F.lit(eval_run_id))
    .withColumn("judge_model", F.lit(JUDGE_MODEL))
    .withColumn("eval_ts", F.current_timestamp())
)

summary_table = f"`{catalog}`.`{slr_schema}`.eval_accuracy_summary"
field_acc_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(summary_table)

print(f"Wrote field summary to {summary_table}")

# COMMAND ----------

# DBTITLE 1,Final summary
print(f"\n{'='*70}")
print("UNIFIED ACCURACY EVALUATION — COMPLETE")
print(f"{'='*70}")
print(f"Eval run:              {eval_run_id}")
print(f"Prompts evaluated:     {len(prompts_with_scores)}")
print(f"Fields scored:         {len(scores_df)}")
print(f"  LLM judge calls:     {llm_judge_calls}")
print(f"  Deterministic:       {deterministic_calls}")
print(f"Overall accuracy:      {scores_df['score'].mean():.2%}")
print(f"Fields passing:        {field_accuracy['meets_threshold'].sum()}/{len(field_accuracy)}")

if len(failing) > 0:
    print(f"\nFAILING FIELDS ({len(failing)}):")
    for _, row in failing.iterrows():
        crit = " [CRITICAL]" if row["is_critical"] else ""
        print(f"  {row['prompt_name']:25s} / {row['field']:25s}  "
              f"{row['mean_score']:.2%} (need {row['threshold']:.0%}){crit}")
else:
    print("\nAll fields meet their thresholds!")

print(f"\nDelta tables:")
print(f"  Scores:  {scores_table}")
print(f"  Summary: {summary_table}")
print(f"MLflow:    /Shared/eval_accuracy_unified")
