# Databricks notebook source

# CELL 1 — Get widgets (UNCHANGED)
# ============================================================================
# get widgets
catalog = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")
job_run_id = dbutils.widgets.get("job_run_id")

source_table = "godoc_category"

# CELL 2 — Set widgets (UNCHANGED)
# ============================================================================
# set widgets
dbutils.widgets.text("catalog", "comm-afl-dev", "catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz", "brz_schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "slr_schema")
# parameter set in Job UI, set value then add to delta table for lookup in downstream notebooks
dbutils.widgets.text("job_run_id", "")

# CELL 3 — Imports (UNCHANGED)
# ============================================================================
# imports
import time
import numpy as np
import pandas as pd
import mlflow
from collections import Counter
from datetime import datetime
import pyspark.sql.functions as F
import pyspark.sql.types as T
import json
import sys

sys.path.append("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load")
from Utilities.logs import log_event

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[-1]
log_event(job_run_id, 'INFO', notebook_path, 'Running', 'Starting notebook')

NUM_RUNS = 5  # number of repeated LLM calls per document
EVAL_EXPERIMENT_NAME = "/Shared/ai_eval_proposal_extraction"

# CELL 4 — Pipeline args & notebook runs (UNCHANGED)
# ============================================================================
pipeline_args = {
    "catalog": catalog,
    "brz_schema": brz_schema,
    "slr_schema": slr_schema,
    "job_run_id": job_run_id,
}

dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/Pipeline/Utilities/date_functions", timeout_seconds=3600, arguments=pipeline_args)
dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Bronze/load_adls_files", timeout_seconds=3600, arguments=pipeline_args)
dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/01.silver_ddl", timeout_seconds=3600, arguments=pipeline_args)
dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/02.extract_text", timeout_seconds=3600, arguments=pipeline_args)
dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/03.categorise_image", timeout_seconds=3600, arguments=pipeline_args)
dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/04.categorise_other_doc_type", timeout_seconds=3600, arguments=pipeline_args)
dbutils.notebook.run("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/05.classify_asset", timeout_seconds=3600, arguments=pipeline_args)

# CELL 5 — Add working hours UDF (UNCHANGED)
# ============================================================================
add_working_hours_udf = F.udf(
    lambda x: add_working_hours(x) if x is not None else None, T.TimestampType()
)

# Filter files for those categorised as proposals and grab the extracted text
file_category = spark.sql(
    "select proposal_id, document_id, received_dt, category, context_parsed from `{catalog}`.`{slr_schema}`.`{source_table}` where "
    "lower(category) = 'proposal' or lower(category) = 'email_with_proposal' ".format(
        catalog=catalog, slr_schema=slr_schema, source_table=source_table
    )
)  # only interested in proposal

if file_category.isEmpty():
    print("No files found")
    log_event(job_run_id, 'INFO', notebook_path, 'Exit', 'No proposal files found within the proposal folder')
    # exit notebook without error
    dbutils.notebook.exit("No proposal files found within the proposal folder, exiting job and proceeding to next notebook")

proposal_id = file_category.select("PROPOSAL_ID").first()["PROPOSAL_ID"]

# default to new business
deal_type = "[New Business, New Business with annual review, LOC Increase, LOC Increase with annual review, Annual review, Refinance, Reschedule, Substitution]"

# will have to handle original list in later MVP
# original_list="[Finance Lease, Operating Lease, Hire Purchase, Hire, Sale and HP Back, Sale and Leaseback, LOC Revolving, LOC Non-Revolving, Loan]"
finance_type = "[Finance Lease, Hire Purchase, Other]"

# get list of legal forms from lookup table
legal_form = spark.sql("select name from `comm-afl-dev`.`brkrflw-lkh-brz`.lookup_legal_entity") \
    .select(F.collect_list("name").alias("legal_forms")) \
    .collect()[0]["legal_forms"]
# create in the proper format for injection into prompt
legal_form_list = "[" + ", ".join(f'"{lf}"' for lf in legal_form) + "]"

# CELL 6 — READ PROMPT, RESPONSE_FORMAT, MODEL_ENDPOINT, TEMPERATURE FROM UC TABLE
# ============================================================================
# *** MODIFIED: Read from the current_prompt table instead of hardcoding ***

prompt_table = f"`{catalog}`.`{slr_schema}`.current_prompt"

prompt_row = spark.sql(f"""
    SELECT PROMPT, RESPONSE_FORMAT, MODEL_ENDPOINT, TEMPERATURE
    FROM {prompt_table}
    WHERE PROMPT_TYPE = 'proposal_extraction'
      AND CURRENT = true
""").first()

if prompt_row is None:
    raise ValueError(
        "No current prompt found in the prompt table for PROMPT_TYPE = 'proposal_extraction'. "
        "Please ensure a row exists with CURRENT = true."
    )

PROMPT = prompt_row["PROMPT"]
response_format = prompt_row["RESPONSE_FORMAT"]  # This should already be a JSON string
LLM_ENDPOINT_NAME = prompt_row["MODEL_ENDPOINT"]
TEMPERATURE = float(prompt_row["TEMPERATURE"])

# If response_format is stored as a JSON string, parse it to validate, then keep as string
# for injection into the ai_query call
if isinstance(response_format, str):
    response_format_dict = json.loads(response_format)  # validate it's valid JSON
    response_format_str = response_format  # keep the raw string for ai_query
else:
    response_format_str = json.dumps(response_format)
    response_format_dict = response_format

print(f"Loaded prompt for: proposal_extraction")
print(f"  Model endpoint:  {LLM_ENDPOINT_NAME}")
print(f"  Temperature:     {TEMPERATURE}")
print(f"  Prompt length:   {len(PROMPT)} chars")
print(f"  Response format: {list(response_format_dict.get('json_schema', {}).get('schema', {}).get('properties', {}).keys())[:5]}...")

# CELL 7 — ai_query expression & LLM call (MODIFIED to use table values)
# ============================================================================
from pyspark.sql.functions import col, from_json

# Build the json_schema from the response_format for parsing the response
# Extract the field names from the response_format to build the Spark schema string
schema_properties = response_format_dict.get("json_schema", {}).get("schema", {}).get("properties", {})
json_schema = "STRUCT<" + ", ".join(f"{k} STRING" for k in schema_properties.keys()) + ">"

# Define query with a low temperature as we are interested in recall not generation
ai_query_expr = f"""
    ai_query(
        endpoint => '{LLM_ENDPOINT_NAME}',
        request => CONCAT('{PROMPT.replace("'", "\\'")}', context_parsed),
        responseFormat => '{response_format_str.replace("'", "\\'")}',
        modelParameters => named_struct('temperature', {TEMPERATURE})
    ) AS ai_response
"""

# file_category = file_category.withColumn("context_parsed", F.regexp_replace("context_parsed", "'", "''"))
# run the batch query and unpack the response
info_extracted = file_category.selectExpr("*", ai_query_expr).withColumn("parsed_response", from_json(col("ai_response"), json_schema))
log_event(job_run_id, notebook_path, 'Running', 'LLM call completed')

# CELL 8 — Add working hours UDF (UNCHANGED)
# ============================================================================
add_working_hours_udf = F.udf(
    lambda x: add_working_hours(x) if x is not None else None, T.TimestampType()
)

# CELL 9 — Expected keys and field groups (MODIFIED to derive from response_format)
# ============================================================================
# *** MODIFIED: Derive EXPECTED_KEYS from the response_format loaded from the table ***
EXPECTED_KEYS = list(schema_properties.keys())

FIELD_GROUPS = {
    "identity":   ["broker", "proposer", "crn", "sic_code", "legal_form",
                   "proposer_year_established"],
    "contact":    ["proposer_address", "proposer_trading_address",
                   "proposer_registered_address", "proposer_email",
                   "proposer_phone", "proposer_website"],
    "financial":  ["loan_amount", "currency", "finance_type", "rate_type",
                   "deal_type", "loan_period", "payment_frequency",
                   "initial_payment", "vat_payment", "vat_deferral",
                   "monthly_payment", "balloon_payment"],
    "metadata":   ["loan_date", "term_type", "regulated", "vat_number",
                   "confidence_score"],
}

# CELL 10 — LLM caller for evaluation (MODIFIED to use table values)
# ============================================================================
# LLM caller for evaluation
def call_llm_for_eval(context_text: str) -> dict:
    """
    Call the LLM once via ai_query on a single document.
    Returns the parsed JSON dict.
    """
    prompt_df = spark.createDataFrame([(context_text,)], ["context_parsed"])
    result_df = prompt_df.selectExpr(f"""
        ai_query(
            endpoint => '{LLM_ENDPOINT_NAME}',
            request => CONCAT('{PROMPT.replace("'", "\\'")}', context_parsed),
            responseFormat => '{response_format_str.replace("'", "\\'")}',
            modelParameters => named_struct('temperature', {TEMPERATURE})
        ) AS ai_response
    """)
    raw = result_df.collect()[0]["ai_response"]
    return json.loads(raw)

# CELL 11 — Consistency metric functions (UNCHANGED)
# ============================================================================
# Consistency metric functions
def _norm(s):
    """Lowercase + strip for comparison."""
    return str(s).lower().strip() if s is not None else ""

def exact_match_rate(values):
    """Fraction of runs returning the most common value."""
    if not values:
        return 0.0
    counter = Counter(_norm(v) for v in values)
    return counter.most_common(1)[0][1] / len(values)

def null_rate(values):
    """Fraction of runs returning null / None / empty."""
    return sum(1 for v in values if v is None or _norm(v) in ("null", "none", "")) / len(values)

def unique_ratio(values):
    """Distinct values / total runs (lower = more consistent)."""
    return len(set(_norm(v) if v else "__null__" for v in values)) / len(values)

def numeric_cv(values):
    """Coefficient of variation for numeric fields. None if not numeric."""
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(nums) < 2:
        return None
    mean = np.mean(nums)
    return float(np.std(nums) / abs(mean)) if mean != 0 else 0.0

def _jaccard(a, b):
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def pairwise_jaccard_mean(values):
    n = len(values)
    if n < 2:
        return 1.0
    sims = [_jaccard(str(values[i]), str(values[j]))
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))

# CELL 12 — Run consistency evaluation (UNCHANGED logic, uses dynamic EXPECTED_KEYS)
# ============================================================================
# Run consistency evaluation
# Collect documents to driver for iteration
eval_docs = file_category.select("proposal_id", "context_parsed").toPandas().to_dict("records")
print(f"Evaluating {len(eval_docs)} document(s) x {NUM_RUNS} runs each\n")

mlflow.set_experiment(EVAL_EXPERIMENT_NAME)

all_field_rows = []
all_doc_rows = []

with mlflow.start_run(run_name=f"consistency_{datetime.now():%Y%m%d_%H%M%S}") as parent_run:
    mlflow.log_params({
        "num_runs": NUM_RUNS,
        "num_documents": len(eval_docs),
        "llm_endpoint": LLM_ENDPOINT_NAME,
        "temperature": TEMPERATURE,
    })

    for doc_idx, doc in enumerate(eval_docs):
        pid = doc["proposal_id"]
        ctx = doc["context_parsed"]
        print(f"{'='*50}\nDocument {doc_idx+1}/{len(eval_docs)}: {pid}\n{'='*50}")

        # --- Collect N responses ---
        responses, latencies = [], []
        for ri in range(NUM_RUNS):
            t0 = time.time()
            try:
                resp = call_llm_for_eval(ctx)
            except Exception as e:
                print(f"  Run {ri+1} FAILED: {e}")
                resp = {k: None for k in EXPECTED_KEYS}
            elapsed = time.time() - t0
            latencies.append(elapsed)
            responses.append(resp)
            print(f"  Run {ri+1}/{NUM_RUNS} – {elapsed:.1f}s")

        # --- Per-field metrics ---
        doc_rows = []
        for key in EXPECTED_KEYS:
            vals = [r.get(key) for r in responses]
            cv = numeric_cv(vals)
            doc_rows.append({
                "proposal_id": pid,
                "field": key,
                "exact_match_rate": round(exact_match_rate(vals), 4),
                "null_rate": round(null_rate(vals), 4),
                "unique_ratio": round(unique_ratio(vals), 4),
                "modal_value": Counter(_norm(v) for v in vals).most_common(1)[0][0],
                "all_values": str([_norm(v) for v in vals]),
                "jaccard_mean": round(pairwise_jaccard_mean(vals), 4),
            })
        all_field_rows.extend(doc_rows)

        # --- Document-level summary ---
        fdf = pd.DataFrame(doc_rows)
        summary = {
            "proposal_id": pid,
            "mean_exact_match_rate": round(fdf["exact_match_rate"].mean(), 4),
            "min_exact_match_rate": round(fdf["exact_match_rate"].min(), 4),
            "mean_null_rate": round(fdf["null_rate"].mean(), 4),
            "fully_consistent_fields": int((fdf["exact_match_rate"] == 1.0).sum()),
            "total_fields": len(EXPECTED_KEYS),
            "pct_fully_consistent": round(
                (fdf["exact_match_rate"] == 1.0).sum() / len(EXPECTED_KEYS) * 100, 2),
            "mean_latency_s": round(np.mean(latencies), 3),
        }
        all_doc_rows.append(summary)

        # log child run per document
        with mlflow.start_run(run_name=f"doc_{pid}", nested=True):
            mlflow.log_metrics({k: v for k, v in summary.items()
                                if isinstance(v, (int, float)) and k != "total_fields"})
            resp_path = f"/tmp/responses_{pid}.json"
            with open(resp_path, "w") as f:
                json.dump(responses, f, indent=2, default=str)
            mlflow.log_artifact(resp_path)

    # --- Aggregate across all documents ---
    field_metrics_df = pd.DataFrame(all_field_rows)
    doc_metrics_df = pd.DataFrame(all_doc_rows)
    field_ranking = (field_metrics_df.groupby("field")["exact_match_rate"]
                     .mean().sort_values().reset_index()
                     .rename(columns={"exact_match_rate": "mean_exact_match_rate"}))

    agg_metrics = {
        "overall_mean_exact_match": round(field_metrics_df["exact_match_rate"].mean(), 4),
        "overall_pct_fully_consistent": round(
            (field_metrics_df["exact_match_rate"] == 1.0).sum() / len(field_metrics_df) * 100, 2),
        "overall_mean_jaccard": round((field_metrics_df["jaccard_mean"].mean()), 4),
    }
    mlflow.log_metrics(agg_metrics)

    # Save CSV artifacts
    for name, df in [("field_metrics", field_metrics_df),
                     ("doc_metrics", doc_metrics_df),
                     ("field_ranking", field_ranking)]:
        path = f"/tmp/{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)

# CELL 13 — Evaluation summary (UNCHANGED)
# ============================================================================
# Evaluation summary
print(f"{'='*60}")
print("CONSISTENCY EVALUATION SUMMARY")
print(f"{'='*60}")
print(f"Documents evaluated:  {len(eval_docs)}")
print(f"Runs per document:    {NUM_RUNS}")
print(f"Overall exact match:  {agg_metrics['overall_mean_exact_match']:.2%}")
print(f"Overall Jaccard mean: {agg_metrics['overall_mean_jaccard']:.2%}")
print(f"Fully consistent:     {agg_metrics['overall_pct_fully_consistent']:.1f}% of field-runs")

# CELL 14 — Least consistent fields (UNCHANGED)
# ============================================================================
# Least consistent fields (worst first)
display(spark.createDataFrame(field_ranking))

# CELL 15 — Per-document results (UNCHANGED)
# ============================================================================
# Per-document results
display(spark.createDataFrame(doc_metrics_df))

# CELL 16 — Full field-level detail (UNCHANGED)
# ============================================================================
# Full field-level detail
display(spark.createDataFrame(field_metrics_df.drop(columns=["all_values"])))
