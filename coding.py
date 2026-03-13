# MAGIC %md # Consistency Evaluation — Person Party Extraction
# MAGIC %md 
# MAGIC %md **Purpose:** Measure how stable the LLM's extraction outputs are across
# MAGIC %md repeated runs of the same document. Because the model is probabilistic,
# MAGIC %md identical input can produce slightly different results. This notebook
# MAGIC %md quantifies that variance.
# MAGIC %md 
# MAGIC %md **Key principle:** A high consistency score does **not** mean the extracted
# MAGIC %md data is correct — it means the model reliably produces the same answer.
# MAGIC %md Always read consistency alongside accuracy.
# MAGIC %md 
# MAGIC %md **Target:** >= 85% consistency across all fields. Fields below this threshold
# MAGIC %md are flagged for prompt review.
# MAGIC %md 
# MAGIC %md **Prompt under test:** `extract_person_party_identify_role` — LLM structured extraction.
# MAGIC %md 
# MAGIC %md ---

# COMMAND ----------

# MAGIC %md ## 1. Widgets & Configuration

# COMMAND ----------

# COMMAND ----------
dbutils.widgets.text("catalog", "comm-afl-dev", "Catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz", "Bronze Schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "Silver Schema")

catalog    = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")

# Evaluation config
NUM_RUNS              = 5       # number of repeated runs per document
CONSISTENCY_THRESHOLD = 0.85    # 85% agreement target

print(f"Catalog:    {catalog}")
print(f"BRZ Schema: {brz_schema}")
print(f"SLR Schema: {slr_schema}")
print(f"Runs/doc:   {NUM_RUNS}")
print(f"Threshold:  {CONSISTENCY_THRESHOLD:.0%}")

# COMMAND ----------

# MAGIC %md ## 2. Imports

# COMMAND ----------

# COMMAND ----------
import json
import time
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

sns.set_theme(style="whitegrid", palette="muted")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md ## 3. Load Prompt from Unity Catalog Prompt Table
# MAGIC %md 
# MAGIC %md Read the current prompt, response format, model endpoint, and temperature
# MAGIC %md from the `current_prompt` table — same pattern as the other evaluation notebooks.

# COMMAND ----------

# COMMAND ----------
prompt_table = f"{catalog}.{slr_schema}.current_prompt"

prompt_row = spark.sql(f"""
    SELECT PROMPT_ID, PROMPT_TYPE, MODEL_ENDPOINT, PROMPT, RESPONSE_FORMAT, TEMPERATURE
    FROM {prompt_table}
    WHERE PROMPT_TYPE = 'person_party_extraction'
      AND CURRENT = true
""").collect()

assert len(prompt_row) > 0, f"No current prompt found for PROMPT_TYPE='person_party_extraction' in {prompt_table}"
prompt_row = prompt_row[0]

PROMPT_ID       = prompt_row["PROMPT_ID"]
PROMPT_TYPE     = prompt_row["PROMPT_TYPE"]
MODEL_ENDPOINT  = prompt_row["MODEL_ENDPOINT"]
PROMPT          = prompt_row["PROMPT"]
RESPONSE_FORMAT = json.loads(prompt_row["RESPONSE_FORMAT"]) if isinstance(prompt_row["RESPONSE_FORMAT"], str) else prompt_row["RESPONSE_FORMAT"]
TEMPERATURE     = float(prompt_row["TEMPERATURE"])

print(f"Prompt ID:       {PROMPT_ID}")
print(f"Prompt Type:     {PROMPT_TYPE}")
print(f"Model Endpoint:  {MODEL_ENDPOINT}")
print(f"Temperature:     {TEMPERATURE}")
print(f"Prompt preview:  {PROMPT[:200]}...")
print(f"Response format: {json.dumps(RESPONSE_FORMAT, indent=2)[:300]}...")

# COMMAND ----------

# MAGIC %md ## 4. Define Output Fields

# COMMAND ----------

# COMMAND ----------
OUTPUT_FIELDS = [
    "role",
    "salutation",
    "first_name",
    "last_name",
    "company_name",
    "gender",
    "date_of_birth",
    "country_code",
    "mob",
    "email",
    "job_title",
    "street_address",
    "city_name",
    "postal_code",
    "country_name",
    "reasoning",
]

print(f"Output fields ({len(OUTPUT_FIELDS)}): {OUTPUT_FIELDS}")

# COMMAND ----------

# MAGIC %md ## 5. Load Evaluation Documents
# MAGIC %md 
# MAGIC %md Read proposal/email documents from `doc_category` — the `extract_person_party_identify_role`
# MAGIC %md notebook reads from this table to extract structured party information.

# COMMAND ----------

# COMMAND ----------
source_table = f"{catalog}.{slr_schema}.doc_category"

eval_docs_df = spark.sql(f"""
    SELECT PROPOSAL_ID, DOCUMENT_ID, SOURCE_PATH, FILE_NAME, FILE_EXT,
           CONTEXT_PARSED, CATEGORY, MODIFICATION_TS
    FROM {source_table}
    WHERE lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details')
      AND CONTEXT_PARSED IS NOT NULL
      AND length(CONTEXT_PARSED) > 10
""")

eval_docs_count = eval_docs_df.count()
print(f"Total documents available for evaluation: {eval_docs_count}")

eval_docs_df.groupBy("CATEGORY").count().orderBy("count", ascending=False).show()

eval_docs = [row.asDict() for row in eval_docs_df.collect()]
print(f"Collected {len(eval_docs)} documents for evaluation.")

# COMMAND ----------

# MAGIC %md ## 6. ai_query SQL Expression (Reference)
# MAGIC %md 
# MAGIC %md This is the Spark SQL expression used in the production `extract_person_party_identify_role`
# MAGIC %md notebook. We replicate it in `call_llm_for_eval` below.

# COMMAND ----------

# COMMAND ----------
ai_query_sql = f"""
SELECT ai_query(
    '{MODEL_ENDPOINT}',
    CONCAT('{PROMPT}', '\n\nDocument text:\n', <text_column>),
    responseFormat => '{json.dumps(RESPONSE_FORMAT)}'
) AS llm_response
"""

print("ai_query SQL expression (reference):")
print(ai_query_sql[:500])

# COMMAND ----------

# MAGIC %md ## 7. LLM Call Function
# MAGIC %md 
# MAGIC %md This function calls the LLM endpoint for a single document and returns
# MAGIC %md the parsed response as a dict. It mirrors the production `extract_person_party_identify_role`
# MAGIC %md notebook's logic.

# COMMAND ----------

# COMMAND ----------
def call_llm_for_eval(text_content: str, file_name: str) -> dict:
    """
    Call the LLM endpoint for a single document and return parsed response.
    Returns dict with keys matching OUTPUT_FIELDS, or dict of Nones on failure.
    """
    try:
        escaped_text = text_content.replace("'", "\\'").replace("\n", " ")[:8000]
        response_format_str = json.dumps(RESPONSE_FORMAT).replace("'", "\\'")

        result_df = spark.sql(f"""
            SELECT ai_query(
                '{MODEL_ENDPOINT}',
                CONCAT(
                    '{PROMPT.replace("'", "\\'")}',
                    '\n\nDocument text:\n',
                    '{escaped_text}'
                ),
                responseFormat => '{response_format_str}'
            ) AS llm_response
        """)

        raw_response = result_df.collect()[0]["llm_response"]

        if isinstance(raw_response, str):
            parsed = json.loads(raw_response)
        elif isinstance(raw_response, dict):
            parsed = raw_response
        else:
            parsed = json.loads(str(raw_response))

        return {field: parsed.get(field, None) for field in OUTPUT_FIELDS}

    except Exception as e:
        print(f"  ERROR on {file_name}: {e}")
        return {field: None for field in OUTPUT_FIELDS}


print("call_llm_for_eval() defined.")

# COMMAND ----------

# MAGIC %md ## 8. Consistency Metric Functions
# MAGIC %md 
# MAGIC %md Same metric functions as the other evaluation notebooks.

# COMMAND ----------

# COMMAND ----------
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


def null_rate(values: list) -> float:
    """Fraction of values that are null/None."""
    return sum(1 for v in values if _norm(v) == "__NULL__") / len(values) if values else 0.0


def unique_ratio(values: list) -> float:
    """Number of unique normalised values / total values."""
    normed = [_norm(v) for v in values]
    return len(set(normed)) / len(normed) if normed else 0.0


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def pairwise_jaccard_mean(values: list) -> float:
    """Mean pairwise Jaccard similarity across all value pairs."""
    normed = [_norm(v) for v in values]
    if len(normed) < 2:
        return 1.0
    scores = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            scores.append(_jaccard(normed[i], normed[j]))
    return np.mean(scores) if scores else 1.0


print("Metric functions defined: exact_match_rate, null_rate, unique_ratio, pairwise_jaccard_mean")

# COMMAND ----------

# MAGIC %md ## 9. Run Consistency Evaluation
# MAGIC %md 
# MAGIC %md For each document, run the LLM call NUM_RUNS times and compute per-field
# MAGIC %md consistency metrics.

# COMMAND ----------

# COMMAND ----------
doc_rows = []
all_run_data = []

total_calls = len(eval_docs) * NUM_RUNS
print(f"Starting consistency evaluation: {len(eval_docs)} documents x {NUM_RUNS} runs = {total_calls} LLM calls")
print("=" * 70)

for idx, doc in enumerate(eval_docs):
    pid = doc["PROPOSAL_ID"]
    did = doc["DOCUMENT_ID"]
    fname = doc["FILE_NAME"]
    text_content = doc["CONTEXT_PARSED"]

    print(f"\n[{idx+1}/{len(eval_docs)}] {fname} (proposal={pid[:12]}...)")

    run_results = []
    for run_i in range(NUM_RUNS):
        t0 = time.time()
        result = call_llm_for_eval(text_content, fname)
        elapsed = time.time() - t0
        result["_run"] = run_i
        result["_elapsed_s"] = round(elapsed, 2)
        run_results.append(result)
        print(f"  Run {run_i+1}/{NUM_RUNS}: first_name={str(result.get('first_name', '?')):<40s} ({elapsed:.1f}s)")

    for r in run_results:
        all_run_data.append({"proposal_id": pid, "document_id": did, "file_name": fname, **r})

    for key in OUTPUT_FIELDS:
        vals = [r[key] for r in run_results]
        doc_rows.append({
            "proposal_id": pid, "document_id": did, "file_name": fname, "field": key,
            "exact_match_rate": round(exact_match_rate(vals), 4),
            "null_rate": round(null_rate(vals), 4),
            "unique_ratio": round(unique_ratio(vals), 4),
            "jaccard_mean": round(pairwise_jaccard_mean([str(v) for v in vals]), 4),
            "modal_value": Counter(_norm(v) for v in vals).most_common(1)[0][0],
            "all_values": str([_norm(v) for v in vals]),
        })

print("\n" + "=" * 70)
print("Evaluation complete.")

field_metrics_df = pd.DataFrame(doc_rows)
run_data_df = pd.DataFrame(all_run_data)
print(f"Field metrics: {len(field_metrics_df)} rows")
print(f"Raw run data:  {len(run_data_df)} rows")

# COMMAND ----------

# MAGIC %md ## 10. Aggregate Metrics & MLflow Logging
# MAGIC %md 
# MAGIC %md Compute overall summary metrics and log to MLflow — same pattern as the
# MAGIC %md other evaluation notebooks.

# COMMAND ----------

# COMMAND ----------
agg_metrics = {
    "overall_mean_exact_match": round(field_metrics_df["exact_match_rate"].mean(), 4),
    "overall_pct_fully_consistent": round(
        (field_metrics_df["exact_match_rate"] == 1.0).sum() / len(field_metrics_df) * 100, 2),
    "overall_mean_jaccard": round(field_metrics_df["jaccard_mean"].mean(), 4),
}

field_summary = field_metrics_df.groupby("field").agg(
    mean_exact_match=("exact_match_rate", "mean"),
    mean_jaccard=("jaccard_mean", "mean"),
    mean_null_rate=("null_rate", "mean"),
    mean_unique_ratio=("unique_ratio", "mean"),
).round(4)

print("=" * 60)
print("CONSISTENCY EVALUATION SUMMARY — PERSON PARTY EXTRACTION")
print("=" * 60)
print(f"Documents evaluated:  {len(eval_docs)}")
print(f"Runs per document:    {NUM_RUNS}")
print(f"Prompt ID:            {PROMPT_ID}")
print(f"Model endpoint:       {MODEL_ENDPOINT}")
print(f"Overall exact match:  {agg_metrics['overall_mean_exact_match']:.2%}")
print(f"Fully consistent:     {agg_metrics['overall_pct_fully_consistent']:.1f}% of field-runs")
print(f"Overall Jaccard mean: {agg_metrics['overall_mean_jaccard']:.2%}")
print()
print("Per-field breakdown:")
print(field_summary.to_string())

experiment_name = "/Evaluation/extract_person_party_consistency"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"extract_person_party_consistency_{PROMPT_ID}_{datetime.now():%Y%m%d_%H%M}"):
    mlflow.log_param("prompt_id", PROMPT_ID)
    mlflow.log_param("prompt_type", PROMPT_TYPE)
    mlflow.log_param("model_endpoint", MODEL_ENDPOINT)
    mlflow.log_param("temperature", TEMPERATURE)
    mlflow.log_param("num_runs", NUM_RUNS)
    mlflow.log_param("num_documents", len(eval_docs))
    mlflow.log_param("consistency_threshold", CONSISTENCY_THRESHOLD)
    mlflow.log_metrics(agg_metrics)
    for field_name, row in field_summary.iterrows():
        for metric_name, value in row.items():
            mlflow.log_metric(f"{field_name}__{metric_name}", value)
    field_metrics_df.to_csv("/tmp/person_party_field_metrics.csv", index=False)
    mlflow.log_artifact("/tmp/person_party_field_metrics.csv")
    run_data_df.to_csv("/tmp/person_party_run_data.csv", index=False)
    mlflow.log_artifact("/tmp/person_party_run_data.csv")

print(f"\nLogged to MLflow experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md ## 11. Write Results to Delta for Ground Truth Review
# MAGIC %md 
# MAGIC %md Persist the field-level metrics and raw run data to Delta tables in the
# MAGIC %md evaluation schema, so results accumulate across prompt versions.

# COMMAND ----------

# COMMAND ----------
from pyspark.sql import functions as F

eval_schema = f"{catalog}.{slr_schema}"

field_metrics_sdf = spark.createDataFrame(field_metrics_df)
field_metrics_sdf = field_metrics_sdf.withColumn("prompt_id", F.lit(PROMPT_ID)) \
                                     .withColumn("prompt_type", F.lit(PROMPT_TYPE)) \
                                     .withColumn("model_endpoint", F.lit(MODEL_ENDPOINT)) \
                                     .withColumn("num_runs", F.lit(NUM_RUNS)) \
                                     .withColumn("eval_ts", F.current_timestamp())

field_metrics_sdf.write.format("delta") \
    .mode("append") \
    .saveAsTable(f"{eval_schema}.eval_person_party_consistency_metrics")

run_data_sdf = spark.createDataFrame(run_data_df.astype(str))
run_data_sdf = run_data_sdf.withColumn("prompt_id", F.lit(PROMPT_ID)) \
                           .withColumn("eval_ts", F.current_timestamp())

run_data_sdf.write.format("delta") \
    .mode("append") \
    .saveAsTable(f"{eval_schema}.eval_person_party_consistency_runs")

print(f"Results written to:")
print(f"  {eval_schema}.eval_person_party_consistency_metrics")
print(f"  {eval_schema}.eval_person_party_consistency_runs")

# COMMAND ----------

# MAGIC %md ## 12. Visualisation: Per-Field Consistency
# MAGIC %md 
# MAGIC %md Bar chart showing mean exact match rate per field, with the 85% threshold line.

# COMMAND ----------

# COMMAND ----------
fig, ax = plt.subplots(figsize=(12, max(5, len(field_summary) * 0.4)))

field_order = field_summary.sort_values("mean_exact_match", ascending=True).index
colours = ["#e74c3c" if field_summary.loc[f, "mean_exact_match"] < CONSISTENCY_THRESHOLD
           else "#2ecc71" for f in field_order]

ax.barh(field_order, field_summary.loc[field_order, "mean_exact_match"], color=colours)
ax.axvline(x=CONSISTENCY_THRESHOLD, color="orange", linestyle="--", linewidth=2, label=f"Threshold ({CONSISTENCY_THRESHOLD:.0%})")
ax.set_xlabel("Mean Exact Match Rate")
ax.set_title(f"Person Party Extraction — Per-Field Consistency (n={len(eval_docs)} docs, {NUM_RUNS} runs each)")
ax.set_xlim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## 13. Side-by-Side Run Comparison (HTML)
# MAGIC %md 
# MAGIC %md For each document, show all NUM_RUNS responses side by side so you can
# MAGIC %md visually inspect where the LLM diverges.

# COMMAND ----------

# COMMAND ----------
from IPython.display import HTML

def build_comparison_html(run_data_df, output_fields, max_docs=20):
    """Build an HTML table comparing runs for each document."""
    html_parts = ['<style>table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ddd;padding:6px;text-align:left;font-size:12px;}th{background:#f5f5f5;}.mismatch{background:#ffe0e0;}.match{background:#e0ffe0;}</style>']

    doc_ids = run_data_df["document_id"].unique()[:max_docs]

    for did in doc_ids:
        doc_runs = run_data_df[run_data_df["document_id"] == did].sort_values("_run")
        fname = doc_runs.iloc[0].get("file_name", did[:12])
        html_parts.append(f"<h3>{fname}</h3>")
        html_parts.append("<table><tr><th>Field</th>")
        for run_i in range(NUM_RUNS):
            html_parts.append(f"<th>Run {run_i+1}</th>")
        html_parts.append("</tr>")

        for field in output_fields:
            vals = [_norm(doc_runs.iloc[r].get(field, "")) if r < len(doc_runs) else "" for r in range(NUM_RUNS)]
            all_same = len(set(vals)) == 1
            css = "match" if all_same else "mismatch"
            html_parts.append(f"<tr><td><b>{field}</b></td>")
            for v in vals:
                display_v = str(v)[:80]
                html_parts.append(f'<td class="{css}">{display_v}</td>')
            html_parts.append("</tr>")
        html_parts.append("</table>")

    return "".join(html_parts)

comparison_html = build_comparison_html(run_data_df, OUTPUT_FIELDS)
displayHTML(comparison_html)

# COMMAND ----------

# MAGIC %md ## 14. Worst Performers
# MAGIC %md 
# MAGIC %md Documents and fields with the lowest consistency scores — candidates for
# MAGIC %md prompt engineering review.

# COMMAND ----------

# COMMAND ----------
worst = field_metrics_df[field_metrics_df["exact_match_rate"] < CONSISTENCY_THRESHOLD] \
    .sort_values("exact_match_rate") \
    .head(20)

if len(worst) == 0:
    print(f"All (document, field) pairs meet the {CONSISTENCY_THRESHOLD:.0%} threshold.")
else:
    print(f"Found {len(worst)} (document, field) pairs below {CONSISTENCY_THRESHOLD:.0%} threshold:")
    print()
    for _, row in worst.iterrows():
        print(f"  {row['file_name']:<40s} | {row['field']:<30s} | "
              f"exact_match={row['exact_match_rate']:.2%} | modal={row['modal_value'][:40]}")
        print(f"    values: {row['all_values'][:120]}")

# COMMAND ----------

# MAGIC %md ## Final Summary

# COMMAND ----------

# COMMAND ----------
print("=" * 70)
print("PERSON PARTY EXTRACTION — CONSISTENCY EVALUATION COMPLETE")
print("=" * 70)
print(f"Prompt ID:            {PROMPT_ID}")
print(f"Model:                {MODEL_ENDPOINT}")
print(f"Temperature:          {TEMPERATURE}")
print(f"Documents evaluated:  {len(eval_docs)}")
print(f"Runs per document:    {NUM_RUNS}")
print(f"Total LLM calls:      {len(eval_docs) * NUM_RUNS}")
print(f"Overall exact match:  {agg_metrics['overall_mean_exact_match']:.2%}")
print(f"Fully consistent:     {agg_metrics['overall_pct_fully_consistent']:.1f}%")
print(f"Overall Jaccard mean: {agg_metrics['overall_mean_jaccard']:.2%}")
print()

below_threshold = field_metrics_df[field_metrics_df["exact_match_rate"] < CONSISTENCY_THRESHOLD]
if len(below_threshold) == 0:
    print(f"RESULT: PASS — all (document, field) pairs meet the {CONSISTENCY_THRESHOLD:.0%} threshold.")
else:
    n_below = len(below_threshold)
    n_total = len(field_metrics_df)
    print(f"RESULT: REVIEW NEEDED — {n_below}/{n_total} ({n_below/n_total:.1%}) of (document, field) pairs "
          f"are below the {CONSISTENCY_THRESHOLD:.0%} threshold.")
    print()
    field_fail_counts = below_threshold.groupby("field").size().sort_values(ascending=False)
    print("Fields with most below-threshold documents:")
    for f, cnt in field_fail_counts.items():
        print(f"  {f:<30s}: {cnt} documents below threshold")

# COMMAND ----------
