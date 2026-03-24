# Databricks notebook source

# MAGIC %md
# MAGIC # Unified Consistency Evaluation — All Prompts
# MAGIC
# MAGIC Evaluates consistency of **4 text-based LLM prompts** across all documents
# MAGIC in a single notebook run. No per-proposal looping — reads from Silver Delta tables.
# MAGIC
# MAGIC | # | Prompt | Task | Source table |
# MAGIC |---|--------|------|-------------|
# MAGIC | 1 | `categorise_other_doc_type` | Classification | `doc_category` |
# MAGIC | 2 | `classify_asset` | Classification | `doc_category` |
# MAGIC | 3 | `extract_account_proposal_information` | Extraction (30 fields) | `doc_category` |
# MAGIC | 4 | `extract_corporate_party_identify_role` | Extraction | `doc_category` |
# MAGIC | 5 | `extract_person_party_identify_role` | Extraction | `doc_category` |
# MAGIC
# MAGIC **Note:** `categorise_image` (VLM) is excluded — it requires binary image input
# MAGIC and is evaluated for accuracy only in Notebook 2.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. Load all documents from Silver tables (one read, all proposals)
# MAGIC 2. Loop through the prompt catalogue
# MAGIC 3. For each prompt × each document: call `ai_query` N times
# MAGIC 4. Compute consistency metrics per field per document
# MAGIC 5. Log everything to MLflow + Delta

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "comm-afl-dev", "Catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz", "Bronze Schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "Silver Schema")
dbutils.widgets.text("num_runs", "5", "Runs per document")

catalog = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")
NUM_RUNS = int(dbutils.widgets.get("num_runs"))

print(f"Catalog:     {catalog}")
print(f"Silver:      {slr_schema}")
print(f"Runs/doc:    {NUM_RUNS}")

# COMMAND ----------

# DBTITLE 1,Imports
import json
import time
import re
import html as html_mod
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Catalogue
# MAGIC
# MAGIC One dict per prompt. Both notebooks (consistency + accuracy) share this structure.
# MAGIC Each entry defines: which `current_prompt` row to load, which fields to evaluate,
# MAGIC which Silver table to read documents from, and filtering criteria.

# COMMAND ----------

# DBTITLE 1,Prompt catalogue definition
PROMPT_CATALOGUE = {
    # ── Classification prompts ──────────────────────────────────────────────
    "categorise_doc": {
        "prompt_type": "text_categorisation",  # PROMPT_TYPE in current_prompt table
        "task_type": "classification",
        "source_table": "eval_snap_doc_category",
        "source_filter": "CONTEXT_PARSED IS NOT NULL AND length(CONTEXT_PARSED) > 10",
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "category", "image_description", "document_type",
            "bank_statement_period", "bank_statement_bank_name",
            "text", "reasoning",
        ],
        "primary_field": "category",
    },

    "classify_asset": {
        "prompt_type": "asset_extraction",
        "task_type": "classification",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            lower(CATEGORY) IN ('asset_image', 'proposal', 'email_body_with_proposal_details')
            AND CONTEXT_PARSED IS NOT NULL
            AND length(CONTEXT_PARSED) > 10
        """,
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "asset_type", "asset_description", "manufacturer",
            "model", "year", "registration", "reasoning",
        ],
        "primary_field": "asset_type",
    },

    # ── Extraction prompts ──────────────────────────────────────────────────
    "extract_proposal": {
        "prompt_type": "account_extraction",
        "task_type": "extraction",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            (lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details'))
            AND CONTEXT_PARSED IS NOT NULL
            AND length(CONTEXT_PARSED) > 10
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
            "vat_payment", "vat_deferral", "monthly_payment",
            "balloon_payment", "confidence_score",
        ],
        "primary_field": "proposer",
    },

    "extract_corporate_party": {
        "prompt_type": "corporate_extraction",
        "task_type": "extraction",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            (lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details'))
            AND CONTEXT_PARSED IS NOT NULL
            AND length(CONTEXT_PARSED) > 10
        """,
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "company_role", "company_name", "company_number",
            "street_name", "city_name", "postal_code", "country_name",
            "trading_address", "registered_address",
            "mob", "email", "reasoning",
        ],
        "primary_field": "company_name",
    },

    "extract_person_party": {
        "prompt_type": "person_extraction",
        "task_type": "extraction",
        "source_table": "eval_snap_doc_category",
        "source_filter": """
            (lower(CATEGORY) IN ('proposal', 'email_body_with_proposal_details'))
            AND CONTEXT_PARSED IS NOT NULL
            AND length(CONTEXT_PARSED) > 10
        """,
        "text_column": "CONTEXT_PARSED",
        "id_columns": ["PROPOSAL_ID", "DOCUMENT_ID", "FILE_NAME"],
        "output_fields": [
            "role", "salutation", "first_name", "last_name",
            "company_name", "gender", "date_of_birth",
            "country_code", "mob", "email", "job_title",
            "street_address", "city_name", "postal_code", "country_name",
            "reasoning",
        ],
        "primary_field": "first_name",
    },
}

print(f"Prompt catalogue: {len(PROMPT_CATALOGUE)} prompts")
for name, cfg in PROMPT_CATALOGUE.items():
    print(f"  {name:30s}  {cfg['task_type']:15s}  {len(cfg['output_fields'])} fields")

# COMMAND ----------

# DBTITLE 1,Select which prompts to evaluate this run
# By default, evaluate ALL prompts. To run a subset, override here:
PROMPTS_TO_EVALUATE = list(PROMPT_CATALOGUE.keys())
# PROMPTS_TO_EVALUATE = ["extract_proposal"]  # uncomment to run just one

print(f"Will evaluate: {PROMPTS_TO_EVALUATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consistency Metric Functions

# COMMAND ----------

# DBTITLE 1,Shared metric functions
CONSISTENCY_THRESHOLD = 0.85

def _norm(val):
    """Normalise a value for comparison."""
    if val is None:
        return "__NULL__"
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match_rate(values: list) -> float:
    """Fraction of values that equal the mode."""
    normed = [_norm(v) for v in values]
    if not normed:
        return 0.0
    mode_count = Counter(normed).most_common(1)[0][1]
    return mode_count / len(normed)


def null_rate(values: list) -> float:
    """Fraction of null/empty values."""
    return sum(1 for v in values if _norm(v) in ("__null__", "null", "none", "")) / len(values)


def unique_ratio(values: list) -> float:
    """Distinct values / total (lower = more consistent)."""
    normed = [_norm(v) for v in values]
    return len(set(normed)) / len(normed)


def pairwise_jaccard_mean(values: list) -> float:
    """Mean pairwise token-level Jaccard similarity."""
    n = len(values)
    if n < 2:
        return 1.0
    def _jaccard(a, b):
        sa, sb = set(_norm(a).split()), set(_norm(b).split())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)
    sims = [_jaccard(str(values[i]), str(values[j]))
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))


def entropy(values: list) -> float:
    """Shannon entropy — 0 = perfectly consistent."""
    normed = [_norm(v) for v in values]
    n = len(normed)
    if n == 0:
        return 0.0
    counts = Counter(normed)
    probs = [c / n for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


print("Consistency metrics defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Prompt Configs from Unity Catalog

# COMMAND ----------

# DBTITLE 1,Load all prompt configs at once
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
        "model_endpoint": row["MODEL_ENDPOINT"],
        "prompt_text": row["PROMPT"],
        "response_format": resp_fmt,
        "temperature": float(row["TEMPERATURE"]),
    }
    print(f"  Loaded: {prompt_name:30s} model={row['MODEL_ENDPOINT']}  temp={row['TEMPERATURE']}")

print(f"\n{len(prompt_configs)}/{len(PROMPTS_TO_EVALUATE)} prompt configs loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load All Documents from Silver Tables

# COMMAND ----------

# DBTITLE 1,Load documents for all prompts (one read per source table)
all_eval_data = {}

# De-duplicate source table reads — multiple prompts may read from the same table
source_tables = {}
for prompt_name in prompt_configs:
    cfg = PROMPT_CATALOGUE[prompt_name]
    table_key = cfg["source_table"]
    if table_key not in source_tables:
        source_tables[table_key] = []
    source_tables[table_key].append((prompt_name, cfg))

for table_name, prompt_list in source_tables.items():
    full_table = f"`{catalog}`.`{slr_schema}`.{table_name}"

    # Read the full table once
    base_df = spark.table(full_table)
    print(f"\nTable: {full_table} ({base_df.count()} total rows)")

    for prompt_name, cfg in prompt_list:
        # Apply prompt-specific filter
        filtered_df = base_df.filter(cfg["source_filter"])
        count = filtered_df.count()

        # Collect the columns we need
        select_cols = cfg["id_columns"] + [cfg["text_column"]]
        docs = [row.asDict() for row in filtered_df.select(*select_cols).collect()]

        all_eval_data[prompt_name] = docs
        print(f"  {prompt_name:30s} → {count} documents loaded")

print(f"\nTotal: {sum(len(v) for v in all_eval_data.values())} document-prompt pairs")
total_llm_calls = sum(len(v) * NUM_RUNS for v in all_eval_data.values())
print(f"Total LLM calls needed: {total_llm_calls} ({sum(len(v) for v in all_eval_data.values())} docs x {NUM_RUNS} runs)")
print(f"\nDocument-to-prompt routing:")
print(f"{'='*70}")
for pname, docs in all_eval_data.items():
    cfg = PROMPT_CATALOGUE[pname]
    print(f"  {pname:30s}  {len(docs):4d} docs  filter: {cfg['source_filter'].strip()[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Call Function

# COMMAND ----------

# DBTITLE 1,Generic LLM caller via ai_query
def call_llm_for_eval(text_content: str, prompt_name: str) -> dict:
    """
    Call the LLM for a given prompt config on a single document.
    Returns parsed JSON dict with keys matching the prompt's output_fields.
    """
    pcfg = prompt_configs[prompt_name]
    ccfg = PROMPT_CATALOGUE[prompt_name]
    output_fields = ccfg["output_fields"]

    try:
        escaped_text = text_content.replace("'", "\\'").replace("\n", " ")[:8000]
        prompt_escaped = pcfg["prompt_text"].replace("'", "\\'")
        resp_fmt_str = json.dumps(pcfg["response_format"]).replace("'", "\\'")

        result_df = spark.sql(f"""
            SELECT ai_query(
                '{pcfg["model_endpoint"]}',
                CONCAT('{prompt_escaped}', '\n\nDocument text:\n', '{escaped_text}'),
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
        print(f"    ERROR: {e}")
        return {field: None for field in output_fields}


print("call_llm_for_eval() defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Consistency Evaluation
# MAGIC
# MAGIC Main loop: for each prompt × each document × N runs.

# COMMAND ----------

# DBTITLE 1,Main evaluation loop
eval_run_id = f"consistency_{datetime.now():%Y%m%d_%H%M%S}"
mlflow.set_experiment(f"/Shared/eval_consistency_unified")

all_field_metrics = []    # Per field per document
all_prompt_summaries = [] # Per prompt aggregate
all_run_data = []         # Raw run outputs

print(f"Eval run: {eval_run_id}")
print(f"{'='*70}")

for prompt_name in prompt_configs:
    cfg = PROMPT_CATALOGUE[prompt_name]
    docs = all_eval_data.get(prompt_name, [])
    output_fields = cfg["output_fields"]
    text_col = cfg["text_column"]
    id_cols = cfg["id_columns"]

    if not docs:
        print(f"\n[{prompt_name}] No documents — skipping.")
        continue

    total_calls = len(docs) * NUM_RUNS
    print(f"\n{'='*70}")
    print(f"[{prompt_name}] {len(docs)} docs x {NUM_RUNS} runs = {total_calls} LLM calls")
    print(f"{'='*70}")

    prompt_field_metrics = []
    t_start = time.time()

    with mlflow.start_run(run_name=f"{prompt_name}_{eval_run_id}"):
        mlflow.log_params({
            "eval_run_id": eval_run_id,
            "prompt_name": prompt_name,
            "prompt_type": cfg["prompt_type"],
            "prompt_id": prompt_configs[prompt_name]["prompt_id"],
            "model_endpoint": prompt_configs[prompt_name]["model_endpoint"],
            "temperature": prompt_configs[prompt_name]["temperature"],
            "num_runs": NUM_RUNS,
            "num_docs": len(docs),
        })

        for doc_idx, doc in enumerate(docs):
            doc_id = doc.get("DOCUMENT_ID", doc.get("PROPOSAL_ID", f"doc_{doc_idx}"))
            file_name = doc.get("FILE_NAME", "unknown")
            text_content = doc.get(text_col, "")

            if doc_idx % 5 == 0:
                elapsed = time.time() - t_start
                print(f"  [{doc_idx+1}/{len(docs)}] {file_name[:40]} ({elapsed:.0f}s elapsed)")

            # Run N times
            run_results = []
            for run_i in range(NUM_RUNS):
                result = call_llm_for_eval(text_content, prompt_name)
                result["_run"] = run_i
                run_results.append(result)

                # Store raw output
                raw_row = {
                    "eval_run_id": eval_run_id,
                    "prompt_name": prompt_name,
                    "document_id": doc_id,
                    "file_name": file_name,
                    "run_number": run_i,
                }
                for f in output_fields:
                    raw_row[f"val_{f}"] = str(result.get(f))
                all_run_data.append(raw_row)

            # Compute per-field consistency
            for field in output_fields:
                values = [r.get(field) for r in run_results]
                normed = [_norm(v) for v in values]
                modal_val = Counter(normed).most_common(1)[0][0]

                row = {
                    "eval_run_id": eval_run_id,
                    "prompt_name": prompt_name,
                    "document_id": doc_id,
                    "file_name": file_name,
                    "field": field,
                    "exact_match_rate": exact_match_rate(values),
                    "null_rate": null_rate(values),
                    "unique_ratio": unique_ratio(values),
                    "jaccard_mean": pairwise_jaccard_mean(values),
                    "entropy": entropy(values),
                    "modal_value": modal_val,
                    "num_runs": NUM_RUNS,
                }
                all_field_metrics.append(row)
                prompt_field_metrics.append(row)

        # Aggregate for this prompt
        pfm_df = pd.DataFrame(prompt_field_metrics)
        agg = {
            "mean_exact_match": pfm_df["exact_match_rate"].mean(),
            "mean_jaccard": pfm_df["jaccard_mean"].mean(),
            "mean_entropy": pfm_df["entropy"].mean(),
            "pct_above_threshold": (pfm_df["exact_match_rate"] >= CONSISTENCY_THRESHOLD).mean() * 100,
            "num_docs": len(docs),
            "num_fields": len(output_fields),
            "total_calls": total_calls,
            "elapsed_s": round(time.time() - t_start, 1),
        }

        mlflow.log_metrics({
            f"mean_exact_match": round(agg["mean_exact_match"], 4),
            f"mean_jaccard": round(agg["mean_jaccard"], 4),
            f"pct_above_threshold": round(agg["pct_above_threshold"], 1),
        })

        # Per-field metrics to MLflow
        field_ranking = (
            pfm_df.groupby("field")["exact_match_rate"]
            .mean()
            .sort_values()
            .reset_index()
            .rename(columns={"exact_match_rate": "mean_exact_match"})
        )
        for _, fr in field_ranking.iterrows():
            mlflow.log_metric(f"field_{fr['field']}", round(fr["mean_exact_match"], 4))

        # Log CSV artifacts
        for name, df in [("field_metrics", pfm_df), ("field_ranking", field_ranking)]:
            path = f"/tmp/{prompt_name}_{name}.csv"
            df.to_csv(path, index=False)
            mlflow.log_artifact(path)

        all_prompt_summaries.append({"prompt_name": prompt_name, **agg})
        print(f"  Done: {agg['mean_exact_match']:.2%} exact match, {agg['pct_above_threshold']:.0f}% above threshold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# DBTITLE 1,Prompt-level summary
summary_df = pd.DataFrame(all_prompt_summaries)
print("PROMPT-LEVEL CONSISTENCY SUMMARY")
print("=" * 80)
for _, row in summary_df.iterrows():
    status = "PASS" if row["pct_above_threshold"] >= 80 else "REVIEW"
    print(f"  [{status:6s}] {row['prompt_name']:30s}  exact={row['mean_exact_match']:.2%}  "
          f"jaccard={row['mean_jaccard']:.2%}  above_thresh={row['pct_above_threshold']:.0f}%  "
          f"({row['elapsed_s']:.0f}s)")

display(spark.createDataFrame(summary_df))

# COMMAND ----------

# DBTITLE 1,Worst fields across all prompts
all_fm_df = pd.DataFrame(all_field_metrics)

worst_fields = (
    all_fm_df.groupby(["prompt_name", "field"])
    .agg(
        mean_exact_match=("exact_match_rate", "mean"),
        mean_jaccard=("jaccard_mean", "mean"),
        mean_entropy=("entropy", "mean"),
        doc_count=("document_id", "nunique"),
    )
    .sort_values("mean_exact_match")
    .reset_index()
)

print("\nWORST FIELDS (bottom 20):")
display(spark.createDataFrame(worst_fields.head(20)))

# COMMAND ----------

# DBTITLE 1,Per-field bar chart by prompt
prompts_evaluated = summary_df["prompt_name"].tolist()
n_prompts = len(prompts_evaluated)

if n_prompts > 0:
    fig, axes = plt.subplots(1, n_prompts, figsize=(6 * n_prompts, 8), squeeze=False)

    for i, pname in enumerate(prompts_evaluated):
        ax = axes[0][i]
        subset = worst_fields[worst_fields["prompt_name"] == pname].sort_values("mean_exact_match")
        colors = ["#E24B4A" if v < CONSISTENCY_THRESHOLD else "#639922"
                  for v in subset["mean_exact_match"]]
        ax.barh(subset["field"], subset["mean_exact_match"], color=colors)
        ax.axvline(x=CONSISTENCY_THRESHOLD, color="#854F0B", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.05)
        ax.set_title(pname, fontsize=11)
        if i == 0:
            ax.set_xlabel("Mean exact match rate")

    plt.tight_layout()
    display(fig)
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta

# COMMAND ----------

# DBTITLE 1,Persist field metrics to Delta
field_metrics_sdf = spark.createDataFrame(all_fm_df.astype(str))
field_metrics_sdf = (
    field_metrics_sdf
    .withColumn("eval_ts", F.current_timestamp())
)

metrics_table = f"`{catalog}`.`{slr_schema}`.eval_consistency_metrics"

field_metrics_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(metrics_table)

print(f"Wrote {len(all_fm_df)} field metric rows to {metrics_table}")

# COMMAND ----------

# DBTITLE 1,Persist raw run data to Delta
if all_run_data:
    run_data_df = pd.DataFrame(all_run_data)
    run_data_sdf = spark.createDataFrame(run_data_df.astype(str))
    run_data_sdf = run_data_sdf.withColumn("eval_ts", F.current_timestamp())

    runs_table = f"`{catalog}`.`{slr_schema}`.eval_consistency_runs"

    run_data_sdf.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(runs_table)

    print(f"Wrote {len(run_data_df)} raw run rows to {runs_table}")

# COMMAND ----------

# DBTITLE 1,Final summary
print(f"\n{'='*70}")
print("UNIFIED CONSISTENCY EVALUATION — COMPLETE")
print(f"{'='*70}")
print(f"Eval run:           {eval_run_id}")
print(f"Prompts evaluated:  {len(prompts_evaluated)}")
print(f"Total documents:    {sum(len(all_eval_data.get(p, [])) for p in prompt_configs)}")
print(f"Total LLM calls:    {sum(s['total_calls'] for s in all_prompt_summaries)}")
print(f"Total time:         {sum(s['elapsed_s'] for s in all_prompt_summaries):.0f}s")
print(f"\nResults:")
print(f"  Metrics: {metrics_table}")
if all_run_data:
    print(f"  Runs:    {runs_table}")
print(f"  MLflow:  /Shared/eval_consistency_unified")
