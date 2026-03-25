# Databricks notebook source

# MAGIC %md
# MAGIC # Accuracy Evaluation — LLM-as-Judge on Consistency Results
# MAGIC
# MAGIC Reads the extracted values from `eval_01_consistency` runs, takes the **modal value**
# MAGIC (most common answer across N runs) for each field, and asks a stronger model (120b)
# MAGIC to judge whether each extraction is correct by reading the source document.
# MAGIC
# MAGIC **No re-extraction. No golden dataset. Just: read the document, read the extraction, judge it.**
# MAGIC
# MAGIC **Flow:**
# MAGIC 1. Load consistency run results from `eval_consistency_runs` (output of eval_01)
# MAGIC 2. Compute modal value per (document, field) across the N runs
# MAGIC 3. Load original source document text from snapshot tables
# MAGIC 4. For each (document, field): ask 120b "given this document, is this extraction correct?"
# MAGIC 5. Score: 1.0 = correct, 0.5 = partially correct, 0.0 = wrong
# MAGIC 6. Write results to Delta + MLflow

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "comm-afl-dev", "Catalog")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "Silver Schema")
dbutils.widgets.text("consistency_run_id", "", "Consistency eval_run_id (leave blank for latest)")

catalog = dbutils.widgets.get("catalog")
slr_schema = dbutils.widgets.get("slr_schema")
consistency_run_id = dbutils.widgets.get("consistency_run_id")

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
JUDGE_MODEL = "proposal-load-gpt-oss-120b"
print(f"Judge model: {JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Consistency Run Results

# COMMAND ----------

# DBTITLE 1,Load raw run data from eval_01
runs_table = f"`{catalog}`.`{slr_schema}`.eval_consistency_runs"

# Find the eval_run_id to use
if consistency_run_id:
    EVAL_RUN_ID = consistency_run_id
    print(f"Using specified run: {EVAL_RUN_ID}")
else:
    latest = spark.sql(f"""
        SELECT eval_run_id, MAX(eval_ts) as latest
        FROM {runs_table}
        GROUP BY eval_run_id
        ORDER BY latest DESC
        LIMIT 1
    """).collect()

    if not latest:
        print("ERROR: No consistency runs found. Run eval_01 first.")
        dbutils.notebook.exit("No consistency runs")

    EVAL_RUN_ID = latest[0]["eval_run_id"]
    print(f"Using latest run: {EVAL_RUN_ID}")

# Load all raw runs
runs_df = spark.sql(f"""
    SELECT * FROM {runs_table}
    WHERE eval_run_id = '{EVAL_RUN_ID}'
""").toPandas()

print(f"Loaded {len(runs_df)} raw run rows")
print(f"Prompts: {runs_df['prompt_name'].unique().tolist()}")
print(f"Documents: {runs_df['document_id'].nunique()}")

# COMMAND ----------

# DBTITLE 1,Identify which prompts and fields are available
# The runs table has columns like val_field_name for each extracted field
val_columns = [c for c in runs_df.columns if c.startswith("val_")]
field_names = [c.replace("val_", "") for c in val_columns]

prompts_available = runs_df["prompt_name"].unique().tolist()

print(f"Available prompts: {prompts_available}")
print(f"Fields found: {len(field_names)}")
print(f"  {field_names[:10]}{'...' if len(field_names) > 10 else ''}")

# COMMAND ----------

# DBTITLE 1,Select prompts to evaluate
PROMPTS_TO_EVALUATE = prompts_available  # evaluate everything from eval_01
# PROMPTS_TO_EVALUATE = ["extract_proposal"]  # or just one
print(f"Will evaluate: {PROMPTS_TO_EVALUATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Modal Values
# MAGIC
# MAGIC For each (document, field), find the most common value across the N consistency runs.
# MAGIC This is what the judge will assess.

# COMMAND ----------

# DBTITLE 1,Compute modal (most common) value per document per field per prompt
def _norm(val):
    if val is None:
        return "__NULL__"
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


modal_values = []

for prompt_name in PROMPTS_TO_EVALUATE:
    prompt_runs = runs_df[runs_df["prompt_name"] == prompt_name]
    doc_ids = prompt_runs["document_id"].unique()

    # Figure out which val_ columns have data for this prompt
    prompt_val_cols = [c for c in val_columns
                       if prompt_runs[c].notna().any()
                       and not (prompt_runs[c] == "None").all()]

    for doc_id in doc_ids:
        doc_runs = prompt_runs[prompt_runs["document_id"] == doc_id]
        file_name = doc_runs["file_name"].iloc[0] if "file_name" in doc_runs.columns else ""

        for val_col in prompt_val_cols:
            field = val_col.replace("val_", "")
            values = doc_runs[val_col].tolist()
            normed = [_norm(v) for v in values]

            # Modal value = most common
            counter = Counter(normed)
            modal_normed = counter.most_common(1)[0][0]

            # Get the original (non-normalised) version of the modal value
            modal_original = None
            for v, n in zip(values, normed):
                if n == modal_normed:
                    modal_original = v
                    break

            # Consistency rate for this field
            consistency = counter.most_common(1)[0][1] / len(normed)

            modal_values.append({
                "prompt_name": prompt_name,
                "document_id": doc_id,
                "file_name": file_name,
                "field": field,
                "modal_value": modal_original,
                "modal_value_normed": modal_normed,
                "consistency_rate": consistency,
                "num_runs": len(values),
            })

modal_df = pd.DataFrame(modal_values)
print(f"Computed {len(modal_df)} modal values")
print(f"  Prompts: {modal_df['prompt_name'].nunique()}")
print(f"  Documents: {modal_df['document_id'].nunique()}")
print(f"  Unique fields: {modal_df['field'].nunique()}")

# Filter out fields where modal value is NULL across all docs
null_fields = modal_df.groupby("field")["modal_value_normed"].apply(
    lambda x: (x == "__NULL__").all()
)
fields_all_null = null_fields[null_fields].index.tolist()
if fields_all_null:
    print(f"\n  Skipping {len(fields_all_null)} always-null fields: {fields_all_null[:5]}...")
    modal_df = modal_df[~modal_df["field"].isin(fields_all_null)]
    print(f"  Remaining: {len(modal_df)} values to judge")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Source Documents

# COMMAND ----------

# DBTITLE 1,Load source document text for judging
doc_source_table = f"`{catalog}`.`{slr_schema}`.eval_snap_doc_category"

doc_texts_df = spark.sql(f"""
    SELECT DOCUMENT_ID, PROPOSAL_ID, CONTEXT_PARSED, CATEGORY, FILE_NAME
    FROM {doc_source_table}
    WHERE CONTEXT_PARSED IS NOT NULL
""").toPandas()

# Build lookup: document_id -> source text
doc_text_lookup = {}
for _, row in doc_texts_df.iterrows():
    doc_text_lookup[row["DOCUMENT_ID"]] = row["CONTEXT_PARSED"]

print(f"Loaded source text for {len(doc_text_lookup)} documents")

# Check coverage
doc_ids_needed = set(modal_df["document_id"].unique())
doc_ids_available = set(doc_text_lookup.keys())
coverage = len(doc_ids_needed & doc_ids_available) / len(doc_ids_needed) * 100 if doc_ids_needed else 0
print(f"Document coverage: {coverage:.0f}% ({len(doc_ids_needed & doc_ids_available)}/{len(doc_ids_needed)})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Judge
# MAGIC
# MAGIC The 120b model reads the source document and the extracted value,
# MAGIC then judges whether the extraction is correct.

# COMMAND ----------

# DBTITLE 1,Judge prompt and function
JUDGE_PROMPT_TEMPLATE = """You are a quality reviewer for a UK loan document extraction system.

You are given:
1. A SOURCE DOCUMENT (the original text from a loan proposal)
2. A FIELD NAME (what was being extracted)
3. AN EXTRACTED VALUE (what the AI extracted for this field)

Your task: Read the source document carefully and determine whether the extracted value is correct.

Field name: {field_name}
Extracted value: {extracted_value}

Scoring:
- 1.0 = CORRECT — the extracted value accurately reflects what is in the document
- 0.5 = PARTIALLY CORRECT — the core information is right but there are minor issues (formatting, abbreviations, missing detail)
- 0.0 = INCORRECT — the value is wrong, fabricated, or not found in the document

Rules:
- If the field genuinely has no value in the document and the extraction is null/None, score 1.0
- If the field has a value in the document but the extraction is null/None, score 0.0
- For UK addresses: abbreviations like St/Street, Rd/Road are acceptable (1.0)
- For company names: Ltd/Limited, PLC/Public Limited Company are equivalent (1.0)
- For phone numbers: formatting differences (+44 vs 0) are acceptable (1.0)
- For monetary values: minor rounding differences are acceptable (1.0), wrong amounts are 0.0
- Read the ENTIRE document before scoring — the relevant information may appear anywhere

SOURCE DOCUMENT:
{document_text}

Respond ONLY with JSON: {{"score": <float>, "reasoning": "<one sentence explaining your judgement>"}}"""


def judge_extraction(field_name: str, extracted_value: str, document_text: str) -> tuple:
    """
    Ask the 120b judge to assess whether an extracted value is correct
    given the source document.
    Returns: (score, reasoning)
    """
    # Skip judging null extractions from null-like fields
    if extracted_value is None or _norm(extracted_value) == "__null__":
        # We can't judge null extractions without reading the doc — let the LLM decide
        pass

    try:
        # Truncate document to fit context window
        doc_truncated = document_text[:6000] if document_text else ""
        extracted_str = str(extracted_value) if extracted_value else "null"

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            field_name=field_name,
            extracted_value=extracted_str,
            document_text=doc_truncated,
        )

        prompt_df = spark.createDataFrame([(prompt,)], ["judge_prompt"])

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

        result_df = prompt_df.selectExpr(f"""
            ai_query('{JUDGE_MODEL}', judge_prompt,
                     responseFormat => '{resp_fmt}') AS resp
        """)

        raw = result_df.collect()[0]["resp"]
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        reasoning = str(parsed.get("reasoning", ""))
        return score, reasoning

    except Exception as e:
        print(f"    JUDGE ERROR on {field_name}: {e}")
        return 0.0, f"Judge failed: {e}"


print(f"judge_extraction() defined — using {JUDGE_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Accuracy Evaluation

# COMMAND ----------

# DBTITLE 1,Main judge loop
eval_run_id = f"judge_{datetime.now():%Y%m%d_%H%M%S}"
all_scores = []
judge_calls = 0

# Only judge documents where we have source text
judgeable = modal_df[modal_df["document_id"].isin(doc_text_lookup)]
total_judgements = len(judgeable)

print(f"Eval run: {eval_run_id}")
print(f"Judge model: {JUDGE_MODEL}")
print(f"Total field-values to judge: {total_judgements}")
print("=" * 70)

t_start = time.time()

for prompt_name in PROMPTS_TO_EVALUATE:
    prompt_modal = judgeable[judgeable["prompt_name"] == prompt_name]

    if len(prompt_modal) == 0:
        print(f"\n[{prompt_name}] No judgeable values — skipping.")
        continue

    doc_ids = prompt_modal["document_id"].unique()
    print(f"\n{'='*70}")
    print(f"[{prompt_name}] {len(doc_ids)} docs, {len(prompt_modal)} field-values to judge")
    print(f"{'='*70}")

    prompt_t0 = time.time()

    for doc_idx, doc_id in enumerate(doc_ids):
        doc_text = doc_text_lookup.get(doc_id, "")
        doc_fields = prompt_modal[prompt_modal["document_id"] == doc_id]

        if doc_idx % 5 == 0:
            elapsed = time.time() - prompt_t0
            print(f"  [{doc_idx+1}/{len(doc_ids)}] ({elapsed:.0f}s elapsed, {judge_calls} judge calls)")

        for _, row in doc_fields.iterrows():
            field = row["field"]
            modal_val = row["modal_value"]

            score, reasoning = judge_extraction(field, modal_val, doc_text)
            judge_calls += 1

            all_scores.append({
                "eval_run_id": eval_run_id,
                "consistency_run_id": EVAL_RUN_ID,
                "prompt_name": prompt_name,
                "document_id": doc_id,
                "file_name": row.get("file_name", ""),
                "field": field,
                "modal_value": str(modal_val),
                "score": score,
                "reasoning": reasoning,
                "consistency_rate": row["consistency_rate"],
                "num_runs": row["num_runs"],
            })

    elapsed = time.time() - prompt_t0
    prompt_scores = [s for s in all_scores if s["prompt_name"] == prompt_name]
    mean = np.mean([s["score"] for s in prompt_scores]) if prompt_scores else 0
    print(f"  Done: mean accuracy={mean:.2%} ({elapsed:.0f}s)")

total_elapsed = time.time() - t_start
print(f"\nTotal: {len(all_scores)} scores, {judge_calls} judge calls, {total_elapsed:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

# DBTITLE 1,Build results
if not all_scores:
    print("No scores generated.")
    dbutils.notebook.exit("No scores")

scores_df = pd.DataFrame(all_scores)
print(f"Overall accuracy (judged by 120b): {scores_df['score'].mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Per-prompt accuracy
prompt_accuracy = (
    scores_df.groupby("prompt_name")
    .agg(
        mean_accuracy=("score", "mean"),
        pct_correct=("score", lambda x: (x >= 0.5).mean()),
        pct_perfect=("score", lambda x: (x >= 1.0).mean()),
        mean_consistency=("consistency_rate", "mean"),
        fields_judged=("score", "count"),
    )
    .sort_values("mean_accuracy")
    .reset_index()
)

print("PER-PROMPT ACCURACY (judged by 120b)")
print("=" * 80)
for _, row in prompt_accuracy.iterrows():
    print(f"  {row['prompt_name']:30s}  accuracy={row['mean_accuracy']:.2%}  "
          f"correct={row['pct_correct']:.2%}  perfect={row['pct_perfect']:.2%}  "
          f"consistency={row['mean_consistency']:.2%}  n={row['fields_judged']:.0f}")

display(spark.createDataFrame(prompt_accuracy))

# COMMAND ----------

# DBTITLE 1,Per-field accuracy
field_accuracy = (
    scores_df.groupby(["prompt_name", "field"])
    .agg(
        mean_accuracy=("score", "mean"),
        pct_correct=("score", lambda x: (x >= 0.5).mean()),
        pct_perfect=("score", lambda x: (x >= 1.0).mean()),
        mean_consistency=("consistency_rate", "mean"),
        count=("score", "count"),
    )
    .sort_values("mean_accuracy")
    .reset_index()
)

print(f"\nPer-field accuracy (bottom 20):")
display(spark.createDataFrame(field_accuracy.head(20)))

# COMMAND ----------

# DBTITLE 1,Consistency vs Accuracy cross-reference
# Fields that are consistent but inaccurate are systematically wrong
# Fields that are inconsistent but accurate (on average) have variance issues
cross_ref = field_accuracy.copy()
cross_ref["consistent_but_wrong"] = (cross_ref["mean_consistency"] >= 0.85) & (cross_ref["mean_accuracy"] < 0.75)
cross_ref["inconsistent_but_right"] = (cross_ref["mean_consistency"] < 0.85) & (cross_ref["mean_accuracy"] >= 0.75)

sys_wrong = cross_ref[cross_ref["consistent_but_wrong"]]
if len(sys_wrong) > 0:
    print("SYSTEMATICALLY WRONG (consistent but inaccurate — prompt issue):")
    for _, row in sys_wrong.iterrows():
        print(f"  {row['prompt_name']:25s} / {row['field']:25s}  "
              f"consistency={row['mean_consistency']:.2%}  accuracy={row['mean_accuracy']:.2%}")

var_issues = cross_ref[cross_ref["inconsistent_but_right"]]
if len(var_issues) > 0:
    print("\nVARIANCE ISSUES (inconsistent but often correct — temperature/sampling issue):")
    for _, row in var_issues.iterrows():
        print(f"  {row['prompt_name']:25s} / {row['field']:25s}  "
              f"consistency={row['mean_consistency']:.2%}  accuracy={row['mean_accuracy']:.2%}")

if len(sys_wrong) == 0 and len(var_issues) == 0:
    print("No systematic issues detected.")

# COMMAND ----------

# DBTITLE 1,Lowest-scoring extractions with judge reasoning
worst = scores_df[scores_df["score"] < 1.0].sort_values("score")
if len(worst) > 0:
    cols = ["prompt_name", "file_name", "field", "modal_value",
            "score", "reasoning", "consistency_rate"]
    display(spark.createDataFrame(worst.head(30)[cols]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualisations

# COMMAND ----------

# DBTITLE 1,Per-prompt field accuracy charts
prompts = prompt_accuracy["prompt_name"].tolist()
n = len(prompts)

if n > 0:
    fig, axes = plt.subplots(1, min(n, 3), figsize=(6 * min(n, 3), 8), squeeze=False)
    for i, pname in enumerate(prompts[:3]):
        ax = axes[0][i]
        subset = field_accuracy[field_accuracy["prompt_name"] == pname].sort_values("mean_accuracy")
        colors = ["#E24B4A" if v < 0.75 else "#639922" for v in subset["mean_accuracy"]]
        ax.barh(subset["field"], subset["mean_accuracy"], color=colors)
        ax.axvline(x=0.90, color="#185FA5", linestyle="--", linewidth=1, label="90%")
        ax.axvline(x=0.75, color="#854F0B", linestyle=":", linewidth=1, label="75%")
        ax.set_xlim(0, 1.05)
        ax.set_title(pname, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    display(fig)
    plt.close()

# COMMAND ----------

# DBTITLE 1,Consistency vs Accuracy scatter
if len(field_accuracy) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))

    for pname in prompts:
        subset = field_accuracy[field_accuracy["prompt_name"] == pname]
        ax.scatter(subset["mean_consistency"], subset["mean_accuracy"],
                   label=pname, s=60, alpha=0.7)

    ax.axhline(y=0.75, color="#854F0B", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=0.85, color="#854F0B", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Consistency (from eval_01)")
    ax.set_ylabel("Accuracy (judged by 120b)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Consistency vs Accuracy per field")

    # Quadrant labels
    ax.text(0.42, 0.35, "Inconsistent + Inaccurate", fontsize=9, color="#888", ha="center")
    ax.text(0.95, 0.35, "Consistent + Inaccurate", fontsize=9, color="#888", ha="center")
    ax.text(0.42, 0.90, "Inconsistent + Accurate", fontsize=9, color="#888", ha="center")
    ax.text(0.95, 0.90, "Consistent + Accurate", fontsize=9, color="#888", ha="center")

    plt.tight_layout()
    display(fig)
    plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow & Delta

# COMMAND ----------

# DBTITLE 1,Log to MLflow
mlflow.set_experiment("/Shared/eval_accuracy_judge")

with mlflow.start_run(run_name=f"judge_{eval_run_id}"):
    mlflow.log_metrics({
        "overall_accuracy": round(scores_df["score"].mean(), 4),
        "pct_correct": round((scores_df["score"] >= 0.5).mean(), 4),
        "pct_perfect": round((scores_df["score"] >= 1.0).mean(), 4),
        "total_judged": len(scores_df),
        "judge_calls": judge_calls,
        "prompts_evaluated": len(prompts),
    })
    mlflow.log_params({
        "eval_run_id": eval_run_id,
        "consistency_run_id": EVAL_RUN_ID,
        "judge_model": JUDGE_MODEL,
        "prompts": ",".join(prompts),
    })
    for name, df in [("prompt_accuracy", prompt_accuracy),
                     ("field_accuracy", field_accuracy)]:
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
    "mergeSchema", "true").saveAsTable(scores_table)
print(f"Wrote {len(scores_df)} scores to {scores_table}")

# COMMAND ----------

# DBTITLE 1,Write field summary to Delta
field_acc_sdf = spark.createDataFrame(field_accuracy.astype(str))
field_acc_sdf = (field_acc_sdf
    .withColumn("eval_run_id", F.lit(eval_run_id))
    .withColumn("consistency_run_id", F.lit(EVAL_RUN_ID))
    .withColumn("judge_model", F.lit(JUDGE_MODEL))
    .withColumn("eval_ts", F.current_timestamp()))

summary_table = f"`{catalog}`.`{slr_schema}`.eval_accuracy_summary"
field_acc_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true").saveAsTable(summary_table)
print(f"Wrote summary to {summary_table}")

# COMMAND ----------

# DBTITLE 1,Final summary
print(f"\n{'='*70}")
print("LLM-AS-JUDGE ACCURACY EVALUATION — COMPLETE")
print(f"{'='*70}")
print(f"Eval run:              {eval_run_id}")
print(f"Consistency run:       {EVAL_RUN_ID}")
print(f"Judge model:           {JUDGE_MODEL}")
print(f"Prompts evaluated:     {len(prompts)}")
print(f"Fields judged:         {len(scores_df)}")
print(f"Judge calls:           {judge_calls}")
print(f"")
print(f"Overall accuracy:      {scores_df['score'].mean():.2%}")
print(f"Correct (>=0.5):       {(scores_df['score'] >= 0.5).mean():.2%}")
print(f"Perfect (=1.0):        {(scores_df['score'] >= 1.0).mean():.2%}")

# Per-prompt summary
print(f"\nPer-prompt:")
for _, row in prompt_accuracy.iterrows():
    print(f"  {row['prompt_name']:30s}  accuracy={row['mean_accuracy']:.2%}  "
          f"consistency={row['mean_consistency']:.2%}")

# Worst fields
worst_fields = field_accuracy[field_accuracy["mean_accuracy"] < 0.75]
if len(worst_fields) > 0:
    print(f"\nFIELDS BELOW 75% ACCURACY ({len(worst_fields)}):")
    for _, row in worst_fields.iterrows():
        print(f"  {row['prompt_name']:25s} / {row['field']:25s}  {row['mean_accuracy']:.2%}")

print(f"\nDelta: {scores_table}")
print(f"       {summary_table}")
print(f"MLflow: /Shared/eval_accuracy_judge")
