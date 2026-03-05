# COMMAND ----------

# DBTITLE 1,Eval config & additional imports
import time
import numpy as np
import pandas as pd
import mlflow
from collections import Counter
from datetime import datetime

NUM_RUNS = 5  # number of repeated LLM calls per document
EVAL_EXPERIMENT_NAME = "/Shared/llm_extraction_consistency_eval"

EXPECTED_KEYS = [
    "loan_date", "broker", "proposer", "proposer_address",
    "proposer_trading_address", "proposer_registered_address",
    "proposer_email", "proposer_phone", "proposer_website",
    "sic_code", "crn", "proposer_year_established", "legal_form",
    "loan_amount", "currency", "finance_type", "regulated",
    "rate_type", "deal_type", "term_type", "loan_period",
    "payment_frequency", "vat_number", "initial_payment",
    "vat_payment", "vat_deferral", "monthly_payment",
    "balloon_payment", "confidence_score",
]

FIELD_GROUPS = {
    "identity":  ["broker", "proposer", "crn", "sic_code", "legal_form",
                   "proposer_year_established"],
    "contact":   ["proposer_address", "proposer_trading_address",
                   "proposer_registered_address", "proposer_email",
                   "proposer_phone", "proposer_website"],
    "financial": ["loan_amount", "currency", "finance_type", "rate_type",
                   "deal_type", "loan_period", "payment_frequency",
                   "initial_payment", "vat_payment", "vat_deferral",
                   "monthly_payment", "balloon_payment"],
    "metadata":  ["loan_date", "term_type", "regulated", "vat_number",
                   "confidence_score"],
}

# COMMAND ----------

# DBTITLE 1,LLM caller for evaluation
def call_llm_for_eval(context_text: str) -> dict:
    """
    Call the LLM once via ai_query on a single document.
    Returns the parsed JSON dict.
    """
    prompt_df = spark.createDataFrame([(context_text,)], ["context_parsed"])
    result_df = prompt_df.selectExpr(f"""
        ai_query(
            endpoint => '{LLM_ENDPOINT_NAME}',
            request => CONCAT('{PROMPT}', context_parsed),
            responseFormat => '{response_format}',
            modelParameters => named_struct('temperature', 0.3)
        ) AS ai_response
    """)
    raw = result_df.collect()[0]["ai_response"]
    return json.loads(raw)

# COMMAND ----------

# DBTITLE 1,Consistency metric functions
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

def _jaccard(a, b):
    """Token-level Jaccard similarity."""
    sa, sb = set(_norm(a).split()), set(_norm(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def pairwise_jaccard_mean(values):
    """Mean pairwise Jaccard across all run pairs."""
    n = len(values)
    if n < 2:
        return 1.0
    sims = [_jaccard(str(values[i]), str(values[j]))
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))

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

def entropy(values):
    """Shannon entropy — 0 means perfectly consistent."""
    n = len(values)
    if n == 0:
        return 0.0
    counter = Counter(_norm(v) for v in values)
    probs = [c / n for c in counter.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))

# COMMAND ----------

# DBTITLE 1,Run consistency evaluation
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
        "temperature": 0.3,
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
            print(f"  Run {ri+1}/{NUM_RUNS} — {elapsed:.1f}s")

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
                "pairwise_jaccard": round(pairwise_jaccard_mean(vals), 4),
                "entropy": round(entropy(vals), 4),
                "coefficient_of_variation": round(cv, 4) if cv is not None else None,
                "modal_value": Counter(_norm(v) for v in vals).most_common(1)[0][0],
                "all_values": str([_norm(v) for v in vals]),
            })
        all_field_rows.extend(doc_rows)

        # --- Document-level summary ---
        fdf = pd.DataFrame(doc_rows)
        summary = {
            "proposal_id": pid,
            "mean_exact_match_rate": round(fdf["exact_match_rate"].mean(), 4),
            "min_exact_match_rate": round(fdf["exact_match_rate"].min(), 4),
            "mean_pairwise_jaccard": round(fdf["pairwise_jaccard"].mean(), 4),
            "mean_entropy": round(fdf["entropy"].mean(), 4),
            "mean_null_rate": round(fdf["null_rate"].mean(), 4),
            "fully_consistent_fields": int((fdf["exact_match_rate"] == 1.0).sum()),
            "total_fields": len(EXPECTED_KEYS),
            "pct_fully_consistent": round(
                (fdf["exact_match_rate"] == 1.0).sum() / len(EXPECTED_KEYS) * 100, 2),
            "mean_latency_s": round(np.mean(latencies), 3),
        }
        for gname, gkeys in FIELD_GROUPS.items():
            gdf = fdf[fdf["field"].isin(gkeys)]
            if len(gdf) > 0:
                summary[f"em_{gname}"] = round(gdf["exact_match_rate"].mean(), 4)
        all_doc_rows.append(summary)

        # Log child run per document
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
        "overall_mean_jaccard": round(field_metrics_df["pairwise_jaccard"].mean(), 4),
        "overall_mean_entropy": round(field_metrics_df["entropy"].mean(), 4),
        "overall_pct_fully_consistent": round(
            (field_metrics_df["exact_match_rate"] == 1.0).sum() / len(field_metrics_df) * 100, 2),
    }
    mlflow.log_metrics(agg_metrics)

    # Save CSV artifacts
    for name, df in [("field_metrics", field_metrics_df),
                     ("doc_metrics", doc_metrics_df),
                     ("field_ranking", field_ranking)]:
        path = f"/tmp/{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)

# COMMAND ----------

# DBTITLE 1,Evaluation summary
print(f"{'='*60}")
print("CONSISTENCY EVALUATION SUMMARY")
print(f"{'='*60}")
print(f"Documents evaluated:  {len(eval_docs)}")
print(f"Runs per document:    {NUM_RUNS}")
print(f"Overall exact match:  {agg_metrics['overall_mean_exact_match']:.2%}")
print(f"Overall Jaccard:      {agg_metrics['overall_mean_jaccard']:.2%}")
print(f"Overall entropy:      {agg_metrics['overall_mean_entropy']:.4f}")
print(f"Fully consistent:     {agg_metrics['overall_pct_fully_consistent']:.1f}% of field-runs")

# COMMAND ----------

# DBTITLE 1,Least consistent fields (worst first)
display(spark.createDataFrame(field_ranking))

# COMMAND ----------

# DBTITLE 1,Per-document results
display(spark.createDataFrame(doc_metrics_df))

# COMMAND ----------

# DBTITLE 1,Full field-level detail
display(spark.createDataFrame(field_metrics_df.drop(columns=["all_values"])))
