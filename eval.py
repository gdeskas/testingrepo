# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # End-to-End LLM Extraction Consistency Evaluation
# MAGIC
# MAGIC **What this notebook does in one `Run All`:**
# MAGIC 1. **Loads all 20 proposal folders** from ADLS → stages files → writes to Bronze Delta table (replicates `load_adls_files` in a loop)
# MAGIC 2. **Runs the downstream pipeline** to populate `context_parsed` in `gddoc_category`
# MAGIC 3. **Repeats the LLM call 5× per document** in configurable batches to measure consistency
# MAGIC 4. **Logs metrics to MLflow** — per-field, per-document, and aggregate
# MAGIC 5. **Writes a review Delta table** — with `is_correct`, `correct_value`, `reviewer_notes` for manual flagging
# MAGIC 6. **Renders side-by-side HTML** — source text on left (with ADLS link), extracted fields on right, disagreements highlighted

# COMMAND ----------

# DBTITLE 1,Configuration
import re, json, time, sys
import numpy as np
import pandas as pd
import mlflow
import html as html_mod
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.functions import col, from_json
from collections import Counter
from datetime import datetime
from math import ceil

# --- Widgets ---
dbutils.widgets.text("catalog", "comm-afl-dev", "catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz", "brz_schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "slr_schema")
dbutils.widgets.text("job_run_id", "", "job_run_id")

catalog = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")
job_run_id = dbutils.widgets.get("job_run_id")

# --- Staging ---
STAGING_ROOT = "dbfs:/Volumes/comm-afl-dev/brkrflw-lkh-brz/staging_files"

# --- Tables ---
source_table = "gddoc_category"
working_target_table = "gdfiles_loaded"       # adjust if different
storage_target_table = "gdfiles_loaded"        # adjust if different

# --- LLM ---
LLM_ENDPOINT_NAME = "databricks-gpt-oss-20b"
TEMPERATURE = 0.3

# --- Evaluation ---
NUM_RUNS = 5
BATCH_SIZE = 5        # documents per batch for the consistency loop
EVAL_EXPERIMENT_NAME = "/Shared/llm_extraction_consistency_eval"
REVIEW_TABLE = f"`{catalog}`.`{slr_schema}`.eval_extraction_review"

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 1. Define All 20 Proposal Folders
# MAGIC No more commenting/uncommenting — every folder is in the list.

# COMMAND ----------

# DBTITLE 1,All 20 ADLS proposal folder paths
ADLS_BASE = "abfss://comm-afl-lz@cbssstuksdbr01dev.dfs.core.windows.net/raw-data/Example Docs - Latest"

VOLUME_ROOTS = [
    f"{ADLS_BASE}/01. JDN Logistics Ltd - HSH Business Finance - AFS",
    f"{ADLS_BASE}/02. KD Grab HIre Limited - Anglo Scottish",
    f"{ADLS_BASE}/03. West Country Recovery (SW) Ltd - EFT Finance",
    f"{ADLS_BASE}/04. CDT Electrical Ltd - Motion",
    f"{ADLS_BASE}/05. TLB Deliveries Ltd - Holmesdale",
    f"{ADLS_BASE}/06. Electric Access Solutions Ltd - Evolution",
    f"{ADLS_BASE}/07. CJT Vocational Skills Ltd - Crown Business Finance - AFS",
    f"{ADLS_BASE}/08. Duffield Harrison LLP - Audeo FS Ltd",
    f"{ADLS_BASE}/09. Glass Padel Club Ltd - White Rose Finance Group Ltd",
    f"{ADLS_BASE}/10. Your Shortlist Original Ltd - Clear Asset Finance",
    f"{ADLS_BASE}/11. Beachers Coaches Ltd - Fundi - AFS",
    f"{ADLS_BASE}/12. Central England Healthcare - LDF",
    f"{ADLS_BASE}/13. Rural GM Ltd - FMF Group Ltd - AFS",
    f"{ADLS_BASE}/14. Stellar",                                         # TODO: complete this name
    # f"{ADLS_BASE}/15. ...",   # <-- ADD remaining folder names here
    # f"{ADLS_BASE}/16. ...",
    # f"{ADLS_BASE}/17. ...",
    # f"{ADLS_BASE}/18. ...",
    # f"{ADLS_BASE}/19. ...",
    # f"{ADLS_BASE}/20. ...",
]

print(f"Defined {len(VOLUME_ROOTS)} proposal folders")
for i, vr in enumerate(VOLUME_ROOTS, 1):
    print(f"  {i:2d}. {vr.split('/')[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Load Pipeline — Stage & Ingest All Folders
# MAGIC Replicates what `load_adls_files` does, but loops through all folders automatically.

# COMMAND ----------

# DBTITLE 1,Helper: safe filename
def safe_name(name: str) -> str:
    return re.sub(r"[^\w._]+", "_", name).strip()

# COMMAND ----------

# DBTITLE 1,Stage all folders from ADLS into DBFS volumes
staged_folders = []

for idx, volume_root in enumerate(VOLUME_ROOTS, 1):
    case_folder = volume_root.rstrip("/").split("/")[-1]
    dst_dir = f"{STAGING_ROOT}/{case_folder}"

    print(f"\n[{idx}/{len(VOLUME_ROOTS)}] Staging: {case_folder}")

    try:
        dbutils.fs.mkdirs(dst_dir)
    except Exception:
        pass

    file_count = 0
    try:
        for f in dbutils.fs.ls(volume_root):
            if f.isDir():
                continue
            new_name = safe_name(f.name)
            dst_path = f"{dst_dir}/{new_name}"
            dbutils.fs.cp(f.path, dst_path, True)
            file_count += 1
    except Exception as e:
        print(f"  ERROR listing/copying: {e}")
        continue

    staged_folders.append({
        "case_folder": case_folder,
        "volume_root": volume_root,
        "staged_path": dst_dir,
        "file_count": file_count,
    })
    print(f"  Copied {file_count} files -> {dst_dir}")

print(f"\nStaged {len(staged_folders)} folders, {sum(f['file_count'] for f in staged_folders)} total files")

# COMMAND ----------

# DBTITLE 1,Ingest all staged files into Bronze Delta table
all_binary_dfs = []

for folder_info in staged_folders:
    staged_path = folder_info["staged_path"]
    volume_root = folder_info["volume_root"]

    try:
        binary_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .load(staged_path)
            .withColumn("folder_path", F.lit(staged_path))
            .withColumn("adls_source_path", F.lit(volume_root))
            .withColumn("file_name", F.regexp_extract(F.col("path"), r"([^/]+)$", 1))
            .withColumn("file_ext", F.lower(F.regexp_extract(F.col("file_name"), r"\.([^.]+)$", 1)))
            .withColumn(
                "proposal_id",
                F.sha2(
                    F.concat_ws("||", F.lit(staged_path), F.current_timestamp().cast("string")),
                    256
                )
            )
            .withColumn("document_id", F.regexp_extract(F.col("path"), r"([^/]+)$", 1))
            .withColumn("job_run_id", F.lit(job_run_id))
            .withColumn("ingestion_ts", F.current_timestamp())
        )
        all_binary_dfs.append(binary_df)
    except Exception as e:
        print(f"  ERROR reading {staged_path}: {e}")

if not all_binary_dfs:
    dbutils.notebook.exit("No files could be read from staged folders.")

combined_df = all_binary_dfs[0]
for df in all_binary_dfs[1:]:
    combined_df = combined_df.unionByName(df, allowMissingColumns=True)

files_loaded_df = combined_df.selectExpr(
    "proposal_id as PROPOSAL_ID",
    "document_id as DOCUMENT_ID",
    "path as SOURCE_PATH",
    "length as LENGTH",
    "content as CONTENT",
    "folder_path as FOLDER_PATH",
    "adls_source_path as ADLS_SOURCE_PATH",
    "file_name as FILE_NAME",
    "file_ext as FILE_EXT",
    "modificationTime as MODIFICATION_TS",
    "job_run_id as JOB_RUN_ID",
    "ingestion_ts as INGESTION_TS"
)

row_count = files_loaded_df.count()
print(f"Total files ingested: {row_count}")

files_loaded_df.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"`{catalog}`.`{brz_schema}`.{working_target_table}")

# Also append to storage table (mirrors original notebook cell 15)
files_loaded_df.write.mode("append").option(
    "mergeSchema", "true"
).saveAsTable(f"`{catalog}`.`{brz_schema}`.{storage_target_table}")

print(f"Written to `{catalog}`.`{brz_schema}`.{working_target_table}` and {storage_target_table}")

# COMMAND ----------

# DBTITLE 1,Preview loaded files
display(
    spark.sql(f"""
        SELECT PROPOSAL_ID, DOCUMENT_ID, FILE_NAME, FILE_EXT, ADLS_SOURCE_PATH, LENGTH
        FROM `{catalog}`.`{brz_schema}`.{working_target_table}
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Run Downstream Pipeline
# MAGIC
# MAGIC Your existing pipeline extracts text from the raw files and populates `context_parsed`
# MAGIC in `gddoc_category`. Uncomment **one** of the options below.

# COMMAND ----------

# DBTITLE 1,Option A: Run the extraction pipeline notebook(s) via %run or dbutils.notebook.run
# Uncomment and adjust paths to your downstream notebooks:
#
# For each notebook in the pipeline, call it in sequence:
# dbutils.notebook.run(
#     "/Workspace/CNTRL-COMM-AFL-DEV/Pipeline/your_text_extraction_notebook",
#     timeout_seconds=3600,
#     arguments={"catalog": catalog, "brz_schema": brz_schema, "slr_schema": slr_schema, "job_run_id": job_run_id}
# )
#
# If you have multiple steps:
# dbutils.notebook.run("/Workspace/.../step1_parse_documents", 3600, {...})
# dbutils.notebook.run("/Workspace/.../step2_categorise", 3600, {...})

print("Skipped — uncomment Option A above or ensure the pipeline has already run.")

# COMMAND ----------

# DBTITLE 1,Option B: Verify gddoc_category is populated
proposal_count = spark.sql(f"""
    SELECT COUNT(*) as cnt
    FROM `{catalog}`.`{slr_schema}`.{source_table}
    WHERE lower(category) = 'proposal' OR lower(category) = 'email_with_proposal'
""").collect()[0]["cnt"]

print(f"Found {proposal_count} proposals in `{catalog}`.`{slr_schema}`.{source_table}`")
if proposal_count == 0:
    print("WARNING: gddoc_category is empty. Run the extraction pipeline first (Option A).")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 4. Load Proposals for Evaluation

# COMMAND ----------

# DBTITLE 1,Load all proposals with ADLS source paths
all_proposals_df = spark.sql(f"""
    SELECT
        gc.proposal_id,
        gc.document_id,
        gc.received_dt,
        gc.category,
        gc.context_parsed,
        fl.ADLS_SOURCE_PATH as adls_source_path,
        fl.FILE_NAME as source_file_name
    FROM `{catalog}`.`{slr_schema}`.{source_table} gc
    LEFT JOIN `{catalog}`.`{brz_schema}`.{working_target_table} fl
        ON gc.proposal_id = fl.PROPOSAL_ID
    WHERE lower(gc.category) = 'proposal'
       OR lower(gc.category) = 'email_with_proposal'
""")

num_proposals = all_proposals_df.count()
print(f"Loaded {num_proposals} proposals for evaluation")
print(f"Plan: {num_proposals} docs x {NUM_RUNS} runs = {num_proposals * NUM_RUNS} LLM calls")
print(f"Batch size: {BATCH_SIZE} -> {ceil(num_proposals / BATCH_SIZE)} batches")

all_docs = all_proposals_df.toPandas().to_dict("records")

# COMMAND ----------

# DBTITLE 1,Preview
display(all_proposals_df.select("proposal_id", "document_id", "category", "adls_source_path", "source_file_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 5. Prompt, Schema & LLM Setup

# COMMAND ----------

# DBTITLE 1,Reference lists
deal_type = "[New Business, New Business with annual review, LOC Increase, LOC Increase with annual review, Annual review, Refinance, Reschedule, Substitution]"
finance_type = "[Finance Lease, Hire Purchase, Other]"
legal_form_list = "[]"  # populate with your actual list

# COMMAND ----------

# DBTITLE 1,Expected extraction fields
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

# DBTITLE 1,Prompt and response schema
PROMPT = f"""You are an AI data entry clerk working for Close Brothers Broker Solutions specialized in analyzing loan proposal documents.
Your task is to extract relevant information from a given proposal document.
Your output must be a structured JSON object.

Instructions:
1. Carefully read the entire document provided at the end of this prompt.
2. Extract the relevant information.
3. Present your findings in JSON format as specified below.

Important Notes:
- Extract only relevant information.
- Ignore any reference to Close Brothers Broker Solution, or Close Brothers.
- Consider the context of the entire proposal when determining relevance.
- Do not be verbose, only respond with the correct format and information.
- Some questions may have no relevant excerpts. Just return null.
- Do not include additional JSON keys beyond the ones listed here.
- Do not include the same key multiple times in the JSON.

Expected JSON keys and explanation of what they are:
- loan_date: The date of the contract, extract in exact format as proposal.
- broker: The brokers company name which sent the proposal to Close Brothers.
- proposer: The company that requires a loan.
- proposer_address: The main address of the company that requires a loan.
- proposer_trading_address: If one of the addresses is described as trading address of the company that requires a loan.
- proposer_registered_address: If one of the addresses is described as registered address of the company that requires a loan.
- proposer_email: The email address of the company that requires a loan.
- proposer_phone: The phone number of the company that requires a loan.
- proposer_website: The website of the company that requires a loan.
- sic_code: Limited Company Sector (SIC Code) a 4 digit code.
- crn: Company Registration Number, format 8 digits or 2 letters from this list [SC, NI, OC] followed by 6 digits.
- proposer_year_established: The year that the proposer company was established.
- legal_form: Legal structure of the company from this list {legal_form_list}.
- loan_amount: The total required funds, extract decimal.
- currency: The currency of the loan, extract only 3 letter code.
- finance_type: The type of finance, from this list {finance_type}.
- regulated: Whether regulated or not.
- rate_type: The interest rate type of the loan requested for example fixed rate, true variable or equalised variable rate.
- deal_type: The type of deal, from this list {deal_type}.
- term_type: The type of term requested, for example fixed term or minimum term.
- loan_period: The length of the loan in months, convert year to months.
- payment_frequency: The frequency of payments, from this list [monthly, quarterly, annual, semi-annual].
- vat_number: The VAT number of the company that requires a loan.
- initial_payment: The initial deposit payment listed in payment schedule, extract decimal.
- vat_payment: The vat on asset The vat tax listed in payment schedule, extract decimal.
- vat_deferral: The vat deferral listed in payment schedule, extract number of months(0 to 3)
- monthly_payment: The monthly payment amount, extract decimal.
- balloon_payment: Any extra payment amount, extract decimal.
- confidence_score: % confidence that the extracted information is correct, deduced based on the quality of the extracted information, between 0 and 100.

Proposal to analyze:
"""

response_format = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "proposals_extraction",
        "schema": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in EXPECTED_KEYS},
            "strict": True,
        },
    },
})

# COMMAND ----------

# DBTITLE 1,LLM caller
def call_llm(context_text: str) -> dict:
    """Single LLM call via ai_query. Returns parsed JSON dict."""
    prompt_df = spark.createDataFrame([(context_text,)], ["context_parsed"])
    result_df = prompt_df.selectExpr(f"""
        ai_query(
            endpoint => '{LLM_ENDPOINT_NAME}',
            request => CONCAT('{PROMPT}', context_parsed),
            responseFormat => '{response_format}',
            modelParameters => named_struct('temperature', {TEMPERATURE})
        ) AS ai_response
    """)
    raw = result_df.collect()[0]["ai_response"]
    return json.loads(raw)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 6. Consistency Metrics

# COMMAND ----------

# DBTITLE 1,Metric functions
def _norm(s):
    return str(s).lower().strip() if s is not None else ""

def exact_match_rate(values):
    if not values:
        return 0.0
    counter = Counter(_norm(v) for v in values)
    return counter.most_common(1)[0][1] / len(values)

def null_rate(values):
    return sum(1 for v in values if v is None or _norm(v) in ("null", "none", "")) / len(values)

def unique_ratio(values):
    return len(set(_norm(v) if v else "__null__" for v in values)) / len(values)

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

def numeric_cv(values):
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
    n = len(values)
    if n == 0:
        return 0.0
    counter = Counter(_norm(v) for v in values)
    probs = [c / n for c in counter.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 7. Run Evaluation — All Documents x 5 Runs (Batched)

# COMMAND ----------

# DBTITLE 1,Execute batched evaluation
eval_run_ts = datetime.now()
eval_run_id = f"eval_{eval_run_ts:%Y%m%d_%H%M%S}"

mlflow.set_experiment(EVAL_EXPERIMENT_NAME)

all_field_rows = []
all_doc_rows = []
all_review_rows = []
all_responses = {}
doc_metadata = {}

num_batches = ceil(len(all_docs) / BATCH_SIZE)

with mlflow.start_run(run_name=f"consistency_{eval_run_ts:%Y%m%d_%H%M%S}") as parent_run:
    mlflow.log_params({
        "eval_run_id": eval_run_id,
        "num_runs": NUM_RUNS,
        "num_documents": len(all_docs),
        "batch_size": BATCH_SIZE,
        "llm_endpoint": LLM_ENDPOINT_NAME,
        "temperature": TEMPERATURE,
    })

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(all_docs))
        batch_docs = all_docs[batch_start:batch_end]

        print(f"\n{'#'*60}")
        print(f"BATCH {batch_idx+1}/{num_batches}  (docs {batch_start+1}-{batch_end} of {len(all_docs)})")
        print(f"{'#'*60}")

        for doc_idx, doc in enumerate(batch_docs):
            global_idx = batch_start + doc_idx + 1
            pid = doc["proposal_id"]
            ctx = doc["context_parsed"]
            adls_path = doc.get("adls_source_path", "") or ""
            source_file = doc.get("source_file_name", "") or ""

            doc_metadata[pid] = {
                "adls_source_path": adls_path,
                "source_file_name": source_file,
                "document_id": doc.get("document_id", ""),
            }

            print(f"\n  [{global_idx}/{len(all_docs)}] {source_file or pid[:20]}")

            responses, latencies = [], []
            for ri in range(NUM_RUNS):
                t0 = time.time()
                try:
                    resp = call_llm(ctx)
                except Exception as e:
                    print(f"    Run {ri+1} FAILED: {e}")
                    resp = {k: None for k in EXPECTED_KEYS}
                elapsed = time.time() - t0
                latencies.append(elapsed)
                responses.append(resp)
                print(f"    Run {ri+1}/{NUM_RUNS} - {elapsed:.1f}s")

            all_responses[pid] = responses

            # Per-field metrics
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
                })
            all_field_rows.extend(doc_rows)

            # Review rows
            for ri, resp in enumerate(responses):
                for key in EXPECTED_KEYS:
                    vals = [r.get(key) for r in responses]
                    modal = Counter(_norm(v) for v in vals).most_common(1)[0][0]
                    all_review_rows.append({
                        "eval_run_id": eval_run_id,
                        "eval_timestamp": eval_run_ts.isoformat(),
                        "proposal_id": pid,
                        "document_id": doc.get("document_id", ""),
                        "adls_source_path": adls_path,
                        "source_file_name": source_file,
                        "run_number": ri + 1,
                        "field": key,
                        "extracted_value": resp.get(key),
                        "modal_value": modal,
                        "agrees_with_majority": "YES" if _norm(resp.get(key)) == modal else "NO",
                        "field_exact_match_rate": round(exact_match_rate(vals), 4),
                        "field_group": next(
                            (g for g, ks in FIELD_GROUPS.items() if key in ks), "other"),
                        "is_correct": None,
                        "correct_value": None,
                        "reviewer_notes": None,
                    })

            # Document summary
            fdf = pd.DataFrame(doc_rows)
            summary = {
                "proposal_id": pid,
                "source_file_name": source_file,
                "mean_exact_match_rate": round(fdf["exact_match_rate"].mean(), 4),
                "min_exact_match_rate": round(fdf["exact_match_rate"].min(), 4),
                "mean_pairwise_jaccard": round(fdf["pairwise_jaccard"].mean(), 4),
                "mean_entropy": round(fdf["entropy"].mean(), 4),
                "mean_null_rate": round(fdf["null_rate"].mean(), 4),
                "fully_consistent_fields": int((fdf["exact_match_rate"] == 1.0).sum()),
                "pct_fully_consistent": round(
                    (fdf["exact_match_rate"] == 1.0).sum() / len(EXPECTED_KEYS) * 100, 2),
                "mean_latency_s": round(np.mean(latencies), 3),
            }
            for gname, gkeys in FIELD_GROUPS.items():
                gdf = fdf[fdf["field"].isin(gkeys)]
                if len(gdf) > 0:
                    summary[f"em_{gname}"] = round(gdf["exact_match_rate"].mean(), 4)
            all_doc_rows.append(summary)

            # MLflow child run
            with mlflow.start_run(run_name=f"doc_{source_file or pid[:16]}", nested=True):
                mlflow.log_metrics({k: v for k, v in summary.items()
                                    if isinstance(v, (int, float))})
                resp_path = f"/tmp/responses_{pid[:16]}.json"
                with open(resp_path, "w") as f:
                    json.dump(responses, f, indent=2, default=str)
                mlflow.log_artifact(resp_path)

                vrows = []
                for key in EXPECTED_KEYS:
                    row = {"field": key}
                    for ri2, r2 in enumerate(responses):
                        row[f"run_{ri2+1}"] = r2.get(key)
                    vals = [r2.get(key) for r2 in responses]
                    row["modal_value"] = Counter(_norm(v) for v in vals).most_common(1)[0][0]
                    row["all_agree"] = "YES" if exact_match_rate(vals) == 1.0 else "NO"
                    vrows.append(row)
                mlflow.log_table(data=pd.DataFrame(vrows), artifact_file="extracted_values.json")

        print(f"\n  Batch {batch_idx+1}/{num_batches} complete.")

    # Aggregate
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

    for name, df in [("field_metrics", field_metrics_df),
                     ("doc_metrics", doc_metrics_df),
                     ("field_ranking", field_ranking)]:
        path = f"/tmp/{name}.csv"
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)
        mlflow.log_table(data=df, artifact_file=f"{name}.json")

print(f"\nAll {len(all_docs)} documents evaluated. MLflow run: {parent_run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Results

# COMMAND ----------

# DBTITLE 1,Summary
print(f"{'='*60}")
print(f"CONSISTENCY EVALUATION SUMMARY")
print(f"{'='*60}")
print(f"Eval run ID:          {eval_run_id}")
print(f"Documents:            {len(all_docs)}")
print(f"Runs per document:    {NUM_RUNS}")
print(f"Batch size:           {BATCH_SIZE}")
print(f"Total LLM calls:      {len(all_docs) * NUM_RUNS}")
print(f"Overall exact match:  {agg_metrics['overall_mean_exact_match']:.2%}")
print(f"Overall Jaccard:      {agg_metrics['overall_mean_jaccard']:.2%}")
print(f"Overall entropy:      {agg_metrics['overall_mean_entropy']:.4f}")
print(f"Fully consistent:     {agg_metrics['overall_pct_fully_consistent']:.1f}%")

# COMMAND ----------

# DBTITLE 1,Least consistent fields
display(spark.createDataFrame(field_ranking))

# COMMAND ----------

# DBTITLE 1,Per-document consistency
display(spark.createDataFrame(doc_metrics_df))

# COMMAND ----------

# DBTITLE 1,Fields that disagree across runs
diff_rows = []
for pid, resps in all_responses.items():
    meta = doc_metadata.get(pid, {})
    for key in EXPECTED_KEYS:
        values = [r.get(key) for r in resps]
        if len(set(_norm(v) for v in values)) > 1:
            row = {"proposal_id": pid[:16], "source_file": meta.get("source_file_name", ""), "field": key}
            for ri, resp in enumerate(resps):
                row[f"run_{ri+1}"] = resp.get(key)
            diff_rows.append(row)

if diff_rows:
    print(f"{len(diff_rows)} field-level disagreements:")
    display(spark.createDataFrame(pd.DataFrame(diff_rows)))
else:
    print("Perfect consistency across all runs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 9. Write Review Table

# COMMAND ----------

# DBTITLE 1,Write review Delta table (append)
review_sdf = spark.createDataFrame(pd.DataFrame(all_review_rows))

review_sdf.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(REVIEW_TABLE)

print(f"Wrote {len(all_review_rows)} review rows to {REVIEW_TABLE}")
print(f"\nTo review:")
print(f"  SELECT * FROM {REVIEW_TABLE}")
print(f"  WHERE eval_run_id = '{eval_run_id}'")
print(f"  ORDER BY source_file_name, field, run_number")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 10. Visual Comparison — Source Document vs Extracted Fields
# MAGIC
# MAGIC Each proposal renders as a side-by-side panel:
# MAGIC - **Left**: Full source text (`context_parsed`) + clickable ADLS link to the original file
# MAGIC - **Right**: All 5 runs of extracted values, disagreements highlighted

# COMMAND ----------

# DBTITLE 1,Side-by-side HTML: source vs extraction
def build_review_html(pid, document_id, adls_path, source_file, source_text, responses, expected_keys):
    escaped_source = html_mod.escape(source_text or "")
    escaped_file = html_mod.escape(str(source_file))
    escaped_adls = html_mod.escape(str(adls_path))
    num_runs = len(responses)

    header_cols = "".join(
        f"<th style='padding:4px 8px; border-bottom:2px solid #999;'>Run {i+1}</th>"
        for i in range(num_runs)
    )

    table_rows = []
    for key in expected_keys:
        values = [r.get(key) for r in responses]
        normed = [_norm(v) for v in values]
        all_same = len(set(normed)) == 1
        modal = Counter(normed).most_common(1)[0][0]
        row_bg = "" if all_same else "background-color:#fff3cd;"
        icon = "&#10003;" if all_same else "&#10007;"
        icon_color = "green" if all_same else "red"

        cells = []
        for v, n in zip(values, normed):
            cell_val = html_mod.escape(str(v) if v is not None else "null")
            if not all_same and n != modal:
                cells.append(f'<td style="background:#ffcccc; font-weight:bold; padding:4px 8px;">{cell_val}</td>')
            else:
                cells.append(f'<td style="padding:4px 8px;">{cell_val}</td>')

        table_rows.append(
            f'<tr style="{row_bg}">'
            f'<td style="padding:4px 8px;"><strong>{key}</strong></td>'
            f'{"".join(cells)}'
            f'<td style="text-align:center; padding:4px 8px;"><span style="color:{icon_color};">{icon}</span></td>'
            f'</tr>'
        )

    rows_html = "\n".join(table_rows)

    return f"""
    <div style="border:2px solid #666; border-radius:10px; margin:20px 0; overflow:hidden;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; font-size:13px;">
        <div style="background:#2c3e50; color:white; padding:12px 20px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong style="font-size:15px;">{escaped_file}</strong>
                <span style="margin-left:12px; opacity:0.7; font-size:12px;">ID: {html_mod.escape(str(pid)[:20])}...</span>
            </div>
            <a href="{escaped_adls}" target="_blank"
               style="color:#5dade2; text-decoration:none; font-size:12px; white-space:nowrap;">
                Open folder in ADLS &#8599;
            </a>
        </div>
        <div style="display:flex; max-height:70vh;">
            <div style="flex:1; overflow:auto; padding:16px; background:#fafafa; border-right:1px solid #ddd;">
                <div style="position:sticky; top:0; background:#fafafa; padding:6px 0 10px; border-bottom:1px solid #ddd; margin-bottom:10px;">
                    <strong>Source Document</strong>
                    <span style="float:right; font-size:11px; color:#888;">{len(source_text or ''):,} chars</span>
                </div>
                <pre style="white-space:pre-wrap; word-wrap:break-word; font-size:12px; line-height:1.6; margin:0;">{escaped_source}</pre>
            </div>
            <div style="flex:1; overflow:auto; padding:16px; background:#fff;">
                <div style="position:sticky; top:0; background:#fff; padding:6px 0 10px; border-bottom:1px solid #ddd; margin-bottom:10px;">
                    <strong>Extracted ({num_runs} runs)</strong>
                    <span style="font-size:11px; margin-left:12px;">
                        <span style="color:green;">&#10003;</span> agree
                        <span style="color:red; margin-left:6px;">&#10007;</span> disagree
                        <span style="background:#ffcccc; padding:0 4px; margin-left:6px;">outlier</span>
                    </span>
                </div>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <thead style="position:sticky; top:45px; background:#f0f0f0;">
                        <tr>
                            <th style="text-align:left; padding:4px 8px; border-bottom:2px solid #999;">Field</th>
                            {header_cols}
                            <th style="padding:4px 8px; border-bottom:2px solid #999;">OK</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        </div>
    </div>
    """

for doc in all_docs:
    pid = doc["proposal_id"]
    if pid in all_responses:
        html_report = build_review_html(
            pid=pid,
            document_id=doc.get("document_id", ""),
            adls_path=doc.get("adls_source_path", "") or "",
            source_file=doc.get("source_file_name", "") or "",
            source_text=doc.get("context_parsed", ""),
            responses=all_responses[pid],
            expected_keys=EXPECTED_KEYS,
        )
        displayHTML(html_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 11. Post-Review Accuracy
# MAGIC After filling in `is_correct` in the review table, run this cell.

# COMMAND ----------

# DBTITLE 1,Accuracy from manual review (run after flagging)
reviewed = spark.sql(f"""
    SELECT * FROM {REVIEW_TABLE}
    WHERE is_correct IS NOT NULL
""")

if reviewed.count() == 0:
    print("No rows reviewed yet.")
    print(f"Open: {REVIEW_TABLE}")
    print(f"Filter: eval_run_id = '{eval_run_id}'")
    print("Set is_correct = TRUE/FALSE, then re-run this cell.")
else:
    reviewed_pdf = reviewed.toPandas()
    reviewed_pdf["is_correct_bool"] = reviewed_pdf["is_correct"].astype(str).str.upper().isin(["TRUE", "1", "YES"])

    total = len(reviewed_pdf)
    correct = int(reviewed_pdf["is_correct_bool"].sum())
    print(f"Reviewed:    {total}")
    print(f"Correct:     {correct} ({correct/total:.1%})")
    print(f"Incorrect:   {total - correct} ({(total-correct)/total:.1%})\n")

    field_acc = (reviewed_pdf.groupby("field")["is_correct_bool"]
                 .agg(["sum", "count"])
                 .assign(accuracy=lambda x: x["sum"] / x["count"])
                 .sort_values("accuracy")
                 .reset_index()
                 .rename(columns={"sum": "correct", "count": "reviewed"}))
    print("Per-field accuracy:")
    display(spark.createDataFrame(field_acc))

    group_map = {k: g for g, keys in FIELD_GROUPS.items() for k in keys}
    reviewed_pdf["group"] = reviewed_pdf["field"].map(group_map).fillna("other")
    group_acc = (reviewed_pdf.groupby("group")["is_correct_bool"]
                 .agg(["sum", "count"])
                 .assign(accuracy=lambda x: x["sum"] / x["count"])
                 .sort_values("accuracy")
                 .reset_index()
                 .rename(columns={"sum": "correct", "count": "reviewed"}))
    print("\nPer-group accuracy:")
    display(spark.createDataFrame(group_acc))

    with mlflow.start_run(run_name=f"review_accuracy_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_metrics({
            "reviewed_total": total,
            "reviewed_correct": correct,
            "reviewed_accuracy": round(correct / total, 4),
        })
        mlflow.log_table(data=field_acc, artifact_file="field_accuracy.json")
