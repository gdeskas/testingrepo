# Databricks notebook source

# MAGIC %md
# MAGIC # eval_00 — Pipeline Orchestrator (build `eval_snap_*` tables)
# MAGIC
# MAGIC One `Run All` processes **all proposal folders** end-to-end and produces
# MAGIC accumulated snapshot tables that the model-selection eval reads from.
# MAGIC
# MAGIC **The problem:** Production Silver notebooks use **overwrite** mode — each
# MAGIC pipeline run replaces the previous proposal's data.
# MAGIC
# MAGIC **The solution:** After each pipeline run, this notebook snapshots the
# MAGIC current Silver table contents into `eval_snap_*` tables using **append**.
# MAGIC The production notebooks are untouched.
# MAGIC
# MAGIC **Flow per proposal:**
# MAGIC 1. Stage files from ADLS → ingest to Bronze working table (overwrite)
# MAGIC 2. Run full production Silver pipeline — this is where the **production
# MAGIC    LLM calls** happen (categorise_image, classify_asset, extract_*).
# MAGIC    These writes go to the production Silver tables (overwrite mode).
# MAGIC 3. Snapshot Silver tables → append to `eval_snap_*` tables
# MAGIC 4. Move to next proposal
# MAGIC
# MAGIC After all proposals, the `eval_snap_*` tables contain accumulated results
# MAGIC across every proposal — ready for `eval_model_selection.py` to read from.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog",    "comm-afl-dev",     "Catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz",  "Bronze Schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr",  "Silver Schema")
dbutils.widgets.text("job_run_id", "",                 "Job Run ID")

catalog    = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")

from datetime import datetime
job_run_id = dbutils.widgets.get("job_run_id") \
    or f"eval_batch_{datetime.now():%Y%m%d_%H%M%S}"

print(f"Catalog:    {catalog}")
print(f"Bronze:     {brz_schema}")
print(f"Silver:     {slr_schema}")
print(f"Job Run ID: {job_run_id}")

# COMMAND ----------

# DBTITLE 1,Imports
import re
import time
import pandas as pd
from datetime import datetime
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Proposal Folders
# MAGIC
# MAGIC Update `VOLUME_ROOTS` with the list of folders you want to process.

# COMMAND ----------

# DBTITLE 1,ADLS proposal folder paths
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
    # Add the remaining folders here once finalised:
    # f"{ADLS_BASE}/14. ...",
    # f"{ADLS_BASE}/15. ...",
    # ...
    # f"{ADLS_BASE}/20. ...",
]

print(f"Defined {len(VOLUME_ROOTS)} proposal folders")
for i, vr in enumerate(VOLUME_ROOTS, 1):
    print(f"  {i:2d}. {vr.split('/')[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pipeline Configuration

# COMMAND ----------

# DBTITLE 1,Pipeline notebook paths and staging config
STAGING_ROOT  = "dbfs:/Volumes/comm-afl-dev/brkrflw-lkh-brz/staging_files"
WORKING_TABLE = "files_loaded"   # Bronze table the pipeline writes to (overwrite)

# Pipeline notebooks — execution order
PIPELINE_NOTEBOOKS = [
    ("/Workspace/CNTRL-COMM-AFL-DEV/Pipeline/Utilities/date_functions",                                "date_functions"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/01.silver_ddl",                            "silver_ddl"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/02.extract_text",                          "extract_text"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/03.categorise_image",                      "categorise_image"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/04.categorise_other_doc_type",             "categorise_doc"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/05.classify_asset",                        "classify_asset"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/06.extract_account_proposal_information",  "extract_proposal"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/07.extract_corporate_party_identify_role", "extract_corp_party"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/08.extract_person_party_identify_role",    "extract_person_party"),
]

pipeline_args = {
    "catalog":    catalog,
    "brz_schema": brz_schema,
    "slr_schema": slr_schema,
    "job_run_id": job_run_id,
}

def safe_name(name: str) -> str:
    """Sanitise filenames the same way load_adls_files does."""
    return re.sub(r"[^\w._]+", "_", name).strip("_")

print(f"Pipeline:  {len(PIPELINE_NOTEBOOKS)} notebooks")
print(f"Staging:   {STAGING_ROOT}")
print(f"Bronze WT: {WORKING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Snapshot Configuration
# MAGIC
# MAGIC Maps each production Silver table to its eval-snapshot equivalent. After
# MAGIC every proposal runs through the pipeline, we read from these production
# MAGIC tables and **append** to the snapshot tables (which accumulate across
# MAGIC all proposals in the batch).

# COMMAND ----------

# DBTITLE 1,Snapshot table mapping
SNAPSHOT_TABLES = {
    # production table       : eval snapshot table
    "files_loaded"            : "eval_snap_files_loaded",
    "text_extract"            : "eval_snap_text_extract",
    "img_category"            : "eval_snap_img_category",
    "doc_category"            : "eval_snap_doc_category",
    "asset_extract"           : "eval_snap_asset_extract",
    "proposals_extract"       : "eval_snap_proposals_extract",
    "corporate_roles"         : "eval_snap_corporate_roles",
    "person_roles"            : "eval_snap_person_roles",
}

# Note: files_loaded lives in the Bronze schema; everything else in Silver.
SNAPSHOT_SCHEMA = {
    "files_loaded"      : brz_schema,
    "text_extract"      : brz_schema,   # text_extract appears in both layers
    "img_category"      : slr_schema,
    "doc_category"      : slr_schema,
    "asset_extract"     : slr_schema,
    "proposals_extract" : slr_schema,
    "corporate_roles"   : slr_schema,
    "person_roles"      : slr_schema,
}

# All snapshots are written to the Silver schema for simplicity.
def snap_full_name(snap_table):
    return f"`{catalog}`.`{slr_schema}`.{snap_table}"

print(f"Will snapshot {len(SNAPSHOT_TABLES)} tables → eval_snap_*")

# COMMAND ----------

# DBTITLE 1,Snapshot function
def snapshot_silver_tables(case_folder: str, batch_id: str):
    """For each production table, append its current contents to the
    corresponding eval_snap_* table with traceability columns."""
    for prod_table, snap_table in SNAPSHOT_TABLES.items():
        prod_schema = SNAPSHOT_SCHEMA[prod_table]
        prod_full = f"`{catalog}`.`{prod_schema}`.{prod_table}"
        snap_full = snap_full_name(snap_table)
        try:
            df = spark.table(prod_full)
            row_count = df.count()
            if row_count == 0:
                print(f"    {prod_table:<22s}  (empty — skipped)")
                continue
            df_snap = (
                df.withColumn("_source_folder", F.lit(case_folder))
                  .withColumn("_batch_id",      F.lit(batch_id))
                  .withColumn("_snapshot_ts",   F.current_timestamp())
            )
            df_snap.write.format("delta").mode("append").option(
                "mergeSchema", "true"
            ).saveAsTable(snap_full)
            print(f"    {prod_table:<22s}  {row_count} rows → {snap_table}")
        except Exception as e:
            print(f"    {prod_table:<22s}  ERROR — {e}")

print("snapshot_silver_tables() defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Optional: Clear Snapshot Tables
# MAGIC
# MAGIC Run this **only** to start fresh. Uncomment the block, run once, then
# MAGIC re-comment.

# COMMAND ----------

# DBTITLE 1,OPTIONAL — Clear eval_snap_* tables
# for prod_table, snap_table in SNAPSHOT_TABLES.items():
#     snap_full = snap_full_name(snap_table)
#     try:
#         spark.sql(f"DROP TABLE IF EXISTS {snap_full}")
#         print(f"  Dropped {snap_full}")
#     except Exception as e:
#         print(f"  Skip {snap_full}: {e}")
# print("Snapshot tables cleared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Process All Proposals
# MAGIC
# MAGIC For each folder:
# MAGIC 1. Stage files into a per-folder staging directory
# MAGIC 2. Read as binary, write to the Bronze working table (overwrite)
# MAGIC 3. Run all 9 production pipeline notebooks (this is where production
# MAGIC    LLM calls happen and Silver tables get populated)
# MAGIC 4. Snapshot the Silver tables → append to eval_snap_*

# COMMAND ----------

# DBTITLE 1,Main loop
results = []
batch_start = time.time()
batch_id = f"batch_{datetime.now():%Y%m%d_%H%M%S}"

for idx, volume_root in enumerate(VOLUME_ROOTS, 1):
    case_folder = volume_root.rstrip("/").split("/")[-1]
    dst_dir = f"{STAGING_ROOT}/{case_folder}"
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"[{idx}/{len(VOLUME_ROOTS)}] {case_folder}")
    print(f"{'='*70}")

    # ── 1. Stage files from ADLS ─────────────────────────────────────────
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
            dbutils.fs.cp(f.path, f"{dst_dir}/{new_name}", True)
            file_count += 1
        print(f"  ✓ Staged {file_count} files")
    except Exception as e:
        print(f"  ✗ STAGE failed: {e}")
        results.append({"folder": case_folder, "status": "STAGE_FAILED",
                        "error": str(e), "elapsed_s": round(time.time() - t0, 1)})
        continue

    # ── 2. Ingest to Bronze working table (overwrite) ────────────────────
    try:
        binary_df = (
            spark.read.format("binaryFile")
                 .option("recursiveFileLookup", "true")
                 .load(dst_dir)
                 .withColumn("folder_path",      F.lit(dst_dir))
                 .withColumn("adls_source_path", F.lit(volume_root))
                 .withColumn("file_name",        F.regexp_extract(F.col("path"), r"([^/]+)$", 1))
                 .withColumn("file_ext",         F.lower(F.regexp_extract(F.col("file_name"), r"\.([^.]+)$", 1)))
                 .withColumn("proposal_id",      F.sha2(F.concat(F.col("folder_path"), F.lit(batch_id)), 256))
                 .withColumn("document_id",      F.sha2(F.col("path"), 256))
                 .withColumn("job_run_id",       F.lit(job_run_id))
                 .withColumn("ingestion_ts",     F.current_timestamp())
        )

        bronze_target = f"`{catalog}`.`{brz_schema}`.{WORKING_TABLE}"
        (binary_df.selectExpr(
            "proposal_id   AS PROPOSAL_ID",
            "document_id   AS DOCUMENT_ID",
            "path          AS SOURCE_PATH",
            "file_name     AS FILE_NAME",
            "file_ext      AS FILE_EXT",
            "length        AS LENGTH",
            "content       AS CONTENT",
            "modificationTime AS MODIFICATION_TIME",
            "folder_path   AS FOLDER_PATH",
            "adls_source_path AS ADLS_SOURCE_PATH",
            "job_run_id    AS JOB_RUN_ID",
            "ingestion_ts  AS INGESTION_TS",
        ).write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(bronze_target))

        ingested = spark.table(bronze_target).count()
        print(f"  ✓ Ingested {ingested} rows → {WORKING_TABLE}")
    except Exception as e:
        print(f"  ✗ INGEST failed: {e}")
        results.append({"folder": case_folder, "status": "INGEST_FAILED",
                        "error": str(e), "elapsed_s": round(time.time() - t0, 1)})
        continue

    # ── 3. Run production pipeline notebooks ─────────────────────────────
    pipeline_failed = False
    for nb_path, nb_label in PIPELINE_NOTEBOOKS:
        nb_t0 = time.time()
        try:
            dbutils.notebook.run(nb_path, timeout_seconds=1800,
                                 arguments=pipeline_args)
            print(f"  ✓ {nb_label:<22s}  ({time.time() - nb_t0:.0f}s)")
        except Exception as e:
            print(f"  ✗ {nb_label:<22s}  FAILED — {e}")
            pipeline_failed = True
            results.append({"folder": case_folder, "status": f"PIPELINE_FAILED:{nb_label}",
                            "error": str(e)[:200],
                            "elapsed_s": round(time.time() - t0, 1)})
            break

    if pipeline_failed:
        continue

    # ── 4. Snapshot Silver tables (append to eval_snap_*) ────────────────
    try:
        print(f"  Snapshotting Silver tables...")
        snapshot_silver_tables(case_folder, batch_id)
        elapsed = round(time.time() - t0, 1)
        results.append({"folder": case_folder, "status": "OK",
                        "files": file_count, "elapsed_s": elapsed})
        print(f"  ✓ Done in {elapsed}s")
    except Exception as e:
        print(f"  ✗ SNAPSHOT failed: {e}")
        results.append({"folder": case_folder, "status": "SNAPSHOT_FAILED",
                        "error": str(e)[:200],
                        "elapsed_s": round(time.time() - t0, 1)})

batch_elapsed = round(time.time() - batch_start, 1)
print(f"\n{'='*70}")
print(f"BATCH COMPLETE  |  batch_id={batch_id}  |  elapsed={batch_elapsed}s")
print(f"{'='*70}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Batch Summary

# COMMAND ----------

# DBTITLE 1,Per-proposal results
results_df = pd.DataFrame(results)
display(results_df)

ok_count   = (results_df["status"] == "OK").sum() if not results_df.empty else 0
fail_count = len(results_df) - ok_count
print(f"\nOK:     {ok_count}/{len(results_df)}")
print(f"Failed: {fail_count}/{len(results_df)}")

# COMMAND ----------

# DBTITLE 1,Snapshot table row counts
print(f"{'Snapshot table':<35s}  {'Rows':>8s}")
print("-" * 50)
for prod_table, snap_table in SNAPSHOT_TABLES.items():
    snap_full = snap_full_name(snap_table)
    try:
        n = spark.table(snap_full).count()
        print(f"{snap_table:<35s}  {n:>8d}")
    except Exception:
        print(f"{snap_table:<35s}  (does not exist)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ready for Evaluation
# MAGIC
# MAGIC The `eval_snap_*` tables now contain accumulated results from all proposals.
# MAGIC Next step:
# MAGIC
# MAGIC > Run **`eval_model_selection.py`** — it reads from `eval_snap_*` and
# MAGIC > compares each candidate model against the production output for every
# MAGIC > evaluated prompt.
