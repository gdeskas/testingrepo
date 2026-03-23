# Databricks notebook source

# MAGIC %md
# MAGIC # Pipeline Orchestrator — Process All Proposals for Evaluation
# MAGIC
# MAGIC One `Run All` processes **all 20 proposals** end-to-end and builds
# MAGIC evaluation-ready snapshot tables.
# MAGIC
# MAGIC **The problem:** Production Silver notebooks (categorise_image, classify_asset, etc.)
# MAGIC use **overwrite** mode — each pipeline run replaces the previous proposal's data.
# MAGIC
# MAGIC **The solution:** After each pipeline run, this notebook snapshots the current
# MAGIC Silver table contents into **eval accumulation tables** using append. The production
# MAGIC notebooks are untouched.
# MAGIC
# MAGIC **Flow per proposal:**
# MAGIC 1. Stage files from ADLS → ingest to Bronze working table (overwrite)
# MAGIC 2. Run full production pipeline (9 notebooks — overwrites Silver tables)
# MAGIC 3. Snapshot Silver tables → append to `eval_snap_*` tables in Silver schema
# MAGIC 4. Move to next proposal
# MAGIC
# MAGIC After all 20 proposals, the `eval_snap_*` tables contain accumulated results
# MAGIC from all proposals — ready for `eval_01_consistency` and `eval_02_accuracy_llm_judge`.

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("catalog", "comm-afl-dev", "Catalog")
dbutils.widgets.text("brz_schema", "brkrflw-lkh-brz", "Bronze Schema")
dbutils.widgets.text("slr_schema", "brkrflw-lkh-slr", "Silver Schema")
dbutils.widgets.text("job_run_id", "", "Job Run ID")

catalog = dbutils.widgets.get("catalog")
brz_schema = dbutils.widgets.get("brz_schema")
slr_schema = dbutils.widgets.get("slr_schema")
job_run_id = dbutils.widgets.get("job_run_id") or f"eval_batch_{__import__('datetime').datetime.now():%Y%m%d_%H%M%S}"

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
    f"{ADLS_BASE}/14. Stellar",                                         # TODO: complete
    # f"{ADLS_BASE}/15. ...",   # ADD remaining folder names
    # f"{ADLS_BASE}/16. ...",
    # f"{ADLS_BASE}/17. ...",
    # f"{ADLS_BASE}/18. ...",
    # f"{ADLS_BASE}/19. ...",
    # f"{ADLS_BASE}/20. ...",
]

print(f"Defined {len(VOLUME_ROOTS)} proposal folders:")
for i, vr in enumerate(VOLUME_ROOTS, 1):
    print(f"  {i:2d}. {vr.split('/')[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

# DBTITLE 1,Pipeline config
STAGING_ROOT = "dbfs:/Volumes/comm-afl-dev/brkrflw-lkh-brz/staging_files"
WORKING_TABLE = "gdfiles_loaded"

PIPELINE_NOTEBOOKS = [
    ("/Workspace/CNTRL-COMM-AFL-DEV/Pipeline/Utilities/date_functions",                      "date_functions"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/01.silver_ddl",                  "silver_ddl"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/02.extract_text",                "extract_text"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/03.categorise_image",            "categorise_image"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/04.categorise_other_doc_type",   "categorise_doc"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/05.classify_asset",              "classify_asset"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/06.extract_account_proposal_information", "extract_proposal"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/07.extract_corporate_party_identify_role", "extract_corp_party"),
    ("/Workspace/CNTRL-COMM-AFL-DEV/gd_Proposal_Load/Silver/08.extract_person_party_identify_role",    "extract_person_party"),
]

pipeline_args = {
    "catalog": catalog,
    "brz_schema": brz_schema,
    "slr_schema": slr_schema,
    "job_run_id": job_run_id,
}

# Silver tables to snapshot after each pipeline run
# Key = production table, Value = eval snapshot table
SNAPSHOT_TABLES = {
    "img_category":      "eval_snap_img_category",
    "doc_category":      "eval_snap_doc_category",
    "text_extract":      "eval_snap_text_extract",
    "asset_extract":     "eval_snap_asset_extract",
    "proposals_extract": "eval_snap_proposals_extract",
    # Add party tables once you confirm their names:
    # "corporate_party_extract": "eval_snap_corporate_party",
    # "person_party_extract":    "eval_snap_person_party",
}


def safe_name(name: str) -> str:
    return re.sub(r"[^\w._]+", "_", name).strip("_")


print(f"Pipeline: {len(PIPELINE_NOTEBOOKS)} notebooks")
print(f"Snapshot: {len(SNAPSHOT_TABLES)} Silver tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Snapshot Function
# MAGIC
# MAGIC After each pipeline run, reads the current production Silver table
# MAGIC (which contains only this proposal's data) and appends it to the
# MAGIC eval snapshot table with traceability columns.

# COMMAND ----------

# DBTITLE 1,Snapshot function
def snapshot_silver_tables(case_folder: str, batch_id: str):
    """
    Read each production Silver table and append to eval_snap_* table.
    Adds _source_folder, _batch_id, _snapshot_ts for traceability.
    """
    for prod_table, snap_table in SNAPSHOT_TABLES.items():
        prod_full = f"`{catalog}`.`{slr_schema}`.{prod_table}"
        snap_full = f"`{catalog}`.`{slr_schema}`.{snap_table}"

        try:
            df = spark.table(prod_full)
            row_count = df.count()

            if row_count == 0:
                print(f"    {prod_table}: empty — skip")
                continue

            df_snap = (
                df
                .withColumn("_source_folder", F.lit(case_folder))
                .withColumn("_batch_id", F.lit(batch_id))
                .withColumn("_snapshot_ts", F.current_timestamp())
            )

            df_snap.write.format("delta").mode("append").option(
                "mergeSchema", "true"
            ).saveAsTable(snap_full)

            print(f"    {prod_table}: {row_count} rows → {snap_table}")

        except Exception as e:
            print(f"    {prod_table}: ERROR — {e}")


print("snapshot_silver_tables() defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Optional: Clear Snapshot Tables
# MAGIC
# MAGIC Run this **only** to start fresh.

# COMMAND ----------

# DBTITLE 1,OPTIONAL — Clear eval snapshot tables
# Uncomment to drop snapshot tables before the batch run.

# for _, snap_table in SNAPSHOT_TABLES.items():
#     snap_full = f"`{catalog}`.`{slr_schema}`.{snap_table}"
#     try:
#         spark.sql(f"DROP TABLE IF EXISTS {snap_full}")
#         print(f"  Dropped {snap_full}")
#     except Exception as e:
#         print(f"  Skip: {e}")
# print("Snapshot tables cleared.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Process All Proposals

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

    # ── Stage files ──────────────────────────────────────────────────────
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
        print(f"  Staged {file_count} files")
    except Exception as e:
        print(f"  ERROR staging: {e}")
        results.append({"folder": case_folder, "status": "STAGE_FAILED",
                        "error": str(e), "elapsed_s": round(time.time() - t0, 1)})
        continue

    # ── Ingest to Bronze ─────────────────────────────────────────────────
    try:
        staged_path = f"{STAGING_ROOT}/{case_folder}"

        binary_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            .load(staged_path)
            .withColumn("folder_path", F.lit(staged_path))
            .withColumn("adls_source_path", F.lit(volume_root))
            .withColumn("file_name", F.regexp_extract(F.col("path"), r"([^/]+)$", 1))
            .withColumn("file_ext", F.lower(F.regexp_extract(F.col("file_name"), r"\.([^.]+)$", 1)))
            .withColumn("proposal_id",
                F.sha2(F.concat_ws("||", F.lit(staged_path), F.current_timestamp().cast("string")), 256))
            .withColumn("document_id", F.sha2(F.col("path"), 256))
            .withColumn("job_run_id", F.lit(job_run_id))
            .withColumn("ingestion_ts", F.current_timestamp())
        )

        files_loaded_df = binary_df.selectExpr(
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
            "ingestion_ts as INGESTION_TS",
        )

        row_count = files_loaded_df.count()

        files_loaded_df.write.mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(f"`{catalog}`.`{brz_schema}`.{WORKING_TABLE}")

        print(f"  Ingested {row_count} files to Bronze")

    except Exception as e:
        print(f"  ERROR ingesting: {e}")
        results.append({"folder": case_folder, "status": "INGEST_FAILED",
                        "error": str(e), "elapsed_s": round(time.time() - t0, 1)})
        continue

    # ── Run production pipeline ──────────────────────────────────────────
    pipeline_ok = True
    for nb_path, nb_label in PIPELINE_NOTEBOOKS:
        try:
            dbutils.notebook.run(nb_path, timeout_seconds=3600, arguments=pipeline_args)
            print(f"    {nb_label} — OK")
        except Exception as e:
            print(f"    {nb_label} — FAILED: {e}")
            results.append({"folder": case_folder, "status": f"FAILED_AT_{nb_label}",
                            "error": str(e), "elapsed_s": round(time.time() - t0, 1)})
            pipeline_ok = False
            break

    if not pipeline_ok:
        continue

    # ── Snapshot Silver tables ───────────────────────────────────────────
    print(f"  Snapshotting Silver tables:")
    snapshot_silver_tables(case_folder, batch_id)

    elapsed = round(time.time() - t0, 1)
    results.append({"folder": case_folder, "status": "OK",
                    "error": None, "elapsed_s": elapsed})
    print(f"  DONE ({elapsed}s)")

total_elapsed = round(time.time() - batch_start, 1)
print(f"\n{'='*70}")
print(f"Batch complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Batch Results

# COMMAND ----------

# DBTITLE 1,Processing summary
results_df = pd.DataFrame(results)

ok_count = (results_df["status"] == "OK").sum()
fail_count = len(results_df) - ok_count

print(f"BATCH PROCESSING SUMMARY")
print(f"{'='*70}")
print(f"Total proposals:   {len(results_df)}")
print(f"Succeeded:         {ok_count}")
print(f"Failed:            {fail_count}")
print(f"Total time:        {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
print()

for _, row in results_df.iterrows():
    icon = "OK" if row["status"] == "OK" else "FAIL"
    print(f"  [{icon:4s}] {row['folder'][:50]:50s}  {row['elapsed_s']:6.0f}s  {row['status']}")

if fail_count > 0:
    print(f"\nFailed proposals:")
    for _, row in results_df[results_df["status"] != "OK"].iterrows():
        print(f"  {row['folder']}: {row['error'][:120]}")

display(spark.createDataFrame(results_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify Snapshot Tables

# COMMAND ----------

# DBTITLE 1,Check eval snapshot tables
print("EVAL SNAPSHOT TABLE VERIFICATION")
print("=" * 70)

for prod_table, snap_table in SNAPSHOT_TABLES.items():
    snap_full = f"`{catalog}`.`{slr_schema}`.{snap_table}"
    try:
        count = spark.sql(f"SELECT COUNT(*) as n FROM {snap_full}").collect()[0]["n"]
        proposals = spark.sql(f"SELECT COUNT(DISTINCT PROPOSAL_ID) as n FROM {snap_full}").collect()[0]["n"]
        folders = spark.sql(f"SELECT COUNT(DISTINCT _source_folder) as n FROM {snap_full}").collect()[0]["n"]
        print(f"  {snap_table:35s}  {count:5d} rows  {proposals:3d} proposals  {folders:3d} folders")
    except Exception:
        print(f"  {snap_table:35s}  (not yet created)")

# COMMAND ----------

# DBTITLE 1,Preview doc_category snapshot
try:
    display(spark.sql(f"""
        SELECT _source_folder, CATEGORY, COUNT(*) as docs
        FROM `{catalog}`.`{slr_schema}`.eval_snap_doc_category
        GROUP BY _source_folder, CATEGORY
        ORDER BY _source_folder, docs DESC
    """))
except Exception:
    print("Not yet created — run the batch first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ready for Evaluation
# MAGIC
# MAGIC Update the `PROMPT_CATALOGUE` in eval notebooks to point at snapshot tables:
# MAGIC
# MAGIC | Production table | Eval snapshot table |
# MAGIC |-----------------|-------------------|
# MAGIC | `doc_category` | `eval_snap_doc_category` |
# MAGIC | `img_category` | `eval_snap_img_category` |
# MAGIC | `proposals_extract` | `eval_snap_proposals_extract` |
# MAGIC | `asset_extract` | `eval_snap_asset_extract` |
# MAGIC | `text_extract` | `eval_snap_text_extract` |
# MAGIC
# MAGIC Then run:
# MAGIC 1. **`eval_01_consistency.py`** — consistency (4 text-based prompts)
# MAGIC 2. **`eval_02_accuracy_llm_judge.py`** — accuracy with LLM-as-judge (all 5 prompts)

