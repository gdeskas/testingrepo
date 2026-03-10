from datetime import datetime as dt

# Add metadata to each first-run response
run_timestamp = dt.now().isoformat()
for row in all_first_run_responses:
    row["eval_timestamp"] = run_timestamp
    row["llm_endpoint"] = LLM_ENDPOINT_NAME
    row["prompt_type"] = "proposal_extraction"

# Create DataFrame and write to Delta
first_run_df = spark.createDataFrame(pd.DataFrame(all_first_run_responses))

ground_truth_table = f"`{catalog}`.`{slr_schema}`.eval_proposal_first_run"

first_run_df.write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(ground_truth_table)

print(f"Saved {first_run_df.count()} rows to {ground_truth_table}")
display(first_run_df)
