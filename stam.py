# Save first-run field metrics to Delta for ground truth review
from datetime import datetime as dt

# Add metadata
field_metrics_df["eval_timestamp"] = dt.now().isoformat()
field_metrics_df["llm_endpoint"] = LLM_ENDPOINT_NAME
field_metrics_df["prompt_type"] = "proposal_extraction"

# Write to Delta
ground_truth_table = f"`{catalog}`.`{slr_schema}`.eval_proposal_field_metrics"

spark.createDataFrame(field_metrics_df).write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(ground_truth_table)

print(f"Saved {len(field_metrics_df)} rows to {ground_truth_table}")
display(spark.createDataFrame(field_metrics_df))
