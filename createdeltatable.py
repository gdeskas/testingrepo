# Save chunks and embeddings to a Delta table
chunk_df = pd.DataFrame({
    "chunk_id": list(range(len(chunk_texts))),
    "complaint_id": [m["complaint_id"] for m in chunk_metadata],
    "section": [m["section"] for m in chunk_metadata],
    "chunk_index": [m["chunk_index"] for m in chunk_metadata],
    "chunk_text": chunk_texts,
    "embedding": [emb.tolist() for emb in embeddings]
})

# Convert to Spark DataFrame and save as Delta table
CATALOG = "your_catalog"
SCHEMA = "your_schema"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.complaint_chunks"

spark_chunk_df = spark.createDataFrame(chunk_df)
spark_chunk_df.write.format("delta").mode("overwrite").saveAsTable(CHUNKS_TABLE)

print(f"✓ Saved {len(chunk_texts)} chunks to {CHUNKS_TABLE}")
