from databricks.vector_search.client import VectorSearchClient

# Initialize client
vsc = VectorSearchClient()

# Your endpoint name (from Step 2)
VECTOR_SEARCH_ENDPOINT = "complaints-vector-endpoint"

# Index name
INDEX_NAME = f"{CATALOG}.{SCHEMA}.complaint_chunks_index"

# Create the index
vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    source_table_name=CHUNKS_TABLE,
    pipeline_type="TRIGGERED",  # or "CONTINUOUS" for auto-sync
    primary_key="chunk_id",
    embedding_dimension=embeddings.shape[1],
    embedding_vector_column="embedding"
)

print(f"✓ Creating index {INDEX_NAME}... (this may take a few minutes)")
