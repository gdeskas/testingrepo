def retrieve_similar_chunks(
    reference_number: str,
    k: int = K_NEIGHBORS,
    exclude_source: bool = True
) -> List[Tuple[str, float, str]]:
    """Retrieve similar chunks using Vector Search."""
    
    # Get a chunk from the source complaint to use as query
    source_chunks = spark.sql(f"""
        SELECT chunk_text, embedding 
        FROM {CHUNKS_TABLE} 
        WHERE reference_number = '{reference_number}'
        LIMIT 1
    """).toPandas()
    
    if len(source_chunks) == 0:
        return []
    
    # Convert embedding to list if it's a numpy array
    query_embedding = source_chunks.iloc[0]['embedding']
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    elif not isinstance(query_embedding, list):
        query_embedding = list(query_embedding)
    
    # Call Vector Search API
    url = f"{WORKSPACE_URL}/api/2.0/vector-search/indexes/{VECTOR_SEARCH_INDEX}/query"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query_vector": query_embedding,
        "num_results": k * 5,  # Get more to filter
        "columns": ["chunk_id", "reference_number", "section", "chunk_text"]
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    
    if not response.ok:
        raise RuntimeError(f"Vector Search failed: {response.status_code} {response.text[:500]}")
    
    results_raw = response.json()
    
    # Process results
    seen_refs = set([reference_number]) if exclude_source else set()
    results = []
    
    for row in results_raw.get("result", {}).get("data_array", []):
        chunk_id, ref_num, section, chunk_text, score = row
        
        # Skip source complaint
        if exclude_source and ref_num == reference_number:
            continue
        
        # Get full document for unique complaints
        if ref_num not in seen_refs:
            seen_refs.add(ref_num)
            full_doc = doc_dict.get(ref_num, chunk_text)
            results.append((ref_num, float(score), full_doc))
        
        if len(results) >= k:
            break
    
    return results
