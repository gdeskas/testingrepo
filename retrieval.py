def retrieve_similar_chunks(
    complaint_id: str,
    query_text: Optional[str] = None,
    k: int = 10,
    return_full_complaints: bool = False,
    exclude_duplicates: bool = True
) -> List[Tuple[str, float, str, Dict]]:
    """
    Retrieve similar chunks using Mosaic AI Vector Search REST API.
    
    Args:
        complaint_id: The complaint ID to find similar items for
        query_text: Optional custom query text
        k: Number of results to return
        return_full_complaints: If True, return full complaint documents
        exclude_duplicates: If True, filter out chunks marked as duplicates
    
    Returns:
        List of (complaint_id, score, text, metadata) tuples
    """
    # Build query vector
    if query_text is not None:
        q_vec = embed_databricks([query_text])[0].tolist()
    else:
        # Find a chunk from this complaint to use as query
        complaint_chunk_indices = [
            i for i, meta in enumerate(chunk_metadata) 
            if meta['complaint_id'] == complaint_id
        ]
        if not complaint_chunk_indices:
            return []
        q_idx = complaint_chunk_indices[0]
        q_vec = embeddings[q_idx].tolist()
    
    # Call Vector Search REST API
    # Request more results if filtering duplicates
    search_k = k * 5 if exclude_duplicates else k * 2
    
    url = f"{WORKSPACE_URL}/api/2.0/vector-search/indexes/{VECTOR_SEARCH_INDEX}/query"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query_vector": q_vec,
        "num_results": search_k,
        "columns": ["chunk_id", "complaint_id", "section", "chunk_index", "chunk_text"]
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    
    if not response.ok:
        raise RuntimeError(f"Vector Search failed: {response.status_code} {response.text[:500]}")
    
    results_raw = response.json()
    
    seen_complaints = set([complaint_id])
    results = []
    
    for row in results_raw.get("result", {}).get("data_array", []):
        # Columns returned in order: chunk_id, complaint_id, section, chunk_index, chunk_text, score
        chunk_id, chunk_cid, section, chunk_idx, text, score = row
        
        # Skip duplicates if requested
        if exclude_duplicates and int(chunk_id) in duplicate_exclusion_set:
            continue
        
        # Skip chunks from the source complaint
        if chunk_cid == complaint_id:
            continue
        
        meta = {
            "complaint_id": chunk_cid,
            "section": section,
            "chunk_index": chunk_idx
        }
        
        if return_full_complaints:
            if chunk_cid not in seen_complaints:
                seen_complaints.add(chunk_cid)
                try:
                    doc_idx = all_ids.index(chunk_cid)
                    full_doc = docs[doc_idx]
                except ValueError:
                    full_doc = text
                results.append((chunk_cid, float(score), full_doc, meta))
        else:
            results.append((chunk_cid, float(score), text, meta))
        
        if len(results) >= k:
            break
    
    return results
