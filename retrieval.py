def retrieve_similar_chunks(
    complaint_id: str,
    query_text: Optional[str] = None,
    k: int = 10,
    return_full_complaints: bool = False
) -> List[Tuple[str, float, str, Dict]]:
    """
    Retrieve similar chunks using Mosaic AI Vector Search REST API.
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
    url = f"{WORKSPACE_URL}/api/2.0/vector-search/indexes/{VECTOR_SEARCH_INDEX}/query"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "query_vector": q_vec,
        "num_results": k * 2,  # Get extra to allow filtering
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
