# Set the reference number to view
VIEW_REFERENCE_NUMBER = "CDCR-23502631"  # Change this to view different complaints

# Generate summary for the specified complaint
if VIEW_REFERENCE_NUMBER in all_ids:
    print("="*80)
    print(f"COMPLAINT: {VIEW_REFERENCE_NUMBER}")
    print("="*80)
    print()
    
    # Show original complaint
    target_doc = docs[all_ids.index(VIEW_REFERENCE_NUMBER)]
    print("ORIGINAL COMPLAINT:")
    print("-"*80)
    print(target_doc[:2000])  # Show first 2000 characters
    if len(target_doc) > 2000:
        print(f"\n... (showing first 2000 of {len(target_doc)} characters)")
    print()
    
    # Generate summary using Claude Sonnet 4.5 Conservative
    print("="*80)
    print("GENERATING SUMMARY (Claude Sonnet 4.5 Conservative)...")
    print("="*80)
    print()
    
    summary = rag_summarize(
        reference_number=VIEW_REFERENCE_NUMBER,
        llm_endpoint=AVAILABLE_LLMS["claude-sonnet-4.5"],
        k_neighbors=4,
        temperature=0.3,
        max_tokens=5000
    )
    
    print("SUMMARY:")
    print("-"*80)
    print(summary)
    print()
    print("="*80)
    
    # Show similar complaints used
    similar = retrieve_similar_chunks(
        reference_number=VIEW_REFERENCE_NUMBER,
        k=4,
        return_full_complaints=True,
        exclude_duplicates=True
    )
    
    print()
    print("Similar complaints used for context:")
    for i, (ref, score, doc, meta) in enumerate(similar):
        print(f"  {i+1}. {ref} (Similarity: {score:.3f})")
    
else:
    print(f"ERROR: Complaint {VIEW_REFERENCE_NUMBER} not found.")
    print(f"Available reference numbers: {all_ids[:10]}...")
