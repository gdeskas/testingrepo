print(f"Processing complaint: {REFERENCE_NUMBER}")
print(f"Using model: Claude Sonnet 4.5 (Temperature: {LLM_TEMPERATURE})")
print(f"Retrieving {K_NEIGHBORS} similar complaints...")
print()

# Verify complaint exists
if REFERENCE_NUMBER not in doc_dict:
    print(f"ERROR: Complaint {REFERENCE_NUMBER} not found in pipeline.")
    print(f"Available reference numbers: {list(doc_dict.keys())[:10]}...")
    print()
    print("The complaint may not have been included in the pipeline setup.")
    print("Try running the 'RAG Pipeline Setup' notebook again to refresh the data.")
else:
    # Show original complaint
    print("="*80)
    print(f"ORIGINAL COMPLAINT: {REFERENCE_NUMBER}")
    print("="*80)
    print()
    target_doc = doc_dict[REFERENCE_NUMBER]
    print(target_doc[:5000])  # Show first 5000 characters
    if len(target_doc) > 5000:
        print(f"\n... (showing first 5000 of {len(target_doc)} characters)")
    print()
    
    # Generate summary
    print("="*80)
    print("GENERATING SUMMARY...")
    print("="*80)
    print()
    
    summary = generate_summary(REFERENCE_NUMBER, k_neighbors=K_NEIGHBORS)
    
    print("SUMMARY:")
    print("-"*80)
    print(summary)
    print()
    print("="*80)
