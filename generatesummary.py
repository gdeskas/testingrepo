
# Show the original complaint text
if REFERENCE_NUMBER in doc_dict:
    print("="*80)
    print(f"ORIGINAL COMPLAINT: {REFERENCE_NUMBER}")
    print("="*80)
    print()
    
    target_doc = doc_dict[REFERENCE_NUMBER]
    print(target_doc[:5000])  # Show first 5000 characters
    
    if len(target_doc) > 5000:
        print(f"\n... (showing first 5000 of {len(target_doc)} characters)")
    
    print()
    print("="*80)
else:
    print(f"ERROR: Complaint {REFERENCE_NUMBER} not found.")
