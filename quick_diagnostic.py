for pname in ["classify_asset", "extract_corporate_party", "extract_person_party"]:
    docs = all_eval_data.get(pname, [])
    print(f"\n{pname}: {len(docs)} docs")
    for d in docs[:3]:
        text = d.get("CONTEXT_PARSED", "")
        print(f"  {d.get('DOCUMENT_ID', '?')[:20]}  text_len={len(text) if text else 0}  preview={str(text)[:80]}")
