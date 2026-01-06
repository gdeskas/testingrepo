# Create a lookup table for full documents (deduplicated)
doc_lookup_dict = {}
for ref_num, doc in zip(df['reference_number'], df['all_columns'].fillna("(no content)")):
    if ref_num not in doc_lookup_dict:
        doc_lookup_dict[ref_num] = doc

doc_lookup_df = pd.DataFrame({
    "reference_number": list(doc_lookup_dict.keys()),
    "full_document": list(doc_lookup_dict.values())
})

DOC_LOOKUP_TABLE = f"`{CATALOG}`.`{SCHEMA}`.`complaint_documents`"

print(f"Saving document lookup to: {DOC_LOOKUP_TABLE}")
spark_doc_df = spark.createDataFrame(doc_lookup_df)
spark_doc_df.write.format("delta").mode("overwrite").saveAsTable(DOC_LOOKUP_TABLE)

print("Document lookup saved successfully")
