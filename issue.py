corporate_roles_df = corporate_roles_df.withColumn(
    header, 
    F.when(F.size(F.col("values")) > idx, F.col("values").getItem(idx)).otherwise(F.lit(None))
)
