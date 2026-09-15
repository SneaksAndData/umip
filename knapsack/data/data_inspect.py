import polars as pl

df = pl.read_parquet("knapsack.parquet")

print(f"Rows: {df.height}")
print(f"Columns: {df.width}")
print(df.columns)
print(df)

print(df["Profit"])
