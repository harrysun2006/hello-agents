# 学习 parquet
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# read parquet file using pandas
def t01():
    df = pd.read_parquet("gsm8k.parquet")
    print(df.columns)
    print(df.head())

# write pandas df to a parquet file
def t02():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [88.5, 92.0, 79.5]
    })

    df.to_parquet("test.parquet", engine="pyarrow", compression="snappy")

# read parquet with pyarrow
def t05():
    table = pq.read_table("gsm8k.parquet")
    df = table.to_pandas()
    print(df.columns)
    print(df.head())

# write to parquet with pyarrow
def t06():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [88.5, 92.0, 79.5]
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "test2.parquet", compression="zstd")

# read parquet with polars
def t07():
    df = pl.read_parquet("gsm8k.parquet")
    print(df.columns)
    print(df.head())
    for col in df.columns:
        print(f"{col}[0]: {df[col][0]}")

# write to parquet with polars
def t08():
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "score": [88.5, 92.0, 79.5]
    })

    df.write_parquet("test2.parquet", compression="zstd")

def t09():
    df = pl.scan_parquet("gsm8k.parquet")
    # print(df.explain(optimized=True))
    df.collect()
    print(df.head())
    cols = df.collect_schema().names()
    print(cols)
    for col in cols:
        print(f"{col}: {df.collect()[col][0]}")

if __name__ == "__main__":
    # t01()
    # t02()
    # t05()
    # t06()
    t07()
    t08()
    t09()
    pass