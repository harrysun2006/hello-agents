Q: 请介绍一下 parquet 文件格式以及用 python 如何进行读写

===

下面我从**是什么 → 为什么用 → 文件结构 → Python 读写（pandas / pyarrow / polars）→ 实战建议**，给你一份**工程师可直接上手**的 Parquet 说明。

---

# 一、Parquet 是什么？

**Parquet** 是一种 **列式（columnar）存储的二进制文件格式**，最初由 Twitter & Cloudera 推动，现为 **Apache 顶级项目**，是大数据 / 数据分析领域的事实标准之一。

> 一句话：
> **Parquet = 为分析型查询优化的列存储格式**

---

# 二、为什么用 Parquet？（相比 CSV / JSON）

| 特性     | CSV / JSON | Parquet                        |
| ------ | ---------- | ------------------------------ |
| 存储方式   | 行式         | **列式**                         |
| 文件大小   | 大          | **小很多（压缩）**                    |
| 读取速度   | 慢          | **快（按列读）**                     |
| Schema | 无          | **强类型 Schema**                 |
| 压缩     | 弱          | **内建（Snappy / ZSTD 等）**        |
| 大数据生态  | 一般         | **Spark / Hive / DuckDB 原生支持** |

### 列式的核心优势

* 只读你需要的列（projection pushdown）
* 更高压缩率（同一列数据类型一致）
* 非常适合：

  * OLAP
  * 数据仓库
  * 特征存储
  * 离线分析

---

# 三、Parquet 文件内部结构（概念）

```
Parquet File
├── Row Groups（行组）
│   ├── Column Chunk（每列一个）
│   │   ├── Pages（数据页）
│   │   └── Statistics（min/max/null count）
├── File Metadata（schema、编码、压缩）
```

关键点：

* **Row Group**：读取/并行的基本单位
* **Column Chunk**：真正的列式存储
* **Statistics**：支持谓词下推（过滤加速）

---

# 四、Python 里如何读写 Parquet（重点）

## 1️⃣ 安装依赖（任选其一或组合）

```bash
pip install pandas pyarrow fastparquet polars
```

> 推荐：**pyarrow（官方、最全）**

---

## 2️⃣ 用 pandas 读写 Parquet（最常见）

### 写入

```python
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [88.5, 92.0, 79.5]
})

df.to_parquet("data.parquet", engine="pyarrow", compression="snappy")
```

### 读取

```python
df = pd.read_parquet("data.parquet")
```

### 只读部分列（非常快）

```python
df = pd.read_parquet("data.parquet", columns=["id", "score"])
```

---

## 3️⃣ 用 pyarrow（更底层 & 强大）

### 写入

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pandas(df)
pq.write_table(table, "data.parquet", compression="zstd")
```

### 读取

```python
table = pq.read_table("data.parquet")
df = table.to_pandas()
```

### 读取 + 条件过滤（predicate pushdown）

```python
table = pq.read_table(
    "data.parquet",
    filters=[("score", ">", 80)]
)
```

> 这一步在大文件上**非常关键**，只扫描必要的 row group。

---

## 4️⃣ 用 Polars（速度怪兽，强烈推荐）

### 写入

```python
import polars as pl

df = pl.DataFrame({
    "id": [1, 2, 3],
    "score": [88.5, 92.0, 79.5]
})

df.write_parquet("data.parquet", compression="zstd")
```

### 读取

```python
df = pl.read_parquet("data.parquet")
```

### **惰性读取（Lazy + 超快）**

```python
df = pl.scan_parquet("data.parquet")
df = df.filter(pl.col("score") > 80).select(["id", "score"])
df.collect()
```

> `scan_parquet` 是真正的 **query planner + pushdown**，非常适合大数据。

---

## 5️⃣ 分区 Parquet（数据仓库常用）

### 写成目录结构

```python
df.to_parquet(
    "dataset/",
    partition_cols=["date", "country"]
)
```

结果：

```
dataset/
├── date=2025-01-01/country=CN/part-0.parquet
├── date=2025-01-01/country=US/part-0.parquet
```

### 读取（自动识别分区列）

```python
pd.read_parquet("dataset/")
```

---

# 五、Parquet vs CSV：真实场景对比

假设 1 亿行、10 列数据：

| 操作   | CSV      | Parquet     |
| ---- | -------- | ----------- |
| 磁盘占用 | 20+ GB   | **3–5 GB**  |
| 读两列  | 全表扫描     | **只读两列**    |
| 过滤   | Python 层 | **文件级统计跳过** |
| 解析   | 文本解析     | **二进制直读**   |

👉 在分析型任务中，**Parquet 是数量级优势**。

---

# 六、常见坑 & 注意事项

### 1️⃣ Parquet ≠ 行级随机写

* 不适合频繁 append 单行
* 正确方式：**批量写 + 分文件**

---

### 2️⃣ Schema 变更要谨慎

* 同一 dataset 下 schema 不一致会炸
* 新增列 OK，改类型风险大

---

### 3️⃣ 压缩算法选择

| 算法     | 特点           |
| ------ | ------------ |
| snappy | 快，默认         |
| zstd   | **压缩率高（推荐）** |
| gzip   | 压缩高但慢        |

---

# 七、典型使用场景

* 数据仓库（Hive / Spark / DuckDB）
* 特征存储（ML features）
* 日志分析
* OLAP 查询
* Python / Pandas 大数据替代 CSV

---

# 八、一句话总结

> **Parquet 是为“读多、分析型、列查询”而生的格式**
> 在 Python 里：
>
> * **小中数据：pandas + pyarrow**
> * **大数据 / 高性能：polars + scan_parquet**

---

如果你愿意，我可以：

* 给你 **CSV → Parquet 的最佳实践脚本**
* 演示 **DuckDB / Spark / Arrow Flight** 读 Parquet
* 帮你设计 **数据湖目录结构（partition / schema 演进）**

你现在是打算用 Parquet 做什么？（日志 / 特征 / 分析 / 数据仓库）
