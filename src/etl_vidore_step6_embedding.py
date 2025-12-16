from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ==========================================
# GLOBAL MODEL (LOAD 1 LẦN / EXECUTOR)
# ==========================================
tokenizer = None
model = None
DEVICE = "cpu"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================================
# 1. Khởi tạo Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("MED-ETL-ViDoRe-Step6-Embedding") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .config("spark.executor.memory", "2g") \
    .config("spark.task.cpus", "2") \
    .getOrCreate()

print("SparkSession created")

# ==========================================
# 2. Đọc dữ liệu (OCR + Queries)
# ==========================================
df_docs = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_ocr"
).select(
    col("doc_id"),
    col("text_ocr").alias("text_content")
)

df_queries = spark.read.parquet(
    "hdfs:///bigdata/vidore/english-queries/*.parquet"
).select(
    col("id").alias("query_id"),
    col("text").alias("text_content")
)

print("Data loaded successfully")

# ==========================================
# 3. Mean Pooling
# ==========================================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ==========================================
# 4. Pandas UDF Embedding (FIXED)
# ==========================================
@pandas_udf(ArrayType(FloatType()))
def embedding_udf(text_series: pd.Series) -> pd.Series:
    global tokenizer, model

    # ---- LOAD MODEL 1 LẦN / EXECUTOR ----
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

    results = []

    for text in text_series:
        try:
            if not text or not str(text).strip():
                results.append(None)
                continue

            encoded = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                output = model(**encoded)

            emb = mean_pooling(output, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

            results.append(emb[0].cpu().tolist())

        except Exception:
            results.append(None)

    return pd.Series(results)

# ==========================================
# 5. Thực thi Embedding
# ==========================================
print("Embedding Documents...")
df_docs_repart = df_docs.repartition(4)
df_doc_emb = df_docs_repart.withColumn(
    "embedding", embedding_udf(col("text_content"))
).filter(col("embedding").isNotNull())

print("Embedding Queries...")
df_queries_repart = df_queries.repartition(2)
df_query_emb = df_queries_repart.withColumn(
    "embedding", embedding_udf(col("text_content"))
).filter(col("embedding").isNotNull())

# ==========================================
# 6. Lưu kết quả
# ==========================================
df_doc_emb.write \
    .mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_doc_embeddings")

print("Saved Document Embeddings")

df_query_emb.write \
    .mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_query_embeddings")

print("Saved Query Embeddings")

spark.stop()
