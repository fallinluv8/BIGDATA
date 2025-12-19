from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# GLOBAL MODEL (LOAD 1 LẦN / EXECUTOR)

tokenizer = None
model = None

DEVICE = "cpu"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 512      # CẮT TEXT OCR SỚM
MAX_TOKENS = 128

# ==========================================
# 1. Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("MED-ETL-ViDoRe-Step6-Embedding") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .config("spark.executor.memory", "2g") \
    .config("spark.task.cpus", "1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession created")

# ==========================================
# 2. Load data
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
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ==========================================
# 4. Pandas UDF Embedding (OPTIMIZED)
# ==========================================
@pandas_udf(ArrayType(FloatType()))
def embedding_udf(text_series: pd.Series) -> pd.Series:
    global tokenizer, model

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

    texts = []
    valid_idx = []

    # ---- PRE-CLEAN & CUT TEXT ----
    for i, text in enumerate(text_series):
        if text and str(text).strip():
            texts.append(str(text)[:MAX_CHARS])
            valid_idx.append(i)
        else:
            texts.append(None)

    results = [None] * len(text_series)

    if not valid_idx:
        return pd.Series(results)

    # ---- BATCH TOKENIZE ----
    encoded = tokenizer(
        [texts[i] for i in valid_idx],
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    for idx, emb in zip(valid_idx, embeddings):
        results[idx] = emb.cpu().numpy().astype("float32").tolist()

    return pd.Series(results)

# ==========================================
# 5. Run Embedding
# ==========================================
print("Embedding Documents...")
df_doc_emb = df_docs.repartition(4) \
    .withColumn("embedding", embedding_udf(col("text_content"))) \
    .filter(col("embedding").isNotNull())

print("Embedding Queries...")
df_query_emb = df_queries.repartition(2) \
    .withColumn("embedding", embedding_udf(col("text_content"))) \
    .filter(col("embedding").isNotNull())

# ==========================================
# 6. Save
# ==========================================
df_doc_emb.coalesce(2).write \
    .mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_doc_embeddings")

print("Saved Document Embeddings")

df_query_emb.coalesce(1).write \
    .mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_query_embeddings")

print("Saved Query Embeddings")

spark.stop()
