from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

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
# 2. Đọc dữ liệu (OCR Text + Queries)
# ==========================================
# Đọc kết quả OCR từ bước 5
df_docs = spark.read.parquet("hdfs:///bigdata/processed/vidore_ocr") \
    .select(col("doc_id"), col("text_ocr").alias("text_content"))

# Đọc câu hỏi (Queries)
# [SỬA LỖI TẠI ĐÂY]: Đổi col("_id") thành col("id")
df_queries = spark.read.parquet("hdfs:///bigdata/vidore/english-queries/*.parquet") \
    .select(col("id").alias("query_id"), col("text").alias("text_content"))

print("Data loaded successfully")

# ==========================================
# 3. Định nghĩa UDF Embedding (Dùng 'transformers' thuần)
# ==========================================
# Hàm Mean Pooling để lấy vector đại diện cho câu
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@pandas_udf(ArrayType(FloatType()))
def embedding_udf(text_series: pd.Series) -> pd.Series:
    # Cấu hình Device
    device = "cpu"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Load model (chỉ load 1 lần mỗi batch)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        # Trả về list rỗng nếu lỗi load model
        return pd.Series([None] * len(text_series))

    results = []
    
    # Batch processing for loop
    for text in text_series:
        try:
            if not text or len(str(text).strip()) == 0:
                results.append(None)
                continue
                
            # Tokenize
            encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # Inference
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings (để tính Cosine Similarity bằng Dot Product)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # Convert to list
            emb_list = sentence_embeddings[0].cpu().numpy().tolist()
            results.append(emb_list)
            
        except Exception:
            results.append(None)
            
    return pd.Series(results)

# ==========================================
# 4. Thực thi Embedding
# ==========================================
# 4.1. Embed Documents
print("Embedding Documents...")
df_docs_repart = df_docs.repartition(8)
df_doc_emb = df_docs_repart.withColumn("embedding", embedding_udf(col("text_content"))) \
                           .filter(col("embedding").isNotNull())

# 4.2. Embed Queries
print("Embedding Queries...")
df_queries_repart = df_queries.repartition(4)
df_query_emb = df_queries_repart.withColumn("embedding", embedding_udf(col("text_content"))) \
                                .filter(col("embedding").isNotNull())

# ==========================================
# 5. Lưu kết quả
# ==========================================
# Lưu Document Embeddings
df_doc_emb.write.mode("overwrite").parquet("hdfs:///bigdata/processed/vidore_doc_embeddings")
print("Saved Document Embeddings")

# Lưu Query Embeddings
df_query_emb.write.mode("overwrite").parquet("hdfs:///bigdata/processed/vidore_query_embeddings")
print("Saved Query Embeddings")

spark.stop()