from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

# ==========================================
# 1. Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("Export-Data-For-Streamlit") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .getOrCreate()

output_dir = "/app/src/result"
os.makedirs(output_dir, exist_ok=True)

print("=== BAT DAU EXPORT DATA CHO STREAMLIT ===")

# ==========================================
# 2. EXPORT RETRIEVAL RESULTS (TOP-K)
# ==========================================
print("1. Export Retrieval Results...")
df_results = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_retrieval_results"
)

# Lấy 1000 dòng để demo / frontend
df_results_export = df_results.limit(1000)

df_results_export.toPandas().to_csv(
    os.path.join(output_dir, "web_data_results.csv"),
    index=False
)

print("-> OK: web_data_results.csv")

# ==========================================
# 3. EXPORT OCR TEXT (doc_id + text_ocr)
# ==========================================
print("2. Export OCR Text...")
df_ocr = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_ocr"
).select(
    "doc_id", "text_ocr"
)

# Chỉ lấy OCR của những doc xuất hiện trong kết quả retrieval
df_ocr_export = df_ocr.join(
    df_results_export.select("doc_id").distinct(),
    "doc_id",
    "left_semi"
)

df_ocr_export.toPandas().to_csv(
    os.path.join(output_dir, "web_data_ocr.csv"),
    index=False
)

print("-> OK: web_data_ocr.csv")

# ==========================================
# 4. EXPORT QUERIES (query_id + query_text)
# ==========================================
print("3. Export Queries...")
queries_path = "hdfs:///bigdata/vidore/english-queries/*.parquet"

df_queries = spark.read.parquet(queries_path).select(
    col("id").alias("query_id"),
    col("text").alias("query_text")
)

df_queries_export = df_queries.join(
    df_results_export.select("query_id").distinct(),
    "query_id",
    "left_semi"
)

df_queries_export.toPandas().to_csv(
    os.path.join(output_dir, "web_data_queries.csv"),
    index=False
)

print("-> OK: web_data_queries.csv")

# ==========================================
# 5. EXPORT METRICS (Precision@10)
# ==========================================
print("4. Export Evaluation Metrics...")
df_metrics = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_evaluation_metrics"
)

df_metrics.toPandas().to_csv(
    os.path.join(output_dir, "web_data_metrics.csv"),
    index=False
)

print("-> OK: web_data_metrics.csv")

print("=== EXPORT HOAN TAT ===")
spark.stop()
