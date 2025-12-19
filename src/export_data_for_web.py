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

spark.sparkContext.setLogLevel("WARN")

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

df_results_export = (
    df_results
    .orderBy("query_id", "rank")
    .limit(1000)
    .cache()
)

print(f"   -> Rows: {df_results_export.count()}")

df_results_export.toPandas().to_csv(
    os.path.join(output_dir, "web_data_results.csv"),
    index=False
)

print("-> OK: web_data_results.csv")

# ==========================================
# 3. EXPORT OCR TEXT
# ==========================================
print("2. Export OCR Text...")

df_ocr = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_ocr"
).select("doc_id", "text_ocr")

df_ocr_export = df_ocr.join(
    df_results_export.select("doc_id").distinct(),
    "doc_id",
    "left_semi"
)

print(f"   -> OCR rows: {df_ocr_export.count()}")

df_ocr_export.toPandas().to_csv(
    os.path.join(output_dir, "web_data_ocr.csv"),
    index=False
)

print("-> OK: web_data_ocr.csv")

# ==========================================
# 4. EXPORT QUERIES
# ==========================================
print("3. Export Queries...")

df_queries = spark.read.parquet(
    "hdfs:///bigdata/vidore/english-queries/*.parquet"
).select(
    col("id").alias("query_id"),
    col("text").alias("query_text")
)

df_queries_export = df_queries.join(
    df_results_export.select("query_id").distinct(),
    "query_id",
    "left_semi"
)

print(f"   -> Queries rows: {df_queries_export.count()}")

df_queries_export.toPandas().to_csv(
    os.path.join(output_dir, "web_data_queries.csv"),
    index=False
)

print("-> OK: web_data_queries.csv")

# ==========================================
# 5. EXPORT METRICS
# ==========================================
print("4. Export Evaluation Metrics...")

df_metrics = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_evaluation_metrics"
).select(
    "query_id", "precision_at_10"
)

df_metrics.toPandas().to_csv(
    os.path.join(output_dir, "web_data_metrics.csv"),
    index=False
)

print("-> OK: web_data_metrics.csv")

print("=== EXPORT HOAN TAT ===")
spark.stop()
