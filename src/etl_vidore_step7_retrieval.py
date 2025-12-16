from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, expr, explode, collect_list, struct
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# ==========================================
# 1. Khởi tạo Spark Session
# ==========================================
# ĐÃ SỬA: Xóa bỏ dòng cấu hình cứng RAM 4GB để tránh treo máy
spark = SparkSession.builder \
    .appName("MED-ETL-ViDoRe-Step7-Retrieval-Evaluation") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .getOrCreate()

print("Step 7: Retrieval System Started...")

# ==========================================
# 2. Load Dữ Liệu Embeddings
# ==========================================
# Load Doc Embeddings
df_docs = spark.read.parquet("hdfs:///bigdata/processed/vidore_doc_embeddings") \
    .select(col("doc_id"), col("embedding").alias("doc_vec"))

# Load Query Embeddings
df_queries = spark.read.parquet("hdfs:///bigdata/processed/vidore_query_embeddings") \
    .select(col("query_id"), col("embedding").alias("query_vec"))

# Load Ground Truth (Qrels)
df_qrels = spark.read.parquet("hdfs:///bigdata/vidore/english-qrels/*.parquet") \
    .select(col("query-id").alias("query_id"), col("corpus-id").alias("doc_id"), col("score"))

print(f"Loaded {df_docs.count()} docs and {df_queries.count()} queries.")

# ==========================================
# 3. Thực hiện Tìm kiếm (Exact Search - Brute Force)
# ==========================================

# Để tối ưu, ta chỉ lấy mẫu 100 câu hỏi để demo metrics (tránh treo máy nếu RAM yếu)
df_queries_sample = df_queries.limit(100).cache()

# Cross Join: Query x Docs
df_joined = df_queries_sample.crossJoin(df_docs)

# Hàm tính Cosine Similarity (Dot Product)
@F.udf(FloatType())
def dot_product_udf(v1, v2):
    if not v1 or not v2: return 0.0
    return float(sum(a*b for a, b in zip(v1, v2)))

print("Calculating similarity scores...")
df_scores = df_joined.withColumn("similarity", dot_product_udf(col("query_vec"), col("doc_vec")))

# ==========================================
# 4. Lấy Top-10 Kết quả cho mỗi Query
# ==========================================
windowSpec = Window.partitionBy("query_id").orderBy(col("similarity").desc())

# Lấy Top 10 documents có điểm cao nhất cho mỗi query
df_top_k = df_scores.withColumn("rank", F.rank().over(windowSpec)) \
    .filter(col("rank") <= 10) \
    .select("query_id", "doc_id", "similarity", "rank")

# ==========================================
# 5. Đánh giá (Evaluation)
# ==========================================
# Join kết quả tìm kiếm (Prediction) với Đáp án (Qrels - Truth)
df_eval = df_top_k.join(df_qrels, ["query_id", "doc_id"], "left") \
    .withColumn("is_relevant", F.when(col("score") > 0, 1).otherwise(0)) \
    .fillna(0, subset=["is_relevant"])

# Tính Precision@10 cho từng query
df_metrics = df_eval.groupBy("query_id") \
    .agg(F.sum("is_relevant").alias("relevant_retrieved"), F.count("*").alias("k_retrieved")) \
    .withColumn("precision_at_10", col("relevant_retrieved") / 10.0)

# Tính trung bình (Mean Precision@10) toàn tập
# Kiểm tra nếu dataframe rỗng để tránh lỗi index
avg_precision_row = df_metrics.agg(F.avg("precision_at_10")).collect()
avg_precision = avg_precision_row[0][0] if avg_precision_row else 0.0

print("="*50)
print(f"EVALUATION RESULT (Sample 100 Queries)")
print(f"Average Precision@10: {avg_precision}")
print("="*50)

# ==========================================
# 6. Lưu kết quả cuối cùng
# ==========================================
df_top_k.write.mode("overwrite").parquet("hdfs:///bigdata/processed/vidore_retrieval_results")
df_metrics.write.mode("overwrite").parquet("hdfs:///bigdata/processed/vidore_evaluation_metrics")

print("Results saved to HDFS.")
spark.stop()