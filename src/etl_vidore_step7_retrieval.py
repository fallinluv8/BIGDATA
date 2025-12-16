from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

# ==========================================
# 1. Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("MED-ETL-ViDoRe-Step7-Retrieval-Evaluation") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .getOrCreate()

print("Step 7: Retrieval & Evaluation Started")

# ==========================================
# 2. Load Embeddings & Ground Truth
# ==========================================
df_docs = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_doc_embeddings"
).select(
    col("doc_id"),
    col("embedding").alias("doc_vec")
)

df_queries = spark.read.parquet(
    "hdfs:///bigdata/processed/vidore_query_embeddings"
).select(
    col("query_id"),
    col("embedding").alias("query_vec")
)

df_qrels = spark.read.parquet(
    "hdfs:///bigdata/vidore/english-qrels/*.parquet"
).select(
    col("query-id").alias("query_id"),
    col("corpus-id").alias("doc_id"),
    col("score")
)

print(f"Loaded {df_docs.count()} docs, {df_queries.count()} queries")

# ==========================================
# 3. Sample Queries (DEMO – tránh treo máy)
# ==========================================
df_queries_sample = df_queries.limit(100).cache()

# ==========================================
# 4. Brute-force Retrieval (Cosine Similarity)
# ==========================================
# Vì vector đã normalize ở Step 6 => cosine = dot product

@F.udf(FloatType())
def dot_product(v1, v2):
    if not v1 or not v2:
        return 0.0
    return float(sum(a * b for a, b in zip(v1, v2)))

print("Computing similarity scores...")
df_scores = df_queries_sample.crossJoin(df_docs) \
    .withColumn("similarity", dot_product(col("query_vec"), col("doc_vec")))

# ==========================================
# 5. Top-10 Retrieval
# ==========================================
windowSpec = Window.partitionBy("query_id").orderBy(col("similarity").desc())

df_top_k = df_scores \
    .withColumn("rank", F.row_number().over(windowSpec)) \
    .filter(col("rank") <= 10) \
    .select("query_id", "doc_id", "similarity", "rank")

print("Top-10 retrieval done")

# ==========================================
# 6. Evaluation – Precision@10
# ==========================================
df_eval = df_top_k.join(
    df_qrels,
    ["query_id", "doc_id"],
    how="left"
).withColumn(
    "is_relevant",
    F.when(col("score") > 0, 1).otherwise(0)
).fillna(0, subset=["is_relevant"])

df_metrics = df_eval.groupBy("query_id") \
    .agg(
        F.sum("is_relevant").alias("relevant_retrieved"),
        F.count("*").alias("k_retrieved")
    ).withColumn(
        "precision_at_10",
        col("relevant_retrieved") / F.least(col("k_retrieved"), F.lit(10))
    )

avg_precision = df_metrics.agg(
    F.avg("precision_at_10")
).collect()[0][0]

print("=" * 60)
print("EVALUATION RESULT (Sample 100 Queries)")
print(f"Mean Precision@10: {avg_precision}")
print("=" * 60)

# ==========================================
# 7. Save Results
# ==========================================
df_top_k.write \
    .mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_retrieval_results")

df_metrics.write \
    .mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_evaluation_metrics")

print("Step 7 finished. Results saved to HDFS.")

spark.stop()
