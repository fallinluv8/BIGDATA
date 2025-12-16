# src/etl_publaynet_step3_layout_stats.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg

spark = SparkSession.builder \
    .appName("MED-ETL-PubLayNet-Step3-Layout-Stats") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("SparkSession created")

# ==========================================
# Read processed layout
# ==========================================
df = spark.read.parquet(
    "hdfs:///bigdata/processed/publaynet_layout"
)

print("Read publaynet_layout")

# ==========================================
# 1. Count layout by category
# ==========================================
df_category_stats = df.groupBy("category_id") \
    .agg(
        count("*").alias("num_regions"),
        avg("area").alias("avg_area")
    ) \
    .orderBy("num_regions", ascending=False)

# Trigger job
df_category_stats.show(truncate=False)

# ==========================================
# 2. Save stats to HDFS
# ==========================================
output_path = "hdfs:///bigdata/processed/publaynet_stats_by_category"

df_category_stats.write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"Layout stats written to {output_path}")

spark.stop()
