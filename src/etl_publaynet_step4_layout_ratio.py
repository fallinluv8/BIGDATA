# src/etl_publaynet_step4_layout_ratio.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder \
    .appName("MED-ETL-PubLayNet-Step4-Layout-Ratio") \
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
# Total number of regions
# ==========================================
total_regions = df.count()
print(f"Total layout regions: {total_regions}")

# ==========================================
# Calculate ratio by category
# ==========================================
df_ratio = df.groupBy("category_id") \
    .agg(
        count("*").alias("num_regions")
    ) \
    .withColumn(
        "ratio",
        col("num_regions") / total_regions
    ) \
    .orderBy(col("ratio").desc())

# Trigger job & show result
df_ratio.show(truncate=False)

# ==========================================
# Write result to HDFS
# ==========================================
output_path = "hdfs:///bigdata/processed/publaynet_layout_ratio"

df_ratio.write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"Layout ratio written to {output_path}")

spark.stop()
