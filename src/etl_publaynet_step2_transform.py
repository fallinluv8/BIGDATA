# src/etl_publaynet_step2_transform.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col

spark = SparkSession.builder \
    .appName("MED-ETL-PubLayNet-Step2-Transform") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("SparkSession created")

# ==========================================
# Read raw PubLayNet
# ==========================================
df = spark.read.parquet("hdfs:///bigdata/publaynet/*.parquet")
print("Read raw PubLayNet")

# ==========================================
# Explode annotations 
# ==========================================
df_exploded = df.select(
    col("id").alias("image_id"),
    col("image.path").alias("image_path"),
    explode("annotations").alias("annotation")
)

# ==========================================
# Normalize annotation fields
# ==========================================
df_layout = df_exploded.select(
    "image_id",
    "image_path",
    col("annotation.bbox").alias("bbox"),
    col("annotation.category_id").alias("category_id"),
    col("annotation.area").alias("area"),
    col("annotation.iscrowd").alias("iscrowd")
)

# Trigger job an toàn
row_count = df_layout.count()
print(f"Number of layout rows: {row_count}")

# ==========================================
# Write to HDFS
# ==========================================
output_path = "hdfs:///bigdata/processed/publaynet_layout"

df_layout.write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"Processed data written to {output_path}")

spark.stop()
