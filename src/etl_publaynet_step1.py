# src/etl_publaynet_step1.py
from pyspark.sql import SparkSession

# ==========================================
# STEP 1 – READ PUBLAYNET PARQUET (MED CLUSTER)
# ==========================================
spark = SparkSession.builder \
    .appName("MED-ETL-PubLayNet-Step1-ReadParquet") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .getOrCreate()

print("SparkSession created")

# ==========================================
# Read PubLayNet from HDFS
# ==========================================
publaynet_path = "hdfs:///bigdata/publaynet/*.parquet"

try:
    df = spark.read.parquet(publaynet_path)
    print("Read PubLayNet parquet from HDFS")

    print("\nSchema of PubLayNet:")
    df.printSchema()

    print("\nSample rows:")
    df.show(5, truncate=False)

except Exception as e:
    print(f"Error reading data: {e}")

spark.stop()
