from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType

import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import torch

# ==========================================
# GLOBAL MODEL (LOAD 1 LẦN / EXECUTOR)
# ==========================================
processor = None
model = None
DEVICE = "cpu"
MODEL_NAME = "microsoft/trocr-small-printed"

# ==========================================
# 1. Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("MED-ETL-ViDoRe-Step5-OCR-TrOCR") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .config("spark.executor.memory", "2g") \
    .config("spark.task.cpus", "2") \
    .getOrCreate()

print("SparkSession created")

# ==========================================
# 2. Read ViDoRe data
# ==========================================
input_path = "hdfs:///bigdata/vidore/english-corpus/*.parquet"

df = spark.read.parquet(input_path)
print("Read ViDoRe parquet successfully")
df.printSchema()

df_select = df.select(
    col("id").alias("doc_id"),
    col("image.bytes").alias("image_bytes")
)

# ==========================================
# 3. OCR Pandas UDF (FIXED)
# ==========================================
@pandas_udf(StringType())
def ocr_trocr_udf(image_bytes_series: pd.Series) -> pd.Series:
    global processor, model

    # ---- LOAD MODEL 1 LẦN / EXECUTOR ----
    if processor is None or model is None:
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(
            MODEL_NAME
        ).to(DEVICE)
        model.eval()

    results = []

    for img_bytes in image_bytes_series:
        try:
            if img_bytes is None:
                results.append("")
                continue

            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pixel_values = processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(DEVICE)

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, max_new_tokens=128
                )

            text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            results.append(text)

        except Exception:
            results.append("")

    return pd.Series(results)

# ==========================================
# 4. Distributed OCR
# ==========================================
df_repart = df_select.repartition(8)

print("Starting OCR processing...")
df_ocr = df_repart.withColumn(
    "text_ocr", ocr_trocr_udf(col("image_bytes"))
)

# ==========================================
# 5. Save to HDFS
# ==========================================
output_path = "hdfs:///bigdata/processed/vidore_ocr"

df_ocr.select("doc_id", "text_ocr") \
    .write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"OCR finished. Data written to {output_path}")

spark.stop()
