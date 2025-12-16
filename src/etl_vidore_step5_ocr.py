from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType

import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
import torch

# ==========================================
# 1. Khởi tạo Spark Session
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
# 2. Đọc dữ liệu ViDoRe từ HDFS
# ==========================================
input_path = "hdfs:///bigdata/vidore/english-corpus/*.parquet"

try:
    df = spark.read.parquet(input_path)
    print("Read ViDoRe parquet successfully")
    df.printSchema()
except Exception as e:
    print(f"Error reading path {input_path}: {e}")
    spark.stop()
    exit(1)

# ==========================================
# [QUAN TRỌNG] SỬA LỖI TẠI ĐÂY
# File gốc có cột 'id', ta đổi tên thành 'doc_id'
# ==========================================
df_select = df.select(
    col("id").alias("doc_id"), 
    col("image.bytes").alias("image_bytes")
)

# ==========================================
# 3. Định nghĩa Pandas UDF (Chạy Model OCR)
# ==========================================
@pandas_udf(StringType())
def ocr_trocr_udf(image_bytes_series: pd.Series) -> pd.Series:
    # Cấu hình device (Docker chạy CPU)
    device = "cpu"
    model_name = "microsoft/trocr-small-printed"
    
    # Load model (chỉ load 1 lần mỗi batch để tối ưu)
    try:
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    except Exception as e:
        return pd.Series([f"Error_Load_Model: {str(e)}"] * len(image_bytes_series))

    results = []
    
    for img_bytes in image_bytes_series:
        try:
            if img_bytes is None:
                results.append("")
                continue
            
            # Convert binary -> Image
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Preprocess & Inference
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_new_tokens=128)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            results.append(generated_text)
        except Exception:
            # Nếu ảnh lỗi thì trả về rỗng để job vẫn chạy tiếp
            results.append("") 
            
    return pd.Series(results)

# ==========================================
# 4. Thực thi OCR phân tán
# ==========================================
# Repartition: Chia dữ liệu thành 8 phần để 2 Worker (mỗi con 2 core) xử lý song song
df_repart = df_select.repartition(8) 

print("Starting OCR processing... (Please wait for model downloading & inference)")
df_ocr = df_repart.withColumn("text_ocr", ocr_trocr_udf(col("image_bytes")))

# ==========================================
# 5. Lưu kết quả xuống HDFS
# ==========================================
output_path = "hdfs:///bigdata/processed/vidore_ocr"

df_ocr.select("doc_id", "text_ocr").write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"OCR finished. Data written to {output_path}")

spark.stop()