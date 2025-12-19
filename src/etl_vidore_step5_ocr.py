from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType
from PIL import Image
import io, re

# =====================================================
# CONFIG
# =====================================================
MAX_TEXT_LEN = 800
MIN_W = 300
MIN_H = 300
MAX_IMG_SIZE = 1000   # ⬅ resize để nhẹ


# 1) Spark

spark = SparkSession.builder \
    .appName("MED-ETL-ViDoRe-Step5-OCR-PaddleOCR") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.task.cpus", "1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("SparkSession created")

#  Accumulator
ocr_counter = spark.sparkContext.accumulator(0)


# 2) Read parquet 

df = spark.read.parquet(
    "hdfs:///bigdata/vidore/english-corpus/*.parquet"
).select(
    "id", "image.bytes"
).withColumnRenamed(
    "id", "doc_id"
).withColumnRenamed(
    "bytes", "image_bytes"
).limit(300)   # giới hạn ẢNH tránh quá nặng

total_images = df.count()
print(f"Total images to OCR (TEST): {total_images}")

# 3) Clean text 
def clean_text(text):
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) < 10:
        return ""
    return t[:MAX_TEXT_LEN]

# 4) OCR partition (LOAD MODEL 1 LẦN / EXECUTOR)

def ocr_partition(rows):
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
    import io

    #  LOAD ĐÚNG THEO YÊU CẦU
    ocr = PaddleOCR(lang="en")

    for row in rows:
        try:
            if not row.image_bytes:
                ocr_counter.add(1)
                yield Row(doc_id=row.doc_id, text_ocr="")
                continue

            img = Image.open(io.BytesIO(row.image_bytes)).convert("RGB")

            if img.width < MIN_W or img.height < MIN_H:
                ocr_counter.add(1)
                yield Row(doc_id=row.doc_id, text_ocr="")
                continue

            # =============================
            # RESIZE ẢNH ĐỂ NHẸ
            # =============================
            w, h = img.size
            scale = min(MAX_IMG_SIZE / max(w, h), 1.0)
            img = img.resize((int(w * scale), int(h * scale)))

            # =============================
            # OCR
            # =============================
            result = ocr.predict(np.array(img))

            texts = []
            if result and "rec_texts" in result[0]:
                texts = result[0]["rec_texts"]

            final_text = clean_text(" ".join(texts))

            ocr_counter.add(1)
            yield Row(doc_id=row.doc_id, text_ocr=final_text)

        except Exception as e:
            ocr_counter.add(1)
            yield Row(doc_id=row.doc_id, text_ocr="")


# 5) RUN OCR (1 PARTITION CHO NHẸ)

df = df.repartition(1)

print("Starting OCR processing...")

rdd_ocr = df.rdd.mapPartitions(ocr_partition)

schema = StructType([
    StructField("doc_id", StringType(), True),
    StructField("text_ocr", StringType(), True)
])

df_ocr = spark.createDataFrame(rdd_ocr, schema)

# =====================================================
# 6) SAVE
# =====================================================
df_ocr.coalesce(1) \
    .write.mode("overwrite") \
    .parquet("hdfs:///bigdata/processed/vidore_ocr")

print("OCR finished successfully")
print(f"TOTAL OCR PROCESSED: {ocr_counter.value} / {total_images}")

spark.stop()
