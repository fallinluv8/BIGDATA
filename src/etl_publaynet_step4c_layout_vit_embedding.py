# src/etl_publaynet_step4c_layout_vit_embedding.py
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType,
    ArrayType, FloatType
)

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# ==========================================
# 1. Spark Session
# ==========================================
spark = SparkSession.builder \
    .appName("ETL-PubLayNet-Step4-CLIP-Embedding") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("SparkSession created")

# ==========================================
# 2. Read processed layout
# ==========================================
df = spark.read.parquet(
    "hdfs:///bigdata/processed/publaynet_layout"
)

df_images = df.select(
    "image_id", "image_path", "category_id"
).distinct()

print("Read publaynet_layout")

# ==========================================
# 3. mapPartitions: LOAD MODEL PER EXECUTOR
# ==========================================
def clip_embedding_partition(iterator):
    device = "cpu"

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    clip_model.eval()

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    for row in iterator:
        image_id = row.image_id
        image_path = row.image_path
        category_id = row.category_id

        try:
            if not image_path or not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():
                features = clip_model.get_image_features(
                    **inputs.to(device)
                )

            embedding = features[0].cpu().numpy().tolist()

            yield (
                image_id,
                category_id,
                image_path,
                embedding
            )

        except Exception:
            continue

# ==========================================
# 4. Run distributed inference
# ==========================================
rdd_embed = df_images.rdd.mapPartitions(
    clip_embedding_partition
)

schema = StructType([
    StructField("image_id", StringType(), True),
    StructField("category_id", IntegerType(), True),
    StructField("image_path", StringType(), True),
    StructField("image_embedding", ArrayType(FloatType()), True),
])

df_embed = spark.createDataFrame(rdd_embed, schema=schema)

print(f"Total embedded rows: {df_embed.count()}")

# ==========================================
# 5. Write to HDFS
# ==========================================
output_path = "hdfs:///bigdata/processed/publaynet_layout_clip_embedding"

df_embed.write \
    .mode("overwrite") \
    .parquet(output_path)

print(f"CLIP embeddings written to {output_path}")

spark.stop()
