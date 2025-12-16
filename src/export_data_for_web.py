from pyspark.sql import SparkSession
import os  # Thư viện để quản lý file và thư mục

# 1. Khởi tạo Spark
spark = SparkSession.builder \
    .appName("Export-Data") \
    .master("spark://med-spark-master:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://med-namenode:9000") \
    .config("spark.driver.host", "med-spark-client") \
    .getOrCreate()

print("Dang xuat du lieu...")

# ==========================================
# CẤU HÌNH THƯ MỤC OUTPUT
# ==========================================
# Định nghĩa đường dẫn thư mục đích
output_dir = "/app/src/result"

# Kiểm tra nếu thư mục chưa tồn tại thì tạo mới
if not os.path.exists(output_dir):
    print(f"Thu muc '{output_dir}' chua ton tai. Dang tao moi...")
    os.makedirs(output_dir)
else:
    print(f"Thu muc '{output_dir}' da ton tai. Se ghi de file vao day.")

# ==========================================
# XUẤT FILE
# ==========================================

# 2. Xuất file Metrics (Đánh giá)
df_metrics = spark.read.parquet("hdfs:///bigdata/processed/vidore_evaluation_metrics")
# Lưu vào đường dẫn mới
file_metrics = os.path.join(output_dir, "web_data_metrics.csv")
df_metrics.toPandas().to_csv(file_metrics, index=False)
print(f"-> Da luu: {file_metrics}")

# 3. Xuất file Kết quả tìm kiếm (Lấy Top 1000 dòng)
df_results = spark.read.parquet("hdfs:///bigdata/processed/vidore_retrieval_results")
file_results = os.path.join(output_dir, "web_data_results.csv")
df_results.limit(1000).toPandas().to_csv(file_results, index=False)
print(f"-> Da luu: {file_results}")

# 4. Xuất file OCR mẫu (Lấy 50 dòng)
df_ocr = spark.read.parquet("hdfs:///bigdata/processed/vidore_ocr")
file_ocr = os.path.join(output_dir, "web_data_ocr.csv")
df_ocr.limit(50).toPandas().to_csv(file_ocr, index=False)
print(f"-> Da luu: {file_ocr}")

print("\n------------------------------------------------")
print(f"XONG! Kiem tra thu muc: {output_dir} tren may .")
print("------------------------------------------------")
spark.stop()