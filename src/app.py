# src/app.py
import sys
import streamlit as st

# Import các module từ thư mục tabs
from tabs import overview, ocr, search
from tabs.utils import get_data_dir, validate_required_files

# =========================
# Debug Python version (optional)
# =========================
st.write(sys.version)

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="BigData Medical Report",
    layout="wide"
)

# =========================
# Sidebar
# =========================
st.sidebar.title(" TABS Navigation")

page = st.sidebar.radio(
    "Chọn nội dung:",
    [
        "1. Tổng quan & Metrics",
        "2. Kiểm tra OCR",
        "3. Demo Tìm kiếm",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Dữ liệu tĩnh (CSV Mode)")

# =========================
# Resolve DATA DIR (CHỖ DUY NHẤT DÙNG os)
# =========================
data_dir = get_data_dir()

ok, missing = validate_required_files(data_dir)
if not ok:
    st.error(" Thiếu file CSV bắt buộc:")
    for f in missing:
        st.write(f"- {f}")
    st.stop()

# =========================
# Routing
# =========================
if page == "1. Tổng quan & Metrics":
    overview.show_page(data_dir)

elif page == "2. Kiểm tra OCR":
    ocr.show_page(data_dir)

elif page == "3. Demo Tìm kiếm":
    search.show_page(data_dir)
