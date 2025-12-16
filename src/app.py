#src/app.py
import streamlit as st
# Import các module từ thư mục tabs
from tabs import overview, ocr, search

import sys
import streamlit as st

st.write(sys.version)
# 1. Cấu hình trang
st.set_page_config(page_title="BigData Medical Report", layout="wide")

st.sidebar.title(" TABS Navigation")
page = st.sidebar.radio("Chọn nội dung:", ["1. Tổng quan & Metrics", "2. Kiểm tra OCR", "3. Demo Tìm kiếm"])

st.sidebar.markdown("---")
st.sidebar.info("Dữ liệu tĩnh (CSV Mode)")

# 2. Điều hướng gọi hàm từ các file con
if page == "1. Tổng quan & Metrics":
    overview.show_page()

elif page == "2. Kiểm tra OCR":
    ocr.show_page()

elif page == "3. Demo Tìm kiếm":
    search.show_page()