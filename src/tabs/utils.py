# src/tabs/utils.py
import os
import pandas as pd
import streamlit as st

REQUIRED_FILES = [
    "web_data_queries.csv",
    "web_data_results.csv",
    "web_data_ocr.csv",
    "web_data_metrics.csv",
]

def get_data_dir() -> str:
    """
    Cho phép user chọn folder data trong sidebar.
    Ưu tiên:
      1) /app/src/result (nếu đang chạy trong docker spark-client)
      2) ./data
    User có thể override bằng text_input.
    """
    default_candidates = ["/app/src/result", "./data", "."]
    default_dir = next((p for p in default_candidates if os.path.isdir(p)), "./data")

    # Lưu trong session để nhớ
    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = default_dir

    data_dir = st.sidebar.text_input("Nhập đường dẫn thư mục chứa CSV:", st.session_state["data_dir"])
    data_dir = data_dir.strip()
    st.session_state["data_dir"] = data_dir
    return data_dir

def validate_required_files(data_dir: str):
    missing = []
    for f in REQUIRED_FILES:
        path = os.path.join(data_dir, f)
        if not os.path.isfile(path):
            missing.append(f)
    return (len(missing) == 0), missing

@st.cache_data(show_spinner=False)
def load_csv(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)

    # Một số chuẩn hoá nhẹ để tránh lỗi kiểu dữ liệu
    if filename == "web_data_results.csv":
        # đảm bảo rank numeric
        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)

    return df

@st.cache_data(show_spinner=False)
def load_all(data_dir: str):
    df_queries = load_csv(data_dir, "web_data_queries.csv")
    df_results = load_csv(data_dir, "web_data_results.csv")
    df_ocr = load_csv(data_dir, "web_data_ocr.csv")
    df_metrics = load_csv(data_dir, "web_data_metrics.csv")
    return df_queries, df_results, df_ocr, df_metrics

def safe_str(x):
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return ""

def is_ocr_error(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("Error_Load_Model") or "Descriptors cannot be created directly" in t

def summarize_text(text: str, max_len: int = 350) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[:max_len] + "…"
