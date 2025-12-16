# src/tabs/utils.py
import os
import pandas as pd
import streamlit as st

REQUIRED_FILES = {
    "queries": "web_data_queries.csv",
    "results": "web_data_results.csv",
    "ocr": "web_data_ocr.csv",
    "metrics": "web_data_metrics.csv",
}

def get_data_dir() -> str:
    """
    Xác định thư mục chứa CSV theo thứ tự ưu tiên:
    1. ENV: DATA_DIR (deploy)
    2. /app/src/result (Docker Spark)
    3. ./data (local)
    """
    if "DATA_DIR" in os.environ:
        base = os.environ["DATA_DIR"]
    elif os.path.isdir("/app/src/result"):
        base = "/app/src/result"
    elif os.path.isdir("./data"):
        base = "./data"
    else:
        base = "."

    if "DATA_DIR_UI" not in st.session_state:
        st.session_state.DATA_DIR_UI = base

    data_dir = st.sidebar.text_input(
        " Data directory (CSV):",
        st.session_state.DATA_DIR_UI
    ).strip()

    st.session_state.DATA_DIR_UI = data_dir
    return data_dir


def csv_path(data_dir: str, key: str) -> str:
    return os.path.join(data_dir, REQUIRED_FILES[key])


def validate_required_files(data_dir: str):
    missing = []
    for name in REQUIRED_FILES.values():
        if not os.path.isfile(os.path.join(data_dir, name)):
            missing.append(name)
    return len(missing) == 0, missing


@st.cache_data(show_spinner=False)
def load_csv(data_dir: str, key: str) -> pd.DataFrame:
    path = csv_path(data_dir, key)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all(data_dir: str):
    """
    TRẢ VỀ ĐÚNG THỨ TỰ:
    df_queries, df_results, df_ocr, df_metrics
    """
    return (
        load_csv(data_dir, "queries"),
        load_csv(data_dir, "results"),
        load_csv(data_dir, "ocr"),
        load_csv(data_dir, "metrics"),
    )


def is_ocr_error(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return (
        text.startswith("Error_Load_Model")
        or "Descriptors cannot be created directly" in text
    )


def summarize_text(text: str, max_len=300):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."
