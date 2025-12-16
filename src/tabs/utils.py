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
    Streamlit Cloud:
    - Working dir = src/
    - CSV nằm ở src/result
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  
    # → trỏ về src/

    default_dir = os.path.join(base_dir, "result")

    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = default_dir

    data_dir = st.sidebar.text_input(
        "Data directory (CSV):",
        st.session_state["data_dir"]
    ).strip()

    st.session_state["data_dir"] = data_dir
    return data_dir


def validate_required_files(data_dir: str):
    missing = []
    for f in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(data_dir, f)):
            missing.append(f)
    return len(missing) == 0, missing


@st.cache_data(show_spinner=False)
def load_csv(data_dir: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, filename))


@st.cache_data(show_spinner=False)
def load_all(data_dir: str):
    return (
        load_csv(data_dir, "web_data_queries.csv"),
        load_csv(data_dir, "web_data_results.csv"),
        load_csv(data_dir, "web_data_ocr.csv"),
        load_csv(data_dir, "web_data_metrics.csv"),
    )


def is_ocr_error(text: str) -> bool:
    t = str(text)
    return t.startswith("Error_Load_Model") or "Descriptors cannot be created directly" in t


def summarize_text(text: str, max_len: int = 350) -> str:
    t = str(text).replace("\n", " ").strip()
    return t if len(t) <= max_len else t[:max_len] + "…"
