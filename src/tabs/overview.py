import streamlit as st
import pandas as pd

def show_page():
    st.header(" Kết quả Đánh giá (Evaluation Metrics)")
    
    try:
        # Đọc file CSV từ thư mục result
        df_metrics = pd.read_csv("result/web_data_metrics.csv")
        
        # Tính toán
        avg_score = df_metrics["precision_at_10"].mean()
        total_queries = len(df_metrics)
        
        # Hiển thị Metrics
        col1, col2 = st.columns(2)
        col1.metric("MAP@10 Score", f"{avg_score:.4f}", delta="Mục tiêu > 0.5")
        col2.metric("Số lượng Query test", f"{total_queries}")
        
        st.subheader("Bảng chi tiết độ chính xác từng câu hỏi")
        st.dataframe(df_metrics, use_container_width=True)
        
    except FileNotFoundError:
        st.error(" Không tìm thấy file 'result/web_data_metrics.csv'. Hãy chạy export data trước.")