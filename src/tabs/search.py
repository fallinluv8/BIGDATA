import streamlit as st
import pandas as pd
import os # <--- Quan trọng

def show_page():
    st.header(" Demo: Semantic Search System")
    st.markdown("Mô phỏng quá trình tìm kiếm dựa trên ngữ nghĩa.")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "../result/web_data_results.csv")
        
        df_results = pd.read_csv(file_path)
        # -------------------------------
        
        unique_queries = df_results['query_id'].unique()
        
        st.subheader(" Thử nghiệm Truy vấn")
        selected_query = st.selectbox("Chọn câu hỏi (Query):", unique_queries)
        
        if st.button("Tìm kiếm ngay", type="primary"):
            results = df_results[df_results['query_id'] == selected_query].sort_values(by='similarity', ascending=False).head(10)
            
            st.write(f"Kết quả cho: **'{selected_query}'**")
            
            st.dataframe(
                results[['doc_id', 'similarity']],
                column_config={
                    "doc_id": "Mã tài liệu",
                    "similarity": st.column_config.ProgressColumn(
                        "Độ tương đồng",
                        format="%.4f",
                        min_value=0,
                        max_value=1,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
    except FileNotFoundError:
        st.error(f" Không tìm thấy file tại: {file_path}")