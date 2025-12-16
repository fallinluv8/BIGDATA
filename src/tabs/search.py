import streamlit as st
import pandas as pd

def show_page():
    st.header(" Demo: Semantic Search System")
    st.markdown("Mô phỏng quá trình tìm kiếm dựa trên ngữ nghĩa.")
    
    try:
        df_results = pd.read_csv("result/web_data_results.csv")
        
        # Lấy danh sách câu hỏi
        unique_queries = df_results['query_id'].unique()
        
        st.subheader(" Thử nghiệm Truy vấn")
        selected_query = st.selectbox("Chọn câu hỏi (Query):", unique_queries)
        
        if st.button("Tìm kiếm ngay", type="primary"):
            # Lọc kết quả
            results = df_results[df_results['query_id'] == selected_query].sort_values(by='similarity', ascending=False).head(10)
            
            st.write(f"Kết quả cho: **'{selected_query}'**")
            
            # Hiển thị bảng
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
        st.error(" Không tìm thấy file 'result/web_data_results.csv'.")