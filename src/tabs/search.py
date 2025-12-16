import streamlit as st
import pandas as pd
import os 

def show_page():
    st.header("🔍 Demo: Semantic Search System")
    st.markdown("Mô phỏng quá trình tìm kiếm dựa trên ngữ nghĩa.")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        path_results = os.path.join(current_dir, "../result/web_data_results.csv")
        
        path_content = os.path.join(current_dir, "../result/web_data_ocr.csv")
        
        # Kiểm tra file
        if not os.path.exists(path_results):
            st.error(f"Thiếu file kết quả: {path_results}")
            return
        if not os.path.exists(path_content):
            st.warning(f"Thiếu file nội dung gốc ({path_content}). Kết quả sẽ chỉ hiện ID.")
            df_content = pd.DataFrame() # Tạo bảng rỗng nếu thiếu
        else:
            df_content = pd.read_csv(path_content)

        df_results = pd.read_csv(path_results)

        # --- XỬ LÝ DỮ LIỆU ---
        # Lấy danh sách câu hỏi duy nhất
        unique_queries = df_results['query_id'].unique()
        
        st.divider()
        st.subheader("1. Thử nghiệm Truy vấn")
        
        # Chọn câu query mẫu
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            selected_query = st.selectbox("Chọn câu hỏi mẫu (Query ID):", unique_queries)
        with col_q2:
            st.write("") # Spacer
            st.write("") 
            btn_search = st.button(" Tìm kiếm", type="primary", use_container_width=True)

        if btn_search:
            # Lọc top 10 kết quả
            results = df_results[df_results['query_id'] == selected_query].sort_values(by='similarity', ascending=False).head(10)
            
            st.write(f"Kết quả tìm thấy cho: **'{selected_query}'**")
            st.markdown("---")


            for index, row in results.iterrows():
                doc_id = row['doc_id']
                score = row['similarity']
                
                # Tìm nội dung text tương ứng với doc_id này
                content_preview = "Nội dung không khả dụng..."
                
                if not df_content.empty and 'doc_id' in df_content.columns:
                    text_col = next((c for c in ['text_ocr', 'text', 'content'] if c in df_content.columns), None)
                    
                    if text_col:
                        # Lấy dòng có doc_id khớp
                        matched_row = df_content[df_content['doc_id'] == doc_id]
                        if not matched_row.empty:
                            full_text = str(matched_row.iloc[0][text_col])
                            content_preview = full_text[:300] + "..." if len(full_text) > 300 else full_text

                # --- GIAO DIỆN HIỂN THỊ TỪNG KẾT QUẢ ---
                # Dùng expander để click vào xem chi tiết
                with st.expander(f" {doc_id} (Độ khớp: {score:.4f})"):
                    st.markdown(f"**Độ tương đồng:** {score*100:.2f}%")
                    st.caption("Nội dung trích dẫn:")
                    st.info(content_preview)
                    st.code(f"ID: {doc_id}", language="text")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {e}")