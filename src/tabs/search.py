#src/tabs/search.py
import streamlit as st
import pandas as pd
import os 

def show_page():
    st.header("Demo: Semantic Search System")
    st.markdown("Mô phỏng quá trình tìm kiếm dựa trên ngữ nghĩa.")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_results = os.path.join(current_dir, "../result/web_data_results.csv")
        path_content = os.path.join(current_dir, "../result/web_data_ocr.csv")
        
        # Kiểm tra file
        if not os.path.exists(path_results):
            st.error(f"Thiếu file kết quả: {path_results}")
            return
            
        # Đọc dữ liệu
        df_results = pd.read_csv(path_results)
        
        df_content = pd.DataFrame()
        if os.path.exists(path_content):
            df_content = pd.read_csv(path_content)
            # --- FIX LỖI QUAN TRỌNG: Chuẩn hóa ID để khớp dữ liệu (xóa khoảng trắng thừa) ---
            if 'doc_id' in df_content.columns:
                df_content['doc_id'] = df_content['doc_id'].astype(str).str.strip()
        else:
             st.warning("Không tìm thấy file nội dung OCR. Kết quả sẽ chỉ hiện ID.")

        # Chuẩn hóa ID bảng kết quả
        df_results['doc_id'] = df_results['doc_id'].astype(str).str.strip()

        # --- XỬ LÝ DỮ LIỆU ---
        unique_queries = df_results['query_id'].unique()
        
        st.divider()
        st.subheader("1. Thử nghiệm Truy vấn")
        
        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            # Nếu có cột 'query_text' trong file result thì dùng, không thì dùng ID
            selected_query = st.selectbox("Chọn câu hỏi mẫu (Query ID):", unique_queries)
        with col_q2:
            st.write("") 
            st.write("") 
            btn_search = st.button("Tìm kiếm", type="primary", use_container_width=True)

        if btn_search:
            results = df_results[df_results['query_id'] == selected_query].sort_values(by='similarity', ascending=False).head(10)
            
            st.write(f"Kết quả tìm thấy cho: **'{selected_query}'**")
            st.markdown("---")

            for index, row in results.iterrows():
                doc_id = row['doc_id']
                score = row['similarity']
                
                # Biến chứa nội dung hiển thị
                content_preview = "Nội dung không khả dụng (Kiểm tra lại mapping ID)"
                header_text = doc_id # Mặc định tiêu đề là ID
                
                # Logic tìm nội dung text từ bảng OCR
                if not df_content.empty and 'doc_id' in df_content.columns:
                    text_col = next((c for c in ['text_ocr', 'text', 'content'] if c in df_content.columns), None)
                    
                    if text_col:
                        matched_row = df_content[df_content['doc_id'] == doc_id]
                        if not matched_row.empty:
                            full_text = str(matched_row.iloc[0][text_col])
                            # Lấy 300 ký tự làm preview
                            content_preview = full_text[:300] + "..." if len(full_text) > 300 else full_text
                            # Lấy 50 ký tự đầu làm tiêu đề thay cho ID
                            header_text = f"[{score:.4f}] {full_text[:60]}..."

                # --- GIAO DIỆN HIỂN THỊ ---
                # Tiêu đề Expander giờ sẽ hiện một phần nội dung thay vì chỉ ID
                with st.expander(header_text):
                    st.markdown(f"**ID Tài liệu:** `{doc_id}` | **Độ tương đồng:** `{score:.4f}`")
                    st.text_area("Nội dung chi tiết:", content_preview, height=100, key=f"txt_{doc_id}_{index}")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {e}")