#src/tabs/ocr.py
import streamlit as st
import pandas as pd
import os 

def show_page():
    st.header("Kiểm tra kết quả OCR")
    st.markdown("Trích xuất văn bản từ hình ảnh tài liệu.")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "../result/web_data_ocr.csv")
        
        if not os.path.exists(file_path):
            st.error(f"Không tìm thấy file tại: {file_path}")
            return

        df_ocr = pd.read_csv(file_path)
        
        if df_ocr.empty:
            st.warning("File dữ liệu rỗng.")
            return

        # Tìm cột chứa text
        possible_text_cols = ['text_ocr', 'text', 'content', 'ocr_text', 'cleaned_text']
        actual_text_col = next((col for col in df_ocr.columns if col in possible_text_cols), None)
        
        if actual_text_col is None:
            st.error("Không tìm thấy cột chứa văn bản.")
            st.write("Các cột hiện có:", list(df_ocr.columns))
            return

        df_ocr['display_label'] = df_ocr.apply(
            lambda x: f"{x['doc_id']} - {str(x[actual_text_col])[:50]}...", axis=1
        )
        
        # Selectbox trả về doc_id nhưng hiển thị display_label
        selected_doc_id = st.selectbox(
            "Chọn tài liệu:", 
            options=df_ocr['doc_id'], 
            format_func=lambda x: df_ocr[df_ocr['doc_id'] == x]['display_label'].values[0]
        )
        
        # Lấy nội dung full
        text_content = df_ocr[df_ocr['doc_id'] == selected_doc_id][actual_text_col].values[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"Document ID: {selected_doc_id}")
        with col2:
            st.text_area("Nội dung trích xuất:", str(text_content), height=400)
            
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")