import streamlit as st
import pandas as pd
import os 

def show_page():
    st.header(" Kiểm tra kết quả OCR")
    st.markdown("Trích xuất văn bản từ hình ảnh tài liệu.")

    try:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "../result/web_data_ocr.csv")
        
        df_ocr = pd.read_csv(file_path)
        # -------------------------------
        
        if df_ocr.empty:
            st.warning("File dữ liệu rỗng.")
            return

        possible_text_cols = ['text', 'content', 'ocr_text', 'cleaned_text']
        actual_text_col = None
        
        for col in df_ocr.columns:
            if col in possible_text_cols:
                actual_text_col = col
                break
        
        if actual_text_col is None:
            st.error(f" Không tìm thấy cột chứa văn bản.")
            st.write(list(df_ocr.columns))
            return

        doc_list = df_ocr['doc_id'].unique()
        selected_doc = st.selectbox("Chọn tài liệu (Document ID):", doc_list)
        
        text_content = df_ocr[df_ocr['doc_id'] == selected_doc][actual_text_col].values[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f" **ID:** {selected_doc}")
        with col2:
            st.text_area("Nội dung trích xuất:", str(text_content), height=400)
            
    except FileNotFoundError:
        st.error(f" Không tìm thấy file tại: {file_path}")