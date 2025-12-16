import streamlit as st
import pandas as pd

def show_page():
    st.header(" Kiểm tra kết quả OCR")
    st.markdown("Trích xuất văn bản từ hình ảnh tài liệu.")

    try:
        # Đọc file CSV
        df_ocr = pd.read_csv("result/web_data_ocr.csv")
        
        # --- ĐOẠN CODE SỬA LỖI ---
        # 1. Kiểm tra xem file có dữ liệu không
        if df_ocr.empty:
            st.warning("File dữ liệu rỗng.")
            return

        # 2. Tự động tìm tên cột chứa nội dung text
        # (Vì có thể Spark lưu là 'content', 'text_content', v.v.)
        possible_text_cols = ['text', 'content', 'ocr_text', 'cleaned_text']
        actual_text_col = None
        
        # Duyệt qua các tên cột thực tế trong file
        for col in df_ocr.columns:
            if col in possible_text_cols:
                actual_text_col = col
                break
        
        # Nếu vẫn không tìm thấy, hiển thị danh sách cột để Debug
        if actual_text_col is None:
            st.error(f" Không tìm thấy cột chứa văn bản (như 'text').")
            st.write("Các cột hiện có trong file CSV là:")
            st.write(list(df_ocr.columns))
            st.info("Hãy sửa lại code trong file ocr.py dòng lấy text theo tên cột đúng ở trên.")
            return
        # -------------------------

        doc_list = df_ocr['doc_id'].unique()
        selected_doc = st.selectbox("Chọn tài liệu (Document ID):", doc_list)
        
        # Lấy text theo tên cột đã tìm thấy (actual_text_col)
        text_content = df_ocr[df_ocr['doc_id'] == selected_doc][actual_text_col].values[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f" **ID:** {selected_doc}")
        with col2:
            st.text_area("Nội dung trích xuất:", str(text_content), height=400)
            
    except FileNotFoundError:
        st.error(" Không tìm thấy file 'result/web_data_ocr.csv'.")
    except Exception as e:
        st.error(f"Lỗi không mong muốn: {e}")