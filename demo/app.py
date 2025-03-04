import streamlit as st
from src.pdf_utils import extract_text_and_images

st.title("Illustrator")

def display_page_text(pdf_text, page_number):
    st.subheader(f"Страница {page_number + 1}:")
    st.write(pdf_text[page_number])


if 'extracted_content' not in st.session_state:
    pdf_path = "data/with.pdf"
    pdf_text = st.session_state['extracted_content'] = extract_text_and_images(pdf_path)
else:
    pdf_text = st.session_state['extracted_content']

if pdf_text:
    num_pages = len(pdf_text)
    page_number = st.sidebar.number_input("Номер страницы", min_value=1, max_value=num_pages, value=1, step=1) - 1

    display_page_text(pdf_text, page_number)

    if st.button("Сгенерировать", key=f"generate_button_{page_number}"):
        st.session_state['generated_image'] = f"Изображение для страницы {page_number + 1}"
        st.session_state['page_number'] = page_number + 1
        st.success(st.session_state['generated_image'])