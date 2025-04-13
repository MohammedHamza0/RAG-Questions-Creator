import streamlit as st
from src.helper import (get_pdf_text, 
                        get_text_chunks, 
                        get_vector_store, 
                        user_query,
                        save_text_to_pdf)

def main():
    st.set_page_config(page_title="Chat PDFs", page_icon="💬")
    
     # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "response" not in st.session_state:
        st.session_state.response = ""

    st.header("Exam Questions Generator🧾")
    st.caption("Hi, I will help you to prepare to you exam..☺")

    with st.sidebar:
        st.title("Menu:")
        doc_files = st.file_uploader(label="Upload your PDF files", accept_multiple_files=True)
        if st.button("Process"):
            if doc_files is not None:
                with st.spinner("Processing...."):
                    raw_text = get_pdf_text(doc_files)
                    if not raw_text.strip():
                        st.error("❌ Couldn't extract any text from the PDF, Please upload a suitable one.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("❌ Failed to split text into chunks.")
                        else:
                            get_vector_store(text_chunks)
                            st.success("✅ Done.")
                    
    question = st.text_input("What do you want me to do?")
    num_questions = st.number_input("Choose the number of questions", min_value=10, max_value=100)
    difficulty_level = st.selectbox("Difficulty Level", options=["easy", "medium", "hard"])
    question_types = st.selectbox("Questions Type", options=["multiple choice", "true/false", "short answer", "essay"])

    generate = st.button("Generate")

    if generate:
          with st.spinner("Generating..."):
               response = user_query(
               question=question,
               num_questions=num_questions,
               difficulty_level=difficulty_level,
               question_types=question_types,
               include_answers="Yes"
               )
               st.session_state.response = response

    if st.session_state.response:
          st.subheader("Generated Questions:")
          st.write(st.session_state.response)
          
          
          file_name = st.text_input("Write the file name (with .pdf)", value="questions.pdf")

          if st.button("Save To PDF"):
               if file_name:
                    file_path = save_text_to_pdf(st.session_state.response, filename=file_name)
                    st.success("✅ PDF saved successfully!")
                    st.write(f"📄 File Path: {file_path}")
               else:
                    st.warning("⚠️ Please write a file name.")

if __name__ == "__main__":
    main()
