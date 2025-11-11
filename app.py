import streamlit as st
from summarizer import main
import os
import traceback

# Streamlit app configuration
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

# App title and description
st.title("📄 PDF Summarizer")
st.markdown("Upload a PDF to get an **AI-generated summary**.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

# Buttons
col1, col2 = st.columns([1, 1])
submit = col1.button("🔍 Summarize", use_container_width=True)
clear = col2.button("🧹 Clear", use_container_width=True)

# Clear session state when needed
if clear:
    st.session_state.clear()
    st.rerun()

# Function to process PDF
def summarize_pdf(file):
    if file is None:
        st.warning("⚠️ Please upload a PDF file.")
        return None, None, None

    try:
        # Save the uploaded PDF temporarily
        temp_path = "uploaded_file.pdf"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Call your main summarization function
        summary = main(temp_path)

        if not summary:
            return "No summary could be generated.", None, None

        # Check for output files
        txt_file = "summary_output.txt" if os.path.exists("summary_output.txt") else None
        pdf_file = "summary_output.pdf" if os.path.exists("summary_output.pdf") else None

        return summary, txt_file, pdf_file

    except Exception as e:
        error_msg = f"❌ Error processing PDF:\n{str(e)}\n\n{traceback.format_exc()}"
        st.error(error_msg)
        return None, None, None


# When user clicks submit
if submit:
    with st.spinner("⏳ Summarizing your PDF... Please wait."):
        summary, txt_path, pdf_path = summarize_pdf(uploaded_file)

        if summary:
            st.success("✅ Summary generated successfully!")
            st.text_area("Generated Summary", summary, height=300)

            if txt_path and os.path.exists(txt_path):
                with open(txt_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Summary (TXT)",
                        data=f,
                        file_name="summary_output.txt",
                        mime="text/plain"
                    )

            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Summary (PDF)",
                        data=f,
                        file_name="summary_output.pdf",
                        mime="application/pdf"
                    )

st.markdown("---")
st.info("⚙️ The summarization process may take a few minutes depending on PDF size.")
