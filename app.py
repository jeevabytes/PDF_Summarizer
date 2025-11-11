import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
import re
import tempfile
from bert_score import score
import numpy as np

# ---------------------------------------------------------
# 1. Extract text from PDF
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# ---------------------------------------------------------
# 2. Clean the text
# ---------------------------------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'Page\s\d+', '', text, flags=re.IGNORECASE)
    return text

# ---------------------------------------------------------
# 3. Chunk the text
# ---------------------------------------------------------
def chunk_text(text, max_words=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk, words = [], "", 0
    for sent in sentences:
        sent_words = len(sent.split())
        if words + sent_words > max_words and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk, words = sent, sent_words
        else:
            current_chunk += " " + sent
            words += sent_words
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ---------------------------------------------------------
# 4. Save summary as PDF
# ---------------------------------------------------------
def save_summary_to_pdf(summary_text, output_pdf="summary_output.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    y = height - 50
    margin = 50

    text_object = c.beginText(margin, y)
    text_object.setFont("Times-Roman", 12)
    wrapped_text = textwrap.wrap(summary_text, width=90)

    for line in wrapped_text:
        text_object.textLine(line)
        if text_object.getY() < margin:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(margin, height - margin)
            text_object.setFont("Times-Roman", 12)

    c.drawText(text_object)
    c.save()
    return output_pdf

# ---------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")
st.title("📘 PDF Summarizer using BART-large-CNN")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("🔍 Extracting and cleaning text..."):
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)

    if not cleaned_text or len(cleaned_text.split()) < 40:
        st.warning("⚠️ Not enough readable text found in the PDF.")
    else:
        st.success(f"✅ Extracted {len(cleaned_text.split())} words from the PDF.")
        st.write("✂️ Splitting text into chunks...")

        chunks = chunk_text(cleaned_text, max_words=400)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

        summaries = []
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            progress.progress((i + 1) / len(chunks))
            if len(chunk.split()) > 40:
                summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                summaries.append(summary)

        final_summary = "\n\n".join(summaries)

        st.subheader("🧾 Generated Summary:")
        st.text_area("", final_summary, height=300)

        # Save outputs
        txt_path = "summary_output.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        pdf_path = save_summary_to_pdf(final_summary)

        col1, col2 = st.columns(2)
        with open(txt_path, "rb") as f1, open(pdf_path, "rb") as f2:
            col1.download_button("⬇️ Download TXT", f1, file_name="summary_output.txt")
            col2.download_button("⬇️ Download PDF", f2, file_name="summary_output.pdf")

        # Optional: Evaluate BERTScore
        if st.checkbox("🔬 Evaluate Summary Quality (BERTScore)"):
            st.info("Evaluating summary vs original text...")
            valid_chunks = chunks[:len(summaries)]
            P, R, F1 = score(summaries, valid_chunks, lang="en", verbose=True)
            st.write(f"**Precision:** {P.mean().item():.4f}")
            st.write(f"**Recall:** {R.mean().item():.4f}")
            st.write(f"**F1 Score:** {F1.mean().item():.4f}")
