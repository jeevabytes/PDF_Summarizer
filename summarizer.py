import fitz  # PyMuPDF for better PDF text extraction
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
import re
from bert_score import score
import sys
import numpy as np

# ---------------------------------------------------------
# 1. Extract text from PDF using PyMuPDF (more accurate)
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF with PyMuPDF: {e}")
    return text

# ---------------------------------------------------------
# 2. Clean the extracted text
# ---------------------------------------------------------
def clean_text(text):
    """Cleans raw text by normalizing whitespace and removing artifacts."""
    # Normalize whitespace to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove page numbers that might appear as "Page X"
    text = re.sub(r'Page\s\d+', '', text, flags=re.IGNORECASE)
    # Further cleaning can be added here if needed
    return text

# ---------------------------------------------------------
# 3. Split text into sentence-based chunks
# ---------------------------------------------------------
def chunk_text(text, max_words=400):
    """Splits text into chunks of a maximum word count."""
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
# 4. Save summary to a PDF file
# ---------------------------------------------------------
def save_summary_to_pdf(summary_text, output_pdf="summary_output.pdf"):
    """Saves the given text to a PDF file."""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    y = height - 50
    margin = 50
    
    text_object = c.beginText(margin, y)
    text_object.setFont("Times-Roman", 12)
    
    wrapped_text = textwrap.wrap(summary_text, width=90)
    
    for line in wrapped_text:
        text_object.textLine(line)
        # Move to the next line, and create a new page if necessary
        if text_object.getY() < margin:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(margin, height - margin)
            text_object.setFont("Times-Roman", 12)
            
    c.drawText(text_object)
    c.save()

# ---------------------------------------------------------
# 5. Main function to orchestrate the summarization
# ---------------------------------------------------------
def main(pdf_path):
    """Main function to run the summarization pipeline."""
    print(f"📖 Reading PDF from: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        print("❌ Could not extract text from the PDF. Exiting.")
        return

    print("🧹 Cleaning extracted text...")
    cleaned_text = clean_text(raw_text)
    print(f"Extracted approx. {len(cleaned_text.split())} cleaned words.")

    print("⚡ Loading BART summarizer on CPU...")
    # Forcing CPU usage as requested
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    print("✂️ Splitting text into manageable chunks...")
    chunks = chunk_text(cleaned_text, max_words=400)

    summaries = []
    print(f"Summarizing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if not chunk or len(chunk.split()) < 40:  # Skip empty or very short chunks
            continue
        print(f"🔹 Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    final_summary = "\n\n".join(summaries)

    # Save to TXT and PDF
    with open("summary_output.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)
    save_summary_to_pdf(final_summary, "summary_output.pdf")

    print("\n✅ Summarization complete!")
    print("📄 Output saved as: summary_output.txt and summary_output.pdf")
    print("\n--- Preview of Summary ---")
    print(textwrap.fill(final_summary[:800], width=100))

    # ---------------------------------------------------------
    # 6. Evaluate with BERTScore (Chunk-level and averaged)
    # ---------------------------------------------------------
    if chunks and summaries:
        print("\n🔍 Evaluating with BERTScore (chunk-level)...")
        # Ensure that we only evaluate as many chunks as we have summaries
        valid_chunks = chunks[:len(summaries)]
        
        P, R, F1 = score(summaries, valid_chunks, lang="en", verbose=True)
        
        print("\n📊 Average BERTScore Evaluation (Summary vs. Original Chunks):")
        print(f"Precision: {P.mean().item():.4f}")
        print(f"Recall:    {R.mean().item():.4f}")
        print(f"F1 Score:  {F1.mean().item():.4f}")

    return final_summary 

# ---------------------------------------------------------
# 7. Entry point for command-line execution
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <path_to_pdf_file>")
    else:
        main(sys.argv[1])
