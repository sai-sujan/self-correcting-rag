import os
import glob
from pathlib import Path
import pymupdf
import pymupdf4llm
from config import DOCS_DIR, MARKDOWN_DIR

"""
PyMuPDF is a high-performance Python library for data extraction, analysis, conversion, and manipulation of PDF (and other) documents.

PyMuPDF4LLM is a specialized library built on top of PyMuPDF. It is designed specifically to convert PDF content into formats that are easier for Large Language Models (LLMs) to understand, such as Markdown.


"""

def pdf_to_markdown(pdf_path, output_dir):
    """Convert single PDF to markdown"""
    doc = pymupdf.open(pdf_path)
    md = pymupdf4llm.to_markdown(
        doc, 
        write_images=False,  # Skip images for now
        ignore_images=True
    )
    
    # Clean text
    md_cleaned = md.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Save as .md file
    output_path = Path(output_dir) / Path(pdf_path).stem
    output_path.with_suffix(".md").write_text(md_cleaned, encoding='utf-8')
    print(f"✓ Converted: {Path(pdf_path).name}")

def convert_all_pdfs():
    """Convert all PDFs in docs/ folder"""
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    
    pdf_files = glob.glob(f"{DOCS_DIR}/*.pdf")
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in {DOCS_DIR}/")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF(s)")
    
    for pdf_path in pdf_files:
        pdf_to_markdown(pdf_path, MARKDOWN_DIR)
    
    print(f"\n✅ All PDFs converted to {MARKDOWN_DIR}/")

if __name__ == "__main__":
    convert_all_pdfs()