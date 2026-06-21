import os
from typing import Optional

import pandas as pd
from docx import Document
from pypdf import PdfReader


def _extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def process_pdf(file_path: str) -> str:
    """Extract plain text from a PDF (used by multi_model_api)."""
    return _extract_pdf_text(file_path)


def process_docx(file_path: str) -> str:
    """Extract plain text from a Word document."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def process_txt(file_path: str) -> str:
    """Read a UTF-8 text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
