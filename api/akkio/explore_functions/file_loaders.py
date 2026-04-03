import os
from typing import Optional, Tuple, List
import pandas as pd
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from docx import Document
from pathlib import Path


def _find_latest_file_in_directory(directory_path: str, extensions: Tuple[str, ...]) -> Optional[str]:
    try:
        if not os.path.isdir(directory_path):
            return None
        candidates = []
        for name in os.listdir(directory_path):
            path = os.path.join(directory_path, name)
            if os.path.isfile(path) and name.lower().endswith(extensions):
                candidates.append((os.path.getmtime(path), path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception:
        return None


def process_pdf(file_path: str) -> str:
    """Extract plain text from a PDF (used by multi_model_api)."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    if not pages:
        return ""
    return "\n".join([p.page_content or "" for p in pages])


def process_docx(file_path: str) -> str:
    """Extract plain text from a Word document."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def process_txt(file_path: str) -> str:
    """Read a UTF-8 text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _process_pdf_file(file_path: str) -> pd.DataFrame:
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        if not pages:
            raise ValueError("PDF file appears to be empty or corrupted")
        full_text = "\n".join([page.page_content for page in pages])
        if not full_text.strip():
            raise ValueError("PDF file contains no extractable text")
        lines = full_text.split('\n')
        df = pd.DataFrame({
            'line_number': range(1, len(lines) + 1),
            'text_content': lines,
            'document_type': 'PDF',
            'total_pages': len(pages),
            'total_lines': len(lines)
        })
        df = df[df['text_content'].str.strip() != '']
        lower = full_text.lower()
        df['is_legal_document'] = any(k in lower for k in ['article', 'section', 'law', 'regulation', 'act', 'decree', 'resolution', 'provision'])
        df['is_contract'] = any(k in lower for k in ['agreement', 'contract', 'terms', 'conditions', 'party', 'signature'])
        df['is_policy'] = any(k in lower for k in ['policy', 'procedure', 'guideline', 'standard', 'requirement'])
        return df
    except Exception as e:
        raise ValueError(f"Error processing PDF file: {str(e)}")


def _process_docx_file(file_path: str) -> pd.DataFrame:
    try:
        doc = Document(file_path)
        paragraphs = doc.paragraphs
        if not paragraphs:
            raise ValueError("DOCX file appears to be empty or corrupted")
        full_text = "\n".join([p.text for p in paragraphs if p.text.strip()])
        if not full_text.strip():
            raise ValueError("DOCX file contains no extractable text")
        lines = full_text.split('\n')
        df = pd.DataFrame({
            'line_number': range(1, len(lines) + 1),
            'text_content': lines,
            'document_type': 'DOCX',
            'total_paragraphs': len(paragraphs),
            'total_lines': len(lines)
        })
        df = df[df['text_content'].str.strip() != '']
        lower = full_text.lower()
        df['is_legal_document'] = any(k in lower for k in ['article', 'section', 'law', 'regulation', 'act', 'decree', 'resolution', 'provision'])
        df['is_contract'] = any(k in lower for k in ['agreement', 'contract', 'terms', 'conditions', 'party', 'signature'])
        df['is_policy'] = any(k in lower for k in ['policy', 'procedure', 'guideline', 'standard', 'requirement'])
        return df
    except Exception as e:
        raise ValueError(f"Error processing DOCX file: {str(e)}")


def load_dataset(preferred_path: Optional[str] = None) -> pd.DataFrame:
    """
    Dynamically load a dataset with the following priority:
    1) preferred_path (if provided and exists)
    2) ENV EXPLORE_DATASET_PATH (if set and exists)
    3) Latest file in ./uploads (csv/xlsx/xls/pdf/docx)
    4) Fallback: ./data.csv (if exists)
    Raises HTTPException(404) if no dataset found, or 400 if empty.
    """
    search_order: List[str] = []
    if preferred_path:
        search_order.append(preferred_path)
    env_path = os.getenv('EXPLORE_DATASET_PATH')
    if env_path:
        search_order.append(env_path)
    # Prefer project-root uploads over CWD to avoid uvicorn cwd issues
    project_root = Path(__file__).resolve().parents[3]
    uploads_primary = project_root / "uploads"
    uploads_fallback = Path(os.getcwd()) / "uploads"
    candidates: List[Tuple[float, str]] = []
    for d in [uploads_primary, uploads_fallback]:
        latest = _find_latest_file_in_directory(str(d), ('.csv', '.xlsx', '.xls', '.pdf', '.docx'))
        if latest:
            try:
                candidates.append((os.path.getmtime(latest), latest))
            except Exception:
                candidates.append((0.0, latest))
    # Add latest uploads (newest first)
    if candidates:
        for _, p in sorted(candidates, key=lambda x: x[0], reverse=True):
            search_order.append(p)
    # Also support data.csv at project root or CWD
    for p in [project_root / "data.csv", Path(os.getcwd()) / "data.csv"]:
        try:
            if p.exists():
                search_order.append(str(p))
        except Exception:
            continue

    seen: set = set()
    ordered_unique_paths: List[str] = []
    for p in search_order:
        try:
            rp = os.path.realpath(p)
        except Exception:
            rp = p
        if rp not in seen:
            seen.add(rp)
            ordered_unique_paths.append(p)

    for path in ordered_unique_paths:
        try:
            if not os.path.exists(path):
                continue
            lower = path.lower()
            if lower.endswith('.csv'):
                df = pd.read_csv(path)
            elif lower.endswith('.xlsx') or lower.endswith('.xls'):
                df = pd.read_excel(path)
            elif lower.endswith('.pdf'):
                df = _process_pdf_file(path)
            elif lower.endswith('.docx'):
                df = _process_docx_file(path)
            else:
                continue
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"Error loading file {path}: {str(e)}")
            continue
    raise HTTPException(status_code=404, detail="No valid dataset found to analyze")







