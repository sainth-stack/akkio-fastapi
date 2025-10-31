import os
import time
import json
from typing import Optional, Dict, Any, List
import re

from fastapi import APIRouter, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic.json import pydantic_encoder

from .sla_apis import manage_session_memory
from ingestion.uae_legislation import (
    PDFMeta,
    discover_pdfs,
    discover_pdfs_async,
    download_pdf,
    pdf_to_chunks,
    ensure_dir,
    logger as ingest_logger,
    upload_to_s3,
)
from rag.chroma_store import upsert_chunks, CHROMA_DIR
from rag.qa import answer_with_rag


legislation_router = APIRouter()


DOWNLOAD_DIR = os.getenv("UAE_LEGISLATION_DOWNLOAD_DIR", os.path.join("uploads", "legislation"))
S3_BUCKET = os.getenv("UAE_S3_BUCKET", "akio-report-data")
S3_PREFIX = os.getenv("UAE_S3_PREFIX", "uae_legislation/")

# In-memory job registry
JOBS: Dict[str, Dict[str, Any]] = {}


def _jsonable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=pydantic_encoder))
    except Exception:
        return obj


def _sanitize_answer(text: Optional[str]) -> str:
    """Remove branding, collapse whitespace, and strip excessive HTML breaks.

    - Removes any occurrence of the company name (case-insensitive), including
      variants like "CXINGULARITY ENHANCED".
    - Replaces multiple <br> tags (any form) with a single space.
    - Removes empty paragraph tags and collapses multiple blank lines.
    """
    if not text:
        return ""

    cleaned = text
    # Remove company branding (case-insensitive)
    cleaned = re.sub(r"(?i)\bCXINGULARITY\b(\s*ENHANCED)?", "", cleaned)

    # Normalize different <br> variants to a single space (avoid large gaps)
    cleaned = re.sub(r"(?i)(<br\s*/?>\s*)+", " ", cleaned)

    # Remove completely empty paragraphs
    cleaned = re.sub(r"(?i)(<p>\s*</p>\s*)+", "", cleaned)

    # Collapse multiple whitespace/newlines
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def _format_legal_response(user_query: str, answer_text: Optional[str], citations: Optional[List[Dict[str, Any]]], language: str = "en") -> str:
    # Sanitize incoming answer first to remove branding and excessive spacing
    answer_text = _sanitize_answer(answer_text)
    # Check if answer_text already contains HTML tags (from LLM response)
    is_html = answer_text and ("<h" in answer_text or "<p>" in answer_text or "<ul>" in answer_text or "<li>" in answer_text)
    
    # Start with HTML wrapper if answer is HTML, otherwise use plain text
    if is_html:
        parts: List[str] = []
    else:
        parts: List[str] = ["<h3>LEGAL AI SYSTEM</h3>\n"]

    has_support = bool(citations) and len([c for c in (citations or []) if c.get("pdf_url")]) > 0
    
    # Show answer if we have answer text OR if we have citations (even if answer says insufficient, it might have useful info)
    # Only use fallback if we have no answer text AND no citations
    if answer_text and answer_text.strip():
        # If answer is already HTML, use it as-is; otherwise wrap in HTML tags
        if is_html:
            parts.append(answer_text.strip())
        else:
            # Convert plain text to HTML paragraphs
            lines = answer_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # Check if line looks like a heading (short and ends with :)
                    if len(line) < 80 and line.endswith(':'):
                        parts.append(f"<h4>{line}</h4>\n")
                    else:
                        parts.append(f"<p>{line}</p>\n")
    elif has_support:
        # If we have citations but no answer, still show a basic response with citations
        if language == "ar":
            parts.append("<p>بناءً على الوثائق القانونية المتاحة، فيما يلي المراجع ذات الصلة:</p>\n")
        else:
            parts.append("<p>Based on the available legal documents, here are the relevant references:</p>\n")
    else:
        # Only show fallback if truly no answer and no citations
        if language == "ar":
            parts.append("<p>مرحباً! يبدو أن سؤالك غير مكتمل أو غير واضح. ومع ذلك، بناءً على الوثائق القانونية المقدمة، يمكنني تقديم ملخص للمفاهيم والإجراءات القانونية ذات الصلة بموضوعك. إذا كان لديك سؤال محدد، يرجى تقديم المزيد من التفاصيل.</p>\n")
        else:
            parts.append("<p>Hello! It seems like your question is incomplete or unclear. However, based on the legal documents provided, I can offer a summary of relevant legal concepts and procedures related to your topic. If you have a specific question, please feel free to provide more details.</p>\n")
    
    # Add footer section
    parts.append("<hr>\n")
    parts.append("<p><em>This information is based on UAE legal documents and is for guidance only. For specific legal matters, please consult with a qualified UAE legal professional.</em></p>\n")
    parts.append("<p><strong>SOURCE:</strong> <a href=\"https://uaelegislation.gov.ae\" target=\"_blank\" style=\"color: #3498db;\">https://uaelegislation.gov.ae</a></p>\n")
    
    # Always show references if we have them (formatted as HTML)
    if has_support:
        if language == "ar":
            parts.append("<h4>المراجع</h4>\n")
        else:
            parts.append("<h4>References</h4>\n")
        parts.append("<ul>\n")
        for c in citations or []:
            url = c.get("pdf_url")
            if not url:
                continue
            title = c.get("title") or "Legal Document"
            law_no = c.get("law_number")
            article = c.get("article")
            pages = c.get("pages")
            
            # Build reference text with details
            ref_parts = [title]
            details = []
            if law_no:
                if language == "ar":
                    details.append(f"قانون {law_no}")
                else:
                    details.append(f"Law {law_no}")
            if article:
                details.append(article)
            if pages:
                details.append(f"Page {pages}" if language == "en" else f"صفحة {pages}")
            
            if details:
                ref_parts.append(" (" + ", ".join(details) + ")")
            
            ref_text = "".join(ref_parts)
            parts.append(f"<li><strong>{ref_text}</strong> — <a href=\"{url}\" target=\"_blank\" style=\"color: #3498db;\">{url}</a></li>\n")
        parts.append("</ul>\n")
    
    # Security/branding lines removed per request
    parts.append(f"<p><em>USER INPUT: {user_query}</em></p>\n")
    
    return "".join(parts)


@legislation_router.post("/api/legislation/ingest")
async def ingest_legislation(
    url: str = Form("https://uaelegislation.gov.ae/ar/legislations?sector=47"),
    force: bool = Form(False),
    max_files: int = Form(0),
    use_browser: bool = Form(False),
    max_pages: int = Form(50),
):
    try:
        t0 = time.perf_counter()
        ensure_dir(DOWNLOAD_DIR)
        url = (url or "").strip().strip('"').strip("'")
        if url.lower().endswith(".pdf"):
            discovered = [PDFMeta(url=url, title=os.path.basename(url), year=None, law_number=None, source_page=url)]
        else:
            if use_browser:
                discovered = await discover_pdfs_async(url, max_detail_pages=max_pages)
            else:
                discovered = discover_pdfs(url, use_browser=False, max_detail_pages=max_pages)
        downloaded_files: List[str] = []
        total_chunks = 0

        uploaded_count = 0
        upload_errors = []
        for idx, meta in enumerate(discovered, 1):
            local_path = download_pdf(meta, DOWNLOAD_DIR)
            if not local_path:
                continue
            downloaded_files.append(local_path)
            
            # Upload to S3
            try:
                s3_url = upload_to_s3(local_path, S3_BUCKET, f"{S3_PREFIX}{os.path.basename(local_path)}")
                if s3_url:
                    uploaded_count += 1
                    ingest_logger.info(f"Uploaded to S3: {s3_url}")
                else:
                    upload_errors.append(f"Failed to upload {os.path.basename(local_path)} (boto3 not installed or AWS credentials not configured)")
            except Exception as e:
                upload_errors.append(f"Error uploading {os.path.basename(local_path)}: {str(e)}")
                ingest_logger.warning(f"S3 upload failed for {local_path}: {e}")
            
            chunks = pdf_to_chunks(local_path, meta)
            # prepare for chroma
            chunk_dicts = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
            total_chunks += upsert_chunks(chunk_dicts)
            if max_files and idx >= max_files:
                break

        elapsed = time.perf_counter() - t0
        ingest_logger.info(
            f"INGEST COMPLETE | files={len(downloaded_files)} chunks={total_chunks} uploaded={uploaded_count} time={elapsed:.2f}s | chroma_dir={CHROMA_DIR}"
        )
        response_data = {
            "message": "Ingest completed",
            "files": len(downloaded_files),
            "chunks": total_chunks,
            "elapsed_sec": round(elapsed, 2),
            "chroma_dir": CHROMA_DIR,
            "uploaded_to_s3": uploaded_count,
            "s3_bucket": S3_BUCKET if uploaded_count > 0 else None,
            "s3_prefix": S3_PREFIX if uploaded_count > 0 else None,
        }
        if upload_errors:
            response_data["upload_warnings"] = upload_errors
            error_count = len(upload_errors)
            response_data["message"] += f" (Note: {error_count} upload failures - check upload_warnings)"
        return JSONResponse(content=response_data, status_code=200)
    except Exception as e:
        ingest_logger.exception(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@legislation_router.post("/api/legislation/chat")
async def legislation_chat(
    query: str = Form(...),
    top_k: int = Form(8),
    session_id: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    law_number: Optional[str] = Form(None),
):
    try:
        filters: Dict[str, Any] = {}
        if year:
            filters["year"] = year
        if law_number:
            filters["law_number"] = law_number

        result = answer_with_rag(query, top_k=top_k, where=filters or None)

        # Manage chat memory
        if not session_id:
            session_id = os.urandom(8).hex()
        answer_text = result.get("answer", "")
        manage_session_memory(session_id, user_message=query, bot_message=answer_text)

        language = result.get("language", "en")
        formatted = _format_legal_response(query, answer_text, result.get("citations"), language=language)
        payload = {"answer": formatted, "citations": result.get("citations"), "session_id": session_id}
        return JSONResponse(content=_jsonable(payload), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Alias route for explore-style querying
@legislation_router.post("/api/legislation/explore")
async def legislation_explore(
    query: str = Form(...),
    top_k: int = Form(8),
    session_id: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    law_number: Optional[str] = Form(None),
):
    return await legislation_chat(query=query, top_k=top_k, session_id=session_id, year=year, law_number=law_number)


async def _ingest_worker(job_id: str, url: str, use_browser: bool, max_pages: int, max_files: int):
    try:
        ensure_dir(DOWNLOAD_DIR)
        JOBS[job_id] = {"status": "discovering", "files": {}, "processed": 0, "discovered": 0}
        url = (url or "").strip().strip('"').strip("'")
        if url.lower().endswith(".pdf"):
            discovered = [PDFMeta(url=url, title=os.path.basename(url), year=None, law_number=None, source_page=url)]
        else:
            if use_browser:
                discovered = await discover_pdfs_async(url, max_detail_pages=max_pages)
            else:
                discovered = discover_pdfs(url, use_browser=False, max_detail_pages=max_pages)
        JOBS[job_id]["discovered"] = len(discovered)
        JOBS[job_id]["status"] = "processing"

        for idx, meta in enumerate(discovered, 1):
            if max_files and idx > max_files:
                break
            info: Dict[str, Any] = {"pdf_url": meta.url, "title": meta.title}
            try:
                local_path = download_pdf(meta, DOWNLOAD_DIR)
                info["local_path"] = local_path
                
                # Upload to S3
                if local_path:
                    try:
                        s3_url = upload_to_s3(local_path, S3_BUCKET, f"{S3_PREFIX}{os.path.basename(local_path)}")
                        if s3_url:
                            info["s3_url"] = s3_url
                            info["s3_upload_status"] = "success"
                            ingest_logger.info(f"[async] Uploaded to S3: {s3_url}")
                        else:
                            info["s3_upload_status"] = "failed"
                            info["s3_upload_error"] = "boto3 not installed or AWS credentials not configured"
                            ingest_logger.warning(f"[async] S3 upload failed for {local_path}: boto3/credentials issue")
                    except Exception as upload_err:
                        info["s3_upload_status"] = "failed"
                        info["s3_upload_error"] = str(upload_err)
                        ingest_logger.warning(f"[async] S3 upload error for {local_path}: {upload_err}")
                
                chunks = pdf_to_chunks(local_path, meta) if local_path else []
                chunk_dicts = [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]
                added = upsert_chunks(chunk_dicts) if chunk_dicts else 0
                info["chunks"] = added
            except Exception as e:
                info["error"] = str(e)
                ingest_logger.error(f"[async] Error processing {meta.url}: {e}")
            JOBS[job_id]["files"][str(idx)] = info
            JOBS[job_id]["processed"] = idx
        JOBS[job_id]["status"] = "completed"
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}


@legislation_router.post("/api/legislation/ingest_async")
async def ingest_legislation_async(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    use_browser: bool = Form(False),
    max_pages: int = Form(50),
    max_files: int = Form(0),
):
    job_id = os.urandom(8).hex()
    url = (url or "").strip().strip('"').strip("'")
    background_tasks.add_task(_ingest_worker, job_id, url, use_browser, max_pages, max_files)
    return JSONResponse(content={"message": "ingest started", "job_id": job_id, "s3_bucket": S3_BUCKET, "s3_prefix": S3_PREFIX})


@legislation_router.get("/api/legislation/ingest_status")
async def ingest_legislation_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JSONResponse(content=_jsonable(job))


__all__ = ["legislation_router"]


