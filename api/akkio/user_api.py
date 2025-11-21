from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from database import PostgresDatabase
import base64
import io
from mimetypes import guess_type
from PIL import Image

user_router = APIRouter()

# Local database instance for this router
db = PostgresDatabase()


@user_router.post("/api/get_user_data")
async def get_user_data(
    email: str = Form(...),
    pdfs: Optional[List[UploadFile]] = File(None, alias="pdfs"),
    files: Optional[List[UploadFile]] = File(None, alias="files")
):
    """
    Returns a unified array of objects with 'name', 'type', and 'subtype'.
    - Includes user items from DB for the given email (name, type, subtype).
    - If PDFs/files are uploaded in the same multipart request, includes their filenames (best-effort type/subtype).
    """
    try:
        # Fetch user-specific uploaded/ingested items with metadata
        db_items = db.get_user_items(email)
        items: List[dict] = list(db_items or [])
        # Attach inline image data URLs for items of type image
        for it in items:
            try:
                if (it.get("type") or "").lower() == "image":
                    raw = db.get_raw_file(it.get("name"))
                    if raw:
                        mime = None
                        try:
                            with Image.open(io.BytesIO(raw)) as img:
                                # Prefer accurate mime from Pillow if available
                                mime = Image.MIME.get(img.format, None)
                        except Exception:
                            mime = None
                        if not mime:
                            # Fallback to extension-based or default to PNG
                            guessed = guess_type((it.get("name") or ""))[0]
                            mime = guessed or "image/png"
                        b64 = base64.b64encode(raw).decode("ascii")
                        it["image_data_url"] = f"data:{mime};base64,{b64}"
            except Exception:
                # Skip embedding if any error
                continue

        # If PDFs/other files are uploaded, list them as well (metadata only)
        uploaded_docs: List[UploadFile] = []
        if pdfs:
            uploaded_docs.extend(pdfs)
        if files:
            uploaded_docs.extend(files)
        if uploaded_docs:
            for file in uploaded_docs:
                if not file:
                    continue
                fname = file.filename or "untitled"
                ext = (fname.split(".")[-1] if "." in fname else "").lower()
                # Best-effort mapping
                ftype = None
                fsub = None
                if ext in {"pdf"}:
                    ftype, fsub = "pdf", "report"
                elif ext in {"csv"}:
                    ftype = "csv"
                elif ext in {"xls", "xlsx"}:
                    ftype = "excel"
                elif ext in {"docx"}:
                    ftype = "word"
                elif ext in {"png", "jpg", "jpeg", "webp"}:
                    ftype = "image"
                items.append({
                    "name": fname.split(".")[0],
                    "type": ftype or "file",
                    "subtype": fsub
                })

        return JSONResponse(content=items)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error retrieving user data: {str(exc)}")


