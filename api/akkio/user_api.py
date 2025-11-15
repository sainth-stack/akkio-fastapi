from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from database import PostgresDatabase

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
    Returns a unified array of objects with 'name' and 'type'.
    - Includes user table names from the database for the given email (type='table').
    - If PDFs are uploaded in the same multipart request, includes their filenames (type='pdf').
    """
    try:
        # Fetch user-specific uploaded/ingested table names
        table_names = db.get_user_tables(email)
        items: List[dict] = []

        # Normalize table names into the required shape
        for name in (table_names or []):
            items.append({"name": str(name), "type": "table"})

        # If PDFs are uploaded, list them as well (metadata only)
        uploaded_docs: List[UploadFile] = []
        if pdfs:
            uploaded_docs.extend(pdfs)
        if files:
            uploaded_docs.extend(files)
        if uploaded_docs:
            for file in uploaded_docs:
                if not file:
                    continue
                items.append({
                    "name": file.filename or "untitled.pdf",
                    "type": "pdf"
                })

        return JSONResponse(content=items)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error retrieving user data: {str(exc)}")


