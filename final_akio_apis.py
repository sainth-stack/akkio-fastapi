import ast
import asyncio
import base64
import io
import os
import re
import smtplib
import sys
import tempfile
import traceback
import uuid
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from dateutil.parser import parse
from datetime import datetime
import boto3
from langchain_community.document_loaders import PyPDFLoader
from docx import Document
from PIL import Image
from fastapi.responses import StreamingResponse, Response
import dateutil.parser
import markdown
import plotly.graph_objects as go
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi import UploadFile, File, Form, HTTPException, Request, status, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from starlette.responses import HTMLResponse, JSONResponse
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from img_datascout import ImageGen_agent
from datascout import DataScout_agent, extract_sections_tool, extract_num_pages_tool, \
    pdf_generator_tool, initialize_llm
from database import PostgresDatabase
from synthetic_data_function import generate_synthetic_data
from models import DBConnectionRequest, GenAIBotRequest, ModelRequest
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from plotly.graph_objs import Figure
import plotly as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import shutil
from api.sla_apis import sla_router
from api.sla_tabs_api import sla_tabs_router
from api.akkio.main import akkio_router
from api.akkio.upload_api import upload_router
from api.akkio.url_scraper import url_router
from api.sharepoint import (
    list_sharepoint_files as sp_list_sharepoint_files,
    get_app_token as sp_get_app_token,
    resolve_site_and_drive as sp_resolve_site_and_drive,
    upload_file_to_folder as sp_upload_file_to_folder,
    download_file_from_path as sp_download_file_from_path,
    select_latest_file as sp_select_latest_file,
    list_folder_children as sp_list_folder_children,
    delete_drive_item as sp_delete_drive_item,
    SHAREPOINT_INPUT_FOLDER,
    SHAREPOINT_OUTPUT_FOLDER,
    start_sharepoint_automation,
    stop_sharepoint_automation,
    run_sharepoint_automation_once,
    get_sharepoint_automation_status,
    process_latest_if_new,
)
from api.chat2doc_fastapi import chat2doc
from api.akkio.usage_tracking import record_llm_usage
from llm_config import get_llm_config, get_api_key, get_model_name, get_default_model_for_provider

"""
=================================================================================
HOW TO USE LLM CONFIGURATION (API Key and Model)
=================================================================================

The system now supports user-specific API keys and models. Use these functions:

1. get_llm_config(user_email) - Returns dict with 'api_key' and 'model'
2. get_api_key(user_email) - Returns just the API key
3. get_model_name(user_email) - Returns just the model name

Example usage:
--------------
# Get full config for a user
config = get_llm_config(user_email)
llm = ChatOpenAI(model=config["model"], openai_api_key=config["api_key"])

# Or get them separately
api_key = get_api_key(user_email)
model = get_model_name(user_email)
llm = ChatOpenAI(model=model, openai_api_key=api_key)

# Without user email (uses defaults)
config = get_llm_config()
llm = ChatOpenAI(model=config["model"], openai_api_key=config["api_key"])

If user has saved custom settings in database, those will be used.
Otherwise, defaults from environment (OPENAI_API_KEY and gpt-4o-mini) are used.
=================================================================================
"""

# Calculate comprehensive metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score
import threading
import time
import uuid
from pathlib import Path
from PyPDF2 import PdfReader
from datetime import datetime, date
from decimal import Decimal

# Supervised AutoML (multi-model training + best-model selection)
from supervised_automl import train_supervised_automl, predict_supervised

load_dotenv()

app = FastAPI()
# Optionally start SharePoint automation on startup if enabled by env
ENABLE_SHAREPOINT_AUTOMATION = os.getenv("ENABLE_SHAREPOINT_AUTOMATION", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}

# --- Enhanced server-side de-dup state for SharePoint uploads (file-based persistence) ---
# Track multiple processed files with timestamps for better de-duplication
PROCESSED_FILES_LOCK = threading.RLock()
DEDUP_CACHE_HOURS = 24  # Keep processed files in cache for 24 hours
PROCESSED_FILES_TXT = Path(__file__).resolve().parent / "uploads_sla" / "processed_files.txt"

def _load_processed_files() -> Dict[str, float]:
    """Load processed files from txt file"""
    try:
        if not PROCESSED_FILES_TXT.exists():
            return {}
        
        processed_files = {}
        with open(PROCESSED_FILES_TXT, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '|' in line:
                    try:
                        filename, timestamp_str = line.split('|', 1)
                        processed_files[filename] = float(timestamp_str)
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines
        return processed_files
    except Exception as e:
        print(f"⚠️ Error loading processed files: {e}")
        return {}

def _save_processed_files(processed_files: Dict[str, float]):
    """Save processed files to txt file"""
    try:
        PROCESSED_FILES_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_FILES_TXT, 'w', encoding='utf-8') as f:
            for filename, timestamp in processed_files.items():
                f.write(f"{filename}|{timestamp}\n")
    except Exception as e:
        print(f"⚠️ Error saving processed files: {e}")

def _cleanup_old_processed_files():
    """Remove processed files older than DEDUP_CACHE_HOURS from cache"""
    current_time = time.time()
    cutoff_time = current_time - (DEDUP_CACHE_HOURS * 3600)
    
    with PROCESSED_FILES_LOCK:
        processed_files = _load_processed_files()
        original_count = len(processed_files)
        
        # Remove expired entries
        processed_files = {filename: timestamp for filename, timestamp in processed_files.items() 
                          if timestamp >= cutoff_time}
        
        if len(processed_files) < original_count:
            _save_processed_files(processed_files)
            removed_count = original_count - len(processed_files)
            print(f"🧹 Cleaned up {removed_count} expired processed files from cache")

def _is_file_recently_processed(filename: str) -> bool:
    print(f"🔄 Checking if file was processed recently: {filename}")
    if not filename:
        return False
    
    normalized_filename = filename.strip().lower()
    current_time = time.time()
    cutoff_time = current_time - (DEDUP_CACHE_HOURS * 3600)
    
    with PROCESSED_FILES_LOCK:
        processed_files = _load_processed_files()
        # Directly apply cutoff without calling nested cleanup to avoid nested locking delays
        timestamp = processed_files.get(normalized_filename)
        if timestamp is not None and timestamp >= cutoff_time:
            return True
        # Opportunistic cleanup: remove any expired entries quickly
        expired_keys = [k for k, ts in processed_files.items() if ts < cutoff_time]
        if expired_keys:
            for k in expired_keys:
                processed_files.pop(k, None)
            _save_processed_files(processed_files)
    
    return False

def _mark_file_as_processed(filename: str):
    """Mark a file as processed with current timestamp"""
    if not filename:
        return
    
    normalized_filename = filename.strip().lower()
    current_time = time.time()
    
    with PROCESSED_FILES_LOCK:
        processed_files = _load_processed_files()
        processed_files[normalized_filename] = current_time
        _save_processed_files(processed_files)
        print(f"📝 Marked file as processed: {filename}")

@app.on_event("startup")
async def _on_startup():
    """Initialize application resources on startup."""
    # Initialize database connection pool
    try:
        PostgresDatabase._ensure_pool()
        print("✅ Database connection pool initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize database pool: {e}")
    
    # Initialize LLM settings table
    try:
        db.ensure_connection()
        db.create_llm_settings_table()
        db.close()
        print("✅ LLM settings table initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize LLM settings table: {e}")
    
    if ENABLE_SHAREPOINT_AUTOMATION:
        # Use backend's default automation interval (configurable in api/sharepoint.py)
        start_sharepoint_automation()

@app.on_event("shutdown")
async def _on_shutdown():
    """Clean up resources on shutdown."""
    if ENABLE_SHAREPOINT_AUTOMATION:
        stop_sharepoint_automation()
    
    # Close all database connections
    try:
        PostgresDatabase.close_all_connections()
        print("✅ Database connections closed cleanly")
    except Exception as e:
        print(f"⚠️ Warning: Error closing database connections: {e}")

# Automation control endpoints
@app.get("/api/sharepoint/automation/status")
async def sharepoint_automation_status():
    return JSONResponse(content=get_sharepoint_automation_status())

@app.post("/api/sharepoint/automation/start")
async def sharepoint_automation_start(every_minutes: int = 10):
    start_sharepoint_automation(every_minutes=every_minutes)
    return JSONResponse(content={"message": "Automation started", **get_sharepoint_automation_status()})

@app.post("/api/sharepoint/automation/stop")
async def sharepoint_automation_stop():
    stop_sharepoint_automation()
    return JSONResponse(content={"message": "Automation stopped", **get_sharepoint_automation_status()})

@app.post("/api/sharepoint/automation/run_once")
async def sharepoint_automation_run_once():
    run_sharepoint_automation_once()
    return JSONResponse(content={"message": "Run once completed", **get_sharepoint_automation_status()})

global connection_obj
# Global variables for chat memory management
CHAT_MEMORY: Dict[str, list] = {}  # In-memory store; replace as needed
CHAT_MEMORY_LOCK = asyncio.Lock()

# Enable CORS for frontend access - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Required when using * - enables true all-origin support
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
db = PostgresDatabase()

# Create client getter function using llm_config
def get_openai_client(user_email: str = None):
    """Get OpenAI client using llm_config"""
    from llm_config import get_api_key
    return OpenAI(api_key=get_api_key(user_email))

client = get_openai_client()  # Default client

app.include_router(sla_router)
app.include_router(sla_tabs_router)
app.include_router(akkio_router, prefix="/api")
app.include_router(url_router)
app.include_router(chat2doc)
app.include_router(upload_router)

from api.app_creator.app_creator import router as app_builder_router
from api.app_creator.apps_api import router as app_builder_apps_router
from api.app_creator.prd_api import router as prd_router
from api.app_creator.agent_api import router as agent_router
from api.app_creator.codegen_api import router as codegen_router
from api.app_creator.planning_api import router as planning_router
from api.app_creator.deployment_api import router as deployment_router
from api.app_creator.github_api import router as github_router
from api.app_creator.test_api import router as test_router

app.include_router(app_builder_router)
app.include_router(app_builder_apps_router, prefix="/api/app-builder")
app.include_router(prd_router)
app.include_router(agent_router)
app.include_router(codegen_router)
app.include_router(planning_router)
app.include_router(deployment_router)
app.include_router(github_router)
app.include_router(test_router)


# Health check endpoints
@app.get("/health")
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to monitor API and database connectivity.
    Returns the status of the API server and database connection pool.
    """
    health_status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": {}
    }
    
    # Check database pool status
    try:
        pool_status = PostgresDatabase.get_pool_status()
        health_status["database"] = pool_status
        
        # Try a simple database query to verify connectivity
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            health_status["database"]["connectivity"] = "connected"
        except Exception as db_err:
            health_status["database"]["connectivity"] = "error"
            health_status["database"]["error"] = str(db_err)
            health_status["api"] = "degraded"
    except Exception as e:
        health_status["database"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["api"] = "degraded"
    
    status_code = 200 if health_status["api"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/api/sharepoint/input_files")
async def get_sharepoint_input_files():
    """Get list of files from SharePoint input folder"""
    try:
        result = sp_list_sharepoint_files(SHAREPOINT_INPUT_FOLDER)
        return JSONResponse(content={
            "success": True,
            "folder": result.get("folder", SHAREPOINT_INPUT_FOLDER),
            "files": result.get("items", []),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/api/sharepoint/delete/{folder_type}/{file_id}")
async def delete_sharepoint_file(folder_type: str, file_id: str):
    """Delete a file from SharePoint by file ID in input or output folder."""
    try:
        if folder_type not in ["input", "output"]:
            raise HTTPException(status_code=400, detail="Invalid folder type. Use 'input' or 'output'")
        token = sp_get_app_token()
        _, drive = sp_resolve_site_and_drive(token)
        drive_id = drive["id"]
        # Ensure the item exists and is in the expected folder (best-effort)
        try:
            item_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}"
            headers = {"Authorization": f"Bearer {token}"}
            item_res = requests.get(item_url, headers=headers)
            if not item_res.ok:
                raise HTTPException(status_code=404, detail="File not found")
            item = item_res.json()
            parent_path = ((item.get("parentReference") or {}).get("path") or "")
            expected_folder = SHAREPOINT_INPUT_FOLDER if folder_type == "input" else SHAREPOINT_OUTPUT_FOLDER
            # parent path comes like /drives/{id}/root:/<path>
            if expected_folder not in parent_path:
                # Not fatal; allow delete anyway if requested
                pass
        except HTTPException:
            raise
        except Exception:
            pass
        # Perform delete
        sp_delete_drive_item(token, drive_id, file_id)
        return JSONResponse(content={
            "success": True,
            "message": "File deleted successfully",
            "folder_type": folder_type,
            "file_id": file_id,
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/sharepoint/output_files")
async def get_sharepoint_output_files():
    """Get list of files from SharePoint output folder"""
    try:
        result = sp_list_sharepoint_files(SHAREPOINT_OUTPUT_FOLDER)
        return JSONResponse(content={
            "success": True,
            "folder": result.get("folder", SHAREPOINT_OUTPUT_FOLDER),
            "files": result.get("items", []),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/sharepoint/all_files")
async def get_all_sharepoint_files():
    """Get list of files from both input and output folders"""
    try:
        result = sp_list_sharepoint_files()  # Lists both folders
        return JSONResponse(content={
            "success": True,
            "input": result.get("input", {"folder": SHAREPOINT_INPUT_FOLDER, "items": []}),
            "output": result.get("output", {"folder": SHAREPOINT_OUTPUT_FOLDER, "items": []}),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/sharepoint/download/{folder_type}/{file_id}")
async def download_sharepoint_file(folder_type: str, file_id: str):
    """Download a file from SharePoint by file ID"""
    try:
        # Validate folder type
        if folder_type not in ["input", "output"]:
            raise HTTPException(status_code=400, detail="Invalid folder type. Use 'input' or 'output'")
        
        folder_path = SHAREPOINT_INPUT_FOLDER if folder_type == "input" else SHAREPOINT_OUTPUT_FOLDER
        
        token = sp_get_app_token()
        _, drive = sp_resolve_site_and_drive(token)
        drive_id = drive["id"]
        
        # Get file details first
        file_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}"
        headers = {"Authorization": f"Bearer {token}"}
        file_response = requests.get(file_url, headers=headers)
        
        if not file_response.ok:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_response.json()
        file_name = file_info.get("name", "download")
        
        # Download file content
        download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"
        content_response = requests.get(download_url, headers=headers)
        
        if not content_response.ok:
            raise HTTPException(status_code=500, detail="Failed to download file content")
        
        # Return file as streaming response
        return Response(
            content=content_response.content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={file_name}"}
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/sharepoint/upload")
async def upload_to_sharepoint_output(
    file: UploadFile = File(...),
    source_input_name: Optional[str] = Form(None),
):
    """Upload file to SharePoint output folder"""
    try:
        print(f"📤 Starting upload process for: {file.filename}")
        
        # Read file bytes
        file_bytes = await file.read()
        size_bytes = len(file_bytes)
        print(f"📤 File read successfully: {file.filename} ({size_bytes} bytes, {size_bytes/1024:.2f} KB)")
        
        # Import SharePoint functions with proper aliases
        from api.sharepoint import (
            get_app_token as sp_get_app_token,
            resolve_site_and_drive as sp_resolve_site_and_drive,
            upload_file_to_folder as sp_upload_file_to_folder,
            try_get_item_by_path as sp_try_get_item_by_path,
        )
        
        print("🔄 Getting SharePoint token...")
        token = sp_get_app_token()
        print("✅ Token obtained")
        
        print("🔄 Resolving site and drive...")
        _, drive = sp_resolve_site_and_drive(token)
        drive_id = drive["id"]
        print(f"✅ Drive resolved: {drive_id}")

        # Enhanced de-duplication based on source input filename
        if source_input_name and _is_file_recently_processed(source_input_name):
            print(f"⏭️ Skipping upload - input file '{source_input_name}' already processed recently")
            return JSONResponse(content={
                "success": True,
                "message": f"Skip upload: input file '{source_input_name}' was already processed recently.",
                "skipped": True,
                "reason": "already_processed_input",
                "source_input_name": source_input_name,
            })

        # Fast idempotency check: direct item-by-path lookup
        print("🔄 Checking if file already exists...")
        try:
            out_path = f"{SHAREPOINT_OUTPUT_FOLDER.rstrip('/')}/{(file.filename or '').strip()}"
            print(f"🔎 Checking path: {out_path}")
            
            existing_item = sp_try_get_item_by_path(token, drive_id, out_path)
            if existing_item:
                print(f"⏭️ File already exists in SharePoint: {file.filename}")
                return JSONResponse(content={
                    "success": True,
                    "message": "File already exists in SharePoint output folder. Skipping upload.",
                    "folder": SHAREPOINT_OUTPUT_FOLDER,
                    "filename": file.filename,
                    "already_exists": True,
                })
        except Exception as e:
            print(f"⚠️ Error checking existing file (non-fatal): {e}")
            # Continue with upload

        print("🔄 Starting SharePoint upload...")
        
        # Add timeout and error handling to the upload
        try:
            uploaded_item = sp_upload_file_to_folder(
                token=token,
                drive_id=drive_id,
                folder_path=SHAREPOINT_OUTPUT_FOLDER,
                file_name=file.filename,
                file_bytes=file_bytes,
            )
            print("✅ Upload completed successfully")
        except Exception as upload_error:
            print(f"❌ Upload failed: {upload_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"SharePoint upload failed: {str(upload_error)}"
            )

        # Mark the source input file as processed for future de-duplication
        if source_input_name:
            print(f"✅ Marking input file as processed: {source_input_name}")
            _mark_file_as_processed(source_input_name)
            
        print(f"✅ Upload process completed for: {file.filename}")
        return JSONResponse(content={
            "success": True,
            "message": "File uploaded to SharePoint output folder successfully",
            "folder": SHAREPOINT_OUTPUT_FOLDER,
            "filename": file.filename,
            "item": uploaded_item,
            "source_input_name": source_input_name,
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        print(f"❌ Unexpected error in upload endpoint: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(exc)}")
    """Upload file to SharePoint output folder"""
    try:
        file_bytes = await file.read()
        token = sp_get_app_token()
        _, drive = sp_resolve_site_and_drive(token)
        drive_id = drive["id"]
        
        size_bytes = len(file_bytes)
        print(
            f"📤 Uploading file to SharePoint output: {file.filename} "
            f"(size: {size_bytes} bytes, {size_bytes/1024:.2f} KB, {size_bytes/1024/1024:.2f} MB)"
        )

        # Enhanced de-duplication based on source input filename (in-memory with timestamps)
        if source_input_name and _is_file_recently_processed(source_input_name):
            return JSONResponse(content={
                "success": True,
                "message": f"Skip upload: input file '{source_input_name}' was already processed recently.",
                "skipped": True,
                "reason": "already_processed_input",
                "source_input_name": source_input_name,
            })

        # Fast idempotency check: direct item-by-path lookup
        try:
            # Build the output path under the drive root
            out_path = f"{SHAREPOINT_OUTPUT_FOLDER.rstrip('/')}/{(file.filename or '').strip()}"
            from api.sharepoint import try_get_item_by_path as sp_try_get_item_by_path
            if sp_try_get_item_by_path(token, drive_id, out_path):
                return JSONResponse(content={
                    "success": True,
                    "message": "File already exists in SharePoint output folder. Skipping upload.",
                    "folder": SHAREPOINT_OUTPUT_FOLDER,
                    "filename": file.filename,
                    "already_exists": True,
                })
        except Exception:
            # Non-fatal; continue to attempt upload
            pass
        uploaded_item = sp_upload_file_to_folder(
            token=token,
            drive_id=drive_id,
            folder_path=SHAREPOINT_OUTPUT_FOLDER,
            file_name=file.filename,
            file_bytes=file_bytes,
        )

        # Mark the source input file as processed for future de-duplication
        if source_input_name:
            _mark_file_as_processed(source_input_name)
        return JSONResponse(content={
            "success": True,
            "message": "File uploaded to SharePoint output folder successfully",
            "folder": SHAREPOINT_OUTPUT_FOLDER,
            "filename": file.filename,
            "item": uploaded_item,
            "source_input_name": source_input_name,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


 


@app.post("/api/tabledata")
async def read_data(tablename: str = Form(...)):
    """
    Retrieve table data with proper error handling and connection management.
    """
    df = None
    try:
        # Validate table name
        if not tablename or not tablename.strip():
            return JSONResponse(
                content={"detail": "Table name is required and cannot be empty"}, 
                status_code=400
            )
        
        print(f"Processing table: {tablename}")
        df = db.get_table_data(tablename)
        
        if df is None or df.empty:
            return JSONResponse(
                content={"detail": "Table is empty or not found"}, 
                status_code=404
            )
        
        print(f"Retrieved {len(df)} rows from table '{tablename}'")
        
        # Safe JSON conversion with proper error handling
        try:
            json_str = df.to_json(orient='records', date_format='iso', default_handler=str)
            result = json.loads(json_str)
        except Exception as json_err:
            print(f"Error converting table data to JSON: {json_err}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error converting table data to JSON format: {str(json_err)}"
            )
        
        print(f"Data for table '{tablename}' processed successfully.")
        
        # Prepare ChromaDB embeddings for text-style docs (non-blocking, best effort)
        try:
            if 'text_content' in df.columns:
                texts = [str(t).strip() for t in df['text_content'].dropna().astype(str).tolist() if str(t).strip()]
                if texts:
                    try:
                        embeddings = OpenAIEmbeddings()
                        persist_dir = str(Path(__file__).resolve().parent / "chroma_store")
                        os.makedirs(persist_dir, exist_ok=True)
                        # Try to tag by email if available from table meta
                        try:
                            meta = db.get_tables_info(tablename)
                            email = (meta[0].get("email") if isinstance(meta, list) and meta else None) or "unknown"
                        except Exception:
                            email = "unknown"
                        safe_email = "".join(ch if ch.isalnum() else "_" for ch in email)
                        safe_name = "".join(ch if ch.isalnum() else "_" for ch in tablename)
                        collection_name = f"{safe_email}__{safe_name}"
                        vectordb = Chroma(collection_name=collection_name, persist_directory=persist_dir, embedding_function=embeddings)
                        ids = [f"{collection_name}_{i}" for i in range(len(texts))]
                        metadatas = [{"email": email, "name": tablename}] * len(texts)
                        vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                        print(f"Successfully embedded {len(texts)} text documents")
                    except Exception as ve:
                        print(f"[WARNING] Chroma ingestion skipped: {ve}")
        except Exception as ve_outer:
            print(f"[WARNING] Vector prep failed for '{tablename}': {ve_outer}")
        
        # Save CSV files (non-blocking, best effort)
        try:
            df.to_csv('data.csv', index=False)
            os.makedirs("uploads", exist_ok=True)
            df.to_csv(os.path.join("uploads", f"{tablename.lower()}.csv"), index=False)
        except Exception as csv_err:
            print(f"[WARNING] CSV export failed: {csv_err}")
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error processing table '{tablename}': {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={
                "detail": "An error occurred while processing the table data. Please try again or contact support if the issue persists.",
                "error": str(e),
                "table": tablename
            }, 
            status_code=500
        )
    finally:
        # Ensure any open resources are cleaned up
        del df


# 3.Deleting the user-specific list of tables-----------------Deleting the list of tables corresponding to the specific user
@app.post("/api/delete_selected_tables")
async def delete_selected_tables_by_name(
        email: str = Form(...),
        table_names: List[str] = Form(...)
):
    try:
        print("Parsing request body for table deletion.")  # Debug statement
        print(f"Received email: {email}, table names: {table_names}")  # Debug statement

        if not email or not table_names:
            print("Missing 'email' or 'table_names' in the request.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'email' and 'table_names' are required"
            )

        print(f"Calling delete_selected_user_tables_by_name with email: {email} and table_names: {table_names}")
        deletion_status = db.delete_tables_data(email, table_names)

        if deletion_status:
            print(f"Deleted {len(table_names)} table(s) for email '{email}'.")
            return JSONResponse(
                content={
                    "message": f"{len(table_names)} table(s) associated with email '{email}' have been deleted."
                },
                status_code=status.HTTP_200_OK
            )
        else:
            print(f"No matching tables found for email '{email}' or the provided table names: {table_names}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No matching tables found for email '{email}' or the provided table names."
            )

    except Exception as e:
        print(f"Exception occurred while deleting selected tables: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )

# 5.Dashboard apis--------------------Dashboard related api for generating the dynamic graphs---------- 5. Have to modify
# Dashboard configuration
CHARTS_DIR = "generated_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Fixed filenames for the charts
FIXED_CHART_FILENAMES = [
    "chart_1.json",
    "chart_2.json",
    "chart_3.json",
    "chart_4.json"
]


@app.post("/api/dashboard")
async def gen_plotly_response(
    count: int = Query(default=4, ge=1, le=10, description="Number of charts to generate"),
    type: str = Query(default="basic", regex="^(basic|advanced)$", description="Chart type: basic or advanced")
) -> JSONResponse:
    try:
        # Load and process data
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)

        # Clean column names
        df.columns = df.columns.str.strip()

        num_plots = count
        chart_type = type.lower()
        file_path = csv_file_path
        
        # Use more data for better insights, but limit to prevent token overflow
        sample_size = min(100, len(df))
        sample_data = df.head(sample_size).to_string()
        data_types_info = df.dtypes.to_string()
        
        # Get basic data statistics for better chart generation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Detect potential date columns
        potential_date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains date-like strings
                sample_vals = df[col].dropna().head(5).astype(str).tolist()
                for val in sample_vals:
                    if any(char in val for char in ['-', '/', ' ']) and any(char.isdigit() for char in val):
                        potential_date_cols.append(col)
                        break
        
        data_shape = f"Rows: {len(df)}, Columns: {len(df.columns)}"
        
        chart_responses = []

        # Create dynamic chart filenames based on count
        chart_filenames = [f"chart_{i+1}.json" for i in range(max(1, num_plots))]
        
        # Initialize all chart files as empty
        for filename in chart_filenames:
            chart_path = os.path.join(CHARTS_DIR, filename)
            with open(chart_path, "w") as f:
                json.dump({}, f)

        # Create optimized prompts based on chart type
        if chart_type == "basic":
            chart_types_instruction = """
            Generate ONLY basic, easy-to-understand charts:
            - Bar charts for categorical data comparisons
            - Line charts for trends over time 
            - Pie charts for proportion/percentage data
            - Scatter plots for simple correlations
            - Simple histograms for distributions
            Focus on clarity and simplicity. Avoid complex features like faceting, animations, or 3D plots.
            """
        else:  # advanced
            chart_types_instruction = """
            Generate advanced, complex charts for deeper insights:
            - Box plots and violin plots for distribution analysis
            - Heatmaps for correlation matrices
            - 3D scatter plots for multi-dimensional analysis
            - Faceted plots for comparative analysis
            - Animated charts for time-series trends
            - Sunburst or treemap for hierarchical data
            - Advanced statistical plots (regression lines, confidence intervals)
            Use sophisticated Plotly features and statistical analysis.
            """

        prompt_eng = f"""
You are a data visualization expert. Generate {num_plots} {chart_type} charts using the provided dataset.

Dataset Info:
- Shape: {data_shape}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Potential date columns: {potential_date_cols}

{chart_types_instruction}

Requirements:
1. Return ONLY valid Python code (no markdown, no explanations outside comments)
2. Initialize: chart_dict = {{}}
3. For each chart:
   - Use try-except blocks: except Exception as e: (capital E)
   - Store as: chart_dict["Chart_Title"] = {{"plot_data": fig.to_json(), "description": "insight"}}
   - Use exact column names from dataset
   - Add meaningful titles and insights as comments

4. Use the FULL dataset (df) for analysis, not just samples
5. Apply data cleaning: df.columns = df.columns.str.strip()
6. Use aggregations (.groupby(), .sum(), .count()) when appropriate
7. Filter data intelligently (top N, remove outliers, date ranges)

DATE FORMATTING REQUIREMENTS:
8. For datetime columns on x-axis (especially: {potential_date_cols}):
   - Convert to datetime: pd.to_datetime(df[col], errors='coerce')
   - Use proper date formatting in Plotly:
     * For daily data: fig.update_xaxes(tickformat='%Y-%m-%d')
     * For monthly data: fig.update_xaxes(tickformat='%b %Y')
     * For yearly data: fig.update_xaxes(tickformat='%Y')
   - Set readable tick angles: fig.update_xaxes(tickangle=45)
   - For time series, use: fig.update_xaxes(type='date')

9. Example date formatting code:
   ```
   fig.update_xaxes(
       tickformat='%Y-%m-%d',
       tickangle=45,
       type='date'
   )
   ```

Data preview:
{sample_data}

Column types:
{data_types_info}

Generate diverse, insightful {chart_type} charts that reveal different data patterns.
IMPORTANT: Always format date axes properly for readability!
"""

        try:
            # Generate code using AI
            generated_code = generate_code4(prompt_eng)
            print(f"Generated code:\n{generated_code}")

            if 'import' not in generated_code.lower():
                raise ValueError("Invalid AI response - missing imports")

            # Execute the generated code
            namespace = {'pd': pd, 'px': px, 'go': go, 'df': df}
            exec(generated_code, namespace)

            # Get the chart dictionary from executed code
            chart_dict = namespace.get("chart_dict", {})

            if not chart_dict or not isinstance(chart_dict, dict):
                raise ValueError("No charts generated - chart_dict is empty or invalid")

            # Process each generated chart with improved error handling
            chart_keys = list(chart_dict.keys())[:num_plots] if isinstance(chart_dict, dict) else []
            failed_charts = []

            for i, chart_key in enumerate(chart_keys):
                try:
                    chart_info = chart_dict[chart_key]
                    chart_data = chart_info.get("plot_data")
                    description = chart_info.get("description", "")

                    if not chart_data:
                        raise ValueError(f"No plot data for chart: {chart_key}")

                    # Make data serializable
                    chart_data_serializable = make_serializable(chart_data)
                    chart_filename = chart_filenames[i]
                    chart_path = os.path.join(CHARTS_DIR, chart_filename)

                    # Save individual chart file
                    with open(chart_path, "w", encoding="utf-8") as f:
                        json.dump(chart_data_serializable, f, indent=2, ensure_ascii=False)

                    chart_responses.append({
                        "timestamp": datetime.now().isoformat(),
                        "chart_title": chart_key,
                        "chart_data": chart_data_serializable,
                        "chart_file": chart_filename,
                        "description": description,
                        "status": "success"
                    })

                except Exception as e:
                    print(f"Error processing chart '{chart_key}': {str(e)}")
                    chart_filename = chart_filenames[i] if i < len(chart_filenames) else f"chart_{i + 1}.json"
                    failed_charts.append(i)
                    chart_responses.append({
                        "chart_file": chart_filename,
                        "chart_title": chart_key if 'chart_key' in locals() else f"Chart {i + 1}",
                        "status": "failed",
                        "error": str(e)
                    })

            # Attempt to regenerate failed charts individually (if any)
            if failed_charts and isinstance(failed_charts, list) and len(failed_charts) < num_plots:
                print(f"Attempting to regenerate {len(failed_charts)} failed charts...")
                for failed_index in failed_charts:
                    try:
                        # Create a simpler prompt for individual chart regeneration
                        simple_prompt = f"""
Generate 1 simple {chart_type} chart for this dataset. Return only Python code.

chart_dict = {{}}
# Generate exactly 1 chart and store as:
# chart_dict["Chart_Name"] = {{"plot_data": fig.to_json(), "description": "insight"}}

Dataset info:
{sample_data[:500]}...

Use basic chart types: bar, line, pie, or scatter plot.

IMPORTANT: For any date columns on x-axis, format them properly:
- Convert to datetime: pd.to_datetime(df[col], errors='coerce')
- Add formatting: fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=45, type='date')
"""
                        regenerated_code = generate_code4(simple_prompt)
                        namespace = {'pd': pd, 'px': px, 'go': go, 'df': df}
                        exec(regenerated_code, namespace)
                        
                        regen_chart_dict = namespace.get("chart_dict", {})
                        if regen_chart_dict:
                            chart_key = list(regen_chart_dict.keys())[0]
                            chart_info = regen_chart_dict[chart_key]
                            chart_data = chart_info.get("plot_data")
                            
                            if chart_data:
                                chart_data_serializable = make_serializable(chart_data)
                                chart_filename = chart_filenames[failed_index]
                                chart_path = os.path.join(CHARTS_DIR, chart_filename)
                                
                                with open(chart_path, "w", encoding="utf-8") as f:
                                    json.dump(chart_data_serializable, f, indent=2, ensure_ascii=False)
                                
                                # Update the failed chart response
                                chart_responses[failed_index] = {
                                    "timestamp": datetime.now().isoformat(),
                                    "chart_title": chart_key,
                                    "chart_data": chart_data_serializable,
                                    "chart_file": chart_filename,
                                    "description": chart_info.get("description", ""),
                                    "status": "success"
                                }
                                print(f"Successfully regenerated chart {failed_index + 1}")
                    except Exception as e:
                        print(f"Failed to regenerate chart {failed_index + 1}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error in chart generation: {str(e)}")
            # Create fallback empty responses
            for i in range(max(1, num_plots)):
                chart_filename = chart_filenames[i] if i < len(chart_filenames) else f"chart_{i+1}.json"
                chart_responses.append({
                    "chart_file": chart_filename,
                    "status": "failed",
                    "error": str(e)
                })

        # Prepare final response
        success_count = len([c for c in chart_responses if c.get("status") == "success"]) if chart_responses else 0
        response_data = {
            "message": f"Chart generation completed - {chart_type} charts",
            "generated_charts": success_count,
            "total_charts": num_plots,
            "chart_type": chart_type,
            "chart_files": chart_filenames if isinstance(chart_filenames, list) else [],
            "charts": chart_responses if isinstance(chart_responses, list) else []
        }

        return JSONResponse(content=response_data, status_code=200)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Data file not found"
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chart generation failed: {str(e)}"
        )


# Function to generate code from OpenAI API
def generate_code4(prompt_eng):
    """Generate Python code for creating Plotly charts using AI"""
    _client = get_openai_client(None)
    _model = get_model_name(None)
    response = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": """
            You are VizCopilot, an expert Python data visualization assistant having 20+ years of experience in specialising in Plotly.

            Your core responsibilities:
            - Generate complete, executable Python code for data visualization
            - Create diverse, insightful charts that reveal different data patterns
            - Use both plotly.express (px) and plotly.graph_objects (go) appropriately
            - Apply data analysis techniques: grouping, filtering, aggregation, transformation

            Code Generation Standards:
            - Always start with necessary imports: pandas, plotly.express, plotly.graph_objects
            - Generate ONLY valid Python code (no markdown, no text outside comments)
            - Use proper exception handling: 'except Exception as e:' (capital E)
            - Create complete, working code blocks with proper indentation
            - Include meaningful chart titles and descriptions
            - Apply best practices for data visualization

            Chart Diversity Requirements:
            - Create different chart types for comprehensive data exploration
            - Use various Plotly features: faceting, animations, multi-series, custom styling
            - Focus on actionable insights: trends, outliers, distributions, correlations
            - Apply appropriate data transformations and filtering

            Technical Requirements:
            - Return charts in a dictionary format: chart_dict[title] = {"plot_data": fig.to_plotly_json(), "description": "insight"}
            - Handle edge cases and data quality issues
            - Use exact column names from provided dataset
            - Ensure all generated code is immediately executable
            - Validate data types and handle datetime conversions properly

            Quality Assurance:
            - Every chart must provide unique insights
            - Code must be syntactically correct and complete
            - No placeholder functions or incomplete logic
            - Proper error handling for robustness
                """},
            {"role": "user", "content": prompt_eng}
        ],
        temperature=0.7,  # Add some randomness for variety in chart generation
        max_tokens=4000
    )

    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message

    print(f"AI Response: {all_text}")

    # Extract Python code from response
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        if code_end == -1:
            code_end = len(all_text)
        code = all_text[code_start:code_end].strip()
    elif "```" in all_text:
        code_start = all_text.find("```") + 3
        code_end = all_text.find("```", code_start)
        if code_end == -1:
            code_end = len(all_text)
        code = all_text[code_start:code_end].strip()
    else:
        code = all_text.strip()

    return code


def make_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable format"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    elif str(type(obj)).startswith("<class 'numpy."):
        return float(obj) if 'float' in str(type(obj)) else int(obj)
    else:
        return obj


# 6..Get summary api-------------getting the summary of the above generated graphs-----------6-----pending.
# In-memory cache for summaries
SUMMARY_CACHE: Dict[str, str] = {}

# In-memory cache for AutoML prediction insights (fast path)
PREDICTION_INSIGHTS_CACHE: Dict[str, Dict[str, Any]] = {}


@app.post("/api/analyze_chart")
async def analyze_chart(
        chart_id: str = Form(...),
        question: Optional[str] = Form(None),
        email: Optional[str] = Form(None)
) -> JSONResponse:
    try:
        # Check if this is a request to generate a new chart
        is_generation_request = False
        if question:
            keywords = ["chart", "graph", "plot", "show me", "visualize"]
            if any(k in question.lower() for k in keywords):
                is_generation_request = True

        # --- CASE 0: NEW CHART GENERATION REQUEST ---
        if is_generation_request:
            print(f"Generating new chart for query: {question}")
            
            # Load the dataset
            csv_file_path = 'data.csv'
            if not os.path.exists(csv_file_path):
                 raise HTTPException(status_code=404, detail="Data file not found")
            
            df = pd.read_csv(csv_file_path)
            
            # Get sample data for prompt
            sample_data = df.head(5).to_string()
            data_types_info = df.dtypes.to_string()
            
            prompt_eng = f"""
            You are a data visualization expert. Generate 1 new chart based on this user request: "{question}"
            
            Dataset Info:
            - Shape: {df.shape}
            - Columns: {list(df.columns)}
            
            Requirements:
            1. Return ONLY valid Python code (no markdown)
            2. Initialize: chart_dict = {{}}
            3. Generate exactly 1 chart and store as:
               chart_dict["Generated Chart"] = {{"plot_data": fig.to_json(), "description": "Analysis of request"}}
            4. Use the FULL dataset (df)
            5. Clean columns: df.columns = df.columns.str.strip()
            6. STRICTLY follow the user's requested chart type (e.g., if "box plot" is requested, generate a box plot).
            7. If no specific chart type is requested, choose the best visualization for the data.
            
            DATE FORMATTING:
            - Convert dates: pd.to_datetime(df[col], errors='coerce')
            - Format axes: fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=45)
            
            Data preview:
            {sample_data}
            """
            
            try:
                # Generate and execute code
                generated_code = generate_code4(prompt_eng)
                namespace = {'pd': pd, 'px': px, 'go': go, 'df': df}
                exec(generated_code, namespace)
                
                chart_dict = namespace.get("chart_dict", {})
                if not chart_dict:
                    raise ValueError("No chart generated")
                
                chart_key = list(chart_dict.keys())[0]
                chart_info = chart_dict[chart_key]
                chart_data = chart_info.get("plot_data")
                
                if chart_data:
                    # Make serializable
                    chart_data_serializable = make_serializable(chart_data)
                    
                    # Return only the chart data without text explanation as requested
                    return JSONResponse(
                        content={
                            "chart_id": "custom",
                            "question": question,
                            "response": "", # Empty response as user requested only graph
                            "plot_data": chart_data_serializable, # Frontend will look for this
                            "type": "answer"
                        },
                        status_code=200
                    )
            except Exception as e:
                print(f"Chart generation failed: {e}")
                # Fallback to normal text answer if chart generation fails
                pass 

        # 1. Validate Chart ID (only if not a custom generation request or if fell through)
        chart_num = 0
        try:
            chart_num = int(chart_id)
        except ValueError:
            pass # might be a string id or invalid, but we'll check validity below if needed

        # 2. Locate and load chart data from file
        chart_json = {}
        # Only try to load if it looks like a valid existing chart ID (1-6)
        if 1 <= chart_num <= 6:
            filename = f"chart_{chart_id}.json"
            chart_path = os.path.join(CHARTS_DIR, filename)

            if os.path.exists(chart_path):
                with open(chart_path, "r", encoding="utf-8") as f:
                    chart_json = json.load(f)
            elif not is_generation_request:
                 # If specifically asking about a chart that doesn't exist and not generating queries
                 raise HTTPException(status_code=404, detail=f"Chart file '{filename}' not found.")

        # 3. Determine action: Summarize or Answer Question
        # --- CASE 1: No question provided -> Generate a detailed summary ---
        if not question or not question.strip():
            # Check cache for existing summary
            if chart_id in SUMMARY_CACHE:
                return JSONResponse(
                    content={
                        "chart_id": chart_id,
                        "response": SUMMARY_CACHE[chart_id],
                        "type": "summary",
                        "cached": True,
                    },
                    status_code=200,
                )

            # Generate and cache a new summary
        if question == "summary":
            prompt = (
                f"You are a data analyst AI. A user selected a chart represented by this Plotly JSON:\n{json.dumps(chart_json)}\n\n"
                f"Analyze and summarize only the insights, patterns, and trends that are directly visible in the chart.\n\n"
                f"Follow this EXACT output structure with exactly 2 bullet points for each heading:\n\n"
                f"Core Insight\n"
                f"• [First key insight from the chart]\n"
                f"• [Second key insight from the chart]\n\n"
                f"Pattern Analysis\n"
                f"• [First pattern or trend observation]\n"
                f"• [Second pattern or trend observation]\n\n"
                f"Business Context\n"
                f"• [First business implication]\n"
                f"• [Second business implication]\n\n"
                f"Recommendations\n"
                f"• [First recommendation based on the data]\n"
                f"• [Second recommendation based on the data]\n\n"
                f"Actions\n"
                f"• [First specific action to take]\n"
                f"• [Second specific action to take]\n\n"
                f"IMPORTANT RULES:\n"
                f"- Use EXACTLY 2 bullet points per section, no more, no less\n"
                f"- Only describe what you directly observe in the chart data\n"
                f"- Do not invent data or make unsupported claims\n"
                f"- Keep bullet points concise but informative\n"
                f"- Format section headings as plain text (not markdown headings)\n"
            )
            summary = generate_text(prompt, email)
            # Convert summary to HTML format
            summary_html = markdown_to_html(summary)
            SUMMARY_CACHE[chart_id] = summary_html

            return JSONResponse(
                content={
                    "chart_id": chart_id,
                    "response": summary_html,
                    "type": "summary",
                    "cached": False,
                },
                status_code=200,
            )

        # --- CASE 2: Question provided -> Generate a targeted answer ---
        else:
            prompt = (
                f"You are a data analyst AI. A user is asking a question about a chart represented by this Plotly JSON:\n{json.dumps(chart_json)}\n\n"
                f"User's Question: {question}\n\n"
                f"Analyze the chart and provide a clear, concise answer to the user's specific question. "
                f"Base your answer only on what is visible in the chart data. Do not invent or assume data.\n\n"
                f"RESPONSE FORMAT:\n"
                f"- Provide a direct answer in 2-4 paragraphs\n"
                f"- Use simple, clear language\n"
                f"- Include specific data points or observations from the chart when relevant\n"
                f"- If the question cannot be answered from the chart data, politely explain why\n"
                f"- If the question is casual (like 'hi' or 'hello'), politely explain that you're here to help analyze the chart data\n\n"
                f"Keep the response concise and focused on answering the specific question asked."
            )
            answer = generate_text(prompt, email)
            # Convert answer to HTML format
            answer_html = markdown_to_html(answer)

            return JSONResponse(
                content={
                    "chart_id": chart_id,
                    "question": question,
                    "response": answer_html,
                    "type": "answer"
                },
                status_code=200,
            )

    except HTTPException:
        raise  # Re-raise exceptions with specific HTTP status codes
    except Exception as e:
        # Catch-all for any other unexpected errors
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


def markdown_to_html(md_text):
    """Convert markdown text to HTML with enhanced formatting for summaries"""
    # Convert markdown to HTML
    html_text = markdown.markdown(md_text, extensions=['nl2br'])
    
    # Enhance section headings with better styling
    # Replace plain text headings with styled headings
    lines = md_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Check if line is a section heading (has no bullets and is followed by bullets)
        if stripped and not stripped.startswith('•') and not stripped.startswith('-'):
            # Check if this looks like a section heading
            if any(heading in stripped for heading in ['Core Insight', 'Pattern Analysis', 'Business Context', 'Recommendations', 'Actions', 'Key Points', 'Summary']):
                formatted_lines.append(f'<h3 style="color: #2c3e50; font-weight: 600; margin-top: 16px; margin-bottom: 8px; font-size: 16px;">{stripped}</h3>')
                continue
        
        # Convert bullet points
        if stripped.startswith('•') or stripped.startswith('-'):
            bullet_content = stripped[1:].strip()
            formatted_lines.append(f'<li style="margin-bottom: 6px; line-height: 1.6;">{bullet_content}</li>')
        elif stripped:
            formatted_lines.append(f'<p style="margin-bottom: 8px;">{stripped}</p>')
    
    # Wrap list items in ul tags
    html_result = []
    in_list = False
    
    for line in formatted_lines:
        if '<li' in line:
            if not in_list:
                html_result.append('<ul style="margin: 8px 0; padding-left: 24px;">')
                in_list = True
            html_result.append(line)
        else:
            if in_list:
                html_result.append('</ul>')
                in_list = False
            html_result.append(line)
    
    if in_list:
        html_result.append('</ul>')
    
    final_html = ''.join(html_result)
    
    # Wrap in a container div
    return f'<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; color: #333; line-height: 1.6;">{final_html}</div>'


def generate_text(prompt: str, email: Optional[str] = None) -> str:
    _client = get_openai_client(email)
    _model = get_model_name(email)
    response = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system",
             "content": "You are a helpful data analyst that explains data visualizations and user queries. Provide clear, accurate analysis based on the data provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    # Track AI credits usage
    record_llm_usage(email, response)
    return response.choices[0].message.content.strip()


def generate_prediction_insights_llm(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate UI-ready insights for prediction in a strict JSON format.
    Uses a small/fast model and caches results by a stable key.
    """
    cache_key = payload.get("cache_key")
    if cache_key and cache_key in PREDICTION_INSIGHTS_CACHE:
        return PREDICTION_INSIGHTS_CACHE[cache_key]

    # Keep prompt compact for speed.
    prompt = (
        "You are a reliability engineer. Using ONLY the provided dataset stats and inputs, "
        "produce a JSON object for a prediction summary UI.\n\n"
        "RULES:\n"
        "1) Output MUST be valid JSON only (no markdown, no commentary).\n"
        "2) Create EXACTLY 4 levels: Low, Medium, High, Critical.\n"
        "3) Each level has numeric min/max. Ranges must be contiguous and cover dataset min..max.\n"
        "4) Provide these arrays with EXACTLY 2 bullet strings each:\n"
        "   - input_analysis\n"
        "   - prediction_interpretation\n"
        "   - business_insights\n"
        "   - recommended_actions\n"
        "5) Determine predicted_level based on predicted_value and the level ranges.\n"
        "6) Do NOT invent sensors or facts; keep it generic if unsure.\n\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "OUTPUT_JSON_SCHEMA:\n"
        "{\n"
        '  "output_levels": {"Low":{"min":0,"max":0},"Medium":{"min":0,"max":0},"High":{"min":0,"max":0},"Critical":{"min":0,"max":0}},\n'
        '  "predicted_level": "Low|Medium|High|Critical|null",\n'
        '  "input_analysis": ["...","..."],\n'
        '  "prediction_interpretation": ["...","..."],\n'
        '  "business_insights": ["...","..."],\n'
        '  "recommended_actions": ["...","..."]\n'
        "}\n"
    )

    _client = get_openai_client(None)
    _model = get_model_name(None)
    resp = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    text = (resp.choices[0].message.content or "").strip()
    data = json.loads(text)

    if cache_key:
        PREDICTION_INSIGHTS_CACHE[cache_key] = data
    return data


# 7..Filling missing data------------------Evaluating the missed data in the dataframe----------- 7.
@app.post("/api/fill_missed_data")
async def missing_data() -> JSONResponse:
    try:
        # Resolve absolute paths relative to this file
        base_dir = Path(__file__).resolve().parent
        csv_file_path = base_dir / 'data.csv'

        # Load and validate data
        if not csv_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )

        df = pd.read_csv(str(csv_file_path))
        print("Original data preview:")
        print(df.head(5))

        # Process missing data
        new_df, html_df, summary = process_missing_data(df.copy())

        # Save processed data
        uploads_dir = base_dir / 'uploads'
        uploads_dir.mkdir(parents=True, exist_ok=True)
        processed_path = uploads_dir / 'processed_data.csv'
        new_df.to_csv(str(processed_path), index=False)
        new_df.to_csv(str(csv_file_path), index=False)

        # Save HTML representation
        mvt_json_path = base_dir / 'mvt_data.json'
        with open(str(mvt_json_path), 'w') as fp:
            json.dump({'data': html_df}, fp, indent=4)

        return JSONResponse(
            content={"df": html_df, "summary": summary},
            status_code=200
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Input CSV file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data processing failed: {str(e)}"
        )


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df, summary = handle_missing_data(df)
    return df, html_df, summary


def convert_to_datetime(df):
    """
    Converts object (string) columns containing dates to datetime format.
    """
    for col in df.columns:
        if df[col].dtype == "object":  # Process only string columns
            if df[col].str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", na=False).any():
                df[col] = df[col].apply(detect_and_parse_date)

    return df


import dateutil.parser


def detect_and_parse_date(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return pd.NaT  # Handle missing values safely

    try:
        # Check if it's a date with hyphens or slashes
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", value):
            day_first = False  # Assume MM-DD-YYYY first

            # Check for an ambiguous case (day > 12) → Must be DD-MM-YYYY
            parts = re.split(r"[-/]", value)
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            if day > 12:
                day_first = True  # Switch to DD-MM-YYYY

            # Parse with detected format
            return dateutil.parser.parse(value, dayfirst=day_first)

        # Otherwise, use default dateutil parsing
        return dateutil.parser.parse(value)

    except ValueError:
        return pd.NaT  # Return NaT if parsing fails


from sklearn.impute import KNNImputer


def handle_missing_data(df):
    try:
        ignore_types = ['object', 'string', 'timedelta', 'complex']
        ignored_columns_info = {}

        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # Be robust to different pandas datetime dtypes
        date_time_cols = df.select_dtypes(include=['datetime64[ns]', 'datetimetz', 'datetime']).columns
        ignored_cols = df.select_dtypes(include=ignore_types).columns
        int_like_cols = [col for col in numeric_cols if is_integer_like(df[col])]

        # Snapshot missing-value mask BEFORE any imputations/fills
        pre_missing_flags = df.isnull().copy()

        for col in ignored_cols:
            ignored_columns_info[col] = f"Ignored because of optional data"

        # Impute numeric columns (if any) and track which cells were imputed based on pre-missing
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            imputed_numeric = imputer.fit_transform(df[numeric_cols])
            imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols, index=df.index).round(2)

            for col in numeric_cols:
                if col in int_like_cols:
                    imputed_numeric_df[col] = imputed_numeric_df[col].round().astype("Int64")

            # Update DataFrame with imputed values
            df[numeric_cols] = imputed_numeric_df

        # Initialize imputed flags as the pre-operation missing mask for all columns
        imputed_flags = pre_missing_flags.copy()

        for col in df.select_dtypes(include='category').columns:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                # Mark where values were missing prior to fill
                prior_missing = pre_missing_flags[col]
                df[col].fillna(mode_val, inplace=True)
                imputed_flags.loc[prior_missing, col] = True
        for col in df.select_dtypes(include='bool').columns:
            if df[col].isnull().any():
                prior_missing = pre_missing_flags[col]
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
                imputed_flags.loc[prior_missing, col] = True

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            # If we cannot infer a reasonable average diff, just forward-fill
            if len(time_diffs) == 0 or pd.isna(time_diffs.mean()):
                prior_missing = pre_missing_flags[col]
                df[col] = df[col].ffill()
                imputed_flags.loc[prior_missing, col] = True
                continue

            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            prior_missing = pre_missing_flags[col]
            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col] = df[col].iloc[i - 1] + avg_diff
            # Mark originally missing positions as imputed
            imputed_flags.loc[prior_missing, col] = True

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []

        for idx, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                value = row[col]
                # Normalize to JSON-friendly value
                if isinstance(value, pd.Timestamp):
                    norm_value = value.strftime('%Y-%m-%d %H:%M:%S')
                elif pd.isna(value):
                    norm_value = None
                elif isinstance(value, np.generic):
                    norm_value = value.item()
                elif isinstance(value, (datetime, date, time)):
                    norm_value = value.isoformat()
                elif isinstance(value, Decimal):
                    norm_value = float(value)
                else:
                    norm_value = value

                row_data[col] = {
                    "value": norm_value,
                    "is_imputed": bool(imputed_flags.at[idx, col]) if col in imputed_flags else False
                }
            data.append(row_data)
        missing_values_summary = summarize_missing_values(imputed_flags)
        missing_values_summary["ignored_columns"] = ignored_columns_info
        return df, data, missing_values_summary
    except Exception as e:
        print(e)


def is_integer_like(series):
    return pd.api.types.is_numeric_dtype(series) and \
        series.dropna().apply(lambda x: float(x).is_integer()).all()


def summarize_missing_values(df):
    try:
        # 1. Total number of missing values
        total_missing = df.sum().sum()

        # 2. Columns with any missing values
        columns_with_missing = df.columns[df.any()].tolist()

        # 3. Count of missing values per column
        missing_count_per_column = df.sum()

        # 4. Percentage of missing values per column (optional)
        missing_percentage = df.mean() * 100

        # Final summary
        summary = {
            "total_missing_values": int(total_missing),
            "columns_with_missing": columns_with_missing,
            "missing_count_per_column": missing_count_per_column.to_dict(),
            "missing_percentage_per_column": missing_percentage.round(2).to_dict()
        }
        return summary
    except Exception as e:
        print(e)
        return {}


def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError("Type not serializable")


#Api for generating the column names for the model training
@app.get("/api/get_columns")
async def get_column_names() -> JSONResponse:
    try:
        # Read only the header to get column names
        df = pd.read_csv("data.csv")
        
        return JSONResponse(content={
            "status": "success",
            "columns": df.columns.tolist()
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="data.csv file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# 8.1 Models related apis:
@app.post("/api/models")
async def models(input: ModelRequest):
    try:
        # Load and clean data
        df = pd.read_csv('data.csv')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        print(numeric_cols)
        if len(numeric_cols) < 1:
            raise HTTPException(400, "Dataset does not meet modeling requirements")

        # Handle RandomForest
        if input.model == 'RandomForest':
            result = random_forest(df, input.col)
            # Handle both old and new return formats for backward compatibility
            if len(result) == 3:
                stat, cols, row_data = result
            elif len(result) == 2:
                stat, cols = result
                row_data = {}
            else:
                stat, cols, row_data = False, [], {}
            
            return {
                'columns': list(df.columns),
                'rf': True,
                'status': stat,
                'rf_cols': cols,
                'feature_columns': cols,
                'row_data': row_data if row_data else {}
            }

        # Handle ARIMA
        elif input.model == 'Arima':
            stat= arima_train_only(df, input.col)
            return {
                'columns': list(df.columns),
                'status': stat,
                'arima': True,
                'message': 'ARIMA model trained successfully.'
            }

        # Handle Supervised AutoML (best model selection across multiple estimators)
        elif input.model in ['AutoML', 'Supervised', 'supervised', 'automl']:
            model_dir = os.path.join("models", "supervised", input.col)
            res = train_supervised_automl(df=df, target_col=input.col, model_dir=model_dir)
            if not res.get("status"):
                raise HTTPException(status_code=400, detail=res.get("message", "AutoML training failed"))

            return {
                'columns': list(df.columns),
                'automl': True,
                'status': True,
                'task_type': res.get("task_type"),
                'best_model': res.get("best_model"),
                'metric_type': res.get("metric_type"),
                'metric': res.get("metric"),
                'feature_columns': res.get("feature_columns", []),
                'row_data': res.get("row_data", {}) or {},
                'skipped_models': res.get("skipped_models", []),
                'reused': bool(res.get("reused", False)),
            }

        # Handle unsupported models
        else:
            raise HTTPException(400, "Unsupported model type")

    except FileNotFoundError:
        raise HTTPException(404, "Data file not found")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/model_predict")
async def model_predict(request: Request):
    try:
        form_data = await request.form()
        form_data = dict(form_data)

        # Validate form_name for both RF and ARIMA
        form_name = form_data.get('form_name')
        if form_name not in ['rf', 'arima', 'supervised', 'automl']:
            raise HTTPException(
                status_code=400,
                detail="Invalid form type, expected 'rf', 'arima', or 'supervised'"
            )

        # Validate targetColumn
        targetcol = form_data.get('targetColumn')
        if not targetcol:
            raise HTTPException(
                status_code=400,
                detail="Target column (targetColumn) is required"
            )

        # Handle Random Forest Prediction
        if form_name == 'rf':
            return await handle_rf_prediction(form_data, targetcol)
        
        # Handle ARIMA Forecasting
        elif form_name == 'arima':
            return await handle_arima_forecast(form_data, targetcol)

        # Handle Supervised AutoML Prediction
        elif form_name in ['supervised', 'automl']:
            return await handle_supervised_prediction(form_data, targetcol)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
            headers={"X-Error-Details": str(e)}
        )


async def handle_rf_prediction(form_data, targetcol):
    """Handle Random Forest prediction logic"""
    # Extract features (exclude form_name and targetColumn)
    features = {k: v for k, v in form_data.items() if k not in ['form_name', 'targetColumn']}
    
    # Load model paths
    model_dir = os.path.join("models", "rf", targetcol)
    pipeline_path = os.path.join(model_dir, "pipeline.pkl")
    deployment_path = os.path.join(model_dir, "deployment.json")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    # Check if model files exist
    if not os.path.exists(pipeline_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model pipeline not found for target column '{targetcol}'"
        )

    if not os.path.exists(deployment_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model deployment metadata not found for target column '{targetcol}'"
        )

    # Load deployment metadata first to understand model type
    with open(deployment_path, "r") as f:
        deployment_data = json.load(f)
    
    model_stats = deployment_data.get("stats", {})
    feature_names = deployment_data.get('feature_names', [])
    is_classification = deployment_data.get('is_classification', False)
    model_type = deployment_data.get('model_type', 'Unknown')

    # Load model pipeline
    loaded_pipeline = joblib.load(pipeline_path)

    # Load label encoder if it exists (for categorical targets)
    label_encoder = None
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)

    # Prepare input data with proper feature ordering
    df_predict = pd.DataFrame([features])
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(df_predict.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {list(missing_features)}"
        )
    
    # Reorder columns to match training data
    df_predict = df_predict[feature_names]

    # Convert numeric strings to appropriate types
    for col in df_predict.columns:
        if col in deployment_data.get('numerical_features', []):
            try:
                df_predict[col] = pd.to_numeric(df_predict[col])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric value for feature '{col}': {df_predict[col].iloc[0]}"
                )

    # Make prediction
    predictions = loaded_pipeline.predict(df_predict)
    predicted_value = predictions[0]

    # Handle prediction probabilities and confidence for classification
    prediction_proba = None
    confidence = None
    predicted_class_proba = None
    
    if is_classification:
        # Get prediction probabilities
        if hasattr(loaded_pipeline, 'predict_proba'):
            try:
                proba = loaded_pipeline.predict_proba(df_predict)[0]
                classes = loaded_pipeline.classes_
                prediction_proba = dict(zip(classes, proba))
                confidence = float(max(proba))
                
                # Get probability for the predicted class
                predicted_class_proba = float(proba[list(classes).index(predicted_value)])
                
            except Exception as e:
                print(f"Warning: predict_proba failed: {e}")

        # Decode prediction if label encoder was used
        if label_encoder is not None:
            predicted_value = label_encoder.inverse_transform([predicted_value])[0]
            
            # Also decode probability classes
            if prediction_proba:
                decoded_proba = {}
                for encoded_class, prob in prediction_proba.items():
                    decoded_class = label_encoder.inverse_transform([encoded_class])[0]
                    decoded_proba[str(decoded_class)] = float(prob)
                prediction_proba = decoded_proba

    # Compute feature importance
    feature_importance = get_feature_importance(loaded_pipeline, feature_names)
    
    # Calculate feature impact
    feature_impact = calculate_feature_impact(loaded_pipeline, df_predict, features, is_classification, label_encoder)

    # Build dataset stats for LLM insights (same as AutoML)
    dataset_stats = {}
    try:
        df_all = pd.read_csv('data.csv')
        if targetcol in df_all.columns:
            y = pd.to_numeric(df_all[targetcol], errors="coerce").dropna()
            if not y.empty:
                q = y.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
                dataset_stats = {
                    "min": float(q.get(0.0)),
                    "q25": float(q.get(0.25)),
                    "median": float(q.get(0.5)),
                    "q75": float(q.get(0.75)),
                    "max": float(q.get(1.0)),
                    "count": int(len(y)),
                }
    except Exception:
        dataset_stats = {}

    # Determine task type
    task_type = "classification" if is_classification else "regression"
    
    # Cache key for insights
    cache_key = f"{targetcol}|{task_type}|{json.dumps(dataset_stats, sort_keys=True)}|{str(predicted_value)[:32]}"

    # Generate LLM insights (same as AutoML)
    try:
        insights = generate_prediction_insights_llm({
            "cache_key": cache_key,
            "target_column": targetcol,
            "task_type": task_type,
            "predicted_value": predicted_value,
            "dataset_stats": dataset_stats,
            "input_sample": {k: features.get(k) for k in list(features.keys())[:10]},
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM insights generation failed: {str(e)}")

    # Convert predicted_value to native Python type for JSON serialization
    if isinstance(predicted_value, (np.integer, np.floating)):
        predicted_value = predicted_value.item()
    elif isinstance(predicted_value, np.ndarray):
        predicted_value = predicted_value.tolist()
    
    # Prepare response in same format as AutoML
    prediction_result = {
        "predicted_value": float(predicted_value) if not is_classification else str(predicted_value),
        "target_column": targetcol,
        "model_type": "RandomForest",
        "task_type": task_type,
        "predicted_level": insights.get("predicted_level"),
    }

    # Get metrics from model stats
    metrics = model_stats.get("metrics", {})
    metric_type = "RMSE" if not is_classification else "Accuracy"
    metric_val = metrics.get('rmse') if not is_classification else metrics.get('accuracy')
    
    # Convert metric_val to native Python type
    if metric_val is not None and isinstance(metric_val, (np.integer, np.floating)):
        metric_val = float(metric_val)

    response = {
        "prediction_result": prediction_result,
        "best_model": "RandomForest",
        "metric_type": metric_type,
        "metric": metric_val,
        "output_levels": insights["output_levels"],
        "predicted_level": insights["predicted_level"],
        "input_analysis": insights["input_analysis"],
        "prediction_interpretation": insights["prediction_interpretation"],
        "business_insights": insights["business_insights"],
        "recommended_actions": insights["recommended_actions"],
    }

    return JSONResponse(content=response)


def train_single_model(df, target_col, model_type):
    """Train a specific model (XGBoost, LightGBM, GradientBoosting) similar to RandomForest"""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError:
        LGBMClassifier = LGBMRegressor = None
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Determine if classification or regression
        is_classification = False
        unique_values = y.nunique()
        if y.dtype == 'object' or (y.dtype in ['int64', 'float64'] and unique_values <= 20):
            is_classification = True
            print(f"[INFO] Detected classification task ({unique_values} unique classes)")
        else:
            print(f"[INFO] Detected regression task")
        
        # Encode target if classification
        label_encoder = None
        if is_classification and y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Select model based on type
        if model_type == 'XGBoost':
            if is_classification:
                model = XGBRegressor(n_estimators=100, random_state=42, eval_metric='logloss')
                # For classification, use XGBClassifier if available
                try:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                except:
                    pass
            else:
                model = XGBRegressor(n_estimators=100, random_state=42)
        elif model_type == 'LightGBM':
            if LGBMClassifier is None or LGBMRegressor is None:
                raise HTTPException(400, "LightGBM not installed. Please install it with: pip install lightgbm")
            if is_classification:
                model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            else:
                model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        elif model_type == 'GradientBoosting':
            if is_classification:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise HTTPException(400, f"Unsupported model type: {model_type}")
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print(f"[INFO] Training {model_type} model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2)
            }
            model_type_str = "Classification"
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'r2_score': round(r2 * 100, 2),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2)
            }
            model_type_str = "Regression"
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        
        # Calculate baseline
        if is_classification:
            baseline_accuracy = max(y_train.value_counts()) / len(y_train)
            baseline_comparison = f"{round((accuracy / baseline_accuracy - 1) * 100, 1)}% better than baseline"
        else:
            baseline_mae = np.mean(np.abs(y_train - np.mean(y_train)))
            baseline_comparison = f"{round((1 - mae / baseline_mae) * 100, 1)}% better than mean baseline"
        
        # Save model
        model_dir = os.path.join("models", model_type.lower(), target_col)
        os.makedirs(model_dir, exist_ok=True)
        
        pipeline_path = os.path.join(model_dir, "pipeline.pkl")
        joblib.dump(pipeline, pipeline_path)
        
        # Save label encoder if used
        if label_encoder:
            label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            joblib.dump(label_encoder, label_encoder_path)
        
        # Get feature names after preprocessing
        feature_names = numerical_features + categorical_features
        
        # Save deployment metadata
        deployment_data = {
            'model_type': model_type_str,
            'is_classification': is_classification,
            'feature_names': feature_names,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'target_column': target_col
        }
        
        deployment_path = os.path.join(model_dir, "deployment.json")
        with open(deployment_path, "w") as f:
            json.dump(deployment_data, f, indent=4)
        
        # Prepare stats
        stats = {
            'model_type': model_type_str,
            'total_samples': len(df),
            'cross_val_mean': round(cv_scores.mean(), 4),
            'cross_val_std': round(cv_scores.std(), 4),
            'baseline_comparison': baseline_comparison,
            'metrics': metrics
        }
        
        # Get sample row data for form pre-filling
        row_data = {}
        if len(X) > 0:
            sample_row = X.iloc[0].to_dict()
            row_data = {k: str(v) for k, v in sample_row.items()}
        
        print(f"[INFO] {model_type} model trained successfully")
        return stats, feature_names, row_data
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, [], {}


async def handle_single_model_prediction(form_data, targetcol, model_type):
    """Handle prediction for XGBoost, LightGBM, GradientBoosting"""
    # Extract features
    features = {k: v for k, v in form_data.items() if k not in ['form_name', 'targetColumn', 'col', 'model']}
    
    # Load model paths
    model_dir = os.path.join("models", model_type.lower(), targetcol)
    pipeline_path = os.path.join(model_dir, "pipeline.pkl")
    deployment_path = os.path.join(model_dir, "deployment.json")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    # Check if model files exist
    if not os.path.exists(pipeline_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model pipeline not found for target column '{targetcol}'. Train the model first."
        )

    if not os.path.exists(deployment_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model deployment metadata not found for target column '{targetcol}'"
        )

    # Load deployment metadata
    with open(deployment_path, "r") as f:
        deployment_data = json.load(f)
    
    feature_names = deployment_data.get('feature_names', [])
    is_classification = deployment_data.get('is_classification', False)
    model_type_str = deployment_data.get('model_type', 'Unknown')

    # Load model pipeline
    loaded_pipeline = joblib.load(pipeline_path)

    # Load label encoder if exists
    label_encoder = None
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)

    # Prepare input data
    df_predict = pd.DataFrame([features])
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(df_predict.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {list(missing_features)}"
        )
    
    # Reorder columns
    df_predict = df_predict[feature_names]

    # Convert numeric strings
    for col in df_predict.columns:
        if col in deployment_data.get('numerical_features', []):
            try:
                df_predict[col] = pd.to_numeric(df_predict[col])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric value for feature '{col}': {df_predict[col].iloc[0]}"
                )

    # Make prediction
    predictions = loaded_pipeline.predict(df_predict)
    predicted_value = predictions[0]

    # Decode prediction if label encoder was used
    if is_classification and label_encoder is not None:
        predicted_value = label_encoder.inverse_transform([predicted_value])[0]

    # Convert predicted_value to native Python type for JSON serialization
    if isinstance(predicted_value, (np.integer, np.floating)):
        predicted_value = predicted_value.item()
    elif isinstance(predicted_value, np.ndarray):
        predicted_value = predicted_value.tolist()

    # Build dataset stats for LLM insights
    dataset_stats = {}
    try:
        df_all = pd.read_csv('data.csv')
        if targetcol in df_all.columns:
            y = pd.to_numeric(df_all[targetcol], errors="coerce").dropna()
            if not y.empty:
                q = y.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
                dataset_stats = {
                    "min": float(q.get(0.0)),
                    "q25": float(q.get(0.25)),
                    "median": float(q.get(0.5)),
                    "q75": float(q.get(0.75)),
                    "max": float(q.get(1.0)),
                    "count": int(len(y)),
                }
    except Exception:
        dataset_stats = {}

    # Determine task type
    task_type = "classification" if is_classification else "regression"
    
    # Cache key for insights
    cache_key = f"{targetcol}|{task_type}|{json.dumps(dataset_stats, sort_keys=True)}|{str(predicted_value)[:32]}"

    # Generate LLM insights (same as AutoML and RandomForest)
    try:
        insights = generate_prediction_insights_llm({
            "cache_key": cache_key,
            "target_column": targetcol,
            "task_type": task_type,
            "predicted_value": predicted_value,
            "dataset_stats": dataset_stats,
            "input_sample": {k: features.get(k) for k in list(features.keys())[:10]},
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM insights generation failed: {str(e)}")

    # Prepare response in same format as AutoML and RandomForest
    prediction_result = {
        "predicted_value": float(predicted_value) if not is_classification else str(predicted_value),
        "target_column": targetcol,
        "model_type": model_type,
        "task_type": task_type,
        "predicted_level": insights.get("predicted_level"),
    }

    # Default metrics - try to load from saved model stats
    metric_type = "RMSE" if not is_classification else "Accuracy"
    metric_val = None

    response = {
        "prediction_result": prediction_result,
        "best_model": model_type,
        "metric_type": metric_type,
        "metric": metric_val,
        "output_levels": insights["output_levels"],
        "predicted_level": insights["predicted_level"],
        "input_analysis": insights["input_analysis"],
        "prediction_interpretation": insights["prediction_interpretation"],
        "business_insights": insights["business_insights"],
        "recommended_actions": insights["recommended_actions"],
    }

    return JSONResponse(content=response)


async def handle_supervised_prediction(form_data, targetcol):
    """Handle Supervised AutoML prediction logic (loads best model from metrics.json)"""
    features = {k: v for k, v in form_data.items() if k not in ['form_name', 'targetColumn']}

    model_dir = os.path.join("models", "supervised", targetcol)
    metrics_path = os.path.join(model_dir, "metrics.json")
    deployment_path = os.path.join(model_dir, "deployment.json")

    if not os.path.exists(metrics_path) or not os.path.exists(deployment_path):
        raise HTTPException(
            status_code=404,
            detail=f"AutoML model not found for target column '{targetcol}'. Train first via /api/models with model='AutoML'."
        )

    with open(deployment_path, "r", encoding="utf-8") as f:
        deployment = json.load(f)

    feature_names = deployment.get("feature_names", [])
    numerical_features = deployment.get("numerical_features", [])
    datetime_features = deployment.get("datetime_features", [])

    df_predict = pd.DataFrame([features])

    missing_features = set(feature_names) - set(df_predict.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {list(missing_features)}"
        )

    df_predict = df_predict[feature_names]

    # Coerce datetime fields to numeric seconds since epoch if user passed strings
    for col in datetime_features or []:
        try:
            parsed = pd.to_datetime(df_predict[col], errors="coerce")
            ns = parsed.view("int64").astype("float64")
            ns[pd.isna(parsed)] = np.nan
            df_predict[col] = ns / 1e9
        except Exception:
            # Leave as-is; the pipeline may drop/handle
            pass

    # Convert numeric strings
    for col in df_predict.columns:
        if col in (numerical_features or []):
            try:
                df_predict[col] = pd.to_numeric(df_predict[col])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric value for feature '{col}': {df_predict[col].iloc[0]}"
                )

    pred_res = predict_supervised(model_dir, df_predict)

    # --- UI-focused response (levels + bullet insights) ---
    best_model = (deployment.get("best_model") or "Unknown")
    metric_type = deployment.get("metric_type")
    metric_val = deployment.get("metric")
    task_type = deployment.get("task_type")

    predicted_value = (pred_res.get("predictions") or [None])[0]
    
    # Convert predicted_value to native Python type for JSON serialization
    if isinstance(predicted_value, (np.integer, np.floating)):
        predicted_value = predicted_value.item()
    elif isinstance(predicted_value, np.ndarray):
        predicted_value = predicted_value.tolist()

    # Build LLM payload grounded in dataset stats for speed + accuracy
    dataset_stats: Dict[str, Any] = {}
    try:
        df_all = pd.read_csv('data.csv')
        if targetcol in df_all.columns:
            y = pd.to_numeric(df_all[targetcol], errors="coerce").dropna()
            if not y.empty:
                q = y.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
                dataset_stats = {
                    "min": float(q.get(0.0)),
                    "q25": float(q.get(0.25)),
                    "median": float(q.get(0.5)),
                    "q75": float(q.get(0.75)),
                    "max": float(q.get(1.0)),
                    "count": int(len(y)),
                }
    except Exception:
        dataset_stats = {}

    # Fast, stable cache key (avoid repeating calls for the same target+stats+rounded prediction)
    cache_key = f"{targetcol}|{task_type}|{json.dumps(dataset_stats, sort_keys=True)}|{str(predicted_value)[:32]}"

    # LLM-only: no hardcoded defaults or fallback.
    # If the LLM fails / returns invalid JSON, we surface an error to the caller.
    try:
        insights = generate_prediction_insights_llm({
            "cache_key": cache_key,
            "target_column": targetcol,
            "task_type": task_type,
            "predicted_value": predicted_value,
            "dataset_stats": dataset_stats,
            "input_sample": {k: features.get(k) for k in list(features.keys())[:10]},
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM insights generation failed: {str(e)}")

    # Ensure predicted_value is the right type for response
    is_classification = task_type == "classification"
    
    prediction_result = {
        "predicted_value": float(predicted_value) if not is_classification else str(predicted_value),
        "target_column": targetcol,
        "model_type": f"AutoML({best_model})",
        "task_type": task_type,
        "predicted_level": insights.get("predicted_level"),
    }
    
    # Convert metric_val to native Python type
    if metric_val is not None and isinstance(metric_val, (np.integer, np.floating)):
        metric_val = float(metric_val)

    response = {
        "prediction_result": prediction_result,
        "best_model": best_model,
        "metric_type": metric_type,
        "metric": metric_val,
        "output_levels": insights["output_levels"],
        "predicted_level": insights["predicted_level"],
        "input_analysis": insights["input_analysis"],
        "prediction_interpretation": insights["prediction_interpretation"],
        "business_insights": insights["business_insights"],
        "recommended_actions": insights["recommended_actions"],
    }

    if "probabilities" in pred_res:
        response["prediction_result"]["class_probabilities"] = pred_res.get("probabilities")

    return JSONResponse(content=response)


async def handle_arima_forecast(form_data, targetcol):
    """Handle ARIMA forecasting logic"""
    
    # Validate ARIMA-specific parameters
    frequency = form_data.get('frequency')
    tenure = form_data.get('tenure')
    
    if not frequency:
        raise HTTPException(
            status_code=400,
            detail="Frequency parameter is required for ARIMA forecasting"
        )
    
    if not tenure:
        raise HTTPException(
            status_code=400,
            detail="Tenure (forecast horizon) parameter is required for ARIMA forecasting"
        )

    try:
        tenure = int(tenure)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Tenure must be a valid integer"
        )

    # Validate that ARIMA model exists
    model_base_path = os.path.join("models", 'Arima', targetcol)
    if not os.path.exists(model_base_path):
        raise HTTPException(
            status_code=404,
            detail=f"ARIMA model not found for column '{targetcol}'"
        )

    # Load model metadata
    results_path = os.path.join(model_base_path, f"{targetcol}_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="ARIMA model metadata not found"
        )

    with open(results_path, 'r') as fp:
        model_metadata = json.load(fp)

    # Perform ARIMA forecasting
    try:
        stat, forecasted_data, img_data = arima_forecast_only(
            targetcol, 
            {
                'time_unit': frequency,
                'forecast_horizon': tenure
            }
        )

        if not stat:
            raise HTTPException(
                status_code=500,
                detail="ARIMA forecasting failed"
            )

        # Prepare ARIMA response
        forecast_result = {
            "target_column": targetcol,
            "model_type": "ARIMA",
            "frequency": frequency,
            "forecast_horizon": tenure,
            "forecast_data": json.loads(forecasted_data.to_json()),
            "chart_path": str(img_data),
            "description": generate_text_from_json(json.loads(forecasted_data.to_json()))
        }

        # Prepare model metadata response
        model_info = {
            "model_type": "ARIMA",
            "data_frequency": model_metadata.get('data_freq', 'N/A'),
            "training_period": {
                "start_date": model_metadata.get('start_date', 'N/A'),
                "end_date": model_metadata.get('end_date', 'N/A')
            },
            "trained_at": model_metadata.get('trained_at', 'N/A')
        }

        response = {
            "prediction_result": forecast_result,
            "model_performance": model_info,
            "forecast_analysis": {
                "forecast_summary": {
                    "periods_forecasted": tenure,
                    "frequency": frequency,
                    "model_path": str(img_data)
                },
                "input_parameters": {
                    "target_column": targetcol,
                    "frequency": frequency,
                    "tenure": tenure
                }
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        print(f"ARIMA forecasting error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ARIMA forecasting failed: {str(e)}"
        )


def arima_forecast_only(target_col, bot_query):
    """
    Load trained ARIMA model and perform forecasting
    """
    try:
        print('ARIMA Forecasting Phase')
        
        # Load model metadata
        metadata_path = os.path.join("models", 'Arima', target_col, target_col + '_results.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model metadata not found for {target_col}")

        with open(metadata_path, 'r') as fp:
            model_metadata = json.load(fp)

        frequency = bot_query['time_unit']
        periods = bot_query['forecast_horizon']
        
        # Load the trained model
        model_path = os.path.join(os.getcwd(), 'models', 'Arima', target_col, frequency, "best_model.pkl")
        print("Model path:", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found for frequency '{frequency}'")

        loaded_model = load_forecast_model(model_path)
        
        freq_map = {
            'hours': 'H',
            'days': 'D',
            'weeks': 'W',
            'months': 'MS',
            'years': 'YS'
        }

        # Generate forecast
        forecasted_data = arima_forecast(loaded_model, periods, freq_map[frequency], target_col)
        print("Forecast generated:", forecasted_data.shape)

        # Generate visualization
        result_graph = plot_graph(forecasted_data, target_col)

        return True, forecasted_data, result_graph

    except Exception as e:
        print(f"Forecasting error: {e}")
        return False, pd.DataFrame(), ''


# 8.Genai bot plotly visualisation.-----------------Prediction and forecasting related api-------8
def extract_model_from_prompt(prompt: str) -> tuple[str, bool]:
    """
    Extract model name from user prompt. 
    Returns (model_name, is_supported) tuple.
    """
    prompt_lower = prompt.lower()
    
    # Supported models
    supported_models = {
        'automl': 'AutoML',
        'auto ml': 'AutoML',
        'random forest': 'RandomForest',
        'randomforest': 'RandomForest',
        'xgboost': 'XGBoost',
        'xgb': 'XGBoost',
        'lightgbm': 'LightGBM',
        'lgbm': 'LightGBM',
        'gradient boost': 'GradientBoosting',
        'gradientboost': 'GradientBoosting',
        'gradientboosting': 'GradientBoosting'
    }
    
    # Check for unsupported models
    unsupported_keywords = [
        'decision tree', 'decisiontree',
        'logistic regression', 'logisticregression',
        'svm', 'support vector',
        'naive bayes', 'naivebayes',
        'knn', 'k-nearest',
        'adaboost', 'ada boost',
        'catboost', 'cat boost',
        'neural network', 'neural net', 'deep learning'
    ]
    
    # Check for unsupported models first
    for keyword in unsupported_keywords:
        if keyword in prompt_lower:
            # Extract the model name mentioned
            for unsup in unsupported_keywords:
                if unsup in prompt_lower:
                    return (unsup.title().replace(' ', ''), False)
    
    # Check for specific model mentions
    for keyword, model_name in supported_models.items():
        if keyword in prompt_lower:
            return (model_name, True)
    
    # Default to AutoML if no specific model mentioned
    return ('AutoML', True)


@app.post("/api/ai_bot")
async def gen_ai_bot(request_body: GenAIBotRequest):
    try:
        # Load and prepare data - try multiple possible locations
        data_file = None
        possible_paths = [
            'data.csv',
            os.path.join(os.getcwd(), 'data.csv'),
            os.path.join(os.path.dirname(__file__), 'data.csv'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        if not data_file:
            error_msg = (
                "No data file found. Please upload your dataset first using the file upload feature. "
                "Once your data is uploaded, I'll be able to make predictions."
            )
            return JSONResponse({
                'text_pre_code_response': error_msg,
                'error': 'no_data',
                'message': 'Please upload data before making predictions'
            }, status_code=404)
        
        df = pd.read_csv(data_file)
        metadata_str = ", ".join(df.columns.tolist())
        sample_data = df.head(2).to_dict(orient='records')

        prompt = request_body.prompt
        session_id = str(uuid.uuid4())

        # Store the new user message in memory
        async with CHAT_MEMORY_LOCK:  # <--- no error now!
            if session_id not in CHAT_MEMORY:
                CHAT_MEMORY[session_id] = []
            CHAT_MEMORY[session_id].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
        # Build LLM messages with full chat history
        messages = []
        for msg in CHAT_MEMORY[session_id]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Use messages as context for the LLM
        # Handle forecasting requests
        if 'forecast' in prompt.lower():
            data = extract_forecast_details_llm(prompt, df.columns, df, request_body.email)
            print("data printed")
            
            # Check if ARIMA model already exists for this target variable
            target_col = data['target_variable']
            model_base_path = os.path.join("models", 'Arima', target_col)
            
            # Step 1: Train model if it doesn't exist
            if not os.path.exists(model_base_path) or not os.path.exists(os.path.join(model_base_path, f"{target_col}_results.json")):
                print(f"[AI_BOT] Training ARIMA model for {target_col}...")
                train_stat = arima_train_only(df, target_col)
                
                if not train_stat:
                    return JSONResponse({
                        'error': f'ARIMA model training failed for {target_col}',
                        'session_id': session_id
                    }, status_code=500)
                
                print(f"ARIMA model training completed for {target_col}")
            
            # Step 2: Perform forecasting
            print(f"Generating forecast for {target_col}...")
            forecast_stat, forecasted_data, img_data = arima_forecast_only(target_col, data)
            
            if not forecast_stat:
                return JSONResponse({
                    'error': f'ARIMA forecasting failed for {target_col}',
                    'session_id': session_id
                }, status_code=500)

            # Store bot response
            bot_content = json.dumps({
                'data': json.loads(forecasted_data.to_json()), 
                'plot': make_serializable(img_data),
                'model_trained': not os.path.exists(model_base_path),  # Indicates if training occurred
                'target_column': target_col
            })
            
            async with CHAT_MEMORY_LOCK:
                CHAT_MEMORY[session_id].append(
                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})

            return JSONResponse({
                'data': json.loads(forecasted_data.to_json()),
                'plot': make_serializable(img_data),
                'session_id': session_id,
                'description': generate_text_from_json(json.loads(forecasted_data.to_json())),
                'model_info': {
                    'target_column': target_col,
                    'frequency': data.get('time_unit', 'N/A'),
                    'forecast_horizon': data.get('forecast_horizon', 'N/A'),
                    'model_trained_this_session': not os.path.exists(model_base_path)
                }
            })


        # Handle prediction requests
        elif 'predict' in prompt.lower():
            # Extract model from prompt and check if supported
            model_name, is_supported = extract_model_from_prompt(prompt)
            print(f"[AI_BOT] Detected model: {model_name}, Supported: {is_supported}")
            
            # If model is not supported, return error with supported models list
            if not is_supported:
                supported_models_list = [
                    "AutoML (Best Model)",
                    "Random Forest",
                    "XGBoost",
                    "LightGBM",
                    "Gradient Boosting"
                ]
                error_msg = (
                    f"'{model_name}' model is not currently supported. "
                    f"Please use one of the following supported models:\n"
                    f"• " + "\n• ".join(supported_models_list)
                )
                bot_content = json.dumps({'text_pre_code_response': error_msg})
                async with CHAT_MEMORY_LOCK:
                    CHAT_MEMORY[session_id].append(
                        {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                return JSONResponse({
                    'text_pre_code_response': error_msg,
                    'session_id': session_id,
                    'error': 'unsupported_model',
                    'supported_models': supported_models_list
                })
            
            data = extract_forecast_details_rf(prompt, df.columns, request_body.email)

            if len(data.get('missing_columns', [])) > 0:
                bot_content = json.dumps({'text_pre_code_response': (
                    f'Prediction failed due to missing fields: {data.get("missing_columns")}. '
                    f'Please retry with all required inputs.')})
                async with CHAT_MEMORY_LOCK:
                    CHAT_MEMORY[session_id].append(
                        {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                return JSONResponse({
                    'text_pre_code_response': (
                        f'Prediction failed due to missing fields: {data.get("missing_columns")}. '
                        f'Please retry with all required inputs.'),
                    'session_id': session_id
                })

            # Determine model directory based on model type
            if model_name == 'AutoML':
                model_folder = 'supervised'
            elif model_name == 'RandomForest':
                model_folder = 'rf'
            elif model_name == 'XGBoost':
                model_folder = 'xgboost'
            elif model_name == 'LightGBM':
                model_folder = 'lightgbm'
            elif model_name == 'GradientBoosting':
                model_folder = 'gradientboosting'
            else:
                model_folder = 'supervised'  # Default to AutoML
                model_name = 'AutoML'
            
            model_path = os.path.join("models", model_folder, data['target_column'])
            pipeline_path = os.path.join(model_path, "pipeline.pkl")
            deployment_path = os.path.join(model_path, "deployment.json")
            label_encoder_path = os.path.join(model_path, "label_encoder.pkl")

            # Check and train model if needed
            model_needs_training = not os.path.exists(deployment_path) or not os.path.exists(pipeline_path)
            
            if model_needs_training:
                df_train = pd.read_csv(data_file)
                print(f"[AI_BOT] Model not found. Training {model_name} model for {data.get('target_column')}...")
                
                # Notify user that training is starting
                training_msg = f"Training {model_name} model for '{data.get('target_column')}' prediction. This may take a moment..."
                
                try:
                    # Train based on model type
                    if model_name == 'AutoML':
                        from supervised_automl import train_supervised_automl
                        res = train_supervised_automl(df=df_train, target_col=data.get('target_column'), model_dir=model_path)
                        if not res.get("status"):
                            error_msg = (
                                f"Failed to train {model_name} model: {res.get('message', 'Unknown error')}. "
                                f"Please check your data and try again."
                            )
                            bot_content = json.dumps({'text_pre_code_response': error_msg})
                            async with CHAT_MEMORY_LOCK:
                                CHAT_MEMORY[session_id].append(
                                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                            return JSONResponse({
                                'text_pre_code_response': error_msg, 
                                'session_id': session_id,
                                'error': 'training_failed'
                            }, status_code=500)
                        model_stats = {}
                    elif model_name == 'RandomForest':
                        model_stats, _ = random_forest(df_train, data.get('target_column'))
                    else:
                        # XGBoost, LightGBM, GradientBoosting
                        model_stats, _, _ = train_single_model(df_train, data.get('target_column'), model_name)
                    
                    print(f"[AI_BOT] {model_name} model trained successfully!")
                    
                except Exception as e:
                    error_msg = (
                        f"An error occurred while training the {model_name} model: {str(e)}. "
                        f"Please check your data format and try again."
                    )
                    bot_content = json.dumps({'text_pre_code_response': error_msg})
                    async with CHAT_MEMORY_LOCK:
                        CHAT_MEMORY[session_id].append(
                            {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                    return JSONResponse({
                        'text_pre_code_response': error_msg, 
                        'session_id': session_id,
                        'error': 'training_error'
                    }, status_code=500)
                
                # Reload deployment data after training
                if not os.path.exists(deployment_path):
                    error_msg = f"Model training completed but deployment data is missing. Please try again."
                    return JSONResponse({
                        'text_pre_code_response': error_msg, 
                        'session_id': session_id,
                        'error': 'deployment_missing'
                    }, status_code=500)
                    
                with open(deployment_path, 'r', encoding='utf-8') as f:
                    deployment_data = json.load(f)
            else:
                # Load existing model statistics and deployment data
                print(f"[AI_BOT] Using existing {model_name} model for {data.get('target_column')}")
                with open(deployment_path, 'r', encoding='utf-8') as f:
                    deployment_data = json.load(f)
                model_stats = deployment_data.get('stats', {})

            # Get model metadata
            is_classification = deployment_data.get('is_classification', False)
            model_type = deployment_data.get('model_type', 'Unknown')
            feature_names = deployment_data.get('feature_names', [])

            # Load the trained pipeline
            loaded_pipeline = load_pipeline(pipeline_path)

            # Load label encoder if it exists (for categorical targets)
            label_encoder = None
            if os.path.exists(label_encoder_path):
                label_encoder = joblib.load(label_encoder_path)

            # Prepare prediction data with proper feature ordering
            features = data.get('features', {})
            df_predict = pd.DataFrame([features])
            
            # Ensure all required features are present and reorder columns
            if feature_names:
                # Check for missing features
                missing_features = set(feature_names) - set(df_predict.columns)
                if missing_features:
                    error_msg = f'Missing required features: {list(missing_features)}'
                    bot_content = json.dumps({'text_pre_code_response': error_msg})
                    async with CHAT_MEMORY_LOCK:
                        CHAT_MEMORY[session_id].append(
                            {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                    return JSONResponse({
                        'text_pre_code_response': error_msg,
                        'session_id': session_id
                    })
                
                # Reorder columns to match training data
                df_predict = df_predict[feature_names]

            # Convert numeric strings to appropriate types for numerical features
            numerical_features = deployment_data.get('numerical_features', [])
            for col in df_predict.columns:
                if col in numerical_features:
                    try:
                        df_predict[col] = pd.to_numeric(df_predict[col])
                    except ValueError:
                        error_msg = f"Invalid numeric value for feature '{col}': {df_predict[col].iloc[0]}"
                        bot_content = json.dumps({'text_pre_code_response': error_msg})
                        async with CHAT_MEMORY_LOCK:
                            CHAT_MEMORY[session_id].append(
                                {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                        return JSONResponse({
                            'text_pre_code_response': error_msg,
                            'session_id': session_id
                        })

            # Make prediction
            predictions = loaded_pipeline.predict(df_predict)
            predicted_value = predictions[0]

            # Handle prediction probabilities and confidence for classification
            prediction_proba = None
            confidence = None
            class_probabilities = None
            
            if is_classification:
                # Get prediction probabilities for classification
                if hasattr(loaded_pipeline, 'predict_proba'):
                    try:
                        proba = loaded_pipeline.predict_proba(df_predict)[0]
                        classes = loaded_pipeline.classes_
                        prediction_proba = dict(zip(classes, proba))
                        confidence = float(max(proba))
                    except Exception as e:
                        print(f"Warning: predict_proba failed: {e}")

                # Decode prediction if label encoder was used
                if label_encoder is not None:
                    predicted_value = label_encoder.inverse_transform([predicted_value])[0]
                    
                    # Also decode probability classes
                    if prediction_proba:
                        decoded_proba = {}
                        for encoded_class, prob in prediction_proba.items():
                            decoded_class = label_encoder.inverse_transform([encoded_class])[0]
                            decoded_proba[str(decoded_class)] = float(prob)
                        class_probabilities = decoded_proba
                else:
                    class_probabilities = prediction_proba

            # Calculate feature importance and impact
            feature_importance = get_feature_importance(loaded_pipeline, feature_names or features.keys())
            feature_impact = calculate_feature_impact(loaded_pipeline, df_predict, features, is_classification, label_encoder)

            # Build dataset stats for LLM insights (same as model_predict)
            dataset_stats = {}
            try:
                if os.path.exists(data_file):
                    df_all = pd.read_csv(data_file)
                    target_col = data.get('target_column')
                    if target_col in df_all.columns:
                        y = pd.to_numeric(df_all[target_col], errors="coerce").dropna()
                        if not y.empty:
                            q = y.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
                            dataset_stats = {
                                "min": float(q.get(0.0)),
                                "q25": float(q.get(0.25)),
                                "median": float(q.get(0.5)),
                                "q75": float(q.get(0.75)),
                                "max": float(q.get(1.0)),
                                "count": int(len(y)),
                            }
            except Exception as e:
                print(f"[AI_BOT] Warning: Could not load dataset stats: {e}")
                dataset_stats = {}

            # Determine task type
            task_type = "classification" if is_classification else "regression"
            
            # Cache key for insights
            cache_key = f"{data.get('target_column')}|{task_type}|{json.dumps(dataset_stats, sort_keys=True)}|{str(predicted_value)[:32]}"

            # Generate LLM insights (same as model_predict endpoints)
            try:
                insights = generate_prediction_insights_llm({
                    "cache_key": cache_key,
                    "target_column": data.get('target_column'),
                    "task_type": task_type,
                    "predicted_value": predicted_value,
                    "dataset_stats": dataset_stats,
                    "input_sample": {k: features.get(k) for k in list(features.keys())[:10]},
                })
            except Exception as e:
                print(f"[AI_BOT] Warning: LLM insights generation failed: {e}")
                # Fallback insights if LLM fails
                insights = {
                    "output_levels": {},
                    "predicted_level": None,
                    "input_analysis": [],
                    "prediction_interpretation": [],
                    "business_insights": [],
                    "recommended_actions": []
                }

            # Convert predicted_value to native Python type for JSON serialization
            if isinstance(predicted_value, (np.integer, np.floating)):
                predicted_value = predicted_value.item()
            elif isinstance(predicted_value, np.ndarray):
                predicted_value = predicted_value.tolist()
            
            # Prepare response in same format as model_predict
            prediction_result = {
                "predicted_value": float(predicted_value) if not is_classification else str(predicted_value),
                "target_column": data.get('target_column'),
                "model_type": model_name,
                "task_type": task_type,
                "predicted_level": insights.get("predicted_level"),
            }
            
            # Get metrics from model stats
            metrics = model_stats.get("metrics", {})
            metric_type = "RMSE" if not is_classification else "Accuracy"
            metric_val = metrics.get('rmse') if not is_classification else metrics.get('accuracy')
            
            # Convert metric_val to native Python type
            if metric_val is not None and isinstance(metric_val, (np.integer, np.floating)):
                metric_val = float(metric_val)
            
            # Text response
            if model_needs_training:
                text_response = f"✓ Trained {model_name} model successfully!\n\nPrediction: {data.get('target_column')} = {predicted_value}"
            else:
                text_response = f"Using {model_name} model: Predicted {data.get('target_column')} value is {predicted_value}"
            
            if insights.get("predicted_level"):
                text_response += f" (Level: {insights.get('predicted_level')})"

            # Create response matching model_predict format
            response_data = {
                "prediction_result": prediction_result,
                "best_model": model_name,
                "metric_type": metric_type,
                "metric": metric_val,
                "output_levels": insights.get("output_levels", {}),
                "predicted_level": insights.get("predicted_level"),
                "input_analysis": insights.get("input_analysis", []),
                "prediction_interpretation": insights.get("prediction_interpretation", []),
                "business_insights": insights.get("business_insights", []),
                "recommended_actions": insights.get("recommended_actions", []),
                "text_pre_code_response": text_response,
                "model_used": model_name,
                "model_trained_this_session": model_needs_training,  # Indicate if model was just trained
                'session_id': session_id
            }

            bot_content = json.dumps(response_data)
            async with CHAT_MEMORY_LOCK:
                CHAT_MEMORY[session_id].append(
                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})

            return JSONResponse(response_data)


        # Handle general data analysis requests
        else:
            # Use full chat history as context for the LLM (model from llm_config)
            _client = get_openai_client(request_body.email)
            _model = get_model_name(request_body.email)
            response = _client.chat.completions.create(
                model=_model,
                messages=messages
            )

            # Track AI credits usage
            record_llm_usage(request_body.email, response)

            pre_code_text, post_code_text, code = process_genai_response(response)
            result: Dict[str, Any] = {}
            result.update({
                'text_pre_code_response': pre_code_text,
                'code': code,
                'text_post_code_response': post_code_text
            })

            if 'import' in code:
                namespace = {}
                try:
                    exec(code, namespace)
                    result['text_output'] = namespace.get('text_output')

                    fig = namespace.get('fig')
                    if fig and isinstance(fig, Figure):
                        result['chart_response'] = make_serializable(fig.to_plotly_json())
                except Exception as e:
                    bot_content = json.dumps({'error': f"Code execution failed: {str(e)}"})
                    async with CHAT_MEMORY_LOCK:
                        CHAT_MEMORY[session_id].append(
                            {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                    raise HTTPException(
                        status_code=500,
                        detail=f"Code execution failed: {str(e)}"
                    )

            # Store bot response
            bot_content = json.dumps(result)
            async with CHAT_MEMORY_LOCK:
                CHAT_MEMORY[session_id].append(
                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})

            result['session_id'] = session_id
            return JSONResponse(result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Data file not found"
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Input CSV file is empty or corrupt"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


def generate_text_from_json(json_data: dict) -> str:
    description = "You are a helpful data analyst. Given the following analysis output, describe the results to the user in plain English within 2 or 3 lines only.."
    data_summary = json.dumps(json_data, indent=2)
    full_prompt = f"{description}\nHere is the analysis output:\n{data_summary}\n"

    _client = get_openai_client(None)
    _model = get_model_name(None)
    response = _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system",
             "content": "You are a helpful data analyst who explains model outputs, data visualizations, and user queries clearly and insightfully."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def load_pipeline(save_path="model_pipeline.pkl"):
    # Load the saved pipeline
    pipeline = joblib.load(save_path)
    print(f"Pipeline loaded from: {save_path}")
    return pipeline


def extract_forecast_details_llm(prompt, column_names, df, email: Optional[str] = None):
    try:
        system_prompt = f""" You are an AI assistant that extracts forecast details from a user's prompt. Given a 
        natural language input and the following column names from the input data, return the following in JSON format:

            1. "target_variable" - The thing being forecasted (e.g., "sales", "revenue"). - If the target variable is 
            misspelled or ambiguous, try to match it to the closest column name from the list below. 2. 
            "forecast_horizon" - The number of time steps. 3. "time_unit" - The unit of time (days, months, years).

            Available column names: {', '.join(column_names)}

            Example Outputs:
            - Input: "Forecast the sales data for 5 years."
              Output: {{"target_variable": "sales", "forecast_horizon": 5, "time_unit": "years"}}

            - Input: "Can you predict electricity demand for the next 12 months?"
              Output: {{"target_variable": "electricity demand", "forecast_horizon": 12, "time_unit": "months"}}

            - Input: "I want to predict CO2 levels for 7 days."
              Output: {{"target_variable": "CO2 levels", "forecast_horizon": 7, "time_unit": "days"}}
            
            -You have to start the forecasting based on the last available date point in the dataset i.e the date at {df.tail(1)}.You do not consider the current date as the starting point for forecasting.
            Ensure that the "target_variable" matches one of the available column names, even if the user misspells it.
            """
        forecast_details = ''
        _client = get_openai_client(email)
        _model = get_model_name(email)
        response = _client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Make it deterministic
        )

        # Track AI credits usage
        record_llm_usage(email, response)
        
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            forecast_details += chunk_message
        print(forecast_details)

        return eval(forecast_details)

    except Exception as e:
        print(e)


def extract_forecast_details_rf(prompt, column_names, email: Optional[str] = None):
    try:
        system_prompt = f"""
            You are an AI assistant that extracts machine learning input features and the target variable from a user's natural language prompt.

            You are provided a list of available column names: {', '.join(column_names)}.

            Your job is to:
            1. **Correct Spelling**: If any feature or target column is misspelled, match it to the closest name from the provided column list.
            2. **Extract Features**: Identify which features and their values are mentioned in the input.
            3. **Detect Missing Features**: If some required features are not mentioned, list them under "missing_columns".
            4. **Identify Target Column**: If the user specifies a column as the one to be predicted or forecasted, include it as "target_column".
            5. **Always Return All Three Fields**: Even if one or more are empty, the response **must always** contain "features", "missing_columns", and "target_column".

            ### Expected Output Formats:
            Return the output strictly as a Python dictionary:

            #### a) All features and target column provided:
            Input: "Predict CO2 level. The temperature is 25 and humidity is 45."
            Output:
            {{
              "features": {{
                "temperature": 25,
                "humidity": 45
              }},
              "missing_columns":[],
              "target_column": "CO2 level"
            }}

            #### b) Some features missing:
            Input: "I want to predict pressure. Set humidity to 50."
            Output:
            {{
              "features": {{
                "humidity": 50
              }},
              "missing_columns": ["temperature", "CO2 level"],
              "target_column": "pressure"
            }}

            #### c) Misspelled entries:
            Input: "Predict temprature using humdity = 60 and presure = 1000"
            Output:
            {{
              "features": {{
                "humidity": 60,
                "pressure": 1000
              }},
              "missing_columns": ["CO2 level"],
              "target_column": "temperature"
            }}

            ### Notes:
            - Always correct any misspelled column names to the closest match in the available list.
            - Use numeric types for numeric values, not strings.
            - If the target column is not explicitly provided, leave "target_column" as null or omit it.
        """

        predict_details = ''
        _client = get_openai_client(email)
        _model = get_model_name(email)
        response = _client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Make it deterministic
        )

        # Track AI credits usage
        record_llm_usage(email, response)
        
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            predict_details += chunk_message
        print(predict_details)

        return eval(predict_details)

    except Exception as e:
        print(e)


def process_genai_response(response):
    all_text = ""
    text_post_code = ''
    code_start = -1
    code_end = -1
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    text_pre_code = all_text[:code_start - 9]
    if code_start != -1:
        text_post_code = all_text[code_end:]
    return text_pre_code, text_post_code, code

def get_feature_importance(pipeline, feature_names):
    """Extract feature importance from the trained model"""
    try:
        # Get the model from the pipeline
        model = pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps['preprocessor']
            feature_names_transformed = []
            
            # Handle numerical features
            if hasattr(preprocessor, 'named_transformers_'):
                if 'num' in preprocessor.named_transformers_:
                    num_features = preprocessor.named_transformers_['num'].feature_names_in_
                    feature_names_transformed.extend(num_features)
                
                # Handle categorical features (one-hot encoded)
                if 'cat' in preprocessor.named_transformers_:
                    cat_transformer = preprocessor.named_transformers_['cat']
                    if hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
                        cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out()
                        feature_names_transformed.extend(cat_features)
            
            # If we can't get transformed names, use original names
            if not feature_names_transformed:
                feature_names_transformed = feature_names[:len(importances)]
            
            # Create importance pairs and sort
            importance_pairs = list(zip(feature_names_transformed, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 10 features
            return [
                {"feature": feature, "importance": round(importance * 100, 2)}
                for feature, importance in importance_pairs[:10]
            ]
    except Exception as e:
        print(f"Error getting feature importance: {e}")
    
    return []



def get_prediction_confidence(prediction_proba):
    """Calculate prediction confidence from probability scores"""
    if prediction_proba is not None and len(prediction_proba) > 0:
        max_proba = max(prediction_proba[0])
        return round(max_proba * 100, 1)
    return None


def calculate_feature_impact(pipeline, df_predict, original_features, is_classification, label_encoder):
    """Calculate the impact of each feature on the prediction"""
    try:
        base_prediction = pipeline.predict(df_predict)[0]
        feature_impacts = {}
        
        for feature, value in original_features.items():
            # Create a copy of the dataframe
            df_modified = df_predict.copy()
            
            # Try to modify the feature value to see impact
            if feature in df_predict.columns:
                # For numerical features, try mean imputation
                if df_predict[feature].dtype in ['int64', 'float64']:
                    df_modified[feature] = df_predict[feature].mean()
                else:
                    # For categorical, use mode or a default value
                    df_modified[feature] = 'unknown'
                
                try:
                    modified_prediction = pipeline.predict(df_modified)[0]
                    
                    if is_classification:
                        # For classification, show if prediction changed
                        if label_encoder is not None:
                            base_pred_decoded = label_encoder.inverse_transform([base_prediction])[0]
                            mod_pred_decoded = label_encoder.inverse_transform([modified_prediction])[0]
                            impact = "Changed" if base_pred_decoded != mod_pred_decoded else "No change"
                        else:
                            impact = "Changed" if base_prediction != modified_prediction else "No change"
                    else:
                        # For regression, show numerical difference
                        impact = round(float(base_prediction - modified_prediction), 4)
                    
                    feature_impacts[feature] = {
                        "current_value": str(value),
                        "impact": impact
                    }
                except Exception:
                    feature_impacts[feature] = {
                        "current_value": str(value),
                        "impact": "Unable to calculate"
                    }
        
        return feature_impacts
    except Exception as e:
        print(f"Error calculating feature impact: {e}")
        return {}


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def parse_mixed_dates_with_format_detection(date_series):
    """
    Enhanced date parsing that detects and handles different date formats
    """
    try:
        # Common date formats to try
        date_formats = [
            '%d-%m-%Y',      # day-month-year (12-05-2025)
            '%d/%m/%Y',      # day/month/year (12/05/2025)
            '%d.%m.%Y',      # day.month.year (12.05.2025)
            '%d-%m-%y',      # day-month-year short (12-05-25)
            '%d/%m/%y',      # day/month/year short (12/05/25)
            '%m-%d-%Y',      # month-day-year (05-12-2025)
            '%m/%d/%Y',      # month/day/year (05/12/2025)
            '%Y-%m-%d',      # year-month-day (2025-05-12)
            '%Y/%m/%d',      # year/month/day (2025/05/12)
            '%d %b %Y',      # day month year (12 May 2025)
            '%d %B %Y',      # day month year full (12 May 2025)
            '%b %d, %Y',     # month day, year (May 12, 2025)
            '%B %d, %Y',     # month day, year full (May 12, 2025)
        ]
        
        # Get a sample of non-null values to test formats
        sample_dates = date_series.dropna().head(10).astype(str)
        
        print(f"Sample dates for format detection: {list(sample_dates)}")
        
        successful_format = None
        
        # Try each format
        for date_format in date_formats:
            try:
                print(f"Trying format: {date_format}")
                # Test with first few samples
                test_parsed = pd.to_datetime(sample_dates, format=date_format, errors='raise')
                successful_format = date_format
                print(f"Successfully detected format: {date_format}")
                break
            except (ValueError, TypeError) as e:
                continue
        
        # If no specific format worked, try pandas' automatic parsing with dayfirst=True
        if successful_format is None:
            print("No specific format worked, trying automatic parsing with dayfirst=True")
            try:
                # For day-first formats (common in many countries)
                parsed_dates = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
                
                # Validate that we got reasonable results
                if not parsed_dates.isnull().all():
                    print("Automatic parsing with dayfirst=True succeeded")
                    return parsed_dates
            except Exception as e:
                print(f"Automatic parsing with dayfirst=True failed: {e}")
        
        # If we found a successful format, parse the entire series
        if successful_format:
            try:
                parsed_dates = pd.to_datetime(date_series, format=successful_format, errors='coerce')
                
                # Validate results
                valid_count = (~parsed_dates.isnull()).sum()
                total_count = len(date_series)
                success_rate = valid_count / total_count if total_count > 0 else 0
                
                print(f"Format {successful_format}: {valid_count}/{total_count} dates parsed ({success_rate:.1%})")
                
                if success_rate > 0.8:  # At least 80% success rate
                    return parsed_dates
                    
            except Exception as e:
                print(f"Error parsing with format {successful_format}: {e}")
        
        # Last resort: try pandas default parsing
        print("Falling back to pandas default parsing")
        return pd.to_datetime(date_series, errors='coerce')
        
    except Exception as e:
        print(f"Error in enhanced date parsing: {e}")
        return pd.to_datetime(date_series, errors='coerce')



# def arima_train(data, target_col, bot_query=None):
#     try:
#         print('ArimaTrain')
#         print("Column dtypes:\n", data.dtypes)

#         # Identify date column
#         date_column = None
#         results = {}

#         if not os.path.exists(os.path.join("models", 'Arima', target_col)):
#             for col in data.columns:
#                 print(f"Checking column '{col}' for dates")
#                 if data.dtypes[col] == 'object':
#                     try:
#                         # First check if it's already datetime
#                         if pd.api.types.is_datetime64_any_dtype(data[col]):
#                             date_column = col
#                             break

#                         # Try parsing with mixed formats handler
#                         print(f"Attempting to parse mixed formats in column '{col}'")
#                         data[col] = parse_mixed_dates(data[col])

#                         # Check if we successfully parsed any dates
#                         if not data[col].isnull().all():
#                             date_column = col
#                             print(f"Successfully parsed datetime column: {col}")
#                             break

#                     except Exception as e:
#                         print(f"Error parsing column '{col}': {e}")
#                         continue

#             if not date_column:
#                 raise ValueError("No datetime column could be parsed from the dataset")

#             data[date_column] = pd.to_datetime(data[date_column], errors='coerce', utc=False)

#             # If some values are tz-aware Python datetime objects, strip tz without shifting
#             def strip_tz(x):
#                 if isinstance(x, datetime) and x.tzinfo is not None:
#                     return x.replace(tzinfo=None)
#                 return x

#             # Apply the function
#             data[date_column] = data[date_column].map(strip_tz)

#             # Standardize the format and set as index
#             data[date_column] = pd.to_datetime(data[date_column],errors='coerce', utc=False)
#             try:
#                 data_actual = data[[date_column,target_col]].copy()
#                 data_actual.columns = ["datetime", 'value']
#                 data_actual.set_index("datetime", inplace=True)

#                 # Check for frequency and handle any irregularities
#                 train_frequency = check_data_frequency(data_actual)

#                 # Ensure no duplicate indices
#                 if data_actual.index.duplicated().any():
#                     print("Warning: Duplicate datetime indices found - aggregating")
#                     data_actual = data_actual.groupby(data_actual.index).mean()

#                 train_models(data_actual, target_col)
#                 print("Getting from the train_models function................")

#                 with open(os.path.join("models", 'Arima', target_col, target_col + '_results.json'), 'w') as fp:
#                     json.dump({
#                         'data_freq': train_frequency,
#                         'date_column': date_column,
#                         'start_date': str(data_actual.index.min()),
#                         'end_date': str(data_actual.index.max())
#                     }, fp, indent=4)

#             except Exception as e:
#                 print(f"Error during model training: {e}")
#                 raise

#         # Forecasting logic remains the same
#         frequency = bot_query['time_unit']
#         periods = bot_query['forecast_horizon']
#         model_path = os.path.join(os.getcwd(), 'models', 'Arima', target_col, frequency, "best_model.pkl")
#         print("model_path", model_path)

#         loaded_model = load_forecast_model(model_path)
#         freq_map = {
#             'hours': 'H',
#             'days': 'D',
#             'weeks': 'W',
#             'months': 'MS',
#             'years': 'YS'
#         }

#         forecasted_data = arima_forecast(loaded_model, periods, freq_map[frequency],target_col)
#         print(forecasted_data)

#         result_graph = plot_graph(forecasted_data,target_col)

#         print(f"Results saved to {os.path.join('models', 'Arima', target_col, target_col + '_results.json')}")
#         return True, forecasted_data, result_graph

#     except Exception as e:
#         print(e)
#         return False, pd.DataFrame, ''


def arima_train_only(data, target_col):
    """
    Train ARIMA model and save it without forecasting
    """
    try:
        print('ARIMA Training Phase')
        print("Column dtypes:\n", data.dtypes)

        # Identify date column
        date_column = None

        # Check if metadata file exists, not just the directory
        metadata_path = os.path.join("models", 'Arima', target_col, target_col + '_results.json')
        if not os.path.exists(metadata_path):
            for col in data.columns:
                print(f"Checking column '{col}' for dates")
                if data.dtypes[col] == 'object':
                    try:
                        # First check if it's already datetime
                        if pd.api.types.is_datetime64_any_dtype(data[col]):
                            date_column = col
                            break

                        # Try parsing with mixed formats handler
                        print(f"Attempting to parse mixed formats in column '{col}'")
                        data[col] = parse_mixed_dates_with_format_detection(data[col])

                        # Check if we successfully parsed any dates
                        if not data[col].isnull().all():
                            date_column = col
                            print(f"Successfully parsed datetime column: {col}")
                            break

                    except Exception as e:
                        print(f"Error parsing column '{col}': {e}")
                        continue

            if not date_column:
                raise ValueError("No datetime column could be parsed from the dataset")

            # Strip timezone information
            def strip_tz(x):
                if isinstance(x, datetime) and x.tzinfo is not None:
                    return x.replace(tzinfo=None)
                return x

            data[date_column] = data[date_column].map(strip_tz)
            
            try:
                data_actual = data[[date_column, target_col]].copy()
                data_actual.columns = ["datetime", 'value']
                data_actual.set_index("datetime", inplace=True)
                
                # Remove any duplicate indices and sort
                data_actual = data_actual[~data_actual.index.duplicated(keep='first')]
                data_actual = data_actual.sort_index()

                # Enhanced frequency detection
                train_frequency = check_data_frequency(data_actual)
                print(f"Detected frequency: {train_frequency}")

                print("First 10 rows after processing:")
                print(data_actual.head(10))
                print("\nLast 10 rows after processing:")
                print(data_actual.tail(10))

                # Train models for all frequencies
                train_models(data_actual, target_col)
                print("Model training completed successfully")

                # Save metadata with enhanced information - ensure valid dates
                start_idx = data_actual.index[0]
                end_idx = data_actual.index[-1]
                
                # Validate and convert to string, fallback to min/max if needed
                try:
                    start_date_str = pd.Timestamp(start_idx).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(start_idx) else str(data_actual.index.min())
                    end_date_str = pd.Timestamp(end_idx).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(end_idx) else str(data_actual.index.max())
                except:
                    start_date_str = str(data_actual.index.min())
                    end_date_str = str(data_actual.index.max())
                
                print(f"Saving metadata: start={start_date_str}, end={end_date_str}")
                
                with open(os.path.join("models", 'Arima', target_col, target_col + '_results.json'), 'w') as fp:
                    json.dump({
                        'data_freq': train_frequency,
                        'date_column': date_column,
                        'start_date': start_date_str,
                        'end_date': end_date_str,
                        'trained_at': str(datetime.now()),
                        'data_points': len(data_actual),
                        'date_range_days': (data_actual.index.max() - data_actual.index.min()).days
                    }, fp, indent=4)

                return True

            except Exception as e:
                print(f"Error during model training: {e}")
                raise

        else:
            print(f"Model and metadata for {target_col} already exists")
            return True

    except Exception as e:
        print(f"Training error: {e}")
        return False


def load_forecast_model(model_path):
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    else:
        print(f"No model found at {model_path}")
        return None


def plot_graph(data, target_col):
    try:

        # Create Plotly figure
        fig = go.Figure()
        try:
            if (data['date'].dt.time != pd.to_datetime('00:00:00').time()).any():
                # If any row has time info other than midnight, include hours
                data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # Otherwise, keep only the date
                data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(e)
        # Actual Data Line
        fig.add_trace(go.Scatter(
            x=data['date'], y=data["forecasted_value"],
            mode='lines+markers', name='Forecast',
            line=dict(color='blue'), marker=dict(symbol='circle')
        ))

        # Forecast Data Line
        # fig.add_trace(go.Scatter(
        #     x=forecast_dates, y=forecast_values,
        #     mode='lines+markers', name='Forecast',
        #     line=dict(color='orange', dash='dash'), marker=dict(symbol='x')
        # ))

        # Layout Settings
        fig.update_layout(
            title=f'Forecasted {target_col} values Over Time',
            xaxis_title='Date',
            yaxis_title='Values',
            xaxis=dict(tickangle=-45, type='category', tickformat='%Y-%m-%d'),
            template="plotly_white",
            width=1000, height=600
        )

        # Convert figure to JSON for frontend rendering
        # Removed fig.show() to prevent opening new browser tab
        return make_serializable(fig.to_json())

    except Exception as e:
        print(e)
        return str(e)


def arima_forecast(model, periods, freq,target_col):
    freq_map = {
        'hours': 'H',
        'days': 'D',
        'weeks': 'W',
        'months': 'M',
        'years': 'YS'
    }
    data_path= os.path.join('models', 'Arima', target_col, target_col + '_results.json')
    print(f"Loading results from: {data_path}")

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)  # Now data is a dictionary
    except FileNotFoundError:
        print(f"Results file not found: {data_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {data_path}")
        return None

    end_date = data.get('end_date')
    
    # Robust handling of end_date - handle NaT, None, or invalid dates
    start_date = None
    try:
        if end_date and end_date != 'NaT' and end_date != 'None':
            parsed_end = pd.to_datetime(end_date, errors='coerce')
            if pd.notna(parsed_end):
                try:
                    frequency = freq_map.get(data.get('data_freq', '').lower())
                    if frequency:
                        start_date = parsed_end + pd.tseries.frequencies.to_offset(frequency)
                    else:
                        start_date = parsed_end + pd.tseries.frequencies.to_offset(freq)
                except:
                    start_date = parsed_end + pd.tseries.frequencies.to_offset(freq)
    except Exception as e:
        print(f"Error parsing end_date '{end_date}': {e}")
    
    # Fallback to current time if start_date couldn't be determined
    if start_date is None or pd.isna(start_date):
        print(f"WARNING: Could not parse end_date '{end_date}', using current time as forecast start")
        start_date = pd.Timestamp.now().normalize()  # Start from today midnight
                 
    future = pd.date_range(start=start_date, periods=periods, freq=freq)
    if (freq=="H") and ((future.hour != 0).any() or (future.minute != 0).any() or (future.second != 0).any()):
        # Include time in formatting
        future = future.strftime('%Y-%m-%d %H:%M:%S').tolist()
    else:
        # Only date
        future = future.strftime('%Y-%m-%d').tolist()
    

    model_type = str(type(model))
    print(f"Detected model type: {model_type}")
    try:
        if 'Prophet' in model_type:
            future_df = pd.DataFrame({'ds': future})
            # Prophet expects 'ds' column and returns 'yhat'
            forecast = model.predict(future_df)
            if 'yhat' in forecast.columns:
                forecast = forecast[['ds', 'yhat']]
                forecast['yhat'] = forecast['yhat'].round(2)
                forecast.columns = ['date', 'forecasted_value']
                return forecast
            else:
                raise ValueError("Prophet output missing 'yhat' column.")

        else:
            # ARIMA, XGBoost, RandomForest — Expect direct prediction
            future_df = pd.DataFrame({'date': future})  # Rename here
            if 'ARIMA' in model_type:
                forecast = model.forecast(steps=future_df.shape[0])
                future_df['forecasted_value'] = np.round(forecast.values, 2)
            else:
                start_idx = model.last_index_ + 1  # get from model
                end_idx = start_idx + len(future_df)
                X_future = np.arange(start_idx, end_idx).reshape(-1, 1)
                print(X_future)
                forecast = model.predict(X_future)
                future_df['forecasted_value'] = np.round(forecast, 2)

            return future_df[['date', 'forecasted_value']]

    except Exception as e:
        print(f"Prediction Error: {e}")


def check_data_frequency(data_actual):
    """
    Enhanced frequency detection with multiple fallback methods
    """
    try:
        # Method 1: Use pandas infer_freq
        inferred_freq = pd.infer_freq(data_actual.index)
        if inferred_freq is not None:
            return inferred_freq
        
        print("pandas infer_freq failed, trying manual detection...")
        
        # Method 2: Calculate most common time difference
        time_diffs = data_actual.index.to_series().diff().dropna()
        
        if len(time_diffs) == 0:
            return "Single data point"
        
        # Get the most common time difference
        most_common_diff = time_diffs.mode()
        
        if len(most_common_diff) > 0:
            diff_seconds = most_common_diff.iloc[0].total_seconds()
            
            # Map common intervals to frequency strings
            freq_mapping = {
                60: '1min',           # 1 minute
                300: '5min',          # 5 minutes
                600: '10min',         # 10 minutes
                900: '15min',         # 15 minutes
                1800: '30min',        # 30 minutes
                3600: '1H',           # 1 hour
                86400: '1D',          # 1 day
                604800: '1W',         # 1 week
                2629746: '1M',        # 1 month (average)
                31556952: '1Y'        # 1 year (average)
            }
            
            if diff_seconds in freq_mapping:
                return freq_mapping[diff_seconds]
            
            # For other intervals, create a descriptive string
            if diff_seconds < 3600:
                minutes = int(diff_seconds / 60)
                return f"{minutes}min"
            elif diff_seconds < 86400:
                hours = int(diff_seconds / 3600)
                return f"{hours}H"
            elif diff_seconds < 604800:
                days = int(diff_seconds / 86400)
                return f"{days}D"
            else:
                return "Irregular"
        
        # Method 3: Analyze the time span and data points
        total_duration = data_actual.index.max() - data_actual.index.min()
        total_points = len(data_actual)
        
        if total_points > 1:
            avg_interval_seconds = total_duration.total_seconds() / (total_points - 1)
            
            if avg_interval_seconds < 3600:
                minutes = int(avg_interval_seconds / 60)
                return f"~{minutes}min"
            elif avg_interval_seconds < 86400:
                hours = int(avg_interval_seconds / 3600)
                return f"~{hours}H"
            else:
                days = int(avg_interval_seconds / 86400)
                return f"~{days}D"
        
        return "Irregular frequency"
        
    except Exception as e:
        print(f"Error in frequency detection: {e}")
        return "Unknown frequency"



def train_models(df, target_col):
    frequencies = ['hours', 'days', 'weeks', 'months','years']
    try:
        for freq in frequencies:
            try:
                print(f"\nTraining {freq} models...")

                # Resample the data for each frequency
                resampled_df = resample_data(df, freq)
                print(f"Resampled data for {freq}:\n", resampled_df.head())
                
                train, test = train_test_split(resampled_df, test_size=0.2, shuffle=False)

                trend = detect_trend(train)
                print("Trend is", trend)
                seasonality = detect_seasonality(train)
                print("Seasonality is", seasonality)

                best_model = None
                best_error = float('inf')
                best_model_name = ""
                scenario = ""

                # Scenario 1: Trend only
                if trend and not seasonality:
                    scenario = "Trend only"
                    arima_model, arima_error = train_arima(train, test)
                    xgb_model, xgb_error = train_xgboost(train, test)

                    if arima_error < xgb_error:
                        best_model, best_error = arima_model, arima_error
                        best_model_name = "ARIMA"
                    else:
                        best_model, best_error = xgb_model, xgb_error
                        best_model_name = "XGBoost"

                # Scenario 2: Seasonality only
                elif seasonality and not trend:  # Added 'elif' for better logic
                    scenario = "Seasonality only"
                    prophet_model, prophet_error = train_prophet(train, test)
                    arima_model, arima_error = train_arima(train, test)

                    if prophet_error < arima_error:
                        best_model, best_error = prophet_model, prophet_error
                        best_model_name = "Prophet"
                    else:
                        best_model, best_error = arima_model, arima_error
                        best_model_name = "ARIMA"

                # Scenario 3: Trend + Seasonality
                elif trend and seasonality:  # Added 'elif' for better logic
                    scenario = "Trend + Seasonality"
                    prophet_model, prophet_error = train_prophet(train, test)
                    arima_model, arima_error = train_arima(train, test)

                    min_error = min(prophet_error, arima_error)
                    if min_error == prophet_error:
                        best_model, best_error = prophet_model, prophet_error
                        best_model_name = "Prophet"
                    elif min_error == arima_error:
                        best_model, best_error = arima_model, arima_error
                        best_model_name = "ARIMA"

                # Scenario 4: No Trend or Seasonality
                else:  # Changed to 'else' since it's the remaining case
                    scenario = "No trend or seasonality"
                    xgb_model, xgb_error = train_xgboost(train, test)
                    rf_model, rf_error = train_randomforest(train, test)

                    if xgb_error < rf_error:
                        best_model, best_error = xgb_model, xgb_error
                        best_model_name = "XGBoost"
                    else:
                        best_model, best_error = rf_model, rf_error
                        best_model_name = "RandomForest"

                # Save the best model
                if best_model:
                    model_dir = f'models/Arima/{target_col}/{freq}'
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, 'best_model.pkl')
                    save_best_model(best_model, model_path)

                    # Save Scenario with Model Name
                    with open(f'scenario_{freq}.json', 'w') as f:
                        json.dump({"scenario": scenario, "model_name": best_model_name}, f)

                    print(f"\n{freq.capitalize()} Training complete. Scenario: {scenario}, Model: {best_model_name}")
                else:
                    print(f"No model could be trained for {freq} frequency")
            except Exception as e:
                print(e)
        return True
    except Exception as e:
        print(f"Error during model training: {e}")
        return False


def train_arima(train, test):
    print("Arima Started training")
    try:
        # Fit ARIMA model
        model = ARIMA(train['value'], order=(1, 1, 1)).fit()
        pred = model.predict(start=test.index[0], end=test.index[-1])

        # Calculate RMSE (handle different sklearn versions)
        try:
            # For newer sklearn versions
            error = mean_squared_error(test['value'], pred, squared=False)
        except TypeError:
            # For older sklearn versions
            error = np.sqrt(mean_squared_error(test['value'], pred))

        print("ARIMA training completed successfully")
        return model, error
    except Exception as e:
        print(f"Error in ARIMA training: {str(e)}")
        raise


# Train Prophet Model
def train_prophet(train, test):
    print("Training Prophet...")
    prophet_df = train.reset_index().rename(columns={'datetime': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(prophet_df)

    future = pd.DataFrame({'ds': test.index})
    forecast = model.predict(future)

    # Calculate RMSE (compatible with all sklearn versions)
    mse = mean_squared_error(test['value'], forecast['yhat'])
    error = np.sqrt(mse)  # Calculate RMSE manually

    print("Returning the parameters from the Prophet model")
    return model, error


# Train XGBoost Model
def train_xgboost(train, test):
    print("Training XGBoost...")
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['value'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    model.last_index_ = len(train) + len(test) - 1
    pred = model.predict(X_test)

    # Calculate RMSE (compatible with all sklearn versions)
    mse = mean_squared_error(test['value'], pred)
    error = np.sqrt(mse)  # Calculate RMSE manually

    return model, error


# Train RandomForest Model
def train_randomforest(train, test):
    print("Training RandomForest...")
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['value'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    model.last_index_ = len(train) + len(test) - 1
    pred = model.predict(X_test)

    # Calculate RMSE (compatible with all sklearn versions)
    mse = mean_squared_error(test['value'], pred)
    error = np.sqrt(mse)  # Calculate RMSE manually

    return model, error


# Save the Best Model
def save_best_model(model, model_path):
    joblib.dump(model, model_path)


def resample_data(df, freq):
    print(f"Resampling data to {freq} frequency")
    if freq == 'hours':
        return df.resample('h').mean().ffill()
    elif freq == 'days':
        return df.resample('D').mean().ffill()
    elif freq == 'weeks':
        return df.resample('W').mean().ffill()
    elif freq == 'months':
        return df.resample('M').mean().ffill()
    elif freq == 'years':
        return df.resample('YE').mean().ffill()
    else:
        raise ValueError("Unsupported frequency")


# Check Trend using Augmented Dickey-Fuller Test
def detect_trend(df):
    print('Detecting Trend...')
    result = adfuller(df['value'])
    p_value = result[1]
    return p_value > 0.05  # If p-value > 0.05 → Trend exists


# Check Seasonality using autocorrelation
def detect_seasonality(df):
    print('Detecting Seasonality...')
    autocorr = df['value'].autocorr(lag=1)
    return abs(autocorr) > 0.3  # If autocorr > 0.3 → Seasonality exists

def is_categorical_target(series):
    """Better detection for categorical vs numerical targets"""
    # Check data type first
    if series.dtype == 'object' or series.dtype.name == 'category':
        return True
    
    # For numeric types, check if it's likely categorical
    unique_count = series.nunique()
    total_count = len(series)
    
    # If unique values are less than 10% of total or max 20 unique values
    if unique_count <= max(10, total_count * 0.1) and unique_count <= 20:
        return True
    
    return False


def random_forest(data, target_column):
    try:
        model_dir = os.path.join("models", "rf", target_column)
        deployment_path = os.path.join(model_dir, 'deployment.json')

        if not os.path.exists(deployment_path):
            os.makedirs(model_dir, exist_ok=True)

            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Better categorical detection for target
            is_classification = is_categorical_target(y)

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # For categorical targets, ensure proper encoding
            label_encoder = None
            label_mapping = None
            if is_classification and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                # Convert numpy arrays to lists for JSON serialization
                label_mapping = {str(cls): int(label_encoder.transform([cls])[0]) 
                               for cls in label_encoder.classes_}

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if is_classification:
                model_type = 'Classification'
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'  # Handle imbalanced classes
                )
            else:
                model_type = 'Regression'
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            cv = min(5, len(X_test))

            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv)
            print(f"Model Performance (CV): {scores.mean():.4f} ± {scores.std():.4f}")

            # Get predictions on test set for detailed metrics
            predictions = pipeline.predict(X_test)

            # Initialize metric variables
            accuracy = None
            precision = None
            recall = None
            f1 = None
            mae = None
            rmse = None
            r2 = None

            if is_classification:
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            else:
                r2 = r2_score(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))

            # Calculate baseline comparison
            def calculate_baseline_comparison(y_true, y_pred, is_classification):
                try:
                    if is_classification:
                        if len(set(y_true)) > 1:
                            baseline_accuracy = max(np.bincount(y_true)) / len(y_true)
                            model_accuracy = accuracy_score(y_true, y_pred)
                            improvement = ((model_accuracy - baseline_accuracy) / baseline_accuracy) * 100
                            return f"{round(improvement, 2)}% better than baseline"
                        else:
                            return "Baseline comparison unavailable (only one class)"
                    else:
                        baseline_mae = mean_absolute_error(y_true, [np.mean(y_true)] * len(y_true))
                        model_mae = mean_absolute_error(y_true, y_pred)
                        improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
                        return f"{round(improvement, 1)}% better than mean baseline"
                except Exception:
                    return "Baseline comparison unavailable"

            baseline_comparison = calculate_baseline_comparison(y_test, predictions, is_classification)

            # Compose model metrics dictionary according to task
            if is_classification:
                model_metrics = {
                    "accuracy": round(accuracy * 100, 1) if accuracy is not None else 'N/A',
                    "precision": round(precision * 100, 1) if precision is not None else None,
                    "recall": round(recall * 100, 1) if recall is not None else None,
                    "f1_score": round(f1 * 100, 1) if f1 is not None else None,
                }
            else:
                model_metrics = {
                    "r2_score": round(r2 * 100, 1) if r2 is not None else None,
                    "mae": round(mae, 2) if mae is not None else None,
                    "rmse": round(rmse, 2) if rmse is not None else None,
                }

            # Create comprehensive model statistics
            model_stats = {
                "model_type": model_type,
                "total_samples": int(len(y_test)),  # Ensure it's a regular int
                "correct_predictions": int(sum(y_test == predictions)) if is_classification else None,
                "cross_val_mean": round(float(scores.mean()), 4),  # Convert numpy float to Python float
                "cross_val_std": round(float(scores.std()), 4),    # Convert numpy float to Python float
                "baseline_comparison": baseline_comparison,
                "metrics": model_metrics,
            }

            # Save the pipeline
            joblib.dump(pipeline, os.path.join(model_dir, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join(model_dir, "pipeline.pkl")}')

            # Save label encoder if used
            if label_encoder is not None:
                joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

            # Get sample row data for form pre-filling
            # Convert the first row of X_train to a dictionary, handling different data types
            row_data = {}
            for col in X_train.columns:
                val = X_train.iloc[0][col]
                # Convert numpy types to Python native types for JSON serialization
                if pd.isna(val):
                    row_data[col] = None
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    row_data[col] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    row_data[col] = float(val)
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_data[col] = str(val)
                else:
                    row_data[col] = str(val)

            # Save enhanced deployment information with statistics
            # Convert all data to JSON-serializable formats
            deployment_data = {
                "columns": [str(col) for col in X_train.columns],  # Ensure strings
                "model_type": str(model_type),
                "target_column": str(target_column),
                "stats": model_stats,
                "feature_names": [str(col) for col in X.columns],  # Ensure strings
                "categorical_features": [str(col) for col in categorical_cols],  # Ensure strings
                "numerical_features": [str(col) for col in numerical_cols],     # Ensure strings
                "label_mapping": label_mapping,
                "is_classification": bool(is_classification),  # Ensure it's a regular bool
                "row_data": row_data  # Sample row for form pre-filling
            }

            # Write JSON with proper encoding to handle special characters
            with open(deployment_path, "w", encoding='utf-8') as fp:
                json.dump(deployment_data, fp, indent=4)

            return model_stats, [str(col) for col in X_train.columns], row_data
        else:
            # Load existing model data and statistics with proper encoding
            try:
                with open(deployment_path, "r", encoding='utf-8') as fp:
                    deployment_data = json.load(fp)
                row_data = deployment_data.get('row_data', {})
                return deployment_data.get('stats', {}), deployment_data.get('columns', []), row_data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error reading deployment file: {e}")
                # If the file is corrupted, delete it and retrain
                os.remove(deployment_path)
                print("Corrupted deployment file deleted. Retraining model...")
                # Recursively call the function to retrain
                return random_forest(data, target_column)

    except Exception as e:
        print(f"Error in random_forest: {e}")
        return False, [], {}

# 9.Data scout api----------------------Data scout api for generating the data------------- 9
@app.post("/api/data_scout")
async def create_data_with_data_scout(
        prompt: str = Form(...),
        data_type: str = Form(...),
) -> JSONResponse:
    if not prompt or not data_type:
        raise HTTPException(
            status_code=400,
            detail="Both prompt and type are required"
        )

    try:
        if data_type == "excel":
            agent1 = DataScout_agent()
            result = agent1.invoke({"input": prompt})
            print("result is", result)

            if isinstance(result, dict) and "output" in result:
                output = result["output"]
                print("output is", output)
                
                if hasattr(output, 'to_dict'):
                    # Use pandas' built-in JSON serialization with date formatting
                    json_str = output.to_json(orient='records', date_format='iso')
                    data_dict = json.loads(json_str)
                    
                    return JSONResponse(content={
                        "message": "Data generated successfully",
                        "data": data_dict,
                        "shape": list(output.shape),
                        "columns": output.columns.tolist()
                    })
                else:
                    return JSONResponse(content={
                        "message": "Output is not a DataFrame",
                        "output_type": str(type(output))
                    })

        elif data_type == "pdf":
            raw_prompt = prompt
            sections = extract_sections_tool(raw_prompt)
            number_of_pages = extract_num_pages_tool(raw_prompt)

            result = pdf_generator_tool(raw_prompt, sections, number_of_pages)

            if isinstance(result, dict) and "title" in result and "sections" in result:
                return JSONResponse(content=result)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate structured PDF content"
            )

        elif data_type == "image":
            agent1 = ImageGen_agent()
            result = agent1.run(prompt)

            # Handle different return formats
            if isinstance(result, list):
                images_data = []
                for item in result:
                    if isinstance(item, dict) and 'image_path' in item:
                        try:
                            with open(item["image_path"], "rb") as img_file:
                                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                                img_data = {
                                    "path": item["image_path"],
                                    "base64": base64_data,
                                    "thumbnail": item.get("thumbnail_path")
                                }
                                images_data.append(img_data)
                        except Exception as e:
                            print(f"Error processing image {item['image_path']}: {e}")
                            continue
                    elif isinstance(item, str):
                        try:
                            with open(item, "rb") as img_file:
                                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                                images_data.append({
                                    "path": item,
                                    "base64": base64_data
                                })
                        except Exception as e:
                            print(f"Error processing image {item}: {e}")
                            continue

                if images_data:
                    return JSONResponse(content={"images": images_data})
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process generated images"
                )

            elif isinstance(result, str) and result.strip():
                try:
                    with open(result.strip(), "rb") as img_file:
                        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                        return JSONResponse(content={
                            "images": [{
                                "path": result.strip(),
                                "base64": base64_data
                            }]
                        })
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to process image: {str(e)}"
                    )

            raise HTTPException(
                status_code=500,
                detail="Failed to generate images - unexpected result format"
            )

        raise HTTPException(
            status_code=400,
            detail="Invalid data type (must be excel, pdf, or image)"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# 17.Data preprocess api----------------------Data preprocessing for the generation of the statistical analysis.-----
@app.get("/api/processing_for_dashboard")
async def perform_statistical_analysis() -> JSONResponse:
    """
    Perform comprehensive statistical analysis on the dataset

    Returns:
        JSONResponse: Contains various statistical metrics and data characteristics
    """
    try:
        # Check if data file exists
        if not os.path.exists('data.csv'):
            raise HTTPException(
                status_code=404,
                detail="Data file not found. Please upload file first."
            )

        # Load and preprocess data
        df = pd.read_csv('data.csv')
        print(df.head(5))
        df = updatedtypes(df)

        print("coming from function",df.dtypes)

        if df.shape[0] == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset contains no data"
            )

        # Basic statistics
        nullvalues = df.isnull().sum().to_dict()
        parameters = list(nullvalues.keys())
        count = list(nullvalues.values())
        total_missing = df.isnull().sum().sum()
        nor = df.shape[0]
        nof = df.shape[1]

        # Remove single value columns
        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]

        # Initialize variables for analysis
        timestamp = 'N'
        boolean = 'N'
        categorical_vars = []
        boolean_vars = []
        numeric_vars = {}
        datetime_vars = []
        text_data = []
        td = None
        stationary = "NA"
        duplicate_records = df[df.duplicated(keep='first')].shape[0]

        # Column type analysis
        for col, dtype in df.dtypes.items():
            dtype_str = str(dtype)
            print(f"Processing column: {col}, dtype: {dtype_str}")
            if dtype_str in ["float64", "int64"]:
                numeric_vars[col] = df[col].describe().to_dict()
            elif dtype_str == "object" and col not in ['Remark']:
                categorical_vars.append({col: df[col].nunique()})
            elif any(dt_format in dtype_str for dt_format in ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns,']):
                if col.upper() in ['DATE', "TIME", "DATE_TIME","START", "END"]:
                    td = col
                datetime_vars.append(col)
            elif dtype_str == "bool":
                boolean_vars.append(col)

        # Additional analyses
        istextdata = 'Y' if len(text_data) > 0 else 'N'
        if len(datetime_vars) > 0:
            timestamp = 'Y'
        if td:
            stationary = adf_test(df, td)
        catvalues = [{'Parameter': list(data.keys())[0], 'Count': list(data.values())[0]} for data in
                        categorical_vars]
        sentiment = checkSentiment(df, categorical_vars)
        if len(catvalues) > 0:
            catdf = pd.DataFrame(catvalues)
        else:
            catdf = pd.DataFrame()
        if len(numeric_vars) > 0:
            numdf = pd.DataFrame(numeric_vars).T
            numdf['ColumnName'] = numdf.index
        else:
            numdf = pd.DataFrame()
        if len(boolean_vars) > 0:
            boolean = 'Y'

        # Prepare missing values data
        missingvalue = pd.DataFrame({
            "Parameters": parameters,
            'Missing Value Count': count
        })

    
        # **** Get the preview data (first 50 rows) ****
        preview = df.head(50)
        preview_data = json.loads(preview.to_json(orient='records'))

        # Build response
        response = {
            'nof_rows': str(nor),
            'nof_columns': str(nof),
            'timestamp': timestamp,
            'Preview_data': preview_data,
            "single_value_columns": ",".join(single_value_columns) if single_value_columns else "NA",
            "sentiment": sentiment,
            "stationary": stationary,
            'catdf': json.loads(catdf.to_json(orient='records')) if not catdf.empty else [],
            'missing_data': str(total_missing),
            'numdf': json.loads(numdf.to_json(orient='records')) if not numdf.empty else [],
            'boolean': boolean,
            'missingvalue': json.loads(missingvalue.to_json(orient='records')),
            'textdata': istextdata,
            'duplicate_records': str(duplicate_records)
        }

        return JSONResponse(content=response)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Data file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Statistical analysis failed: {str(e)}"
        )


def updatedtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                pd.to_datetime(df[col],utc=True)
                # df.drop(col, axis=1, inplace=True)
                print(df.columns)
            except Exception as e:
                pass
    return df


def adf_test(df, kpi):
    df_t = df.set_index(kpi)

    for col in df_t.columns:
        # Check if the column name is not in the specified list and is numeric
        if col.upper() not in ['DATE', 'TIME', 'DATE_TIME'] and pd.api.types.is_numeric_dtype(df_t[col]):
            if df_t[col].nunique() > 1:
                dftest = adfuller(df_t[col], autolag='AIC')
                statistic_value = dftest[0]
                p_value = dftest[1]
                if (p_value > 0.5) and all([statistic_value > j for j in dftest[4].values()]):
                    return "Y"
            else:
                break
    return "N"


def checkSentiment(df, categorical):
    sentiment = 'N'
    for i in categorical:
        # print([j for j in df[i]])
        data = ' '.join([str(j) for j in df[list(i.keys())[0]]]).upper()
        if ('GOOD' in data) | ('BAD' in data) | ('Better' in data):
            sentiment = "Y"
    return sentiment


# 10.Extend synthetic data generation api
class ContentType(Enum):
    DATA = "data"
    IMAGES = "images"
    PDF = "pdf"


@app.post("/api/generate_synthetic_data", response_model=None)
async def generate_synthetic_content(
        files: List[UploadFile],
        user_prompt: str = Form(...)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Unified API for generating synthetic content (data, images, or PDFs)

    Args:
        files: List of uploaded files (CSV/Excel for data, images for image generation, PDF for document generation)
        user_prompt: Instructions for content generation

    Returns:
        StreamingResponse for CSV data or JSONResponse for images/PDFs
    """
    print("[DEBUG] Entered generate_synthetic_content")

    # Validate API key using llm_config
    from llm_config import get_api_key
    openai_api_key = get_api_key(user_email=None)  # Can be parameterized if user email is available
    if not openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not configured"
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded"
        )

    print(f"[DEBUG] Number of files uploaded: {len(files)}")
    print(f"[DEBUG] User prompt: {user_prompt}")

    # Determine content type based on uploaded files
    content_type = determine_content_type(files)
    print(f"[DEBUG] Detected content type: {content_type}")

    try:
        if content_type == ContentType.DATA:
            return await handle_data_generation(files[0], user_prompt, openai_api_key)
        elif content_type == ContentType.IMAGES:
            return await handle_image_generation(files, user_prompt, openai_api_key)
        elif content_type == ContentType.PDF:
            return await handle_pdf_generation(files[0], user_prompt)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type combination"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content generation failed: {str(e)}"
        )


def determine_content_type(files: List[UploadFile]) -> ContentType:
    """Determine the type of content to generate based on uploaded files"""
    if len(files) == 1:
        file = files[0]
        extension = os.path.splitext(file.filename)[1].lower()

        if extension in ['.csv', '.xlsx']:
            return ContentType.DATA
        elif extension == '.pdf':
            return ContentType.PDF
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return ContentType.IMAGES

    # Multiple files - check if all are images
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    if all(os.path.splitext(f.filename)[1].lower() in image_extensions for f in files):
        return ContentType.IMAGES

    raise HTTPException(
        status_code=400,
        detail="Invalid file combination. Upload CSV/Excel for data, images for image generation, or PDF for document generation."
    )


async def handle_data_generation(
        file: UploadFile,
        user_prompt: str,
        openai_api_key: str
) -> JSONResponse:
    """Handle synthetic data generation"""
    print("[DEBUG] Processing data generation request")
    temp_file_name = None

    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        print(f"[DEBUG] File extension: {file_extension}")

        # Read input file
        if file_extension == ".xlsx":
            df = pd.read_excel(file.file)
        elif file_extension == ".csv":
            df = pd.read_csv(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format for data generation. Please upload Excel or CSV."
            )

        print(f"[DEBUG] Original DataFrame shape: {df.shape}")

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file_name = temp_file.name
            if file_extension == ".xlsx":
                df.to_excel(temp_file_name, index=False)
            else:
                df.to_csv(temp_file_name, index=False)
            print(f"[DEBUG] Temporary file saved at: {temp_file_name}")

        # Process user prompt
        num_rows = extract_num_rows_from_prompt1(user_prompt, openai_api_key)
        print(f"[DEBUG] Extracted number of rows: {num_rows}")

        if num_rows is None:
            raise HTTPException(
                status_code=400,
                detail="Could not determine number of rows from prompt"
            )

        if num_rows > 100_000:
            raise HTTPException(
                status_code=400,
                detail="Too many rows requested. Limit is 100,000."
            )

        # Generate synthetic data
        datetime_col = infer_datetime_column(df)
        print(f"[DEBUG] Inferred datetime column: {datetime_col}")

        generated_df = generate_synthetic_data(
            openai_api_key,
            temp_file_name,
            num_rows,
            datetime_col=datetime_col
        )
        print(f"[DEBUG] Generated synthetic data shape: {generated_df.shape}")

        # Combine and return results
        combined_df = pd.concat([df, generated_df], ignore_index=True)
        print(f"[DEBUG] Combined DataFrame shape: {combined_df.shape}")

        # Convert DataFrame to JSON-serializable format
        # Handle datetime columns and other non-serializable types
        def convert_to_serializable(obj):
            """Convert non-serializable objects to serializable format"""
            if pd.isna(obj):
                return None
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Convert DataFrame to records with proper serialization
        data_records = []
        for _, row in combined_df.iterrows():
            record = {}
            for col, value in row.items():
                record[col] = convert_to_serializable(value)
            data_records.append(record)

        result_data = {
            "status": "success",
            "original_rows": len(df),
            "generated_rows": len(generated_df),
            "total_rows": len(combined_df),
            "data": data_records
        }

        print("[DEBUG] Successfully generated synthetic data response.")
        return JSONResponse(content=result_data)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[ERROR] Error in data generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content generation failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)
            print(f"[DEBUG] Temporary file {temp_file_name} deleted.")


async def handle_image_generation(
        images: List[UploadFile],
        user_prompt: str,
        openai_api_key: str
) -> JSONResponse:
    """Handle synthetic image generation"""
    print("[DEBUG] Processing image generation request")

    # Extract number of images from prompt
    requested_count = extract_num_images_from_prompt(user_prompt, openai_api_key)
    if requested_count is None or requested_count <= 0:
        raise HTTPException(
            status_code=400,
            detail="Could not extract valid image count from prompt. Include a clear number like 'generate 20 images'"
        )

    if requested_count > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum limit is 100 images per request"
        )

    # Ensure images folder exists
    folder_path = ensure_images_folder()

    # Analyze all images for comprehensive style
    print("Analyzing uploaded images for style...")
    comprehensive_style = analyze_all_images_for_style(images, openai_api_key)
    print(f"Extracted style: {comprehensive_style}")

    # Get starting index for new images
    start_index = get_next_image_index(folder_path)
    total_generated = 0
    generated_images_info = []

    while total_generated < requested_count:
        remaining = requested_count - total_generated
        batch_size = min(5, remaining)

        # Construct prompt with style guidance
        full_prompt = (
            f"Generate {batch_size} high-quality, stylistically consistent images.\n"
            f"Visual style from references: {comprehensive_style}\n"
            f"Avoid redundant compositions.\n"
            f"Create unique subjects while maintaining style.\n"
            f"User instructions: {user_prompt.strip()}"
        )

        print(f"Generating batch of {batch_size} images...")
        image_urls = generate_images(client, full_prompt, batch_size)

        for url in image_urls:
            try:
                # Download and save image
                response = requests.get(url)
                response.raise_for_status()

                filename = f"synthetic_{start_index + total_generated}.png"
                file_path = Path(folder_path) / filename

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                # Convert to base64
                with open(file_path, 'rb') as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')

                generated_images_info.append({
                    'filename': filename,
                    'path': str(file_path),
                    'base64': base64_image,
                    'index': start_index + total_generated
                })

                total_generated += 1
                print(f"Generated {total_generated}/{requested_count}")

            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

    return JSONResponse(
        content={
            'success': True,
            'content_type': 'images',
            'message': f'Generated {total_generated} images',
            'total_generated': total_generated,
            'images': generated_images_info
        }
    )


async def handle_pdf_generation(
        pdf: UploadFile,
        user_prompt: str
) -> JSONResponse:
    """Handle synthetic PDF generation"""
    print("[DEBUG] Processing PDF generation request")

    # Validate PDF upload
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted for document generation."
        )

    # Extract target pages from prompt
    target_pages = extract_pdf_prompt_semantics(user_prompt)
    if not target_pages:
        raise HTTPException(
            status_code=400,
            detail="Could not determine target page count from prompt"
        )

    # Process uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await pdf.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        # Extract text from PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        current_pages = len(documents)
        full_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)

    # Generate extended document
    final_document = generate_complete_document(
        full_text,
        current_pages,
        target_pages
    )

    # Check for generation errors
    if isinstance(final_document, dict) and final_document.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Document generation failed: {final_document.get('error')}"
        )

    return JSONResponse(
        content={
            'success': True,
            'content_type': 'pdf',
            'message': f'Generated extended PDF document',
            'structured_document': final_document,
            'original_pages': current_pages,
            'target_pages': target_pages
        },
        status_code=200
    )


def infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    print(df.columns)
    
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            # Skip if column has too many nulls
            non_null_ratio = df[col].notna().mean()
            if non_null_ratio < 0.5:  # Skip if more than 50% nulls
                continue
            
            # Get a sample of non-null values for testing
            sample_values = df[col].dropna().head(min(100, len(df[col].dropna())))
            
            # Try multiple datetime parsing approaches
            parsing_methods = [
                # Method 1: Formats matching your data (DD-MM-YYYY HH:MM)
                lambda x: pd.to_datetime(x, format='%d-%m-%Y %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d-%m-%Y %H:%M:%S', errors='coerce'),
                
                # Method 2: Similar formats with different separators
                lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d.%m.%Y %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d.%m.%Y %H:%M:%S', errors='coerce'),
                
                # Method 3: US format variations
                lambda x: pd.to_datetime(x, format='%m-%d-%Y %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%m-%d-%Y %H:%M:%S', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M:%S', errors='coerce'),
                
                # Method 4: ISO-like formats
                lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y/%m/%d %H:%M:%S', errors='coerce'),
                
                # Method 5: Date-only formats
                lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d.%m.%Y', errors='coerce'),
                
                # Method 6: With milliseconds
                lambda x: pd.to_datetime(x, format='%d-%m-%Y %H:%M:%S.%f', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'),
                
                # Method 7: pandas infer_datetime_format
                lambda x: pd.to_datetime(x, infer_datetime_format=True, errors='coerce'),
                
                # Method 8: Flexible parsing (most permissive)
                lambda x: pd.to_datetime(x, errors='coerce', dayfirst=True),  # European format first
                lambda x: pd.to_datetime(x, errors='coerce', dayfirst=False), # US format
            ]
            
            for method in parsing_methods:
                try:
                    parsed_dates = method(sample_values)
                    
                    # Check if a significant portion was successfully parsed
                    success_ratio = parsed_dates.notna().mean()
                    
                    if success_ratio >= 0.8:  # At least 80% successfully parsed
                        # Additional validation: check if parsed dates seem reasonable
                        valid_dates = parsed_dates.dropna()
                        if len(valid_dates) > 0:
                            # Check date range reasonableness (between 1900 and 2100)
                            min_date = valid_dates.min()
                            max_date = valid_dates.max()
                            
                            if (min_date.year >= 1900 and max_date.year <= 2100):
                                return col
                
                except Exception:
                    continue
    
    return None


# Semantic ai related number of rows detection
def extract_num_rows_from_prompt1(prompt: str, api_key: str, user_email: str = None) -> Optional[int]:
    """
    Extracts number of rows to generate using LLM-based semantic parsing only.
    
    NOTE: If user_email is provided, use get_llm_config(user_email) to get user-specific
    API key and model. Otherwise, use the provided api_key parameter for backward compatibility.
    """
    # Use user-specific config if email provided, otherwise use provided api_key
    if user_email:
        config = get_llm_config(user_email)
        api_key = config["api_key"]
        model = config["model"]
    else:
        model = get_llm_config(None)["model"]  # use default from llm_config

    llm = ChatOpenAI(model=model, openai_api_key=api_key)
    messages = [
        SystemMessage(content="You extract the number of rows to generate from user input. Return only the integer."),
        HumanMessage(content=prompt)
    ]
    try:
        response = llm.invoke(messages)
        match = re.search(r'\d+', response.content)
        return int(match.group()) if match else None
    except Exception as e:
        print(f"[ERROR] Semantic extraction failed: {e}")
        return None


GENERATED_IMAGES_FOLDER = "generated_synthetic_images"


def ensure_images_folder():
    """Create the images folder if it doesn't exist"""
    if not os.path.exists(GENERATED_IMAGES_FOLDER):
        os.makedirs(GENERATED_IMAGES_FOLDER)
    return GENERATED_IMAGES_FOLDER


def get_next_image_index(folder_path):
    """Get the next image index based on existing files in the folder"""
    existing_files = [f for f in os.listdir(folder_path) if f.startswith('synthetic_') and f.endswith('.png')]
    if not existing_files:
        return 1

    # Extract numbers from filenames and find the maximum
    indices = []
    for filename in existing_files:
        try:
            # Extract number from filename like "synthetic_123.png"
            number = int(filename.replace('synthetic_', '').replace('.png', ''))
            indices.append(number)
        except ValueError:
            continue

    return max(indices) + 1 if indices else 1


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None


def analyze_all_images_for_style(uploaded_images, openai_api_key):
    """
    Analyze all uploaded images to extract a comprehensive style description.
    """
    individual_descriptions = []

    # Get description for each uploaded image
    for i, image_file in enumerate(uploaded_images):
        try:
            # Read the file content from UploadFile
            image_file.file.seek(0)  # Reset file pointer to beginning
            image_content = image_file.file.read()

            # Create a BytesIO object from the content
            image_bytes = io.BytesIO(image_content)

            # Now open with PIL
            image = Image.open(image_bytes).convert("RGB")
            vision_caption = describe_image(image, openai_api_key)
            individual_descriptions.append({
                'index': i + 1,
                'description': vision_caption
            })
            print(f"Image {i + 1} analysis: {vision_caption}")
        except Exception as e:
            print(f"Error analyzing image {i + 1}: {str(e)}")
            continue

    if not individual_descriptions:
        return "No valid images could be analyzed"
    # Create a comprehensive style analysis prompt
    style_analysis_prompt = f"""
    Analyze the following {len(individual_descriptions)} images and provide a comprehensive style summary that captures the common visual elements, artistic approach, and aesthetic characteristics across all images.

    Individual Image Descriptions:
    """

    for desc in individual_descriptions:
        style_analysis_prompt += f"\nImage {desc['index']}: {desc['description']}"

    style_analysis_prompt += """

    Based on these individual descriptions, provide a unified style analysis that includes:
    1. Common visual elements (colors, lighting, composition patterns)
    2. Consistent artistic style and approach
    3. Technical characteristics (quality, resolution, photographic style)
    4. Subject matter patterns and themes
    5. Overall aesthetic and mood

    Focus on the elements that are consistent across all images to define the core style that should be maintained in generated images. If there are variations, mention the acceptable range of variation within the style.

    Provide a concise but comprehensive style guide that can be used for generating new images in the same style.
    You must provide the final response in 500 characters only.Return the main content only .Don't return headings and unuseful information.
    """

    # Use ChatOpenAI to analyze the combined descriptions
    try:
        from llm_helper import get_llm_for_user
        analysis_llm = get_llm_for_user(user_email=None)  # Can be parameterized if needed
        analysis_messages = [
            SystemMessage(
                content="You are an expert visual style analyst. Analyze multiple image descriptions to extract a comprehensive, unified style guide."),
            HumanMessage(content=style_analysis_prompt)
        ]

        response = analysis_llm.invoke(analysis_messages)
        comprehensive_style = response.content.strip()
        print(f"Comprehensive style analysis: {comprehensive_style}")
        return comprehensive_style

    except Exception as e:
        print(f"Error in comprehensive style analysis: {str(e)}")
        # Fallback: combine individual descriptions
        combined_description = " | ".join([desc['description'] for desc in individual_descriptions])
        return f"Combined style elements from {len(individual_descriptions)} images: {combined_description}"


ROBUST_VISION_SYSTEM_PROMPT = """
You are an expert computer vision analyst with exceptional observational skills. Your task is to provide comprehensive, detailed visual analysis of images with scientific precision and artistic sensitivity.

CORE ANALYSIS FRAMEWORK:

1. COMPOSITIONAL STRUCTURE:
   - Analyze the overall layout, framing, and spatial organization
   - Identify foreground, middle ground, and background elements
   - Describe the visual flow and how elements guide the viewer's eye
   - Note any compositional techniques (rule of thirds, symmetry, leading lines)

2. OBJECTS AND SUBJECTS:
   - Catalog ALL visible objects, people, animals, and entities
   - Describe their positions, sizes, and relationships to each other
   - Identify specific details: clothing, expressions, poses, conditions
   - Note any text, signs, logos, or written elements
   - Mention partially visible or obscured objects

3. VISUAL CHARACTERISTICS:
   - Color palette: dominant colors, color harmony, saturation levels
   - Lighting: source, direction, quality (harsh/soft), shadows, highlights
   - Texture and materials: surfaces, fabrics, finishes
   - Depth and dimensionality: perspective, scale relationships
   - Focus and clarity: sharp vs. blurred areas, depth of field

4. STYLE AND AESTHETIC:
   - Artistic style (realistic, abstract, minimalist, ornate, etc.)
   - Genre or category (portrait, landscape, still life, architectural, etc.)
   - Mood and atmosphere conveyed
   - Cultural or historical context if apparent
   - Technical quality and craftsmanship

5. CONTEXTUAL ELEMENTS:
   - Setting and environment (indoor/outdoor, specific location type)
   - Time indicators (lighting suggests time of day, seasonal clues)
   - Weather conditions if visible
   - Social or cultural context
   - Any narrative or story elements

6. TECHNICAL OBSERVATIONS:
   - Image quality, resolution, and clarity
   - Camera angle and perspective
   - Any visible artifacts, distortions, or technical issues
   - Photographic or artistic techniques employed

RESPONSE GUIDELINES:
- Begin with a concise overview sentence
- Organize observations logically from general to specific
- Use precise, descriptive language without unnecessary adjectives
- Quantify when possible (approximate counts, sizes, proportions)
- Be objective while noting subjective elements like mood or style
- Mention what's NOT present if it's notable or expected
- Conclude with the most striking or significant visual element

ACCURACY REQUIREMENTS:
- Never invent details not visible in the image
- Distinguish between what you can see clearly vs. what you infer
- Use conditional language for uncertain observations ("appears to be", "seems to")
- Prioritize factual description over interpretation
- If image quality limits observation, acknowledge this

Your goal is to create a verbal representation so detailed that someone could understand the image's content, composition, and character without seeing it themselves. Just give the final response in 200 characters  only.
"""


# Usage example with your function:
def describe_image(image: Image.Image, api_key: str) -> str:
    from llm_helper import get_llm_for_user
    vision_llm = get_llm_for_user(user_email=None)  # Can be parameterized if needed
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    base64_image = base64.b64encode(image_bytes.read()).decode('utf-8')

    vision_messages = [
        SystemMessage(content=ROBUST_VISION_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ])
    ]

    try:
        response = vision_llm.invoke(vision_messages)
        return response.content.strip()
    except Exception:
        return "a visually rich and consistent image based on user style"


def generate_images(client: OpenAI, prompt: str, count: int) -> list:
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=count,
        size="512x512"
    )
    return [r.url for r in response.data if r.url]


def extract_num_images_from_prompt(prompt: str, api_key: str) -> Optional[int]:
    try:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None)  # Can be parameterized if needed
        messages = [
            SystemMessage(
                content="You extract the number of images to generate from user input. Return only the integer."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        match = re.search(r'\d+', response.content)
        if match:
            return int(match.group())
    except Exception as e:
        print(f"[ERROR] LLM semantic extraction failed: {e}")

    try:
        fallback_match = re.search(r'(?:generate|create|make)?\s*(\d{1,3})\s*(?:images|pictures)', prompt,
                                   re.IGNORECASE)
        if fallback_match:
            return int(fallback_match.group(1))
    except Exception as e:
        print(f"[ERROR] Regex fallback extraction failed: {e}")

    return None


# --- NEW, UNIFIED FUNCTION TO HANDLE EVERYTHING ---
def generate_complete_document(
        full_text: str, current_pages: int, target_pages: int
) -> Dict[str, Any]:
    """
    Uses a single, powerful LLM call to structure original text and append new,
    generated sections to meet a target page count.
    """
    pages_to_add = max(0, target_pages - current_pages)
    if pages_to_add == 0:
        # If no pages to add, just structure the existing text
        print("No pages to add. Structuring existing content only.")

    prompt = f"""
You are an expert technical writer and document analyst. Your task is to produce a single, complete JSON document that first structures the original content provided, and then seamlessly extends it with new, original sections to meet a target page count.

**Original Raw Text:**
---
{full_text}
---

**Task Details:**
- Current page count of original text: {current_pages}
- Target page count for the final document: {target_pages}
- New pages worth of content to generate and add: {pages_to_add}

**Your Response MUST be a single, cohesive, valid JSON object.**
The final JSON must contain BOTH the structured original content AND the new sections.

**Required JSON Format:**
{{
  "title": "The document's main title",
  "sections": [
    // 1. First, sections from the original text
    {{
      "heading": "Original Section 1 Heading",
      "subsections": [
        {{
          "subheading": "Original Subsection 1.1 Heading",
          "content": "The full, original text content for this subsection..."
        }}
      ]
    }},
    // 2. Then, the NEWLY GENERATED sections
    {{
      "heading": "New, Logically Following Section Heading",
      "subsections": [
        {{
          "subheading": "New Subsection Heading",
          "content": "Full, detailed, and comprehensive written content for the new subsection. This must be entirely new content you generate..."
        }}
      ]
    }}
  ]
}}

**Crucial Instructions:**
1.  **Integrate, Then Extend:** First, accurately place the provided original text into the JSON structure. Then, create and append new sections that logically follow the original content.
2.  **Generate Substantial New Content:** The new sections must be detailed and comprehensive enough to expand the total document to approximately {target_pages} pages. Do not use placeholders.
3.  **Cohesive Final Document:** The final output must be ONE JSON object containing both original and new content seamlessly integrated. Do NOT output only the new parts.
4.  **Strictly JSON:** Your entire response must be a single, valid JSON object. Do not add any explanations, apologies, or markdown (e.g., ```
"""
    llm = initialize_llm()
    try:
        response = llm.invoke(prompt)
        # Attempt to parse the response directly
        return json.loads(response.content)
    except json.JSONDecodeError:
        # If it fails, use our LLM-based repair function as a fallback
        repaired_json_string = repair_json_with_llm(response.content)
        try:
            return json.loads(repaired_json_string)
        except json.JSONDecodeError as e:
            print(f"FATAL: Could not parse even the repaired JSON. Error: {e}")
            return {"error": "Failed to generate and parse final document", "details": str(e)}


def repair_json_with_llm(broken_json_string: str) -> str:
    print("--- Standard JSON parsing failed. Attempting LLM-based repair. ---")

    # Use a different, more capable model for repair if available, or the same one
    repair_llm = initialize_llm()

    # A highly-constrained prompt focused solely on fixing the JSON
    repair_prompt = f"""
            You are a specialized AI assistant that corrects malformed JSON.
            Your single task is to take the provided text and output a valid, well-formed JSON object.

            CRUCIAL INSTRUCTIONS:
            1.  **CORRECT ALL ERRORS:** Fix syntax errors like missing commas, unclosed brackets or braces, incorrect string escaping, and trailing commas.
            2.  **REMOVE EXTRA TEXT:** Delete any text, explanations, or apologies that are outside of the JSON structure.
            3.  **JSON ONLY:** Your entire response MUST be ONLY the corrected JSON object. Do not wrap it in markdown (e.g., ```

            --- MALFORMED JSON INPUT ---
            {broken_json_string}
            --- END MALFORMED JSON INPUT ---

            Your corrected JSON output:
            """

    try:
        response = repair_llm.invoke(repair_prompt)
        # The response content should be the clean JSON string
        repaired_text = response.content.strip()
        return repaired_text
    except Exception as e:
        print(f"--- LLM-based repair also failed: {e} ---")
        # Return a string that represents a JSON error object
        return f'{{"error": "Failed to repair JSON", "details": "{str(e)}"}}'


def extract_pdf_prompt_semantics(user_prompt: str) -> Optional[int]:
    llm = initialize_llm()

    system_instruction = (
        """You are a document request parser. Given a user prompt, extract:
        1. The number of pages requested (as an integer).

        Respond with JSON only in this format:
        {"num_pages": <int or null>}.
        No commentary or markdown.
        """
    )

    response = llm.invoke(f"{system_instruction}\n\nPrompt: {user_prompt}")
    content = response.content if hasattr(response, 'content') else str(response)

    try:
        parsed = json.loads(content)
        return parsed.get("num_pages")
    except Exception:
        return None

# 11.Explore api (moved to api/explore_api.py)


# Report related apis like sending and saving and sending.
# Save report api

# AWS/S3 configuration with your bucket and region
S3_BUCKET = "akio-report-data"
S3_REGION = "ap-southeast-1"
s3_client = boto3.client("s3",
                         aws_access_key_id=os.getenv("REACT_APP_AWS_ACCESS_KEY_ID"),
                         aws_secret_access_key=os.getenv("REACT_APP_AWS_SECRET_ACCESS_KEY"),
                         region_name=S3_REGION)



def upload_file_to_s3(file: UploadFile, bucket: str, key: str) -> str:
    """
    Uploads file to S3 bucket and returns the public URL.
    """
    contents = file.file.read()
    s3_client.put_object(Bucket=bucket, Key=key, Body=contents, ContentType=file.content_type)
    # Construct the S3 object URL for Singapore region
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"


def serialize_report(report: dict) -> dict:
    """Convert datetime objects to ISO format strings in a report dict."""
    for key in ['created_at', 'updated_at']:
        if key in report and isinstance(report[key], datetime):
            report[key] = report[key].isoformat()
    return report


@app.post("/api/save_report")
async def save_report(
        email: str = Form(...),
        title: str = Form(...),
        description: str = Form(...),
        pdf_file: UploadFile = File(...)
) -> JSONResponse:
    """
    Upload a PDF file to S3, extract title/description using LLM, and save metadata in database.
    """
    try:
        # Validate file content type
        if pdf_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        # Read file contents (we'll need this for both S3 upload and text extraction)
        file_contents = await pdf_file.read()
        
        # Reset file pointer for S3 upload
        pdf_file.file = io.BytesIO(file_contents)

        # Generate unique key for S3
        safe_email = email.replace("@", "_at_").replace(".", "_dot_")
        unique_filename = f"reports/{safe_email}/{uuid.uuid4().hex}.pdf"
        print(f"Generated S3 key: {unique_filename}")

        # Upload file to S3
        try:
            s3_url = upload_file_to_s3(pdf_file, S3_BUCKET, unique_filename)
            print(f"S3_url: {s3_url}")
        except Exception as s3exc:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {s3exc}")

        # Insert record in DB with extracted metadata INCLUDING title and description
        result = db.insert_report(email, s3_url, title, description)
        result = serialize_report(result) if result else {}

        # Enhanced response with extracted information
        return JSONResponse(content={
            "status": "success",
            "result": result,
            "pdf_url": s3_url,
            "title": title,
            "description": description
        })

    except HTTPException as he:
        raise he
    except Exception as exc:
        print(f"Unexpected error in save_report: {exc}")
        raise HTTPException(status_code=500, detail=f"Error in save_report: {exc}")



@app.post("/api/get_reports_by_email")
async def get_reports_with_email(
        email: str = Form(...)
) -> JSONResponse:
    """
    Retrieve all saved reports for a given email with title and description.
    """
    try:
        reports = db.get_report_by_email(email)  # Now includes title and description

        if not reports:
            return JSONResponse(content=[])

        # Serialize datetime fields in all reports
        serialized_reports = [serialize_report(r) for r in reports]

        return JSONResponse(content=serialized_reports)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/delete_report_by_id")
async def delete_report_by_id(
        email: str = Form(...),
        report_id: int = Form(...)
) -> JSONResponse:
    """
    Delete a specific report by its ID and the associated email.
    """
    try:
        result = db.delete_user_report_by_id(email, report_id)
        return JSONResponse(content={"status": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Email report to the user
@app.post("/api/email_report")
async def email_report(
        email: str = Form(...),
        report_ids: Optional[str] = Form(None)  # Accept comma-separated string
) -> JSONResponse:
    try:
        if report_ids:
            # Parse comma-separated string into list of ints
            report_ids = [int(rid.strip()) for rid in report_ids.split(",") if rid.strip()]
            reports = db.get_reports_by_ids_and_email(email, report_ids)
        else:
            reports = db.get_report_by_email(email)

        if not reports:
            raise HTTPException(
                status_code=404,
                detail="No reports found for this email with specified report IDs."
            )

        msg = MIMEMultipart()
        msg['Subject'] = 'Selected Graph Reports'
        msg['From'] = os.getenv("EMAIL_USER")
        msg['To'] = email
        msg.attach(MIMEText("Please find attached the selected PDF report(s).", 'plain'))

        for i, report in enumerate(reports, start=1):
            pdf_url = report.get("url")
            if not pdf_url:
                print(f"Skipped report {i}: no URL found")
                continue

            try:
                response = requests.get(pdf_url)
                response.raise_for_status()
                pdf_content = response.content

                pdf_part = MIMEApplication(pdf_content, _subtype="pdf")
                pdf_part.add_header('Content-Disposition', 'attachment', filename=f"report_{report.get('id', i)}.pdf")
                msg.attach(pdf_part)
            except Exception as download_err:
                print(f"Failed to download or attach report {i} from {pdf_url}: {download_err}")
                continue

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(msg)

        return JSONResponse(content={"status": "Email sent successfully."})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


 


# ============================================================================
# LLM Settings API Endpoints
# ============================================================================

@app.post("/api/settings/llm")
async def save_llm_settings(request: Request):
    """Save LLM settings (provider, API key and model) for a user."""
    try:
        data = await request.json()
        email = data.get("email")
        provider = data.get("provider")
        api_key = data.get("api_key")
        model_name = data.get("model_name")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        db.ensure_connection()
        # Create table if it doesn't exist
        try:
            db.create_llm_settings_table()
        except Exception:
            pass
        
        db.save_llm_settings(email, provider, api_key, model_name)
        db.close()
        
        return JSONResponse(content={
            "status": "success",
            "message": "LLM settings saved successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings/llm")
async def get_llm_settings(email: str = Query(...)):
    """Get LLM settings for a user."""
    try:
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        db.ensure_connection()
        # Create table if it doesn't exist
        try:
            db.create_llm_settings_table()
        except Exception:
            pass
        
        settings = db.get_llm_settings(email)
        db.close()
        
        # Return defaults if no settings found (from llm_config)
        if not settings:
            default_config = get_llm_config(None)
            return JSONResponse(content={
                "provider": default_config["provider"],
                "api_key": default_config["api_key"] or "",
                "model_name": default_config["model"],
            })

        return JSONResponse(content={
            "provider": settings.get("provider", "openai"),
            "api_key": settings.get("api_key", ""),
            "model_name": settings.get("model_name") or get_default_model_for_provider(settings.get("provider") or "openai"),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/settings/llm")
async def delete_llm_settings(email: str = Query(...)):
    """Delete LLM settings for a user."""
    try:
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        db.ensure_connection()
        db.delete_llm_settings(email)
        db.close()
        
        return JSONResponse(content={
            "status": "success",
            "message": "LLM settings deleted successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings/llm/providers")
async def get_llm_providers():
    """Get available LLM providers and their models."""
    try:
        from llm_config import get_provider_models

        providers_info = {
            "openai": {
                "name": "OpenAI",
                "description": "To power RAG and/or AI queries",
                "models": get_provider_models("openai"),
                "default_model": get_default_model_for_provider("openai"),
            },
            "anthropic": {
                "name": "Anthropic",
                "description": "To unlock Generative AI models from Anthropic",
                "models": get_provider_models("anthropic"),
                "default_model": get_default_model_for_provider("anthropic"),
            },
            "google": {
                "name": "Google",
                "description": "To unlock Generative AI models from Google",
                "models": get_provider_models("google"),
                "default_model": get_default_model_for_provider("google"),
            },
        }

        return JSONResponse(content=providers_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("final_akio_apis:app", host=host, port=port, reload=True)
