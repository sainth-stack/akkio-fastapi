from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
import io
import uuid
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from docx import Document
from database import PostgresDatabase
from .training import start_training_job
from .training import _detect_task_and_columns
from api.akkio.explore_functions.llm import get_openai_client
import json as _json
import threading
import glob
from pathlib import Path

upload_router = APIRouter()

# Local database instance for this router
db = PostgresDatabase()


def _load_data_from_db_or_uploads(name: str) -> pd.DataFrame:
    """
    Load dataset from database or uploads folder.
    Tries database first, then falls back to uploads folder.
    
    Args:
        name: Table/file name to load
        
    Returns:
        DataFrame with the data
        
    Raises:
        HTTPException: If data not found in either location
    """
    df = None
    
    # Try to get data from database first
    try:
        df = db.get_table_data(name)
        if df is not None and not df.empty:
            print(f"[INFO] Loaded '{name}' from database")
            return df
    except Exception as db_error:
        print(f"[INFO] Database lookup failed for '{name}': {db_error}")
    
    # If not in database, try uploads folder
    print(f"[INFO] Attempting to load '{name}' from uploads folder")
    try:
        # Prefer project-root uploads over CWD
        project_root = Path(__file__).resolve().parents[2]
        uploads_dir_primary = project_root / "uploads"
        uploads_dir_fallback = Path(os.getcwd()) / "uploads"
        search_dirs = [uploads_dir_primary, uploads_dir_fallback] if uploads_dir_primary != uploads_dir_fallback else [uploads_dir_primary]
        
        candidates = []
        for d in search_dirs:
            if d.exists():
                # Try exact match first
                exact_csv = d / f"{name.strip().lower()}.csv"
                if exact_csv.exists():
                    candidates.append(str(exact_csv))
                # Then try pattern match
                pattern = str(d / f"{name.strip().lower()}*.csv")
                candidates.extend(glob.glob(pattern))
        
        if candidates:
            # Choose the most recent file
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            csv_path = candidates[0]
            print(f"[INFO] Loading data from: {csv_path}")
            df = pd.read_csv(csv_path)
            if df is not None and not df.empty:
                return df
        else:
            print(f"[WARNING] No CSV file found for '{name}' in uploads folders")
    except Exception as file_error:
        print(f"[ERROR] Failed to load from uploads: {file_error}")
    
    raise HTTPException(status_code=404, detail=f"No data found for '{name}' in database or uploads folder")

def _spawn_training_thread(email: str, name: str, target: str | None, task: str | None, force: bool):
    """
    Start a dedicated thread for training to avoid blocking the event loop or other APIs.
    Each thread gets its own PostgresDatabase instance to avoid cross-thread connection issues.
    """
    def _runner():
        local_db = PostgresDatabase()
        try:
            start_training_job(local_db, email, name, force_retrain=bool(force), target_override=target, force_task=(task or None))
        except Exception:
            pass
    threading.Thread(target=_runner, daemon=True).start()


def _summarize_df_for_llm(file_name: str, df: pd.DataFrame) -> tuple[str, dict, list[dict]]:
    # Description
    desc = f"File '{file_name}' with {len(df)} rows and {len(df.columns)} columns."
    # Stats per column
    stats = {}
    for col in df.columns:
        series = df[col]
        try:
            nunique = int(series.nunique(dropna=True))
        except Exception:
            nunique = None
        try:
            nulls = int(series.isna().sum())
        except Exception:
            nulls = None
        dtype = str(series.dtype)
        examples = []
        try:
            examples = [series.iloc[i] for i in range(min(3, len(series)))]
        except Exception:
            examples = []
        stats[str(col)] = {
            "dtype": dtype,
            "nunique": nunique,
            "nulls": nulls,
            "examples": examples
        }
    # Sample rows
    try:
        rows = df.head(5).to_dict(orient="records")
    except Exception:
        rows = []
    return desc, stats, rows


def _llm_detect_schema(df: pd.DataFrame, file_name: str) -> dict:
    """
    Ask LLM to identify predictable and forecastable columns.
    Returns dict with keys: predictable_columns, forecastable_columns, ignore_columns (optional).
    Falls back to heuristic when LLM fails.
    """
    desc, stats, rows = _summarize_df_for_llm(file_name, df)
    client = None
    try:
        client = get_openai_client()
    except Exception:
        client = None
    predictable_columns: list[str] = []
    forecastable_columns: list[str] = []
    ignore_columns: list[str] = []
    if client is not None:
        try:
            system_prompt = (
                "You are a skilled data analyst.\n"
                "You will be given three inputs describing a dataset:\n"
                "1. File description\n2. File statistics\n3. Sample rows\n\n"
                "Your task:\n"
                "- Identify most likely dependent variable(s) to be predicted.\n"
                "- If multiple possible, list all.\n"
                "- Identify the independent variables (useful features).\n"
                "- Identify columns to ignore (IDs, timestamps, metadata, irrelevant).\n"
                "- Determine problem type per dependent variable: Regression (numeric) or Classification (categorical).\n\n"
                "Additionally, output two lists:\n"
                "- predictable_columns: all columns that can be targets for prediction (regression/classification).\n"
                "- forecastable_columns: numeric columns that can be forecast over time if a time index/column exists.\n\n"
                "Return STRICT JSON with keys: dependent_variables (array of names), independent_variables (array), "
                "ignore_columns (array), problem_types (object mapping target->'Regression'|'Classification'), "
                "predictable_columns (array), forecastable_columns (array)."
            )
            user_payload = {
                "file_description": desc,
                "file_stats": stats,
                "rows": rows
            }
            # Use chat.completions with JSON object response
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _json.dumps(user_payload)}
                ],
                temperature=0.0
            )
            content = resp.choices[0].message.content if resp and resp.choices else "{}"
            parsed = _json.loads(content)
            predictable_columns = list(map(str, parsed.get("predictable_columns", []) or []))
            forecastable_columns = list(map(str, parsed.get("forecastable_columns", []) or []))
            ignore_columns = list(map(str, parsed.get("ignore_columns", []) or []))
        except Exception:
            # fall back below
            predictable_columns = []
            forecastable_columns = []
            ignore_columns = []
    # Fallback heuristic if needed
    if not predictable_columns and not forecastable_columns:
        _, _, date_col = _detect_task_and_columns(df)
        cols = list(map(str, df.columns.tolist()))
        geospatial_aliases = {"latitude", "longitude", "lat", "lon", "lng"}
        for col in cols:
            if col == date_col or col.lower() in geospatial_aliases:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                predictable_columns.append(col)
            else:
                try:
                    nunique = df[col].nunique(dropna=True)
                    if 2 <= nunique <= 50:
                        predictable_columns.append(col)
                except Exception:
                    pass
        if date_col:
            for col in cols:
                if col == date_col:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    forecastable_columns.append(col)
    # Ensure the suggested columns exist and are valid
    df_cols = set(map(str, df.columns.tolist()))
    predictable_columns = [c for c in predictable_columns if c in df_cols]
    forecastable_columns = [c for c in forecastable_columns if c in df_cols]
    # For forecastable, enforce numeric dtype
    forecastable_columns = [c for c in forecastable_columns if pd.api.types.is_numeric_dtype(df[c])]
    return {
        "predictable_columns": predictable_columns,
        "forecastable_columns": forecastable_columns,
        "ignore_columns": ignore_columns
    }

@upload_router.post("/api/upload_only")
async def upload_only(
    background_tasks: BackgroundTasks,
    mail: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        file_name = file.filename
        if not file_name:
            raise HTTPException(status_code=400, detail="No filename provided")

        file_extension = os.path.splitext(file_name)[1].lower()

        print(f"[DEBUG] Upload request - File: {file_name}, Extension: '{file_extension}'")
        print(f"[DEBUG] File content type: {file.content_type}")
        print(f"[DEBUG] File size: {file.size if hasattr(file, 'size') else 'unknown'}")

        # Check if file has no extension
        if not file_extension:
            raise HTTPException(
                status_code=400,
                detail="File must have an extension (.csv, .xlsx, .xls, .pdf, or .docx)"
            )

        # Read file content into a DataFrame
        content = await file.read()
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_extension in [".xls", ".xlsx"]:
            # Save to temp file because read_excel reads from file path
            temp_path = f"temp{file_extension}"
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)
            df = pd.read_excel(temp_path)
            os.remove(temp_path)
        elif file_extension == ".pdf":
            # Save PDF to temp file for PyPDFLoader
            temp_path = f"temp_pdf_{uuid.uuid4().hex}{file_extension}"
            try:
                print(f"[DEBUG] Processing PDF file: {file_name}")
                print(f"[DEBUG] File size: {len(content)} bytes")

                with open(temp_path, "wb") as temp_file:
                    temp_file.write(content)

                print(f"[DEBUG] PDF saved to temp file: {temp_path}")

                # Extract text from PDF using PyPDFLoader
                try:
                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()
                    print(f"[DEBUG] PDF loaded successfully, {len(pages)} pages found")
                except Exception as pdf_error:
                    print(f"[DEBUG] PDF loading error: {str(pdf_error)}")
                    raise HTTPException(status_code=400, detail=f"Error reading PDF file: {str(pdf_error)}")

                if not pages:
                    raise HTTPException(status_code=400, detail="PDF file appears to be empty or corrupted")

                # Combine all pages text
                full_text = "\n".join([page.page_content for page in pages])
                print(f"[DEBUG] Extracted text length: {len(full_text)} characters")

                if not full_text.strip():
                    raise HTTPException(status_code=400, detail="PDF file contains no extractable text")

                # Create DataFrame with extracted text
                # Split text into lines and create a DataFrame
                lines = full_text.split('\n')
                df = pd.DataFrame({
                    'line_number': range(1, len(lines) + 1),
                    'text_content': lines
                })

                # Remove empty lines
                df = df[df['text_content'].str.strip() != '']
                print(f"[DEBUG] DataFrame created with {len(df)} non-empty lines")

            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as pdf_error:
                print(f"[DEBUG] Unexpected PDF processing error: {str(pdf_error)}")
                raise HTTPException(status_code=400, detail=f"Error processing PDF file: {str(pdf_error)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"[DEBUG] Temp file cleaned up: {temp_path}")
        elif file_extension == ".docx":
            # Save DOCX to temp file for Document processing
            temp_path = f"temp_docx_{uuid.uuid4().hex}{file_extension}"
            try:
                print(f"[DEBUG] Processing DOCX file: {file_name}")
                print(f"[DEBUG] File size: {len(content)} bytes")

                with open(temp_path, "wb") as temp_file:
                    temp_file.write(content)

                print(f"[DEBUG] DOCX saved to temp file: {temp_path}")

                # Extract text from DOCX using python-docx
                try:
                    doc = Document(temp_path)
                    paragraphs = doc.paragraphs
                    print(f"[DEBUG] DOCX loaded successfully, {len(paragraphs)} paragraphs found")
                except Exception as docx_error:
                    print(f"[DEBUG] DOCX loading error: {str(docx_error)}")
                    raise HTTPException(status_code=400, detail=f"Error reading DOCX file: {str(docx_error)}")

                if not paragraphs:
                    raise HTTPException(status_code=400, detail="DOCX file appears to be empty or corrupted")

                # Combine all paragraphs text
                full_text = "\n".join([p.text for p in paragraphs if p.text.strip()])
                print(f"[DEBUG] Extracted text length: {len(full_text)} characters")

                if not full_text.strip():
                    raise HTTPException(status_code=400, detail="DOCX file contains no extractable text")

                # Create DataFrame with extracted text
                # Split text into lines and create a DataFrame
                lines = full_text.split('\n')
                df = pd.DataFrame({
                    'line_number': range(1, len(lines) + 1),
                    'text_content': lines
                })

                # Remove empty lines
                df = df[df['text_content'].str.strip() != '']
                print(f"[DEBUG] DataFrame created with {len(df)} non-empty lines")

            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as docx_error:
                print(f"[DEBUG] Unexpected DOCX processing error: {str(docx_error)}")
                raise HTTPException(status_code=400, detail=f"Error processing DOCX file: {str(docx_error)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"[DEBUG] Temp file cleaned up: {temp_path}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV, Excel, PDF, or DOCX allowed")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data")

        # Insert or update in database ONLY (no local save)
        results = db.insert_or_update(mail, df, file_name)

        # Ensure training tables exist, then detect schema + queue training in a separate thread to avoid blocking this API
        try:
            db.ensure_training_tables()
            # Launch a fast background thread that runs LLM detection, stores schema, and queues training (each training in its own thread)
            def _schema_and_queue():
                try:
                    schema = _llm_detect_schema(df, file_name)
                    all_columns = list(map(str, df.columns.tolist()))
                    db.save_dataset_schema(
                        mail,
                        file_name,
                        predictable_columns=schema.get("predictable_columns", []),
                        forecastable_columns=schema.get("forecastable_columns", []),
                        columns=all_columns,
                        total_columns=len(all_columns)
                    )
                    predictable_columns = schema.get("predictable_columns", []) or []
                    forecastable_columns = schema.get("forecastable_columns", []) or []
                    # Queue jobs and spawn training threads
                    for t in predictable_columns:
                        jid = str(uuid.uuid4())
                        db.upsert_training_job(mail, file_name, jid, status="queued", progress=0, message=f"Queued auto-train (target={t}, task=predict)", model_type="predict", target=t)
                        _spawn_training_thread(mail, file_name, t, "predict", False)
                    for t in forecastable_columns:
                        jid = str(uuid.uuid4())
                        db.upsert_training_job(mail, file_name, jid, status="queued", progress=0, message=f"Queued auto-train (target={t}, task=forecast)", model_type="forecast", target=t)
                        _spawn_training_thread(mail, file_name, t, "forecast", False)
                except Exception:
                    pass
            threading.Thread(target=_schema_and_queue, daemon=True).start()
            training_started = True
            job_id = None
        except Exception as exc:
            # Training scheduling shouldn't block upload success
            job_id = None
            training_started = False

        return JSONResponse(content={
            "message": "File uploaded and data saved to database successfully",
            "db_insert_result": results,
            "training": {
                "job_id": job_id,
                "started": training_started
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@upload_router.get("/api/training_status")
async def get_training_status(
    mail: str = Query(..., alias="mail"),
    name: str = Query(..., alias="name"),
    target: str | None = Query(None, description="Optional: specific target to check status"),
):
    try:
        db.ensure_training_tables()
        status = db.get_training_status(mail, name, target=target)
        if not status:
            return JSONResponse(content=jsonable_encoder({"exists": False, "message": "No training job found"}), status_code=200)
        # Fast overall summary only (omit model details and trained/pending lists for speed)
        overall = db.get_overall_training_summary(mail, name) or {}
        return JSONResponse(content=jsonable_encoder({
            "exists": True,
            "status": status,
            "overall": overall
        }), status_code=200)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@upload_router.post("/api/training_train_all")
async def training_train_all(
    background_tasks: BackgroundTasks,
    mail: str = Form(...),
    name: str = Form(...),
    force: bool = Form(True),
):
    """
    Queue training jobs for all predictable and forecastable targets stored in schema.
    If schema is missing, fall back to quick heuristic detection.
    """
    try:
        db.ensure_training_tables()
        schema = db.get_dataset_schema(mail, name)
        predictable_targets = (schema or {}).get("predictable_columns", []) or []
        forecastable_targets = (schema or {}).get("forecastable_columns", []) or []
        # If no schema stored, compute minimal fallback once and store
        if not predictable_targets and not forecastable_targets:
            df = _load_data_from_db_or_uploads(name)
            fallback = _llm_detect_schema(df, name)
            all_columns = list(map(str, df.columns.tolist()))
            db.save_dataset_schema(
                mail, name,
                predictable_columns=fallback.get("predictable_columns", []),
                forecastable_columns=fallback.get("forecastable_columns", []),
                columns=all_columns,
                total_columns=len(all_columns)
            )
            predictable_targets = fallback.get("predictable_columns", [])
            forecastable_targets = fallback.get("forecastable_columns", [])
        queued = []
        # Queue predict/forecast per target (duplicates ok; background will upsert/overwrite)
        for t in predictable_targets:
            job_id = str(uuid.uuid4())
            db.upsert_training_job(mail, name, job_id, status="queued", progress=0, message=f"Queued auto-train (target={t}, task=predict)", model_type="predict", target=t)
            _spawn_training_thread(mail, name, t, "predict", bool(force))
            queued.append(t)
        for t in forecastable_targets:
            job_id = str(uuid.uuid4())
            db.upsert_training_job(mail, name, job_id, status="queued", progress=0, message=f"Queued auto-train (target={t}, task=forecast)", model_type="forecast", target=t)
            _spawn_training_thread(mail, name, t, "forecast", bool(force))
            queued.append(t)
        # De-duplicate while preserving order
        seen = set()
        unique_targets = []
        for t in queued:
            if t not in seen:
                seen.add(t)
                unique_targets.append(t)
        return JSONResponse(content=jsonable_encoder({
            "message": "Queued training for discovered targets",
            "count": len(unique_targets),
            "targets": unique_targets,
            "source": "schema" if schema else "fallback"
        }), status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@upload_router.get("/api/model_schema")
async def get_model_schema(
    mail: str = Query(..., alias="mail"),
    name: str = Query(..., alias="name"),
):
    """
    Return predictable/forecastable columns stored in DB (from LLM analysis).
    If missing, compute once and store for future calls.
    Also checks uploads folder if data not found in database.
    """
    try:
        schema = db.get_dataset_schema(mail, name)
        if not schema:
            df = _load_data_from_db_or_uploads(name)
            detected = _llm_detect_schema(df, name)
            columns = list(map(str, df.columns.tolist()))
            db.save_dataset_schema(
                mail, name,
                predictable_columns=detected.get("predictable_columns", []),
                forecastable_columns=detected.get("forecastable_columns", []),
                columns=columns,
                total_columns=len(columns)
            )
            schema = db.get_dataset_schema(mail, name)
        return JSONResponse(content=jsonable_encoder({
            "predictable_columns": schema.get("predictable_columns", []),
            "forecastable_columns": schema.get("forecastable_columns", []),
            "columns": schema.get("columns", []),
            "total_columns": schema.get("total_columns", 0)
        }), status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@upload_router.post("/api/training_retrain")
async def retrigger_training(
    background_tasks: BackgroundTasks,
    mail: str = Form(...),
    name: str = Form(...),
    force: bool = Form(True),
    target: str | None = Form(None, description="Optional: explicit target column to train"),
    task: str | None = Form(None, description="Optional: 'predict' or 'forecast'. If omitted, decides based on target data."),
):
    """
    Manually retrigger training for a given (email, file name).
    - If force=True (default), retrains even if a model exists.
    - Responds immediately while training proceeds in background.
    """
    try:
        # Verify data exists for this table name
        try:
            df_meta = db.get_tables_info(db._clean_name(name))
            if df_meta in (None, {}, []):
                raise HTTPException(status_code=404, detail="No uploaded data found for this file")
        except Exception:
            # Try a read to confirm existence; will return None if missing
            df_one = db.read(db._clean_name(name))
            if df_one is None or df_one.empty:
                raise HTTPException(status_code=404, detail="No uploaded data found for this file")

        # Queue job
        job_id = str(uuid.uuid4())
        suffix = []
        if target:
            suffix.append(f"target={target}")
        if task:
            suffix.append(f"task={task}")
        suffix_str = (" (" + ", ".join(suffix) + ")") if suffix else ""
        message = "Manually retriggered" + suffix_str
        db.upsert_training_job(mail, name, job_id, status="queued", progress=0, message=message, model_type=None, target=target)
        _spawn_training_thread(mail, name, target, (task or None), bool(force))
        return JSONResponse(content=jsonable_encoder({
            "message": "Training retriggered",
            "job_id": job_id,
            "force": bool(force),
            "target": target,
            "task": task
        }), status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@upload_router.get("/api/training_model_info")
async def get_trained_model_info(
    mail: str = Query(..., alias="mail"),
    name: str = Query(..., alias="name"),
    target: str | None = Query(None, description="Optional: specific target"),
):
    """
    Return metadata of the trained model for (email, name) without the binary bytes.
    Includes: model_type, framework, metrics, trained_at, updated_at.
    """
    try:
        db.ensure_training_tables()
        model = db.get_existing_model(mail, name, target=target)
        if not model:
            return JSONResponse(content=jsonable_encoder({"exists": False, "message": "No trained model found"}), status_code=200)
        info = {
            "model_type": model.get("model_type"),
            "framework": model.get("framework"),
            "metrics": model.get("metrics"),
            "trained_at": model.get("trained_at"),
            "updated_at": model.get("updated_at"),
            "target": model.get("target"),
        }
        return JSONResponse(content=jsonable_encoder({"exists": True, "info": info}), status_code=200)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# Legacy prediction endpoint removed - use /api/model_predict instead

def _infer_freq(dts: pd.Series) -> str:
    try:
        dts = pd.to_datetime(dts, errors='coerce').dropna()
        if len(dts) < 3:
            return "D"
        freq = pd.infer_freq(dts.sort_values().unique())
        return freq or "D"
    except Exception:
        return "D"

@upload_router.get("/api/forecast")
async def forecast_with_model(
    mail: str = Query(..., alias="mail"),
    name: str = Query(..., alias="name"),
    horizon: int = Query(12, ge=1, le=365),
    target: str | None = Query(None, description="Optional: explicit target column to forecast"),
    date_col_override: str | None = Query(None, description="Optional: explicit date column (auto-detected if omitted)"),
    unit: str | None = Query(None, description="Optional: horizon unit. One of: minutes,hours,days,weeks,months,years"),
    freq: str | None = Query(None, description="Optional: frequency alias to generate future (e.g., '5T','H','D','W','M','Y'). Overrides auto-detected for Prophet"),
):
    """
    Forecast future values using the trained time-series model.
    - For Prophet models: returns ds, yhat.
    - For PyCaret/sktime models: returns step-wise forecast values.
    """
    try:
        db.ensure_training_tables()
        # Try to load an existing forecast model; if missing/invalid we'll fallback later
        framework = None
        model = None
        try:
            model_rec = db.get_existing_model(mail, name, target=target, model_type="forecast") if target else db.get_existing_model(mail, name, model_type="forecast")
        except Exception:
            model_rec = None
        if model_rec:
            try:
                model_type = (model_rec.get("model_type") or "").lower()
                framework = (model_rec.get("framework") or "").lower()
                model_bytes = model_rec.get("model_obj")
                if model_type == "forecast" and model_bytes:
                    import pickle as _p
                    model = _p.loads(model_bytes)
            except Exception:
                model = None

        # Load original dataset to derive date column and frequency
        df = _load_data_from_db_or_uploads(name)
        task, detected_target, detected_date_col = _detect_task_and_columns(df)
        saved_target = model_rec.get("target") if model_rec else None
        used_target = target or saved_target or detected_target
        if (model_rec and saved_target) and target and (target != saved_target):
            raise HTTPException(status_code=400, detail=f"Trained target is '{saved_target}', received '{target}'. Retrain the model for the requested target or omit 'target' to use trained target.")

        date_col = date_col_override or detected_date_col
        if not date_col:
            # try fallback first datetime-like column
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_col = c
                    break
        if not date_col:
            raise HTTPException(status_code=400, detail="Could not determine date column for forecasting")
        # It is enough to compute frequency from date column; the model does not require target at prediction time,
        # but we still validate the target column exists in the dataset if provided/detected.
        if used_target and used_target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{used_target}' not found in dataset")
        ts = df[[date_col]].dropna().copy()
        ts[date_col] = pd.to_datetime(ts[date_col], errors='coerce')
        ts = ts.dropna(subset=[date_col]).sort_values(by=[date_col])
        if ts.empty:
            raise HTTPException(status_code=400, detail="Insufficient time-series data to forecast")

        def _unit_to_freq(u: str) -> str | None:
            m = {
                "minute": "T", "minutes": "T", "min": "T", "mins": "T",
                "hour": "H", "hours": "H", "hr": "H", "hrs": "H",
                "day": "D", "days": "D",
                "week": "W", "weeks": "W",
                "month": "M", "months": "M",
                "year": "Y", "years": "Y"
            }
            return m.get((u or "").strip().lower())

        def _freq_seconds(f: str) -> float | None:
            try:
                from pandas.tseries.frequencies import to_offset
                off = to_offset(f)
                # Some offsets don't have fixed nanos (like M/Y); handle those below
                nanos = getattr(off, "nanos", None)
                if nanos and nanos > 0:
                    return nanos / 1_000_000_000.0
            except Exception:
                pass
            f_norm = (f or "").upper()
            if f_norm.startswith("W"):
                return 7 * 24 * 3600.0
            if f_norm.startswith("D"):
                return 24 * 3600.0
            if f_norm.startswith("H"):
                return 3600.0
            if f_norm.endswith("T") or f_norm.startswith("T"):
                # Treat any "*T" as minutes; try to parse numeric prefix if present (e.g., "5T")
                try:
                    num = int(f_norm[:-1]) if f_norm[:-1] else 1
                except Exception:
                    num = 1
                return num * 60.0
            if f_norm.startswith("M"):  # month approx
                return 30 * 24 * 3600.0
            if f_norm.startswith("Y"):  # year approx
                return 365 * 24 * 3600.0
            return None

        def _unit_seconds(u: str) -> float | None:
            u_norm = (u or "").strip().lower()
            if u_norm in {"minute", "minutes", "min", "mins"}:
                return 60.0
            if u_norm in {"hour", "hours", "hr", "hrs"}:
                return 3600.0
            if u_norm in {"day", "days"}:
                return 24 * 3600.0
            if u_norm in {"week", "weeks"}:
                return 7 * 24 * 3600.0
            if u_norm in {"month", "months"}:
                return 30 * 24 * 3600.0
            if u_norm in {"year", "years"}:
                return 365 * 24 * 3600.0
            return None

        # Fallback: if no trained forecast model available, do a quick one-shot forecast
        if model is None:
            # Prefer Prophet if available; otherwise, naive last-value forecast
            try:
                from prophet import Prophet  # type: ignore
                # Determine/validate target
                if used_target is None:
                    for c in df.columns:
                        if c == date_col:
                            continue
                        if pd.api.types.is_numeric_dtype(df[c]):
                            used_target = c
                            break
                if used_target is None:
                    raise HTTPException(status_code=400, detail="No numeric target column available for forecasting")
                df_p = df[[date_col, used_target]].dropna().copy()
                df_p[date_col] = pd.to_datetime(df_p[date_col], errors='coerce')
                df_p = df_p.dropna(subset=[date_col]).sort_values(by=[date_col])
                df_p = df_p.rename(columns={date_col: "ds", used_target: "y"})
                m = Prophet()
                m.fit(df_p)
                last_dt = df_p["ds"].iloc[-1]
                inferred_freq = _infer_freq(df_p["ds"])
                used_freq = freq or (_unit_to_freq(unit) if unit else inferred_freq) or "D"
                future = pd.date_range(start=last_dt, periods=horizon+1, freq=used_freq)[1:]
                future_df = pd.DataFrame({"ds": future})
                forecast = m.predict(future_df)
                out = pd.DataFrame({"ds": forecast["ds"], "yhat": forecast["yhat"]})
                dates = [d.isoformat() for d in out["ds"].tolist()]
                try:
                    values = [float(v) for v in out["yhat"].tolist()]
                except Exception:
                    values = [v for v in out["yhat"].tolist()]
                return JSONResponse(content=jsonable_encoder({
                    "prediction_result": {
                        "target_column": used_target,
                        "forecast_data": {
                            "date": dates,
                            "forecasted_value": values
                        },
                        "framework": "prophet",
                        "horizon": horizon,
                        "date_col": date_col,
                        "freq": used_freq
                    }
                }), status_code=200)
            except Exception:
                # Naive fallback
                series = df[[date_col, used_target]].dropna().copy() if used_target in df.columns else df.dropna().copy()
                series[date_col] = pd.to_datetime(series[date_col], errors='coerce')
                series = series.dropna(subset=[date_col]).sort_values(by=[date_col])
                if series.empty or used_target is None:
                    raise HTTPException(status_code=400, detail="Insufficient data for naive forecast")
                last_dt = series[date_col].iloc[-1]
                inferred_freq = _infer_freq(series[date_col])
                used_freq = freq or (_unit_to_freq(unit) if unit else inferred_freq) or "D"
                future = pd.date_range(start=last_dt, periods=horizon+1, freq=used_freq)[1:]
                dates = [d.isoformat() for d in future.tolist()]
                try:
                    last_val = float(series[used_target].dropna().iloc[-1])
                except Exception:
                    last_val = series[used_target].dropna().iloc[-1]
                values = [last_val for _ in range(len(dates))]
                return JSONResponse(content=jsonable_encoder({
                    "prediction_result": {
                        "target_column": used_target,
                        "forecast_data": {
                            "date": dates,
                            "forecasted_value": values
                        },
                        "framework": "naive",
                        "horizon": horizon,
                        "date_col": date_col,
                        "freq": used_freq
                    }
                }), status_code=200)

        if framework == "prophet":
            # Prophet expects 'ds' future periods; infer frequency
            last_dt = ts[date_col].iloc[-1]
            inferred_freq = _infer_freq(ts[date_col])
            used_freq = freq or (_unit_to_freq(unit) if unit else inferred_freq)
            if not used_freq:
                used_freq = inferred_freq or "D"
            future = pd.date_range(start=last_dt, periods=horizon+1, freq=used_freq)[1:]
            future_df = pd.DataFrame({"ds": future})
            if not hasattr(model, "predict"):
                raise HTTPException(status_code=500, detail="Loaded Prophet model does not support predict()")
            forecast = model.predict(future_df)
            out = pd.DataFrame({"ds": forecast["ds"], "yhat": forecast["yhat"]})
            dates = [d.isoformat() for d in out["ds"].tolist()]
            try:
                values = [float(v) for v in out["yhat"].tolist()]
            except Exception:
                values = [v for v in out["yhat"].tolist()]
            return JSONResponse(content=jsonable_encoder({
                "prediction_result": {
                    "target_column": used_target,
                    "forecast_data": {
                        "date": dates,
                        "forecasted_value": values
                    },
                    "framework": framework,
                    "horizon": horizon,
                    "date_col": date_col,
                    "freq": used_freq
                }
            }), status_code=200)
        else:
            # Assume sktime-style forecaster saved by PyCaret
            # Translate unit -> number of model steps
            model_freq = _infer_freq(ts[date_col])
            steps = horizon
            if unit:
                unit_secs = _unit_seconds(unit)
                step_secs = _freq_seconds(model_freq) if model_freq else None
                if unit_secs and step_secs:
                    # at least 1 step
                    steps = max(1, int(np.ceil(horizon * unit_secs / step_secs)))
            fh = list(range(1, steps + 1))
            if not hasattr(model, "predict"):
                raise HTTPException(status_code=500, detail="Loaded forecast model does not support predict()")
            try:
                y_pred = model.predict(fh=fh)
            except TypeError:
                y_pred = model.predict(fh)
            try:
                preds = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
            except Exception:
                preds = [float(y_pred)] if np.isscalar(y_pred) else [str(y) for y in y_pred]
            # Generate future dates aligned with inferred frequency
            try:
                last_dt = ts[date_col].iloc[-1]
                used_freq = model_freq or "D"
                future = pd.date_range(start=last_dt, periods=steps+1, freq=used_freq)[1:]
                dates = [d.isoformat() for d in future.tolist()]
            except Exception:
                # Fallback: sequential indices if date generation fails
                dates = list(range(1, len(preds) + 1))
                used_freq = model_freq
            try:
                values = [float(v) for v in preds]
            except Exception:
                values = [v for v in preds]
            return JSONResponse(content=jsonable_encoder({
                "prediction_result": {
                    "target_column": used_target,
                    "forecast_data": {
                        "date": dates,
                        "forecasted_value": values
                    },
                    "framework": framework or "pycaret",
                    "horizon": horizon,
                    "date_col": date_col,
                    "freq": used_freq,
                    "steps": steps,
                    "unit": unit
                }
            }), status_code=200)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


