from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, Body
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
from api.akkio.explore_functions.llm import get_openai_client
import json as _json
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
        # Simple heuristic: numeric columns are predictable
        cols = list(map(str, df.columns.tolist()))
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                predictable_columns.append(col)
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
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(content)

                # Extract text from PDF using PyPDFLoader
                loader = PyPDFLoader(temp_path)
                pages = loader.load()

                if not pages:
                    raise HTTPException(status_code=400, detail="PDF file appears to be empty or corrupted")

                # Combine all pages text
                full_text = "\n".join([page.page_content for page in pages])

                if not full_text.strip():
                    raise HTTPException(status_code=400, detail="PDF file contains no extractable text")

                # Create DataFrame with extracted text
                lines = full_text.split('\n')
                df = pd.DataFrame({
                    'line_number': range(1, len(lines) + 1),
                    'text_content': lines
                })

                # Remove empty lines
                df = df[df['text_content'].str.strip() != '']

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        elif file_extension == ".docx":
            # Save DOCX to temp file for Document processing
            temp_path = f"temp_docx_{uuid.uuid4().hex}{file_extension}"
            try:
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(content)

                # Extract text from DOCX using python-docx
                doc = Document(temp_path)
                paragraphs = doc.paragraphs

                if not paragraphs:
                    raise HTTPException(status_code=400, detail="DOCX file appears to be empty or corrupted")

                # Combine all paragraphs text
                full_text = "\n".join([p.text for p in paragraphs if p.text.strip()])

                if not full_text.strip():
                    raise HTTPException(status_code=400, detail="DOCX file contains no extractable text")

                # Create DataFrame with extracted text
                lines = full_text.split('\n')
                df = pd.DataFrame({
                    'line_number': range(1, len(lines) + 1),
                    'text_content': lines
                })

                # Remove empty lines
                df = df[df['text_content'].str.strip() != '']

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV, Excel, PDF, or DOCX allowed")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data")

        # Insert or update in database
        results = db.insert_or_update(mail, df, file_name)

        return JSONResponse(content={
            "message": "File uploaded and data saved to database successfully",
            "db_insert_result": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ============ LEGACY ENDPOINTS (from final_akio_apis.py) ============

@upload_router.get("/api/get_columns")
async def get_column_names(name: str = Query(None)):
    """Get column names from dataset"""
    try:
        if name:
            df = _load_data_from_db_or_uploads(name)
        else:
            # Load latest from uploads
            project_root = Path(__file__).resolve().parents[2]
            uploads_dir = project_root / "uploads"
            files = list(uploads_dir.glob("*.csv"))
            if not files:
                raise HTTPException(404, "No CSV files found in uploads folder")
            latest = max(files, key=os.path.getmtime)
            df = pd.read_csv(latest)
        
        return JSONResponse(content={
            "status": "success",
            "columns": df.columns.tolist()
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@upload_router.post("/api/models")
async def models(input: dict = Body(...)):
    """Train RandomForest or ARIMA model using legacy code"""
    try:
        model_type = input.get("model")
        target_col = input.get("col")
        name = input.get("name")
        
        # Load data
        if name:
            df = _load_data_from_db_or_uploads(name)
        else:
            # Load latest from uploads
            project_root = Path(__file__).resolve().parents[2]
            uploads_dir = project_root / "uploads"
            files = list(uploads_dir.glob("*.csv"))
            if not files:
                raise HTTPException(404, "No CSV files found in uploads folder")
            latest = max(files, key=os.path.getmtime)
            df = pd.read_csv(latest)
        
        # Save to data.csv for legacy code compatibility
        df.to_csv('data.csv', index=False)
        
        # Import legacy functions from final_akio_apis
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from final_akio_apis import random_forest, arima_train_only
        
        if model_type == 'RandomForest':
            result = random_forest(df, target_col)
            # Handle both old and new return formats for backward compatibility
            if len(result) == 3:
                stat, cols, row_data = result
            elif len(result) == 2:
                stat, cols = result
                row_data = {}
            else:
                stat, cols, row_data = False, [], {}
            
            return JSONResponse(content={
                'columns': list(df.columns),
                'rf': True,
                'status': stat,
                'rf_cols': cols,
                'feature_columns': cols,
                'row_data': row_data if row_data else {}
            })
        elif model_type == 'Arima':
            stat = arima_train_only(df, target_col)
            return JSONResponse(content={
                'columns': list(df.columns),
                'status': stat,
                'arima': True,
                'message': 'ARIMA model trained successfully.'
            })
        else:
            raise HTTPException(400, "Unsupported model type. Use 'RandomForest' or 'Arima'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@upload_router.post("/api/model_predict")
async def model_predict(request: Request):
    """Make predictions using trained RF or ARIMA models"""
    try:
        form_data = await request.form()
        form_data = dict(form_data)
        
        form_name = form_data.get('form_name')
        targetcol = form_data.get('targetColumn')
        
        if not form_name:
            raise HTTPException(400, "form_name is required")
        if not targetcol:
            raise HTTPException(400, "targetColumn is required")
        
        # Import legacy functions
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from final_akio_apis import handle_rf_prediction, handle_arima_forecast
        
        if form_name == 'rf':
            return await handle_rf_prediction(form_data, targetcol)
        elif form_name == 'arima':
            return await handle_arima_forecast(form_data, targetcol)
        else:
            raise HTTPException(400, "Invalid form_name. Use 'rf' or 'arima'")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

