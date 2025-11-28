from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, Body, BackgroundTasks
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
import base64
from typing import List, Tuple, Optional, Dict, Any
from langchain_community.vectorstores import Chroma
try:
    from langchain_openai import OpenAIEmbeddings  # preferred
except Exception:
    from langchain_community.embeddings import OpenAIEmbeddings  # fallback
import wave
import struct

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


def _extract_texts_for_embedding(df: pd.DataFrame) -> List[str]:
    """
    Extract and chunk text from DataFrame for embedding.
    For text-heavy documents (pdf/word), chunks text into ~500 char segments.
    """
    texts: List[str] = []
    try:
        # Priority for text columns
        text_like_cols = [c for c in df.columns if str(c).lower() in {"text", "content", "text_content", "paragraph", "line", "line_text"}]
        if text_like_cols:
            for col in text_like_cols:
                series = df[col].dropna().astype(str)
                raw_texts = series.tolist()
                # Combine into full text and chunk
                full_text = "\n".join(raw_texts)
                # Chunk into ~500 char segments with overlap
                chunk_size = 500
                overlap = 50
                for i in range(0, len(full_text), chunk_size - overlap):
                    chunk = full_text[i:i+chunk_size].strip()
                    if chunk:
                        texts.append(chunk)
        else:
            # Fallback: convert rows to compact JSON lines
            records = df.fillna("").astype(str).to_dict(orient="records")
            texts = [_json.dumps(rec, ensure_ascii=False) for rec in records]
    except Exception:
        try:
            records = df.fillna("").astype(str).to_dict(orient="records")
            texts = [_json.dumps(rec, ensure_ascii=False) for rec in records]
        except Exception:
            texts = []
    # Deduplicate and trim empties
    dedup = []
    seen = set()
    for t in texts:
        ts = t.strip()
        if ts and len(ts) > 20 and ts not in seen:  # min 20 chars
            dedup.append(ts)
            seen.add(ts)
    print(f"[VECTOR][EXTRACT] extracted {len(dedup)} chunks from {len(df)} rows, total_chars={sum(len(t) for t in dedup)}")
    return dedup[:5000]  # hard cap


def _collection_name(email: str, name: str) -> str:
    safe_email = "".join(ch if ch.isalnum() else "_" for ch in (email or "user"))
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in (name or "dataset"))
    return f"{safe_email}__{safe_name}"


def _upsert_to_chromadb(email: str, name: str, texts: List[str], metadata: dict):
    coll_name = _collection_name(email, name)
    try:
        print(f"[VECTOR] Start upsert to Chroma: collection='{coll_name}', texts={len(texts)}")
    except Exception:
        pass
    if not texts:
        try:
            print(f"[VECTOR] No texts to index for collection='{coll_name}'. Skipping.")
        except Exception:
            pass
        return
    try:
        embeddings = OpenAIEmbeddings()
    except Exception:
        # No API key or embeddings unavailable
        try:
            print(f"[VECTOR] OpenAIEmbeddings unavailable; skipping upsert for '{coll_name}'")
        except Exception:
            pass
        return
    persist_dir = str(Path(__file__).resolve().parents[2] / "chroma_store")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    try:
        vectordb = Chroma(collection_name=coll_name, persist_directory=persist_dir, embedding_function=embeddings)
        
        # FIX: Delete existing collection content to ensure clean state
        try:
            # Check if collection has any data
            existing_count = vectordb._collection.count()
            if existing_count > 0:
                print(f"[VECTOR] Deleting {existing_count} existing items in collection='{coll_name}' to avoid duplicates/mixed data")
                vectordb.delete_collection()
                # Re-initialize after deletion
                vectordb = Chroma(collection_name=coll_name, persist_directory=persist_dir, embedding_function=embeddings)
        except Exception as e:
            print(f"[VECTOR] Warning during collection cleanup: {e}")

        ids = [f"{coll_name}_{i}" for i in range(len(texts))]
        vectordb.add_texts(texts=texts, metadatas=[metadata] * len(texts), ids=ids)
        try:
            print(f"[VECTOR] Upserted {len(texts)} chunks into collection='{coll_name}' at '{persist_dir}'")
        except Exception:
            pass
    except Exception as e:
        print(f"[VECTOR] Error during upsert: {e}")
        # Ignore vector errors to not block upload
        pass


def _extract_audio_metadata(audio_bytes: bytes, file_extension: str) -> Dict[str, Any]:
    """
    Extract metadata from audio file including duration, sample rate, channels, and waveform statistics.
    Returns dict with audio properties.
    """
    print(f"[AUDIO][METADATA] Starting metadata extraction for {file_extension} file, size={len(audio_bytes)} bytes")
    
    metadata = {
        "duration_seconds": None,
        "sample_rate_hz": None,
        "channels": None,
        "channel_names": None,
        "bit_depth": None,
        "mean_amplitude": None,
        "std_amplitude": None,
        "audio_quality": None
    }
    
    # Try to extract WAV metadata natively
    if file_extension == ".wav":
        print(f"[AUDIO][METADATA] WAV file detected, using native wave library")
        temp_path = f"temp_audio_meta_{uuid.uuid4().hex}{file_extension}"
        try:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(audio_bytes)
            
            with wave.open(temp_path, 'rb') as wav_file:
                # Basic properties
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                print(f"[AUDIO][METADATA] WAV properties: channels={n_channels}, sample_width={sample_width}, frame_rate={frame_rate}, n_frames={n_frames}")
                
                # Calculate duration
                duration = n_frames / float(frame_rate) if frame_rate > 0 else 0
                print(f"[AUDIO][METADATA] Calculated duration: {duration:.2f} seconds")
                
                # Read audio data for waveform statistics
                print(f"[AUDIO][METADATA] Reading waveform data for statistics...")
                frames = wav_file.readframes(n_frames)
                print(f"[AUDIO][METADATA] Read {len(frames)} bytes of audio data")
                
                # Convert bytes to integers based on sample width
                if sample_width == 1:  # 8-bit
                    print(f"[AUDIO][METADATA] Processing 8-bit audio")
                    fmt = f"{n_frames * n_channels}B"
                    samples = struct.unpack(fmt, frames)
                    samples = [s - 128 for s in samples]  # Convert unsigned to signed
                elif sample_width == 2:  # 16-bit
                    print(f"[AUDIO][METADATA] Processing 16-bit audio")
                    fmt = f"{n_frames * n_channels}h"
                    samples = struct.unpack(fmt, frames)
                else:  # 32-bit or other
                    print(f"[AUDIO][METADATA] Unsupported sample width: {sample_width}, skipping waveform analysis")
                    samples = []
                
                # Calculate statistics
                if samples:
                    print(f"[AUDIO][METADATA] Calculating waveform statistics for {len(samples)} samples...")
                    samples_array = np.array(samples, dtype=float)
                    max_val = 2 ** (8 * sample_width - 1)
                    normalized = samples_array / max_val
                    mean_amp = float(np.mean(normalized))
                    std_amp = float(np.std(normalized))
                    print(f"[AUDIO][METADATA] Waveform stats: mean_amplitude={mean_amp:.4f}, std_amplitude={std_amp:.4f}")
                else:
                    mean_amp = None
                    std_amp = None
                    print(f"[AUDIO][METADATA] No samples to analyze, skipping waveform statistics")
                
                # Determine audio quality
                quality = "Unknown"
                if frame_rate >= 44100:
                    quality = "CD-quality audio"
                elif frame_rate >= 22050:
                    quality = "High-quality audio"
                elif frame_rate >= 16000:
                    quality = "Standard audio"
                else:
                    quality = "Low-quality audio"
                print(f"[AUDIO][METADATA] Audio quality determined: {quality} (sample_rate={frame_rate}Hz)")
                
                # Channel names
                channel_names = "Mono (1 channel)" if n_channels == 1 else f"Stereo ({n_channels} channels)"
                
                metadata.update({
                    "duration_seconds": round(duration, 2),
                    "sample_rate_hz": frame_rate,
                    "channels": n_channels,
                    "channel_names": channel_names,
                    "bit_depth": sample_width * 8,
                    "mean_amplitude": round(mean_amp, 2) if mean_amp is not None else None,
                    "std_amplitude": round(std_amp, 4) if std_amp is not None else None,
                    "audio_quality": quality
                })
                
                print(f"[AUDIO][METADATA] ✓ WAV metadata extraction complete: {duration:.2f}s, {frame_rate}Hz, {n_channels}ch, {quality}")
        except Exception as e:
            print(f"[AUDIO][METADATA] ✗ WAV metadata extraction error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[AUDIO][METADATA] Cleaned up temp file: {temp_path}")
    
    # For non-WAV files, try using a library if available
    else:
        print(f"[AUDIO][METADATA] Non-WAV file, attempting mutagen library extraction...")
        try:
            # Try using mutagen for other formats (optional dependency)
            from mutagen import File as MutagenFile
            temp_path = f"temp_audio_meta_{uuid.uuid4().hex}{file_extension}"
            try:
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(audio_bytes)
                
                audio_file = MutagenFile(temp_path)
                if audio_file and audio_file.info:
                    info = audio_file.info
                    duration = getattr(info, 'length', None)
                    sample_rate = getattr(info, 'sample_rate', None)
                    channels = getattr(info, 'channels', None)
                    
                    print(f"[AUDIO][METADATA] Mutagen extracted: duration={duration}, sample_rate={sample_rate}, channels={channels}")
                    
                    if duration:
                        metadata["duration_seconds"] = round(duration, 2)
                    if sample_rate:
                        metadata["sample_rate_hz"] = sample_rate
                        if sample_rate >= 44100:
                            metadata["audio_quality"] = "CD-quality audio"
                        elif sample_rate >= 22050:
                            metadata["audio_quality"] = "High-quality audio"
                        else:
                            metadata["audio_quality"] = "Standard audio"
                    if channels:
                        metadata["channels"] = channels
                        metadata["channel_names"] = "Mono (1 channel)" if channels == 1 else f"Stereo ({channels} channels)"
                    print(f"[AUDIO][METADATA] ✓ Non-WAV metadata extraction complete")
                else:
                    print(f"[AUDIO][METADATA] Mutagen returned no info for file")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print(f"[AUDIO][METADATA] Cleaned up temp file: {temp_path}")
        except ImportError:
            print("[AUDIO][METADATA] ✗ Mutagen not available for non-WAV metadata extraction")
        except Exception as e:
            print(f"[AUDIO][METADATA] ✗ Non-WAV metadata extraction error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[AUDIO][METADATA] Final metadata: {metadata}")
    return metadata


def _transcribe_audio_with_whisper(audio_bytes: bytes, file_extension: str) -> str:
    """
    Use OpenAI Whisper API to transcribe audio to text.
    Returns transcribed text.
    """
    print(f"[AUDIO][WHISPER] Starting transcription for {file_extension} file, size={len(audio_bytes)} bytes")
    
    try:
        client = get_openai_client()
        print(f"[AUDIO][WHISPER] OpenAI client initialized successfully")
    except Exception as e:
        print(f"[AUDIO][WHISPER] ✗ Failed to get OpenAI client: {e}")
        client = None
    
    if client is None:
        print(f"[AUDIO][WHISPER] ✗ No OpenAI client available, returning empty transcription")
        return ""
    
    try:
        # Save audio to temp file as Whisper API requires file input
        temp_path = f"temp_audio_{uuid.uuid4().hex}{file_extension}"
        print(f"[AUDIO][WHISPER] Creating temp file: {temp_path}")
        
        try:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(audio_bytes)
            print(f"[AUDIO][WHISPER] Temp file written, size={os.path.getsize(temp_path)} bytes")
            
            # Transcribe using Whisper
            print(f"[AUDIO][WHISPER] Calling Whisper API (model=whisper-1)...")
            with open(temp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            print(f"[AUDIO][WHISPER] Whisper API call completed")
            
            # Extract text from response
            if isinstance(transcript, str):
                text = transcript
            else:
                text = getattr(transcript, 'text', str(transcript))
            
            print(f"[AUDIO][WHISPER] ✓ Transcribed {len(text)} characters from audio file")
            print(f"[AUDIO][WHISPER] Transcription preview: {text[:200]}{'...' if len(text) > 200 else ''}")
            return text.strip()
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[AUDIO][WHISPER] Cleaned up temp file: {temp_path}")
    except Exception as e:
        print(f"[AUDIO][WHISPER] ✗ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def _ocr_image_with_llm(image_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Use OpenAI vision model to extract text and classify image subtype (table|chart|other).
    Returns (text, subtype or None).
    """
    def _guess_image_subtype_from_text(text: str) -> Optional[str]:
        try:
            txt = (text or "").lower()
            # Obvious chart keywords
            chart_keywords = ["chart", "graph", "plot", "x-axis", "y-axis", "axis", "legend", "series", "bar ", " line ", " pie ", "scatter", "histogram"]
            if any(k in txt for k in chart_keywords):
                return "chart"
            # Heuristic: many rows with multiple columns => table
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            sample = lines[:20]
            multi_col_lines = 0
            for ln in sample:
                # Count tokens split by 2+ spaces or tabs or commas
                tokens = [t for t in re.split(r"[,\t]| {2,}", ln) if t.strip()]
                if len(tokens) >= 4:
                    multi_col_lines += 1
            if len(sample) >= 5 and multi_col_lines >= max(3, len(sample)//2):
                return "table"
            # Numeric density heuristic
            digits = sum(ch.isdigit() for ch in txt)
            if len(txt) > 0 and digits / max(1, len(txt)) > 0.25 and len(lines) >= 5:
                return "table"
        except Exception:
            return None
        return None
    import re
    try:
        client = get_openai_client()
    except Exception:
        client = None
    if client is None:
        return "", None
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        system_prompt = (
            "You will receive one business screenshot image. First, EXTRACT all visible text verbatim.\n"
            "Second, STRICTLY CLASSIFY the image primary type as one of:\n"
            "- table: grid-like rows/columns, tabular data, spreadsheets, HTML tables, CSV-like layout.\n"
            "- chart: any graph/plot (bar/line/pie/scatter/histogram), with axes, bars, lines, legends, or slices.\n"
            "- other: none of the above.\n"
            "If both appear, choose the primary visual focus (prefer 'chart' over 'table' if a plot is present on the page).\n"
            "Respond as STRICT JSON: {\"text\": \"...\", \"subtype\": \"table|chart|other\"}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract text and classify."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        }
                    ]
                }
            ],
            temperature=0.0
        )
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        parsed = _json.loads(content)
        text = str(parsed.get("text", "") or "").strip()
        subtype = parsed.get("subtype")
        subtype = str(subtype).lower() if subtype else None
        if subtype not in {"table", "chart", "other"}:
            subtype = None
        # Fallback heuristic if model is unsure or says 'other'
        if subtype in (None, "", "other"):
            guess = _guess_image_subtype_from_text(text)
            if guess in {"table", "chart"}:
                subtype = guess
        return text, subtype
    except Exception:
        return "", None


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

        # Check if file has no extension
        if not file_extension:
            raise HTTPException(
                status_code=400,
                detail="File must have an extension (.csv, .xlsx, .xls, .pdf, .docx, images, audio, .json, .xml)"
            )

        # Read file content into a DataFrame and determine type/subtype
        file_type: Optional[str] = None
        file_subtype: Optional[str] = None

        # Read file content into a DataFrame
        content = await file.read()
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
            file_type = "csv"
        elif file_extension in [".xls", ".xlsx"]:
            # Save to temp file because read_excel reads from file path
            temp_path = f"temp{file_extension}"
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)
            df = pd.read_excel(temp_path)
            os.remove(temp_path)
            file_type = "excel"
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
            file_type = "pdf"
            file_subtype = "report"  # treat PDFs as reports by default
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
            file_type = "word"
        elif file_extension in [".png", ".jpg", ".jpeg", ".webp"]:
            # Try OCR via OpenAI Vision
            text, inferred = _ocr_image_with_llm(content)
            file_type = "image"
            # Map subtype
            if inferred in {"table", "chart"}:
                file_subtype = inferred
            else:
                file_subtype = None
            # Build DataFrame
            lines = [ln for ln in (text or "").split("\n")]
            df = pd.DataFrame({
                "line_number": range(1, len(lines) + 1) if lines else [],
                "text_content": lines
            })
        elif file_extension == ".json":
            try:
                parsed = _json.loads(content.decode("utf-8"))
            except Exception as je:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {je}")
            # Normalize into DataFrame
            try:
                if isinstance(parsed, list):
                    df = pd.json_normalize(parsed)
                else:
                    df = pd.json_normalize(parsed)
            except Exception:
                # Fallback: single column text
                df = pd.DataFrame({"text_content": [_json.dumps(parsed, ensure_ascii=False)]})
            file_type = "json"
        elif file_extension == ".xml":
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(content.decode("utf-8"))
            except Exception as xe:
                raise HTTPException(status_code=400, detail=f"Invalid XML: {xe}")
            # Flatten XML into path-text rows
            rows = []
            def walk(node, path):
                current_path = f"{path}/{node.tag}" if path else node.tag
                text_val = (node.text or "").strip()
                if text_val:
                    rows.append({"path": current_path, "text_content": text_val})
                for child in list(node):
                    walk(child, current_path)
            walk(root, "")
            df = pd.DataFrame(rows) if rows else pd.DataFrame({"text_content": []})
            file_type = "xml"
        elif file_extension in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"]:
            print(f"[AUDIO][UPLOAD] ═══════════════════════════════════════════════")
            print(f"[AUDIO][UPLOAD] Audio file detected: {file_name}")
            print(f"[AUDIO][UPLOAD] Extension: {file_extension}, Size: {len(content)} bytes")
            print(f"[AUDIO][UPLOAD] ═══════════════════════════════════════════════")
            
            # Extract audio metadata first
            print(f"[AUDIO][UPLOAD] Step 1/4: Extracting metadata...")
            audio_metadata = _extract_audio_metadata(content, file_extension)
            print(f"[AUDIO][UPLOAD] ✓ Metadata extracted: {audio_metadata}")
            
            # Transcribe audio using OpenAI Whisper
            print(f"[AUDIO][UPLOAD] Step 2/4: Transcribing audio with Whisper...")
            text = _transcribe_audio_with_whisper(content, file_extension)
            file_type = "audio"
            file_subtype = None
            
            if not text.strip():
                print(f"[AUDIO][UPLOAD] ✗ Transcription failed or returned empty text")
                raise HTTPException(status_code=400, detail="Audio file contains no transcribable content or transcription failed")
            
            print(f"[AUDIO][UPLOAD] ✓ Transcription successful: {len(text)} characters")
            
            # Create DataFrame with transcribed text and metadata
            print(f"[AUDIO][UPLOAD] Step 3/4: Creating DataFrame...")
            
            # Split into sentences for better chunking
            sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
            if not sentences:
                sentences = [text]
            print(f"[AUDIO][UPLOAD] Split transcription into {len(sentences)} sentences/segments")
            
            # Build comprehensive DataFrame
            df_data = {
                'segment_number': range(1, len(sentences) + 1),
                'text_content': sentences,
                'duration_seconds': [audio_metadata.get('duration_seconds')] * len(sentences),
                'sample_rate_hz': [audio_metadata.get('sample_rate_hz')] * len(sentences),
                'channels': [audio_metadata.get('channels')] * len(sentences),
                'channel_names': [audio_metadata.get('channel_names')] * len(sentences),
                'audio_quality': [audio_metadata.get('audio_quality')] * len(sentences),
                'bit_depth': [audio_metadata.get('bit_depth')] * len(sentences),
                'mean_amplitude': [audio_metadata.get('mean_amplitude')] * len(sentences),
                'std_amplitude': [audio_metadata.get('std_amplitude')] * len(sentences)
            }
            df = pd.DataFrame(df_data)
            
            print(f"[AUDIO][UPLOAD] ✓ DataFrame created: {len(df)} rows × {len(df.columns)} columns")
            print(f"[AUDIO][UPLOAD] DataFrame columns: {list(df.columns)}")
            print(f"[AUDIO][UPLOAD] DataFrame sample:\n{df.head(2)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: CSV, Excel, PDF, DOCX, images (png/jpg/jpeg/webp), JSON, XML, audio (mp3/wav/m4a/ogg/flac/aac/wma)")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data")

        # Capture metadata for audio files before database insert
        audio_metadata_for_response = None
        if file_type == "audio" and len(df) > 0:
            print(f"[AUDIO][UPLOAD] Step 4/4: Preparing metadata for response...")
            # Extract metadata from first row (all rows have same metadata)
            # Convert numpy/pandas types to native Python types for JSON serialization
            def _to_python_type(val):
                """Convert numpy/pandas types to native Python types"""
                if val is None or pd.isna(val):
                    return None
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    return int(val)
                if isinstance(val, (np.floating, np.float64, np.float32)):
                    return float(val)
                if isinstance(val, (np.bool_, bool)):
                    return bool(val)
                return val
            
            audio_metadata_for_response = {
                "duration_seconds": _to_python_type(df['duration_seconds'].iloc[0]) if 'duration_seconds' in df.columns else None,
                "sample_rate_hz": _to_python_type(df['sample_rate_hz'].iloc[0]) if 'sample_rate_hz' in df.columns else None,
                "channels": _to_python_type(df['channels'].iloc[0]) if 'channels' in df.columns else None,
                "channel_names": str(df['channel_names'].iloc[0]) if 'channel_names' in df.columns and df['channel_names'].iloc[0] is not None else None,
                "audio_quality": str(df['audio_quality'].iloc[0]) if 'audio_quality' in df.columns and df['audio_quality'].iloc[0] is not None else None,
                "bit_depth": _to_python_type(df['bit_depth'].iloc[0]) if 'bit_depth' in df.columns else None,
                "mean_amplitude": _to_python_type(df['mean_amplitude'].iloc[0]) if 'mean_amplitude' in df.columns else None,
                "std_amplitude": _to_python_type(df['std_amplitude'].iloc[0]) if 'std_amplitude' in df.columns else None,
                "transcription": "\n".join(df['text_content'].tolist()) if 'text_content' in df.columns else ""
            }
            print(f"[AUDIO][UPLOAD] ✓ Metadata prepared for response (types converted to JSON-serializable)")
            print(f"[AUDIO][UPLOAD] Response metadata summary: duration={audio_metadata_for_response.get('duration_seconds')}s, "
                  f"quality={audio_metadata_for_response.get('audio_quality')}, "
                  f"transcription_length={len(audio_metadata_for_response.get('transcription', ''))}")
            print(f"[AUDIO][UPLOAD] Metadata types: duration={type(audio_metadata_for_response.get('duration_seconds'))}, "
                  f"sample_rate={type(audio_metadata_for_response.get('sample_rate_hz'))}, "
                  f"channels={type(audio_metadata_for_response.get('channels'))}")

        # Insert or update in database (store raw bytes for image/pdf/word/audio)
        raw_bytes = None
        if file_type in {"image", "pdf", "word", "audio"}:
            raw_bytes = content
            if file_type == "audio":
                print(f"[AUDIO][UPLOAD] Inserting into database with raw audio bytes...")
        results = db.insert_or_update(mail, df, file_name, file_type, file_subtype, raw_bytes)
        if file_type == "audio":
            print(f"[AUDIO][UPLOAD] ✓ Database insert complete: {results}")

        # Schedule vector ingestion to ChromaDB in background for image/pdf/word/audio
        vector_ingest_scheduled = False
        try:
            if file_type in {"image", "pdf", "word", "audio"}:
                if file_type == "audio":
                    print(f"[AUDIO][UPLOAD] ───────────────────────────────────────────────")
                    print(f"[AUDIO][UPLOAD] Starting vector store ingestion process...")
                
                texts = _extract_texts_for_embedding(df)
                print(f"[VECTOR][UPLOAD] file='{file_name}', type={file_type}, subtype={file_subtype}, df_rows={len(df)}, extracted_chunks={len(texts)}")
                
                # Prepare metadata for vector store
                vector_metadata = {
                    "email": mail,
                    "name": Path(file_name).stem,
                    "type": file_type,
                    "subtype": file_subtype
                }
                
                # Add audio-specific metadata for better context
                if file_type == "audio" and audio_metadata_for_response:
                    print(f"[AUDIO][UPLOAD] Adding audio-specific metadata to vector store...")
                    # Audio metadata already converted to Python types above
                    vector_metadata.update({
                        "duration_seconds": audio_metadata_for_response.get("duration_seconds"),
                        "sample_rate_hz": audio_metadata_for_response.get("sample_rate_hz"),
                        "channels": audio_metadata_for_response.get("channels"),
                        "audio_quality": audio_metadata_for_response.get("audio_quality")
                    })
                    print(f"[AUDIO][UPLOAD] Vector metadata: {vector_metadata}")
                
                background_tasks.add_task(
                    _upsert_to_chromadb,
                    email=mail,
                    name=Path(file_name).stem,
                    texts=texts,
                    metadata=vector_metadata
                )
                vector_ingest_scheduled = True
                print(f"[VECTOR] ✓ Scheduled background ingestion for '{file_name}' (type={file_type}, subtype={file_subtype}), texts={len(texts)}")
                
                if file_type == "audio":
                    print(f"[AUDIO][UPLOAD] ✓ Vector ingestion scheduled in background")
                    print(f"[AUDIO][UPLOAD] ───────────────────────────────────────────────")
        except Exception as e:
            print(f"[VECTOR][UPLOAD] ✗ Failed to schedule ingestion: {e}")
            import traceback
            traceback.print_exc()
            vector_ingest_scheduled = False
            if file_type == "audio":
                print(f"[AUDIO][UPLOAD] ✗ Vector ingestion scheduling failed")

        response_data = {
            "message": "File uploaded and data saved to database successfully",
            "db_insert_result": results,
            "type": file_type,
            "subtype": file_subtype,
            "vector_ingest_scheduled": vector_ingest_scheduled
        }
        
        # Add audio metadata to response if available
        if audio_metadata_for_response:
            response_data["audio_metadata"] = audio_metadata_for_response
            print(f"[AUDIO][UPLOAD] ═══════════════════════════════════════════════")
            print(f"[AUDIO][UPLOAD] ✓✓✓ UPLOAD COMPLETE ✓✓✓")
            print(f"[AUDIO][UPLOAD] File: {file_name}")
            print(f"[AUDIO][UPLOAD] Duration: {audio_metadata_for_response.get('duration_seconds')}s")
            print(f"[AUDIO][UPLOAD] Quality: {audio_metadata_for_response.get('audio_quality')}")
            print(f"[AUDIO][UPLOAD] Transcription: {len(audio_metadata_for_response.get('transcription', ''))} characters")
            print(f"[AUDIO][UPLOAD] Vector Ingestion: {'Scheduled' if vector_ingest_scheduled else 'Not Scheduled'}")
            print(f"[AUDIO][UPLOAD] ═══════════════════════════════════════════════")
        
        return JSONResponse(content=response_data)
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
async def get_column_names(name: str = Query(None), mail: str = Query(None)):
    """Get predictable and forecastable columns from LLM analysis"""
    try:
        # Determine dataset name
        dataset_name = name
        if not dataset_name:
            # Load latest from uploads
            project_root = Path(__file__).resolve().parents[2]
            uploads_dir = project_root / "uploads"
            files = list(uploads_dir.glob("*.csv"))
            if not files:
                raise HTTPException(404, "No CSV files found in uploads folder")
            latest = max(files, key=os.path.getmtime)
            dataset_name = latest.stem  # Get filename without extension
        
        # Try to get schema from database first (if mail provided)
        schema = None
        if mail and dataset_name:
            try:
                schema = db.get_dataset_schema(mail, dataset_name)
            except Exception:
                pass
        
        # If no schema, analyze with LLM
        if not schema:
            df = _load_data_from_db_or_uploads(dataset_name)
            detected = _llm_detect_schema(df, dataset_name)
            
            # Save schema if mail provided
            if mail:
                try:
                    columns = list(map(str, df.columns.tolist()))
                    db.save_dataset_schema(
                        mail, dataset_name,
                        predictable_columns=detected.get("predictable_columns", []),
                        forecastable_columns=detected.get("forecastable_columns", []),
                        columns=columns,
                        total_columns=len(columns)
                    )
                except Exception as e:
                    print(f"[WARNING] Could not save schema: {e}")
            
            schema = {
                "predictable_columns": detected.get("predictable_columns", []),
                "forecastable_columns": detected.get("forecastable_columns", []),
                "columns": list(df.columns)
            }
        
        return JSONResponse(content={
            "status": "success",
            "columns": schema.get("columns", []),
            "predictable_columns": schema.get("predictable_columns", []),
            "forecastable_columns": schema.get("forecastable_columns", [])
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
            # Validate target column exists
            if target_col not in df.columns:
                raise HTTPException(400, f"Target column '{target_col}' not found in dataset")
            
            # Check for missing values in target column
            missing_count = df[target_col].isna().sum()
            if missing_count > 0:
                print(f"[WARNING] Target column '{target_col}' has {missing_count} missing values. Dropping rows with missing target.")
                df = df.dropna(subset=[target_col])
            
            if len(df) < 10:
                raise HTTPException(400, f"Insufficient data after removing missing values. Need at least 10 rows, got {len(df)}")
            
            print(f"[INFO] Training RandomForest on {len(df)} rows, target: {target_col}")
            stat, cols = random_forest(df, target_col)
            
            # Get feature columns (all columns except target)
            feature_cols = [col for col in df.columns if col != target_col]
            
            # Check if training succeeded (stat is dict or truthy)
            training_success = bool(stat) and stat is not False
            
            print(f"[INFO] Training {'succeeded' if training_success else 'failed'}")
            if not training_success:
                print(f"[ERROR] Training failed. stat={stat}, cols={cols}")
            
            return JSONResponse(content={
                'columns': list(df.columns),
                'rf': True,
                'status': training_success,
                'rf_cols': feature_cols if training_success else [],
                'target_column': target_col,
                'feature_columns': feature_cols if training_success else [],
                'model_stats': stat if isinstance(stat, dict) else None
            })
        elif model_type == 'Arima':
            stat = arima_train_only(df, target_col)
            
            # For ARIMA, only the target column is needed
            return JSONResponse(content={
                'columns': list(df.columns),
                'status': stat,
                'arima': True,
                'target_column': target_col,
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

