from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import io
import pandas as pd
from datetime import datetime
import json
from langchain_openai import ChatOpenAI
import traceback
from difflib import get_close_matches
import re
import ast
import textwrap
from uuid import uuid4
from collections import defaultdict
import threading
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests
import csv

SESSION_MEMORY = defaultdict(list)
SESSION_MEMORY_LOCK = threading.Lock()
MAX_MEMORY_SIZE = 50  # Maximum number of exchanges per session

def manage_session_memory(session_id: str, user_message: str = None, bot_message: str = None, get_history: bool = False):
    """
    Centralized session memory management with proper logging and cleanup
    """
    with SESSION_MEMORY_LOCK:
        if get_history:
            # Return a copy of the history
            history = list(SESSION_MEMORY.get(session_id, []))
            print(f"Retrieved session memory for {session_id}: {len(history)} messages")
            return history
        
        if user_message is not None:
            # Add user message
            SESSION_MEMORY[session_id].append({"role": "user", "content": user_message})
            print(f"Added user message to session {session_id}: {user_message[:100]}...")
        
        if bot_message is not None:
            # Add bot message
            SESSION_MEMORY[session_id].append({"role": "bot", "content": bot_message})
            print(f"Added bot message to session {session_id}: {bot_message[:100]}...")
        
        # Cleanup old messages if session gets too large
        current_memory = SESSION_MEMORY[session_id]
        if len(current_memory) > MAX_MEMORY_SIZE:
            # Keep the last MAX_MEMORY_SIZE messages
            SESSION_MEMORY[session_id] = current_memory[-MAX_MEMORY_SIZE:]
            print(f"Cleaned up session {session_id}, kept last {MAX_MEMORY_SIZE} messages")
        
        print(f"Session {session_id} now has {len(SESSION_MEMORY[session_id])} total messages")
        return None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import numpy as np

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Chart generation will be disabled.")

sla_router = APIRouter()

# Initialize vector search availability (TF-IDF based)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TFIDF_CHECK  # noqa: F401
    VECTOR_SEARCH_AVAILABLE = True
except Exception:
    VECTOR_SEARCH_AVAILABLE = False
    print("Warning: scikit-learn not available. Vector search will be disabled.")

@sla_router.post("/api/upload_processed_data")
async def upload_processed_data(data: dict):
    """
    Upload pre-processed data from frontend
    """
    try:
        filename = data.get('filename', 'data1.csv')
        records = data.get('records', [])
        headers = data.get('headers', [])
        
        if not records:
            raise HTTPException(status_code=400, detail="No records provided")
        
        # Save to uploads_sla directory (streaming CSV write without pandas for speed)
        upload_dir = "uploads_sla"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Determine headers if not provided
        if not headers:
            if isinstance(records, list) and len(records) > 0 and isinstance(records[0], dict):
                headers = list(records[0].keys())
            elif isinstance(records, list) and len(records) > 0 and isinstance(records[0], (list, tuple)):
                headers = [f"col_{i+1}" for i in range(len(records[0]))]
            else:
                raise HTTPException(status_code=400, detail="No headers provided and unable to infer from records")
        
        # Save as data1.csv (standard filename for processing)
        file_path = os.path.join(upload_dir, "data1.csv")
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(headers)
                # Write rows
                for rec in records:
                    if isinstance(rec, dict):
                        row = [rec.get(h, "") if rec.get(h, "") is not None else "" for h in headers]
                    elif isinstance(rec, (list, tuple)):
                        values = list(rec)
                        # Pad or trim to match header length
                        if len(values) < len(headers):
                            values = values + [""] * (len(headers) - len(values))
                        row = values[:len(headers)]
                    else:
                        row = [str(rec)]
                    # Ensure all values are CSV-safe strings or numbers
                    safe_row = [v if isinstance(v, (int, float)) else ("" if v is None else str(v)) for v in row]
                    writer.writerow(safe_row)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write CSV: {str(e)}")
        
        # Also save processed files list
        processed_files_path = os.path.join(upload_dir, "processed_files.txt")
        with open(processed_files_path, "w", encoding="utf-8") as f:
            f.write(f"{filename}\n")
        
        return JSONResponse({
            "message": "Processed data uploaded successfully",
            "filename": "data1.csv",
            "records": len(records),
            "columns": len(headers)
        })
        
    except Exception as e:
        print(f"Error uploading processed data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading processed data: {str(e)}")

@sla_router.post("/api/upload_data")
async def upload_data_only(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    upload_dir = "uploads_sla"
    os.makedirs(upload_dir, exist_ok=True)
    content = await file.read()

    # Fast path: if CSV, save directly without pandas
    if ext == ".csv":
        if not content:
            raise HTTPException(status_code=400, detail="Empty file or no data found.")
        static_file_path = os.path.join(upload_dir, "data1.csv")
        try:
            with open(static_file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save CSV file: {e}")

        # Track latest processed filename
        processed_files_path = os.path.join(upload_dir, "processed_files.txt")
        try:
            with open(processed_files_path, "w", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        except Exception:
            pass

        return JSONResponse(content={
            "message": "File uploaded successfully",
            "filename": "data1.csv"
        })

    # Excel path: convert to CSV once using pandas
    if ext in [".xls", ".xlsx"]:
        try:
            df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {e}")

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty file or no data found.")

        static_file_path = os.path.join(upload_dir, "data1.csv")
        try:
            df.to_csv(static_file_path, index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save converted CSV: {e}")

        processed_files_path = os.path.join(upload_dir, "processed_files.txt")
        try:
            with open(processed_files_path, "w", encoding="utf-8") as f:
                f.write(f"{filename}\n")
        except Exception:
            pass

        return JSONResponse(content={
            "message": "Excel file uploaded and converted successfully",
            "filename": "data1.csv"
        })

    # Unsupported type
    raise HTTPException(status_code=400, detail="Unsupported file type")

# Handle NaN values and make data JSON serializable - ULTRA ROBUST VERSION
def convert_to_json_safe(df):
    """Convert DataFrame to JSON-safe format with aggressive NaN/inf handling"""
    import json
    import numpy as np
    import pandas as pd
    
    # Step 1: Replace all inf/-inf values with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Convert to records first, then handle NaN values in the records
    try:
        records = df.to_dict(orient="records")
        
        # Step 3: Clean up NaN values in the records
        cleaned_records = []
        for record in records:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    cleaned_record[key] = None
                else:
                    cleaned_record[key] = value
            cleaned_records.append(cleaned_record)
        
        # Step 4: Double-check by attempting JSON serialization
        # This will catch any remaining problematic values
        json.dumps(cleaned_records)
        
        return cleaned_records
    except (ValueError, TypeError) as e:
        print(f"JSON serialization failed, applying string fallback: {e}")
        
        # Ultimate fallback: convert everything to strings and clean
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val) or val is None:
                    record[col] = None
                elif isinstance(val, (int, bool)):
                    record[col] = val
                elif isinstance(val, float):
                    if np.isnan(val) or np.isinf(val):
                        record[col] = None
                    else:
                        record[col] = val
                else:
                    # Convert everything else to string
                    try:
                        record[col] = str(val)
                    except:
                        record[col] = None
            records.append(record)
        
        # Final verification
        try:
            json.dumps(records)
            return records
        except Exception as final_error:
            print(f"Final fallback failed: {final_error}")
            return [{"error": "Data conversion failed", "message": str(final_error)}]
    # Apply the conversion
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame dtypes: {df.dtypes}")
    print(f"NaN values per column: {df.isnull().sum()}")
    
    records = convert_to_json_safe(df)
    print(f"Conversion successful, {len(records)} records created")

    return JSONResponse(content={"records": records})


def safe_execute_pandas_code(code: str, df: pd.DataFrame):
    """
    Safely executes sandboxed Python code for data analysis, primarily using the pandas library.

    This function is designed to run untrusted code in a controlled environment by restricting
    imports, built-in functions, and dangerous operations. It supports optional chart generation
    using the Plotly library.

    Args:
        code: A string containing the Python code to execute. The code should assign its
              final output to a variable named 'result'. If no 'result' variable is found,
              the function will attempt to evaluate the last non-empty line as an expression.
        df: The pandas DataFrame to be used in the execution context, accessible as 'df'.

    Returns:
        The object assigned to the 'result' variable in the executed code, the value of the
        last evaluated expression, or a message indicating successful execution with no
        output.

    Raises:
        ImportError: If a disallowed module is imported.
        ValueError: If the code contains forbidden operations, fails to parse, or if a
                    required library (like Plotly) is not available when needed.
    """
    # Normalize and sanitize code to avoid indentation and fence issues
    def normalize_code(text: str) -> str:
        text = text.replace('\u00A0', ' ').replace('\u200b', '').replace('\ufeff', '')
        if '\\n' in text and text.count('\n') <= 1:
            try:
                text = bytes(text, 'utf-8').decode('unicode_escape')
            except Exception:
                pass
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.strip()
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = text.replace('```', '')
        text = textwrap.dedent(text).expandtabs(4).lstrip('\n')
        text = '\n'.join([ln.rstrip() for ln in text.split('\n')])
        return text

    code_clean = normalize_code(code)

    # 1. Determine if Plotly is needed before attempting to import it
    plotly_keywords = ['px.', 'go.', 'ff.', 'plotly']
    is_plotly_needed = any(keyword in code_clean.lower() for keyword in plotly_keywords)
    px = go = ff = np = None

    if is_plotly_needed:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.figure_factory as ff
            import numpy as np
        except ImportError:
            raise ValueError(
                "Plotly and/or NumPy are required for chart generation but are not installed. "
                "Please install them using: pip install plotly numpy"
            )

    # 2. Define a secure importer to whitelist allowed modules
    def secure_importer(name, globals=None, locals=None, fromlist=(), level=0):
        """A wrapped version of __import__ that only allows whitelisted modules."""
        allowed_modules = {
            'pandas', 'pd', 'numpy', 'np', 'math', 'random', 'datetime',
            'json', 're', 'collections', 'plotly', 'plotly.express',
            'plotly.graph_objects', 'plotly.figure_factory'
        }
        if name not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is disallowed for security reasons.")
        return __import__(name, globals, locals, fromlist, level)

    # 3. Establish a heavily restricted execution environment
    safe_globals = {
        'pd': pd,
        'df': df,
        'px': px,
        'go': go,
        'ff': ff,
        'np': np,
        '__builtins__': {
            '__import__': secure_importer,  # Override the default import
            # Whitelist of safe built-in functions
            'abs': abs, 'dict': dict, 'enumerate': enumerate, 'float': float,
            'int': int, 'len': len, 'list': list, 'max': max, 'min': min,
            'range': range, 'round': round, 'set': set, 'sorted': sorted,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
        }
    }

    # 4. Check for forbidden patterns using a generator expression for efficiency
    # This blacklist prevents access to dangerous functions not covered by the __builtins__ override.
    forbidden_patterns = [
        'import os', 'import sys', 'subprocess', 'shutil', 'open', 'eval', 'exec',
        'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr'
    ]
    if any(pattern in code_clean for pattern in forbidden_patterns):
        raise ValueError(f"Execution failed: Use of a forbidden keyword or function was detected.")

    try:
        # 5. Parse the code to check for syntax errors before execution
        try:
            ast.parse(code_clean)
        except (IndentationError, TabError, SyntaxError):
            lines = code_clean.split('\n')
            non_empty = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith('#')]
            if non_empty:
                def leading_spaces(s: str) -> int:
                    return len(s) - len(s.lstrip(' '))
                min_indent = min(leading_spaces(ln) for ln in non_empty)
                if min_indent > 0:
                    lines = [ln[min_indent:] if ln.startswith(' ' * min_indent) else ln for ln in lines]
                    code_clean = '\n'.join(lines)
            ast.parse(code_clean)

        # Execute the code in the sandboxed environment
        local_vars = {}
        exec(code_clean, safe_globals, local_vars)

        # 6. Refined result retrieval
        if 'result' in local_vars:
            return local_vars['result']

        # If no 'result' variable, try to eval the last line if it's an expression
        lines = code_clean.strip().split('\n')
        last_line = lines[-1].strip()
        if last_line and not last_line.startswith('#'):
            try:
                # Evaluate the last line in the same sandboxed context
                return eval(last_line, safe_globals, local_vars)
            except Exception:
                # The last line was not a valid expression (e.g., an assignment)
                return "Code executed successfully, but no output was returned."

        return "Code executed successfully, but no output was returned."

    except Exception as e:
        # Catch and re-raise exceptions with a clear, unified error message
        raise ValueError(f"Code execution failed with error: {e}")

def classify_query_complexity(query: str, col_names: list) -> tuple:
    """
    Classify if a query can be handled directly or needs LLM code generation.
    Returns (can_handle_directly, structured_result)
    """
    query_lower = query.lower()

    # Simple aggregation patterns that can be handled directly
    simple_patterns = [
        (r"how many unique (.+?)s?", "nunique"),
        (r"unique (.+?)s?", "unique"),
        (r"count of (.+?)s?", "nunique"),
        (r"sum of (.+?)s?", "sum"),
        (r"average of (.+?)s?", "mean"),
        (r"mean of (.+?)s?", "mean"),
        (r"minimum of (.+?)s?", "min"),
        (r"maximum of (.+?)s?", "max"),
        (r"min of (.+?)s?", "min"),
        (r"max of (.+?)s?", "max"),
        (r"list all (.+?)s?", "list"),
        (r"group by (.+?)s?", "groupby"),
        (r"how many (.+?)s?", "count"),
        (r"number of (.+?)s?", "count"),
    ]

    # Check for simple patterns
    for pattern, intent in simple_patterns:
        match = re.search(pattern, query_lower)
        if match:
            col_candidate = match.group(1).strip()
            # Fuzzy match column
            matches = get_close_matches(col_candidate, col_names, n=1, cutoff=0.7)
            if matches:
                return True, (intent, matches[0])

    # Complex patterns that need LLM
    complex_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'correlation', 'regression',
        'filter', 'where', 'condition', 'join', 'merge', 'pivot',
        'percentage', 'ratio', 'trend', 'compare', 'analysis',
        'distribution', 'histogram', 'scatter', 'line chart', 'bar chart'
    ]

    if any(keyword in query_lower for keyword in complex_keywords):
        return False, None

    # If no pattern matches, use LLM for safety
    return False, None


def handle_simple_query(df: pd.DataFrame, intent: str, column: str):
    """
    Handle simple queries directly without LLM.
    """
    try:
        if intent == "nunique":
            result = int(df[column].nunique())
            return {"type": "text", "payload": f"There are {result} unique values in column '{column}'."}

        elif intent == "unique" or intent == "list":
            result = df[column].dropna().unique().tolist()
            # Convert to serializable format
            result = [str(v) if pd.notna(v) else None for v in result]
            return {"type": "table", "payload": [{column: v} for v in result]}

        elif intent == "sum":
            if df[column].dtype in ['int64', 'float64']:
                result = float(df[column].sum())
                return {"type": "text", "payload": f"The sum of column '{column}' is {result}."}
            else:
                return {"type": "text", "payload": f"Cannot calculate sum for non-numeric column '{column}'."}

        elif intent == "mean":
            if df[column].dtype in ['int64', 'float64']:
                result = float(df[column].mean())
                return {"type": "text", "payload": f"The average of column '{column}' is {result:.2f}."}
            else:
                return {"type": "text", "payload": f"Cannot calculate mean for non-numeric column '{column}'."}

        elif intent == "min":
            result = df[column].min()
            return {"type": "text", "payload": f"The minimum value in column '{column}' is {result}."}

        elif intent == "max":
            result = df[column].max()
            return {"type": "text", "payload": f"The maximum value in column '{column}' is {result}."}

        elif intent == "count":
            result = int(df[column].count())
            return {"type": "text", "payload": f"There are {result} non-null records in column '{column}'."}

        elif intent == "groupby":
            group_counts = df.groupby(column).size().reset_index(name='count')
            payload = group_counts.to_dict(orient='records')
            return {"type": "table", "payload": payload}

        else:
            return {"type": "text", "payload": f"Intent '{intent}' not supported for direct handling."}

    except Exception as e:
        return {"type": "text", "payload": f"Error processing query: {str(e)}"}


def convert_plotly_arrays(obj):
    """
    Recursively convert any numpy arrays, pandas Series, or binary-encoded arrays in Plotly figure dicts to plain lists.
    """
    import numpy as np
    import base64
    
    if isinstance(obj, dict):
        # Special case: binary-encoded array (e.g., {'dtype': ..., 'bdata': ...})
        if set(obj.keys()) == {'dtype', 'bdata'}:
            try:
                dtype_str = obj.get('dtype', '')
                bdata = obj.get('bdata', '')
                
                if dtype_str and bdata:
                    # Decode base64 data
                    decoded_data = base64.b64decode(bdata)
                    
                    # Map plotly dtype strings to numpy dtypes
                    dtype_map = {
                        'i1': np.int8,
                        'i2': np.int16, 
                        'i4': np.int32,
                        'i8': np.int64,
                        'u1': np.uint8,
                        'u2': np.uint16,
                        'u4': np.uint32, 
                        'u8': np.uint64,
                        'f4': np.float32,
                        'f8': np.float64,
                    }
                    
                    numpy_dtype = dtype_map.get(dtype_str, np.float64)
                    
                    # Convert to numpy array and then to list
                    arr = np.frombuffer(decoded_data, dtype=numpy_dtype)
                    result = arr.tolist()
                    print(f"Successfully decoded binary data: {dtype_str} -> {len(result)} values")
                    return result
                else:
                    print("Warning: Binary data missing dtype or bdata")
                    return []
            except Exception as e:
                print(f"Failed to decode binary plotly data: {e}")
                return []
        return {k: convert_plotly_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_plotly_arrays(v) for v in obj]
    elif hasattr(obj, 'tolist') and hasattr(obj, 'dtype'):  # numpy array or pandas series
        try:
            return obj.tolist()
        except:
            # If tolist fails, convert to regular list
            return list(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def validate_chart_data(fig_dict):
    """
    Validate that a plotly figure dictionary contains actual data and is not empty.
    Returns True if the chart has valid data, False otherwise.
    """
    try:
        # Check if there's data in the figure
        data_traces = fig_dict.get('data', [])
        if not data_traces:
            return False
            
        # Check each trace for actual data
        for trace in data_traces:
            # Check for different types of data that plotly charts might have
            y_values = trace.get('y', [])
            x_values = trace.get('x', [])
            values = trace.get('values', [])  # For pie charts
            z_values = trace.get('z', [])     # For heatmaps
            
            # Check if any of the data arrays have content
            has_y_data = isinstance(y_values, list) and len(y_values) > 0 and any(v is not None and str(v).strip() != '' for v in y_values)
            has_x_data = isinstance(x_values, list) and len(x_values) > 0 and any(v is not None and str(v).strip() != '' for v in x_values)
            has_values_data = isinstance(values, list) and len(values) > 0 and any(v is not None and str(v).strip() != '' for v in values)
            has_z_data = isinstance(z_values, list) and len(z_values) > 0 and any(v is not None and str(v).strip() != '' for v in z_values)
            
            # For bar/line charts, we need both x and y data, or at least y data
            # For pie charts, we need values data
            # For heatmaps, we need z data
            if has_y_data or has_values_data or has_z_data:
                return True
                
        return False
        
    except Exception as e:
        print(f"Error validating chart data: {e}")
        return False


def format_result_for_response(result):
    """
    Format the execution result into the appropriate response format.
    """
    # Handle plotly figures. Use a specific check for plotly.graph_objects.Figure to avoid
    # incorrectly identifying pandas DataFrames as plotly figures, as both have a to_dict() method.
    if PLOTLY_AVAILABLE and isinstance(result, go.Figure):
        try:
            # It's a plotly figure - use the simpler to_dict() method to avoid binary encoding issues
            if hasattr(result, 'to_dict'):
                # Use to_dict which gives us a cleaner structure without binary encoding
                fig_dict = result.to_dict()
                fig_dict = convert_plotly_arrays(fig_dict)
                
                # Validate chart data before returning
                if validate_chart_data(fig_dict):
                    return {"type": "plotly", "payload": fig_dict}
                else:
                    print("Chart data validation failed - chart has empty data")
                    print(f"Chart traces: {len(fig_dict.get('data', []))}")
                    if fig_dict.get('data'):
                        first_trace = fig_dict['data'][0]
                        print(f"First trace keys: {list(first_trace.keys())}")
                        print(f"First trace y values: {first_trace.get('y', 'No y key')}")
                    return {"type": "text", "payload": "Chart could not be generated due to empty or invalid data. Please try a different query or check if the data contains the requested information."}
            
            elif hasattr(result, 'to_json'):
                # Fallback to to_json if to_dict is not available
                fig_dict = json.loads(result.to_json())
                fig_dict = convert_plotly_arrays(fig_dict)
                
                # Validate chart data before returning
                if validate_chart_data(fig_dict):
                    return {"type": "plotly", "payload": fig_dict}
                else:
                    print("Chart data validation failed - chart has empty data")
                    return {"type": "text", "payload": "Chart could not be generated due to empty or invalid data. Please try a different query or check if the data contains the requested information."}
                    
        except Exception as e:
            print(f"Error converting plotly figure: {e}")
            # Fall through to other handlers

    # Handle pandas DataFrames
    if isinstance(result, pd.DataFrame):
        # Replace NaN and inf values with None for JSON compatibility
        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        if len(result) > 1000:  # Limit large results
            result = result.head(1000)

        # Convert to records format without preserving index
        payload = result.to_dict(orient='records')
        
        # Remove any 'index' columns that might have been added
        if payload and isinstance(payload[0], dict) and 'index' in payload[0]:
            for record in payload:
                record.pop('index', None)

        return {"type": "table", "payload": payload}

    # Handle pandas Series
    if isinstance(result, pd.Series):
        # Replace NaN and inf values with None for JSON compatibility
        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        if len(result) > 1000:
            result = result.head(1000)

        # Convert to DataFrame for consistent display without index
        df_result = pd.DataFrame({
            result.name if hasattr(result, 'name') and result.name else 'value': result.values
        })

        payload = df_result.to_dict(orient='records')
        return {"type": "table", "payload": payload}

    # Handle dictionaries (correlation matrices, etc.)
    if isinstance(result, dict):
        # Try to convert to DataFrame if possible
        try:
            df_result = pd.DataFrame(result)
            payload = df_result.to_dict(orient='records')
            
            # Remove any 'index' columns that might have been added
            if payload and isinstance(payload[0], dict) and 'index' in payload[0]:
                for record in payload:
                    record.pop('index', None)
                    
            return {"type": "table", "payload": payload}
        except:
            # Return as text if can't convert
            return {"type": "text", "payload": str(result)}

    # Handle lists
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            return {"type": "table", "payload": result}
        else:
            return {"type": "text", "payload": str(result)}

    # Handle numpy arrays
    if hasattr(result, 'tolist'):
        try:
            list_result = result.tolist()
            return {"type": "text", "payload": str(list_result)}
        except:
            return {"type": "text", "payload": str(result)}

    # Handle simple values (numbers, strings)
    return {"type": "text", "payload": str(result)}


def process_raw_data_to_unique_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw CSV data to unique tickets with calculated fields, matching frontend logic
    """
    # Column indexes matching frontend COLUMNS constant
    COLUMNS = {
        'CREATION_DATE': 0,
        'TICKET_ID': 3, 
        'PRIORITY': 4,
        'STATUS_FROM': 5,
        'STATUS_TO': 6,
        'STATUS_CHANGE_DATE': 7,
        'MARCO': 9,
        'ASSIGNED_TO': 13,
        'CURRENT_STATUS': 15,
        'ELAPSED_TIME': 32,
        'resolSW': 33,
        'RESP_REM': 35,  # This maps to ResolRem column in CSV (0-based index)
        'REQ_STATUS': df.columns.get_loc("Req. Status - Description") if "Req. Status - Description" in df.columns else None,
        'RESOLUTION_DATE': df.columns.get_loc("Req. Resolution Date") if "Req. Resolution Date" in df.columns else None,
        'REQUEST_TYPE': df.columns.get_loc("Req. Type - Description EN") if "Req. Type - Description EN" in df.columns else None
    }
    
    # Determine key column names where possible for stable sorting
    id_col_name = 'Request - ID' if 'Request - ID' in df.columns else df.columns[COLUMNS['TICKET_ID']]
    change_date_col_name = 'Historical Status - Change Date' if 'Historical Status - Change Date' in df.columns else df.columns[COLUMNS['STATUS_CHANGE_DATE']]
    change_time_col_name = 'Historical Status - Change Time' if 'Historical Status - Change Time' in df.columns else None

    # Build a sortable datetime key to mimic frontend sort by date then time
    def parse_ddmmyyyy(date_str: Any) -> Any:
        try:
            if pd.isna(date_str):
                return pd.NaT
            parts = str(date_str).strip().split('/')
            if len(parts) != 3:
                return pd.NaT
            day, month, year = parts
            return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except Exception:
            return pd.NaT

    def parse_hhmmss(time_val: Any) -> Any:
        try:
            if time_val is None or (isinstance(time_val, float) and np.isnan(time_val)):
                return (0, 0, 0)
            s = str(int(time_val)) if isinstance(time_val, (int, np.integer)) else str(time_val)
            s = ''.join(ch for ch in s if ch.isdigit()).zfill(6)[:6]
            hh = int(s[0:2]); mm = int(s[2:4]); ss = int(s[4:6])
            return (hh, mm, ss)
        except Exception:
            return (0, 0, 0)

    # Create a combined sortable key
    sort_dates = df[change_date_col_name].apply(parse_ddmmyyyy) if change_date_col_name in df.columns else pd.Series([pd.NaT]*len(df))
    if change_time_col_name and change_time_col_name in df.columns:
        times_parsed = df[change_time_col_name].apply(parse_hhmmss)
        sort_times = pd.to_timedelta([pd.Timedelta(hours=h, minutes=m, seconds=s) for (h, m, s) in times_parsed])
    else:
        sort_times = pd.to_timedelta([0]*len(df), unit='s')

    df_sorted = df.copy()
    df_sorted['_sort_ts'] = sort_dates + sort_times

    # Sort by ID then by timestamp (ascending), take the last row per ticket
    df_sorted = df_sorted.sort_values(by=[id_col_name, '_sort_ts'], kind='stable')
    ticket_groups = df_sorted.groupby(id_col_name, sort=False)
    
    processed_tickets = []
    for ticket_id, group in ticket_groups:
        # Get the last row for each ticket (most recent state)
        last_row = group.iloc[-1]
        
        # Create processed ticket with calculated fields matching frontend
        processed_ticket = {
            'ticketId': last_row.iloc[COLUMNS['TICKET_ID']],
            'creationDate': last_row.iloc[COLUMNS['CREATION_DATE']],
            'priority': last_row.iloc[COLUMNS['PRIORITY']],
            'assignedTo': last_row.iloc[COLUMNS['ASSIGNED_TO']],
            'marconaName': last_row.iloc[COLUMNS['MARCO']],
            'currentStatus': last_row.iloc[COLUMNS['CURRENT_STATUS']],
            'elapsedTime': last_row.iloc[COLUMNS['ELAPSED_TIME']],
            # Compute breach status using numeric comparison like frontend
            'isBreached': (pd.to_numeric(last_row.iloc[COLUMNS['RESP_REM']], errors='coerce') < 0) if COLUMNS['RESP_REM'] is not None else False,
            'timeToBreach': last_row.iloc[COLUMNS['RESP_REM']],
            'totalTime': last_row.iloc[COLUMNS['resolSW']],
        }
        
        # Add optional columns if they exist
        if COLUMNS['REQ_STATUS'] is not None:
            processed_ticket['status'] = last_row.iloc[COLUMNS['REQ_STATUS']]
        if COLUMNS['RESOLUTION_DATE'] is not None:
            processed_ticket['resolutionDate'] = last_row.iloc[COLUMNS['RESOLUTION_DATE']]
        if COLUMNS['REQUEST_TYPE'] is not None:
            processed_ticket['requestType'] = last_row.iloc[COLUMNS['REQUEST_TYPE']]
            
        processed_tickets.append(processed_ticket)
    
    result_df = pd.DataFrame(processed_tickets)
    # Cleanup temp columns if leaked (should not exist here)
    return result_df

def process_raw_data_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw CSV data without grouping by ticket. Matches frontend by scanning all rows.

    - Calculates breach based on RespRem (column index 35) per raw row
    - Mirrors existing calculated field names used downstream so analysis code continues to work
    """
    # Column indexes matching frontend COLUMNS constant
    COLUMNS = {
        'CREATION_DATE': 0,
        'TICKET_ID': 3,
        'PRIORITY': 4,
        'STATUS_FROM': 5,
        'STATUS_TO': 6,
        'STATUS_CHANGE_DATE': 7,
        'MARCO': 9,
        'ASSIGNED_TO': 13,
        'CURRENT_STATUS': 15,
        'ELAPSED_TIME': 32,
        'resolSW': 33,
        'RESP_REM': 35,
        'REQ_STATUS': df.columns.get_loc("Req. Status - Description") if "Req. Status - Description" in df.columns else None,
        'RESOLUTION_DATE': df.columns.get_loc("Req. Resolution Date") if "Req. Resolution Date" in df.columns else None,
        'REQUEST_TYPE': df.columns.get_loc("Req. Type - Description EN") if "Req. Type - Description EN" in df.columns else None
    }

    # Build processed rows one-to-one with raw rows
    processed_rows = []
    for _, row in df.iterrows():
        processed_row = {
            'ticketId': row.iloc[COLUMNS['TICKET_ID']] if COLUMNS['TICKET_ID'] is not None else None,
            'creationDate': row.iloc[COLUMNS['CREATION_DATE']] if COLUMNS['CREATION_DATE'] is not None else None,
            'priority': row.iloc[COLUMNS['PRIORITY']] if COLUMNS['PRIORITY'] is not None else None,
            'assignedTo': row.iloc[COLUMNS['ASSIGNED_TO']] if COLUMNS['ASSIGNED_TO'] is not None else None,
            'marconaName': row.iloc[COLUMNS['MARCO']] if COLUMNS['MARCO'] is not None else None,
            'currentStatus': row.iloc[COLUMNS['CURRENT_STATUS']] if COLUMNS['CURRENT_STATUS'] is not None else None,
            'elapsedTime': row.iloc[COLUMNS['ELAPSED_TIME']] if COLUMNS['ELAPSED_TIME'] is not None else None,
            'isBreached': (row.iloc[COLUMNS['RESP_REM']] < 0) if COLUMNS['RESP_REM'] is not None and pd.notna(row.iloc[COLUMNS['RESP_REM']]) else False,
            'timeToBreach': row.iloc[COLUMNS['RESP_REM']] if COLUMNS['RESP_REM'] is not None else None,
            'totalTime': row.iloc[COLUMNS['resolSW']] if COLUMNS['resolSW'] is not None else None,
        }

        if COLUMNS['REQ_STATUS'] is not None:
            processed_row['status'] = row.iloc[COLUMNS['REQ_STATUS']]
        if COLUMNS['RESOLUTION_DATE'] is not None:
            processed_row['resolutionDate'] = row.iloc[COLUMNS['RESOLUTION_DATE']]
        if COLUMNS['REQUEST_TYPE'] is not None:
            processed_row['requestType'] = row.iloc[COLUMNS['REQUEST_TYPE']]

        processed_rows.append(processed_row)

    return pd.DataFrame(processed_rows)

def get_llm_analysis(query: str, df: pd.DataFrame, chat_history: List[Dict[str, str]], assume_ready: bool = False):
    """
    Use LLM to analyze the user's query, determine intent, and generate code if necessary, using chat history for context.
    """
    # If frontend provided the dataset, use it directly. Otherwise, group by ticket.
    processed_df = df if assume_ready else process_raw_data_to_unique_tickets(df)
    
    num_rows, num_cols = processed_df.shape
    col_names = list(processed_df.columns)
    sample_data = processed_df.head(3).to_dict(orient='records')
    dataset_info = f"""
Dataset Information ({'Frontend-Provided Report Dataset' if assume_ready else 'Unique Tickets, Grouped by Ticket ID'}):
- Shape: {num_rows} rows, {num_cols} fields
- Columns: {col_names}
- Sample Data (first 3 rows): {sample_data}
"""

    history_str = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in chat_history])

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0.1)

    llm_prompt = f"""
You are an expert data analyst AI. Your task is to analyze a user's question about a dataset and respond with a structured JSON object. You must also maintain a conversation and remember details from the chat history provided.

**Chat History (for context):**
---
{history_str}
---

**Dataset Information:**
{dataset_info}

**Current User Question:** {query}

**CRITICAL CONTEXT ANALYSIS:**
- If the user query contains ambiguous references like "how many are", "how many of them", "which ones", "those tickets", etc., you MUST look at the chat history to understand what they're referring to
- When the user says "them", "those", "these", they are referring to the subject/filter from the most recent data query
- Example: If previous query was "how many p3 normal tickets are there" and current query is "how many of them related to John", you should count p3 normal tickets that are related to John
- ALWAYS maintain the same filters/conditions from the previous query when processing follow-up questions
- Pay special attention to names, categories, priorities, or any filters mentioned in previous conversations
- If the current query seems like a follow-up, explicitly combine it with the context from the previous query

**IMPORTANT:**
- {'This dataset is provided by the frontend as-is (no backend processing applied). It reflects the current report view.' if assume_ready else 'This dataset is GROUPED by ticket. Each row represents the latest state for a unique ticket (same as frontend table).'}
- NOTE: ChatBot analyzes the COMPLETE unfiltered dataset provided. Frontend table may show fewer results due to active UI filters.
- Use the following column mappings per ticket:
  * ticketId: Unique ticket identifier
  * isBreached: Boolean (True/False) derived from negative timeToBreach
  * timeToBreach: Time remaining to breach (negative = breached)
  * elapsedTime: Total elapsed time for ticket
  * totalTime: Total resolution time
  * currentStatus: Current ticket status
  * assignedTo: Person assigned to ticket
  * priority: Ticket priority level
  * creationDate: When ticket was created
- When the user asks for COUNTS, TOTALS, SUMS, or other NUMERIC AGGREGATIONS, return only the numeric result, NOT the entire table.
- When the user asks to LIST, DISPLAY, or SHOW data, return the appropriate columns.
- CRITICAL: Distinguish between COUNT operations (return numbers) and LIST operations (return tables).
- **BREACH LOGIC**: Use the 'isBreached' column (boolean True/False) for all breach queries. This is already calculated and matches frontend logic.

**Instructions:**

1.  **Analyze the User's Intent:**
    *   If the question requires remembering something from the chat history (e.g., "what's my name?"), answer it based on the history.
    *   If the question is a **data analysis query** (e.g., asking for a count, sum, filter, chart, etc.), you MUST generate the required Python pandas code to answer it.
    *   If the question is a **greeting, small talk, or a general question not answerable from the data or history**, you should respond conversationally.

2.  **For all charts and visualizations, you MUST use Plotly (px, go, or ff). Do NOT use matplotlib or seaborn.**
    *   You MUST support all major chart types: bar, pie, line, scatter, box, histogram, violin, area, heatmap, and any other relevant Plotly chart type.
    *   For each chart, use the most appropriate Plotly function (e.g., px.bar, px.pie, px.line, px.scatter, px.box, px.histogram, px.violin, px.area, go.Heatmap, etc.).
    *   Always use the exact column names from the dataset.
    *   **For every chart, always generate the pandas code to compute the required data (e.g., value_counts, groupby, sum, mean, etc.) before passing it to Plotly. Do not pass empty arrays or raw columns directly to Plotly without aggregation.**
    *   Always check that the data used for chart axes/values is not empty before creating the chart. If the data is empty, set result = None.
    *   For pie/bar/box/line/scatter/histogram/violin charts, ensure the data arrays (like 'values', 'y', etc.) are not empty. If they are, set result = None.
    *   Use .dropna() or .fillna(0) as appropriate to avoid NaN in chart data.
    *   When creating boolean masks for filtering, always use .fillna(False) or .dropna() as appropriate to ensure the mask contains only True/False values (never NaN).
    *   Always set a meaningful chart title and axis labels.
    *   If the chart cannot be generated due to empty or invalid data, set result = None.

3.  **Response Format:**
    *   You MUST respond with a single JSON object containing two keys: `"type"` and `"payload"`.
    *   **For a conversational answer (including from memory):**
        *   `"type"`: "conversational_answer"
        *   `"payload"`: A string with the friendly, conversational response.
    *   **For a data analysis answer:**
        *   `"type"`: "data_analysis_answer"
        *   `"payload"`: A JSON object with two keys:
            *   `"explanation"`: A user-friendly text summary of what the code will do.
            *   `"code"`: A string containing the Python pandas code to execute. The final result must be stored in a variable named `result`.

**Example 1: Count Question (Return Number Only)**
User Question: "how many unique users are there"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "To find the number of unique users, I will count the distinct values in the assignedTo column.",
    "code": "result = df['assignedTo'].nunique()"
  }}
}}

**Example 1a: List Question (Return Table)**
User Question: "list all unique users" or "show me all user names"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here are all the unique users in the dataset.",
    "code": "unique_users = df['assignedTo'].dropna().unique()\\nresult = pd.DataFrame({{'User Name': sorted(unique_users)}})"
  }}
}}

**Example 2: Greeting**
User Question: "hello there"
Your JSON response:
{{
  "type": "conversational_answer",
  "payload": "Hello! How can I help you with your data analysis today?"
}}

**Example 3: Chart Request**
User Question: "show me a bar chart of statuses"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is a bar chart showing the distribution of current ticket statuses.",
    "code": "# Check if currentStatus column exists and has data\\nif 'currentStatus' in df.columns:\\n    status_counts = df['currentStatus'].dropna().value_counts()\\n    if not status_counts.empty and len(status_counts) > 0:\\n        # Convert to regular Python lists to avoid binary encoding\\n        x_values = [str(x) for x in status_counts.index]\\n        y_values = [int(y) for y in status_counts.values]\\n        result = px.bar(x=x_values, y=y_values, title='Ticket Status Distribution', labels={{'x':'Status', 'y':'Count'}})\\n    else:\\n        result = None\\nelse:\\n    result = None"
  }}
}}

**Example 3a: Breach Chart Request**
User Question: "show me a pie chart of breached vs non-breached tickets" or "chart showing breach status"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is a pie chart showing the distribution of breached vs non-breached tickets.",
    "code": "# Check if isBreached column exists\\nif 'isBreached' in df.columns:\\n    df_clean = df.dropna(subset=['isBreached'])\\n    if not df_clean.empty:\\n        df_clean['breach_status'] = df_clean['isBreached'].apply(lambda x: 'Breached' if x else 'Not Breached')\\n        breach_counts = df_clean['breach_status'].value_counts()\\n        if not breach_counts.empty and len(breach_counts) > 0:\\n            # Convert to regular Python lists to avoid binary encoding\\n            values_list = [int(v) for v in breach_counts.values]\\n            names_list = [str(n) for n in breach_counts.index]\\n            result = px.pie(values=values_list, names=names_list, title='Ticket Breach Status Distribution')\\n        else:\\n            result = None\\n    else:\\n        result = None\\nelse:\\n    result = None"
  }}
}}

**Example 4: Safe Boolean Mask**
User Question: "Show all rows where Status is 'Open'"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here are all rows where Status is 'Open'.",
    "code": "mask = (df['Status'] == 'Open').fillna(False)\nresult = df[mask]"
  }}
}}

**Example 5: GroupBy and Count**
User Question: "show me ticket counts by assigned user"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is a breakdown of ticket counts by assigned user.",
    "code": "result = df.groupby('assignedTo').size().reset_index(name='ticket_count').sort_values('ticket_count', ascending=False)"
  }}
}}

**Example 6: Top N and Max Count**
User Question: "Display the user assigned to the most tickets and show top 10"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is the user assigned to the most tickets, along with the top 10 users by ticket count.",
    "code": "user_counts = df['assignedTo'].value_counts()\\ntop_10_users = user_counts.head(10).reset_index()\\ntop_10_users.columns = ['User Name', 'Ticket Count']\\nresult = top_10_users"
  }}
}}

**Example 6a: Just Count (Number Only)**
User Question: "how many tickets are there" or "count of tickets"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is the total count of tickets in the dataset.",
    "code": "result = len(df)"
  }}
}}

**Example 6b: Count by Category (Number Only)**
User Question: "how many high priority tickets are there"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is the count of high priority tickets.",
    "code": "result = len(df[df['priority'].str.contains('High', case=False, na=False)])"
  }}
}}

**Example 6e: Elapsed Time Analysis**
User Question: "what is the average elapsed time for tickets?"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is the average elapsed time for all tickets.",
    "code": "result = df['elapsedTime'].mean()"
  }}
}}

**Example 6f: Time to Breach Analysis**
User Question: "show tickets with negative time to breach"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here are tickets with negative time to breach (already breached).",
    "code": "negative_tickets = df[df['timeToBreach'] < 0]\\nresult = negative_tickets[['ticketId', 'timeToBreach', 'assignedTo', 'priority']].head(50)"
  }}
}}

**Example 6c: Breach Count (Number Only)**
User Question: "how many tickets are breached" or "count of breached tickets"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is the count of breached tickets using the pre-calculated breach status.",
    "code": "result = len(df[df['isBreached'] == True])"
  }}
}}

**Example 8: Follow-up Context Query (CRITICAL for accurate results)**
Previous Query: "how many p3 normal tickets are there"
Current Query: "how many of them related to Shatabdi Roy"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Based on the previous query about p3 normal tickets, I'll count how many of those tickets are related to Shatabdi Roy.",
    "code": "# First apply the filter from previous context: p3 normal tickets\np3_normal_mask = (df['priority'].str.contains('P3', case=False, na=False)) & (df['priority'].str.contains('Normal', case=False, na=False))\np3_normal_tickets = df[p3_normal_mask]\n\n# Then filter for Shatabdi Roy in the relevant columns\nshatabdi_mask = (p3_normal_tickets['assignedTo'].str.contains('Shatabdi Roy', case=False, na=False)) | (p3_normal_tickets['marconaName'].str.contains('Shatabdi Roy', case=False, na=False))\nresult = len(p3_normal_tickets[shatabdi_mask])"
  }}
}}

**Example 9: Another Follow-up Context Query**
Previous Query: "show me high priority incidents"
Current Query: "how many of them are assigned to John"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Based on the previous query about high priority incidents, I'll count how many are assigned to John.",
    "code": "# Apply the filter from previous context: high priority\nhigh_priority_mask = df['priority'].str.contains('High', case=False, na=False)\nhigh_priority_tickets = df[high_priority_mask]\n\n# Then filter for John in assignedTo column\njohn_mask = high_priority_tickets['assignedTo'].str.contains('John', case=False, na=False)\nresult = len(high_priority_tickets[john_mask])"
  }}
}}

**Example 6d: Breach List (Table)**
User Question: "show me breached tickets" or "list all breached tickets"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here are all the breached tickets using the pre-calculated breach status.",
    "code": "breached_df = df[df['isBreached'] == True]\\nresult = breached_df[['ticketId', 'assignedTo', 'priority', 'timeToBreach', 'currentStatus']].head(100)"
  }}
}}

**Example 7: Conversational Memory**
(After user says "my name is Sainath")
User Question: "what's my name?"
Your JSON response:
{{
  "type": "conversational_answer",
  "payload": "Your name is Sainath."
}}

### MUST FOLLOW:
* **CONTEXT CONSISTENCY (CRITICAL):**
  - ALWAYS review the chat history before answering ANY question
  - If a question contains pronouns like "them", "those", "these", you MUST identify what they refer to from previous queries
  - Follow-up questions should ALWAYS give the same result when asked multiple times (consistency is critical)
  - When processing follow-up queries, explicitly combine the current filter with filters from the referenced previous query
* **COUNT vs LIST operations:**
  - For COUNT questions (e.g., "how many", "count of", "total number"): Return ONLY the numeric result (e.g., `result = df['column'].nunique()`)
  - For LIST questions (e.g., "list all", "show me", "display"): Return a DataFrame/table (e.g., `result = pd.DataFrame({{'Column': values}})`)
* While generating result in tabular format, do not generate any index for that, just give the results in the tabular format straight away without any indexes.
* If you dont find any results like empty payload with the user prompts, then you have to give the response as "The Current Query is not processed efficiently, Please try with Other prompts". You have to give this statement as the response only.
* If the user asks about statuses or priorities then you have to look at "Request - Priority Description" only.
* For priority filtering, use column names like 'priority', 'Request - Priority Description', or similar priority-related columns
* When filtering by names (like "Shatabdi Roy"), check both 'assignedTo' and 'marconaName' columns or any name-related columns



Now, provide the JSON response for the given user {query}. Do not include any text or markdown outside of the JSON object.
"""

    response = llm.invoke(llm_prompt)
    content = response.content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return as a conversational response
        print(f"JSON parsing failed for LLM response: {e}. Raw response: {content}")
        return {"type": "conversational_answer", "payload": content}


@sla_router.get("/api/get_csv_data/{filename}")
async def get_csv_data(filename: str):
    """
    Retrieve CSV data from uploads_sla directory based on filename
    """
    try:
        # Sanitize filename to prevent directory traversal
        filename = os.path.basename(filename)
        
        # Ensure the file has a valid extension
        if not filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and Excel files are supported.")
        
        file_path = os.path.join('uploads_sla', filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in uploads directory.")
        
        # Read the file based on its extension
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or contains no data.")
        
        # Convert DataFrame to JSON-safe format
        records = convert_to_json_safe(df)
        
        return JSONResponse(content={
            "filename": filename,
            "records": records,
            "total_rows": len(records),
            "columns": list(df.columns)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@sla_router.post("/api/Explore_sla/")
async def senior_data_analysis_sla(query: str = Form(...), session_id: str = Form(None), dataset: str = Form(None)):
    try:
        # Session handling
        if not session_id:
            session_id = str(uuid4())
            print(f"Generated new session ID: {session_id}")
        
        print(f"Processing query for session {session_id}: {query}")

        # Get chat history for LLM
        chat_history_for_llm = manage_session_memory(session_id, get_history=True)
        
        # Enhanced logging for follow-up question debugging
        if chat_history_for_llm:
            print(f"Session {session_id} has {len(chat_history_for_llm)} messages in history")
            # Look for potential follow-up indicators
            follow_up_indicators = ['them', 'those', 'these', 'it', 'they']
            if any(indicator in query.lower() for indicator in follow_up_indicators):
                print(f"FOLLOW-UP QUERY DETECTED: '{query}'")
                if len(chat_history_for_llm) >= 2:
                    last_user_query = None
                    for msg in reversed(chat_history_for_llm):
                        if msg['role'] == 'user':
                            last_user_query = msg['content']
                            break
                    print(f"Previous user query for context: '{last_user_query}'")
        else:
            print(f"No chat history found for session {session_id}")

        # Determine dataset source: frontend-provided JSON or backend CSV
        df = None
        assume_ready = False
        if dataset:
            try:
                # Expect dataset as JSON string array of objects
                records = json.loads(dataset)
                if not isinstance(records, list) or (len(records) > 0 and not isinstance(records[0], dict)):
                    raise ValueError("dataset must be a JSON array of objects")
                df = pd.DataFrame(records)
                assume_ready = True
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid dataset payload: {str(e)}")
        else:
            # Fallback to CSV persisted on server
            csv_file_path = os.path.join('uploads_sla/data1.csv')
            if not os.path.exists(csv_file_path):
                raise HTTPException(status_code=400, detail="No dataset found. Please upload a file first.")
            df = pd.read_csv(csv_file_path)
            if df.empty:
                raise HTTPException(status_code=400, detail="Dataset is empty")

        # Step 1: Get AI analysis (intent, code, explanation) using chat history
        analysis_result = get_llm_analysis(query, df, chat_history=chat_history_for_llm, assume_ready=assume_ready)
        print("Analysis results",analysis_result)
        response_type = analysis_result.get("type")
        payload = analysis_result.get("payload")

        # Determine bot response text for memory
        bot_response_text = "I have processed your request."
        if response_type == "conversational_answer":
            bot_response_text = payload
        elif response_type == "data_analysis_answer":
            bot_response_text = payload.get("explanation", "I have generated the data analysis you requested.")
        elif isinstance(payload, str):
            bot_response_text = payload

        # Store current exchange in memory using centralized function
        manage_session_memory(session_id, user_message=query, bot_message=bot_response_text)

        if not response_type or not payload:
            return JSONResponse(content={"type": "text",
                                         "payload": "I'm sorry, I couldn't process that request. Please try rephrasing.",
                                         "session_id": session_id}, status_code=200)

        # Step 2: Handle based on response type
        if response_type == "conversational_answer":
            return JSONResponse(content={"type": "text", "payload": payload, "session_id": session_id}, status_code=200)

        elif response_type == "data_analysis_answer":
            explanation = payload.get("explanation")
            code = payload.get("code")

            if not code:
                return JSONResponse(content={"type": "text",
                                             "payload": explanation or "I understood you wanted to analyze the data, but I couldn't generate the right code. Please try again.",
                                             "session_id": session_id}, status_code=200)

            try:
                # Step 3: Use dataset directly if provided by frontend; otherwise group by ticket
                processed_df = df if assume_ready else process_raw_data_to_unique_tickets(df)
                
                # Execute code on processed data
                result = safe_execute_pandas_code(code, processed_df)
                
                # Step 4: Format result
                formatted_response = format_result_for_response(result)

                # Step 5: Add initial explanation and return
                formatted_response['explanation'] = explanation
                formatted_response['session_id'] = session_id
                return JSONResponse(content=formatted_response, status_code=200)

            except Exception as e:
                error_msg = f"There was an error executing the analysis: {str(e)}"
                return JSONResponse(content={"type": "text", "payload": error_msg, "explanation": explanation,
                                             "session_id": session_id}, status_code=200)

        else:
            return JSONResponse(
                content={"type": "text", "payload": f"Unrecognized response type from AI model: {response_type}",
                         "session_id": session_id}, status_code=200)

    except Exception as e:
        print(f"An unexpected error occurred: {traceback.format_exc()}")
        return JSONResponse(content={"error": str(e), "session_id": session_id if 'session_id' in locals() else None},
                            status_code=500)

def is_general_query(query: str) -> bool:
    """
    Check if query is general/global and doesn't need dataset search
    """
    query_lower = query.lower().strip()
    
    # General greetings and conversations
    general_patterns = [
        # Greetings
        r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
        r'\bwhat\'s up\b', r'\bhow are you\b', r'\bwhat can you do\b',
        
        # General questions not related to tickets/data
        r'\bwhat is\b.*\b(python|java|sql|programming|coding|software)\b',
        r'\bhow to\b.*\b(code|program|learn|study)\b',
        r'\bwhat are\b.*\b(best practices|advantages|benefits)\b',
        r'\btell me about\b.*\b(technology|tools|methods)\b',
        
        # Weather, time, general info
        r'\b(weather|time|date|today|tomorrow)\b',
        r'\bwhat\'s the\b.*\b(weather|time|date)\b',
        
        # Help and instructions
        r'\b(help|instruction|guide|tutorial)\b(?!.*\b(ticket|error|issue|problem|sla)\b)',
        
        # Definitions and explanations (non-technical)
        r'\bdefine\b(?!.*\b(error|issue|problem|ticket|sla)\b)',
        r'\bexplain\b(?!.*\b(error|issue|problem|ticket|sla|incident)\b)',
        
        # Personal questions
        r'\b(who are you|what is your name|tell me about yourself)\b'
    ]
    
    # Dataset/ticket-related patterns (should use vector search)
    dataset_patterns = [
        r'\b(ticket|tickets|issue|issues|problem|problems|error|errors)\b',
        r'\b(incident|incidents|sla|breach|breached|priority)\b',
        r'\b(show|list|display|find|search)\b.*\b(ticket|issue|error|incident)\b',
        r'\b(how to fix|resolve|solution|solve)\b.*\b(error|issue|problem)\b',
        r'\b(database|login|payment|connection|timeout|server)\b.*\b(error|issue|problem)\b',
        r'\bsimilar\b.*\b(case|cases|ticket|tickets|issue|issues)\b'
    ]
    
    # Check if it's dataset-related first (higher priority)
    for pattern in dataset_patterns:
        if re.search(pattern, query_lower):
            return False  # Use vector search
    
    # Check if it's general
    for pattern in general_patterns:
        if re.search(pattern, query_lower):
            return True  # Use fast response
    
    # If query is very short (likely general)
    if len(query_lower.split()) <= 3 and not any(word in query_lower for word in ['ticket', 'error', 'issue', 'problem']):
        return True
    
    # Default to vector search for safety
    return False

def get_fast_general_response(query: str) -> dict:
    """
    Generate fast responses for general queries
    """
    query_lower = query.lower().strip()
    
    # Greetings
    if re.search(r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b', query_lower):
        return {
            "type": "text",
            "payload": "Hello! I'm your Knowledge Base assistant. I can help you search for similar tickets, find solutions to technical issues, or answer questions about incidents and SLA management. What would you like to know?",
            "explanation": "Greeting response"
        }
    
    # What can you do
    if re.search(r'\b(what can you do|what are you|who are you|help)\b', query_lower):
        return {
            "type": "text", 
            "payload": """I'm your Knowledge Base Search Assistant! Here's what I can help you with:

🔍 **Search Functions:**
• Find similar tickets and incidents
• Search for solutions to technical problems
• Look up error resolutions and fixes

📊 **Query Examples:**
• "show me tickets about login errors"
• "how to fix database timeout issues"  
• "find similar payment processing problems"
• "list high priority incidents"

💡 **I can provide:**
• Table results with ticket details
• Text explanations and solutions
• Similarity-based recommendations

Just ask me about any technical issues, errors, or incidents you need help with!""",
            "explanation": "Assistant capabilities overview"
        }
    
    # Time/Date
    if re.search(r'\b(time|date|today|what time)\b', query_lower):
        from datetime import datetime
        now = datetime.now()
        return {
            "type": "text",
            "payload": f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            "explanation": "Current time response"
        }
    
    # General tech questions
    if re.search(r'\bwhat is\b.*\b(python|java|sql|programming)\b', query_lower):
        return {
            "type": "text",
            "payload": "I'm specialized in helping with ticket search and incident resolution. For general programming questions, I'd recommend consulting official documentation or programming resources. However, I can help you find similar technical issues and their solutions in our knowledge base!",
            "explanation": "Redirecting to specialized function"
        }
    
    # Default general response
    return {
        "type": "text",
        "payload": "I'm your Knowledge Base assistant, specialized in searching tickets and finding solutions to technical issues. Try asking me about specific errors, incidents, or problems you'd like to resolve. For example: 'show me login error tickets' or 'how to fix database issues'.",
        "explanation": "General guidance response"
    }

@sla_router.post("/api/vector_search/")
async def vector_search_sla(query: str = Form(...), session_id: str = Form(None)):
    """
    Vector-based search API for SLA data - searches for similar tickets and solutions
    """
    try:
        # Session handling
        if not session_id:
            session_id = str(uuid4())
            print(f"Generated new session ID for vector search: {session_id}")
        
        print(f"Processing vector search query for session {session_id}: {query}")

        # Check if this is a general query that doesn't need dataset search
        if is_general_query(query):
            print(f"General query detected: '{query}' - providing fast response")
            fast_response = get_fast_general_response(query)
            
            # Store in memory for context using centralized function
            manage_session_memory(session_id, user_message=query, bot_message=fast_response["explanation"])
            
            return JSONResponse(content={
                "type": fast_response["type"],
                "payload": fast_response["payload"], 
                "explanation": fast_response["explanation"],
                "session_id": session_id
            }, status_code=200)

        # Check if vector search is available
        if not VECTOR_SEARCH_AVAILABLE:
            return JSONResponse(content={
                "type": "text",
                "payload": "Vector search functionality is not available. Please ensure scikit-learn is installed.",
                "session_id": session_id
            }, status_code=200)

        # Get chat history for LLM
        chat_history_for_llm = manage_session_memory(session_id, get_history=True)

        # Load latest uploaded file
        csv_file_path = os.path.join('uploads_sla/data1.csv')
        if not os.path.exists(csv_file_path):
            return JSONResponse(content={
                "type": "text",
                "payload": "No dataset found. Please upload a file first.",
                "session_id": session_id
            }, status_code=200)

        df = pd.read_csv(csv_file_path)
        if df.empty:
            return JSONResponse(content={
                "type": "text",
                "payload": "Dataset is empty. Please upload a valid file.",
                "session_id": session_id
            }, status_code=200)

        # Perform vector search
        search_results = perform_vector_search(query, df)
        
        # Get AI response based on search results
        ai_response = get_vector_search_response(query, search_results, chat_history_for_llm)
        
        # Determine response type and payload
        response_type = ai_response.get("type", "text")
        payload = ai_response.get("payload", "No results found.")
        explanation = ai_response.get("explanation", "")
        
        # Determine bot response text for memory
        bot_response_text = explanation if explanation else str(payload)[:200] + "..." if len(str(payload)) > 200 else str(payload)
        
        # Store current exchange in memory using centralized function
        manage_session_memory(session_id, user_message=query, bot_message=bot_response_text)

        return JSONResponse(content={
            "type": response_type,
            "payload": payload,
            "explanation": explanation,
            "session_id": session_id
        }, status_code=200)

    except Exception as e:
        print(f"Vector search error occurred: {traceback.format_exc()}")
        return JSONResponse(content={
            "type": "text",
            "payload": f"An error occurred during vector search: {str(e)}",
            "session_id": session_id if 'session_id' in locals() else None
        }, status_code=500)

@sla_router.get("/api/predict_incident/")
async def predict_incident():
    """
    Get the latest file from uploads_sla and call external predict_file API
    """
    try:
        # Check if data1.csv exists in uploads_sla
        csv_file_path = os.path.join('uploads_sla', 'data1.csv')
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail="No data file found. Please upload a file first.")
        
        # Read the file
        with open(csv_file_path, 'rb') as file:
            file_content = file.read()
            
        files = {
            'file': ('data1.csv', file_content, 'text/csv')
        }
        
        headers = {
            'Cookie': 'csrftoken=Gz4H911IfjC96YSqY2ToTJcZcbZ5br0F'
        }
        
        # Make the external API call
        external_url = 'http://54.169.213.200:4006/predict_file'
        
        # Use requests to make the external API call
        external_response = requests.post(
            external_url,
            files=files,
            headers=headers,
            timeout=60  # 60 seconds timeout
        )
        
        if not external_response.ok:
            raise HTTPException(
                status_code=external_response.status_code,
                detail=f"External API error: {external_response.text}"
            )
        
        # Parse the response
        try:
            result = external_response.json()
        except ValueError:
            # If not JSON, return as text
            result = {"message": external_response.text}
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "message": "Incident prediction completed successfully"
        })
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"External API call failed: {str(e)}")
    except Exception as e:
        print(f"Predict incident error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@sla_router.post("/api/upload_and_predict/")
async def upload_and_predict(file: UploadFile = File(...)):
    """
    Upload file directly to external predict_file API and return response
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in ['.csv', '.xlsx', '.xls']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV and Excel files are allowed.")

        # Read the file content
        file_content = await file.read()
        
        # Prepare for external API call
        files = {
            'file': (filename, file_content, file.content_type)
        }
        
        headers = {
            'Cookie': 'csrftoken=Gz4H911IfjC96YSqY2ToTJcZcbZ5br0F'
        }
        
        # Make the external API call
        external_url = 'http://54.169.213.200:4006/predict_file'
        
        external_response = requests.post(
            external_url,
            files=files,
            headers=headers,
            timeout=60
        )
        
        if not external_response.ok:
            raise HTTPException(
                status_code=external_response.status_code,
                detail=f"External API error: {external_response.text}"
            )
        
        # Parse the response
        try:
            result = external_response.json()
        except ValueError:
            result = {"message": external_response.text}
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "message": f"File '{filename}' processed successfully",
            "filename": filename
        })
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"External API call failed: {str(e)}")
    except Exception as e:
        print(f"Upload and predict error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@sla_router.get("/api/debug_session_memory/{session_id}")
async def debug_session_memory(session_id: str):
    """
    Debug endpoint to see what's stored in session memory
    """
    try:
        with SESSION_MEMORY_LOCK:
            memory = SESSION_MEMORY.get(session_id, [])
            
        return JSONResponse(content={
            "session_id": session_id,
            "memory_size": len(memory),
            "messages": memory,
            "total_sessions": len(SESSION_MEMORY)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@sla_router.delete("/api/clear_session_memory/{session_id}")
async def clear_session_memory(session_id: str):
    """
    Clear session memory for testing purposes
    """
    try:
        with SESSION_MEMORY_LOCK:
            if session_id in SESSION_MEMORY:
                message_count = len(SESSION_MEMORY[session_id])
                del SESSION_MEMORY[session_id]
                return JSONResponse(content={
                    "session_id": session_id,
                    "cleared_messages": message_count,
                    "status": "cleared"
                })
            else:
                return JSONResponse(content={
                    "session_id": session_id,
                    "cleared_messages": 0,
                    "status": "not_found"
                })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")

@sla_router.post("/api/test_context_consistency/")
async def test_context_consistency(session_id: str = Form(...)):
    """
    Test endpoint to verify context consistency for follow-up questions
    """
    try:
        # Simulate the exact scenario the user mentioned
        test_queries = [
            "how many p3 normal tickets are there",
            "how many of them related to Shatabdi Roy"
        ]
        
        results = []
        for query in test_queries:
            # Make a call to the main analysis endpoint
            import requests
            import json
            
            # This would normally be called internally, but for testing we'll simulate it
            print(f"Testing query: {query}")
            
            # Get current session memory
            memory = manage_session_memory(session_id, get_history=True)
            results.append({
                "query": query,
                "memory_before": len(memory),
                "context_detected": any(indicator in query.lower() for indicator in ['them', 'those', 'these', 'it', 'they'])
            })
            
            # Add to memory for next iteration
            manage_session_memory(session_id, user_message=query, bot_message=f"Test response for: {query}")
        
        return JSONResponse(content={
            "session_id": session_id,
            "test_results": results,
            "final_memory_size": len(manage_session_memory(session_id, get_history=True))
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")

def extract_search_keywords(query: str) -> List[str]:
    """
    Extract specific keywords/topics from the search query for better filtering
    """
    query_lower = query.lower()
    keywords = []
    
    # Common business/technical areas
    business_areas = {
        'scm': ['scm', 'supply chain', 'supply chain management', 'sales order', 'allocation'],
        'sap': ['sap', 'r/3', 'ecc', 's/4hana', 'hana'],
        'login': ['login', 'authentication', 'sign in', 'password', 'credential'],
        'payment': ['payment', 'billing', 'invoice', 'financial'],
        'database': ['database', 'db', 'sql', 'oracle', 'mysql'],
        'network': ['network', 'connection', 'timeout', 'connectivity'],
        'email': ['email', 'mail', 'smtp', 'outlook'],
        'printing': ['print', 'printer', 'printing'],
        'performance': ['performance', 'slow', 'loading', 'timeout'],
        'error': ['error', 'exception', 'failure', 'issue', 'problem']
    }
    
    # Check for business area keywords
    for area, terms in business_areas.items():
        if any(term in query_lower for term in terms):
            keywords.extend(terms)
    
    # Extract quoted terms
    import re
    quoted_terms = re.findall(r'"([^"]*)"', query)
    keywords.extend([term.lower() for term in quoted_terms])
    
    # Extract capitalized terms (likely acronyms or proper nouns)
    cap_terms = re.findall(r'\b[A-Z]{2,}\b', query)
    keywords.extend([term.lower() for term in cap_terms])
    
    return list(set(keywords))  # Remove duplicates

def perform_vector_search(query: str, df: pd.DataFrame, top_k: int = 20):
    """
    Perform vector-based similarity search on the dataset with flexible column detection
    Returns unique tickets based on available ID column with reasonable similarity threshold
    """
    try:
        # Extract keywords for additional filtering
        search_keywords = extract_search_keywords(query)
        print(f"Extracted keywords for filtering: {search_keywords}")
        
        # Print dataset info for debugging
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # Use full dataset for better search coverage
        df_subset = df.copy()
        
        print(f"Processing {len(df_subset)} rows for vector search...")
        
        # Flexible column detection - try multiple possible column names
        text_columns = ['Request - Text Request', 'Request Description', 'Description', 'Issue Description', 
                       'Text Request', 'Problem Description', 'Subject', 'Summary', 'Details']
        id_columns = ['Request - ID', 'Ticket ID', 'ID', 'Request ID', 'ticketId', 'Ticket_ID']
        
        # Find the best text column
        text_col = None
        for col in text_columns:
            if col in df.columns:
                text_col = col
                print(f"Using text column: {text_col}")
                break
        
        # If no standard text column found, try to find columns with text content
        if not text_col:
            best_col = None
            best_score = 0
            
            for col in df.columns:
                if df[col].dtype == 'object':  # String columns
                    non_null_count = df[col].notna().sum()
                    if non_null_count > 0:
                        sample_texts = df[col].dropna().head(10).astype(str).tolist()
                        avg_length = sum(len(text) for text in sample_texts) / len(sample_texts) if sample_texts else 0
                        
                        # Score based on average length and non-null count
                        score = avg_length * (non_null_count / len(df))
                        
                        if score > best_score and avg_length > 10:  # Minimum meaningful text length
                            best_score = score
                            best_col = col
            
            if best_col:
                text_col = best_col
                print(f"Using detected text column: {text_col} (avg length: {best_score:.1f})")
        
        # Find the best ID column
        id_col = None
        for col in id_columns:
            if col in df.columns:
                id_col = col
                print(f"Using ID column: {id_col}")
                break
        
        # Fallback to first column if no ID column found
        if not id_col and len(df.columns) > 0:
            id_col = df.columns[0]
            print(f"Using fallback ID column: {id_col}")
        
        if not text_col:
            print("No suitable text column found for vector search")
            return []
        
        # Create search texts
        search_texts = []
        valid_rows = []
        keyword_matches = []
        request_ids = []
        
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            # Get text content from the identified text column
            request_text = row.get(text_col, '')
            request_id = row.get(id_col, f'row_{idx}') if id_col else f'row_{idx}'
            
            # More flexible text validation - accept shorter texts too
            if (pd.notna(request_text) and str(request_text).strip() and 
                len(str(request_text).strip()) > 3):  # Reduced from 10 to 3 characters
                
                # Clean the request text
                cleaned_text = str(request_text).strip()
                
                # Check for keyword matches in request text
                text_lower = cleaned_text.lower()
                keyword_score = 0
                if search_keywords:
                    keyword_score = sum(1 for keyword in search_keywords if keyword in text_lower)
                
                search_texts.append(cleaned_text)
                valid_rows.append(idx)
                keyword_matches.append(keyword_score)
                request_ids.append(str(request_id).strip())
        
        if not search_texts:
            print("ERROR: No valid request text found for vector search")
            print(f"Text column used: {text_col}")
            print(f"ID column used: {id_col}")
            print("Sample data from text column:")
            if text_col and text_col in df.columns:
                sample_texts = df[text_col].dropna().head(5).tolist()
                for i, text in enumerate(sample_texts):
                    print(f"  Row {i}: '{str(text)[:100]}...' (length: {len(str(text))})")
            else:
                print("  Text column not found in dataset")
            
            print("\nDebugging info:")
            print(f"Total rows in dataset: {len(df)}")
            print(f"Text column '{text_col}' exists: {text_col in df.columns if text_col else False}")
            print(f"Non-null values in text column: {df[text_col].notna().sum() if text_col and text_col in df.columns else 'N/A'}")
            return []
        
        print(f"Created search texts for {len(search_texts)} valid rows")

        # TF-IDF vectorization instead of sentence-transformers
        print("Vectorizing texts with TF-IDF...")
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=50000)
        text_matrix = vectorizer.fit_transform(search_texts)
        query_vector = vectorizer.transform([query])

        print("Calculating similarities...")
        similarities = cosine_similarity(query_vector, text_matrix)[0]
        
        # Combine similarity scores with keyword matches for better ranking
        combined_scores = []
        for i, (sim, keyword_score) in enumerate(zip(similarities, keyword_matches)):
            # Boost similarity score based on keyword matches
            keyword_boost = min(keyword_score * 0.1, 0.3)  # Max 30% boost
            final_score = sim + keyword_boost
            combined_scores.append((i, final_score, sim, keyword_score, request_ids[i]))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results with HIGHER similarity threshold and ensure unique Request IDs
        filtered_results = []
        seen_request_ids = set()
        
        for idx, combined_score, original_sim, keyword_score, request_id in combined_scores:
            # Reasonable similarity threshold - allow more matches for better search results
            min_threshold = 0.1 if keyword_score > 0 else 0.15  # Reduced from 0.3/0.5 to 0.1/0.15 for better recall
            
            # Only include if similarity is above threshold and Request ID is unique
            if original_sim > min_threshold and request_id not in seen_request_ids:
                seen_request_ids.add(request_id)
                
                original_row_idx = valid_rows[idx]
                row_data = df_subset.iloc[original_row_idx]
                
                # Create clean result dict without NaN values
                result = {}
                for key, value in row_data.items():
                    if pd.isna(value) or value is None:
                        result[key] = ""
                    elif isinstance(value, float):
                        # Handle special float values
                        if np.isnan(value) or np.isinf(value):
                            result[key] = ""
                        else:
                            result[key] = float(value)
                    else:
                        result[key] = str(value)
                
                # Add search metadata
                result['similarity_score'] = float(original_sim)
                result['keyword_matches'] = int(keyword_score)
                result['combined_score'] = float(combined_score)
                result['search_text'] = search_texts[idx][:200] + "..." if len(search_texts[idx]) > 200 else search_texts[idx]
                
                filtered_results.append(result)
                
                # Stop when we have enough unique results
                if len(filtered_results) >= top_k:
                    break
        
        print(f"Found {len(filtered_results)} unique similar results with threshold {min_threshold}")
        
        # Log some debug info about top results
        if filtered_results:
            print("Top 3 unique results:")
            for i, result in enumerate(filtered_results[:3]):
                print(f"  {i+1}. ID: {result.get('Request - ID', 'N/A')}, Similarity: {result['similarity_score']:.3f}, Keywords: {result['keyword_matches']}")
                print(f"      Request Text: {result.get('Request - Text Request', 'N/A')[:100]}...")
        
        return filtered_results
        
    except Exception as e:
        print(f"Vector search error: {e}")
        import traceback
        traceback.print_exc()
        return []

def classify_vector_query_intent(query: str) -> str:
    """
    Classify the intent of vector search queries
    Returns: 'count', 'list', 'explain', or 'general'
    """
    query_lower = query.lower().strip()
    
    # Count patterns
    count_patterns = [
        r'\b(how many|count|number of|total)\b',
        r'\b(count of|give count|show count)\b',
        r'\bhow many\b.*\btickets?\b',
        r'\bcount\b.*\btickets?\b'
    ]
    
    # List/display patterns  
    list_patterns = [
        r'\b(show|list|display|find|get)\b.*\btickets?\b',
        r'\b(show me|list all|display all)\b',
        r'\btickets?\b.*\b(about|related to|containing)\b'
    ]
    
    # Explanation patterns
    explain_patterns = [
        r'\b(how to|what causes|why|explain|solution|fix|resolve)\b',
        r'\btell me about\b',
        r'\bwhat is\b.*\b(cause|reason|solution)\b'
    ]
    
    # Check patterns in order of priority
    for pattern in count_patterns:
        if re.search(pattern, query_lower):
            return 'count'
    
    for pattern in list_patterns:
        if re.search(pattern, query_lower):
            return 'list'
            
    for pattern in explain_patterns:
        if re.search(pattern, query_lower):
            return 'explain'
    
    return 'general'

def get_vector_search_response(query: str, search_results: List[Dict], chat_history: List[Dict[str, str]]):
    """
    Generate AI response based on vector search results with better intent classification
    """
    try:
        if not search_results:
            return {
                "type": "text",
                "payload": "No similar tickets or solutions found for your query. Please try rephrasing your question or provide more details.",
                "explanation": "No matching results found in the knowledge base."
            }
        
        # Classify the query intent
        intent = classify_vector_query_intent(query)
        
        # Handle count queries directly
        if intent == 'count':
            count = len(search_results)
            return {
                "type": "text",
                "payload": f"Found {count} tickets matching your search criteria.",
                "explanation": f"Count of matching tickets: {count}"
            }
        
        # Handle list queries - return table
        elif intent == 'list':
            # Clean up the search results for table display
            table_data = []
            for result in search_results[:20]:  # Show up to 20 results for lists
                
                # Helper function to clean field values
                def clean_field(value, max_length=None):
                    if pd.isna(value) or value is None or value == "":
                        return ""
                    str_val = str(value).strip()
                    if max_length and len(str_val) > max_length:
                        return str_val[:max_length] + "..."
                    return str_val
                
                # Use flexible column detection for table display
                clean_result = {
                    "Similarity": f"{result.get('similarity_score', 0):.2f}"
                }
                
                # Add available columns dynamically
                possible_columns = {
                    "Ticket_ID": ['Request - ID', 'Ticket ID', 'ID', 'Request ID', 'ticketId'],
                    "Subject": ['Request - Subject description', 'Subject', 'Summary', 'Title'],
                    "Request": ['Request - Text Request', 'Description', 'Issue Description', 'Problem Description'],
                    "Answer": ['Request - Text Answer', 'Answer', 'Solution', 'Resolution'],
                    "Status": ['Req. Status - Description', 'Status', 'Current Status', 'State'],
                    "Priority": ['Request - Priority Description', 'Priority', 'Ticket Priority'],
                    "Category": ['Request - Category', 'Category', 'Type']
                }
                
                for display_name, possible_cols in possible_columns.items():
                    for col in possible_cols:
                        if col in result:
                            max_length = 150 if display_name in ['Request', 'Answer'] else 100
                            clean_result[display_name] = clean_field(result.get(col, ''), max_length)
                            break
                    else:
                        # If no column found, add empty string
                        clean_result[display_name] = ""
                table_data.append(clean_result)
            
            return {
                "type": "table",
                "payload": table_data,
                "explanation": f"Found {len(search_results)} matching tickets. Showing top {len(table_data)} results."
            }
        
        # Handle explanation queries - only when explicitly asking for explanations
        elif intent == 'explain':
            # Prepare context from search results
            context_info = []
            for i, result in enumerate(search_results[:5]):  # Use top 5 results for context
                context_info.append(f"""
Result {i+1} (Similarity: {result.get('similarity_score', 0):.2f}):
- Ticket ID: {result.get('Request - ID', 'N/A')}
- Subject: {result.get('Request - Subject description', 'N/A')}
- Request: {result.get('Request - Text Request', 'N/A')[:200]}...
- Answer: {result.get('Request - Text Answer', 'N/A')[:200]}...
- Status: {result.get('Req. Status - Description', 'N/A')}
- Priority: {result.get('Request - Priority Description', 'N/A')}
- Category: {result.get('Request - Category', 'N/A')}
""")
            
            context_text = "\n".join(context_info)
            history_str = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in chat_history])
            
            llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo", temperature=0.1)
            
            llm_prompt = f"""
You are a knowledge base search assistant. Based on the user's query and the similar tickets found, provide a helpful explanation or solution.

**Chat History:**
{history_str}

**User Query:** {query}

**Similar Tickets Found:**
{context_text}

**Instructions:**
1. The user is asking for an explanation, solution, or general information about the topic
2. Analyze the similar tickets to understand common patterns, causes, and solutions
3. Provide a helpful text response that explains what you found and any relevant solutions
4. Focus on actionable insights and common patterns from the similar cases

**Response Format:**
Provide a clear, helpful text response that summarizes the key insights from the similar tickets and any relevant solutions or recommendations.

**Example:**
"Based on the similar tickets found, SCM issues are commonly related to sales order allocation problems. The typical causes include incorrect Sold-to party configurations and sales area mismatches. Most cases are resolved by verifying and updating the Sold-to party details for the correct sales area."

Now provide your response:
"""
            
            response = llm.invoke(llm_prompt)
            content = response.content.strip()
            
            return {
                "type": "text",
                "payload": content,
                "explanation": f"Analysis based on {len(search_results)} similar cases found in the knowledge base."
            }
        
        # Handle general queries (direct descriptions) - default to showing list of similar tickets
        else:
            # Clean up the search results for table display
            table_data = []
            for result in search_results[:20]:  # Show up to 20 results for general queries
                
                # Helper function to clean field values
                def clean_field(value, max_length=None):
                    if pd.isna(value) or value is None or value == "":
                        return ""
                    str_val = str(value).strip()
                    if max_length and len(str_val) > max_length:
                        return str_val[:max_length] + "..."
                    return str_val
                
                # Use flexible column detection for table display
                clean_result = {
                    "Similarity": f"{result.get('similarity_score', 0):.2f}"
                }
                
                # Add available columns dynamically
                possible_columns = {
                    "Ticket_ID": ['Request - ID', 'Ticket ID', 'ID', 'Request ID', 'ticketId'],
                    "Subject": ['Request - Subject description', 'Subject', 'Summary', 'Title'],
                    "Request": ['Request - Text Request', 'Description', 'Issue Description', 'Problem Description'],
                    "Answer": ['Request - Text Answer', 'Answer', 'Solution', 'Resolution'],
                    "Status": ['Req. Status - Description', 'Status', 'Current Status', 'State'],
                    "Priority": ['Request - Priority Description', 'Priority', 'Ticket Priority'],
                    "Category": ['Request - Category', 'Category', 'Type']
                }
                
                for display_name, possible_cols in possible_columns.items():
                    for col in possible_cols:
                        if col in result:
                            max_length = 150 if display_name in ['Request', 'Answer'] else 100
                            clean_result[display_name] = clean_field(result.get(col, ''), max_length)
                            break
                    else:
                        # If no column found, add empty string
                        clean_result[display_name] = ""
                table_data.append(clean_result)
            
            return {
                "type": "table",
                "payload": table_data,
                "explanation": f"Found {len(search_results)} similar tickets for your description. Showing top {len(table_data)} results."
            }
            
    except Exception as e:
        print(f"Vector search response generation error: {e}")
        return {
            "type": "text",
            "payload": f"Found {len(search_results)} similar cases. The most relevant case involves: {search_results[0].get('Request - Subject description', 'N/A') if search_results else 'No results'}",
            "explanation": f"Vector search found {len(search_results)} similar results."
        }



