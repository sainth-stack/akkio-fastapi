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
from uuid import uuid4
from collections import defaultdict
import threading
from typing import List, Dict, Any

SESSION_MEMORY = defaultdict(list)
SESSION_MEMORY_LOCK = threading.Lock()

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

@sla_router.post("/upload_data")
async def upload_data_only(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    upload_dir = "uploads_sla"
    os.makedirs(upload_dir, exist_ok=True)
    content = await file.read()

    try:
        if ext == ".csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse or save file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty file or no data found.")

    # Always save to data1.csv regardless of original file type
    static_file_path = os.path.join(upload_dir, "data1.csv")
    df.to_csv(static_file_path, index=False)

    # Handle NaN values and make data JSON serializable
    def make_json_serializable(x):
        # Handle NaN values first
        if pd.isna(x):
            return None
        # Native JSON types are left as is
        if isinstance(x, (str, int, float, bool)) or x is None:
            # Additional check for float NaN and infinity
            if isinstance(x, float) and (pd.isna(x) or x == float('inf') or x == float('-inf')):
                return None
            return x
        # Convert other types (pd.Timestamp, numpy types, etc.) to string
        return str(x)

    # Use applymap for DataFrame element-wise operation
    records = df.astype(object).map(make_json_serializable).to_dict(orient="records")

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
    # 1. Determine if Plotly is needed before attempting to import it
    plotly_keywords = ['px.', 'go.', 'ff.', 'plotly']
    is_plotly_needed = any(keyword in code.lower() for keyword in plotly_keywords)
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
    if any(pattern in code for pattern in forbidden_patterns):
        raise ValueError(f"Execution failed: Use of a forbidden keyword or function was detected.")

    try:
        # 5. Parse the code to check for syntax errors before execution
        ast.parse(code)

        # Execute the code in the sandboxed environment
        local_vars = {}
        exec(code, safe_globals, local_vars)

        # 6. Refined result retrieval
        if 'result' in local_vars:
            return local_vars['result']

        # If no 'result' variable, try to eval the last line if it's an expression
        lines = code.strip().split('\n')
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
    if isinstance(obj, dict):
        # Special case: binary-encoded array (e.g., {'dtype': ..., 'bdata': ...})
        if set(obj.keys()) == {'dtype', 'bdata'}:
            # Can't decode bdata safely here, so just return an empty list (or handle on backend)
            return []
        return {k: convert_plotly_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_plotly_arrays(v) for v in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def format_result_for_response(result):
    """
    Format the execution result into the appropriate response format.
    """
    # Handle plotly figures. Use a specific check for plotly.graph_objects.Figure to avoid
    # incorrectly identifying pandas DataFrames as plotly figures, as both have a to_dict() method.
    if PLOTLY_AVAILABLE and isinstance(result, go.Figure):
        try:
            # It's a plotly figure
            if hasattr(result, 'to_json'):
                fig_dict = json.loads(result.to_json())
                fig_dict = convert_plotly_arrays(fig_dict)
                return {"type": "plotly", "payload": fig_dict}
            elif hasattr(result, 'to_dict'):
                fig_dict = result.to_dict()
                fig_dict = convert_plotly_arrays(fig_dict)
                return {"type": "plotly", "payload": fig_dict}
        except Exception as e:
            print(f"Error converting plotly figure: {e}")
            # Fall through to other handlers

    # Handle pandas DataFrames
    if isinstance(result, pd.DataFrame):
        # Replace NaN and inf values with None for JSON compatibility
        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        if len(result) > 1000:  # Limit large results
            result = result.head(1000)

        # Check if it's a correlation matrix or similar (numeric data with meaningful index/columns)
        if result.index.name or all(isinstance(col, str) for col in result.columns):
            # Convert to records format but preserve index
            result_with_index = result.reset_index()
            payload = result_with_index.to_dict(orient='records')
        else:
            payload = result.to_dict(orient='records')

        return {"type": "table", "payload": payload}

    # Handle pandas Series
    if isinstance(result, pd.Series):
        # Replace NaN and inf values with None for JSON compatibility
        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        if len(result) > 1000:
            result = result.head(1000)

        # Convert to DataFrame for consistent display
        df_result = result.reset_index()

        # Better column naming for series
        if hasattr(result, 'name') and result.name:
            df_result.columns = [result.index.name or 'index', result.name]
        else:
            df_result.columns = [result.index.name or 'index', 'value']

        payload = df_result.to_dict(orient='records')
        return {"type": "table", "payload": payload}

    # Handle dictionaries (correlation matrices, etc.)
    if isinstance(result, dict):
        # Try to convert to DataFrame if possible
        try:
            df_result = pd.DataFrame(result)
            df_result = df_result.reset_index()  # Include index as a column
            payload = df_result.to_dict(orient='records')
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


def get_llm_analysis(query: str, df: pd.DataFrame, chat_history: List[Dict[str, str]]):
    """
    Use LLM to analyze the user's query, determine intent, and generate code if necessary, using chat history for context.
    """
    num_rows, num_cols = df.shape
    col_names = list(df.columns)
    sample_data = df.head(3).to_dict(orient='records')
    dataset_info = f"""
Dataset Information:
- Shape: {num_rows} rows, {num_cols} columns
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

**IMPORTANT:**
-If the user asks any information related tickets,you have to consider and work on the UNIQUE tickets only instead of all the tickets in the dataset.
-If the user asks about the user names then you MUST look at 'Request - Resource Assigned To - Name' only.
-You have to give reply with entire data frame columns rather than the user required columns based on the user question only.Do not give reply in this format for every question.

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

**Example 1: Data Question**
User Question: "how many unique users are there"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "To find the number of unique users, I will count the distinct values in the Request - Resource Assigned To - Name column.",
    "code": "result = df[Request - Resource Assigned To - Name].nunique()"
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
    "explanation": "Here is a bar chart showing the distribution of request statuses.",
    "code": "status_counts = df['Req. Status - Description'].value_counts()\\nif not status_counts.empty:\\n    result = px.bar(x=status_counts.index, y=status_counts.values, title='Request Status Distribution', labels={{'x':'Status', 'y':'Count'}})\\nelse:\\n    result = None"
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
User Question: "list all the request user name ,count and request subject description"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is a list of all request users, their request counts, and the subject of their requests.",
    "code": "result = df.groupby(['Request - Resource Assigned To - Name', 'Request - Subject description']).size().reset_index(name='count')"
  }}
}}

**Example 6: Top N and Max Count**
User Question: "Display the request user name who raised max count of tickets and show the top 10"
Your JSON response:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Here is the user who raised the maximum number of tickets, along with the top 10 users by ticket count.",
    "code": "user_counts = df['Request - Resource Assigned To - Name'].value_counts()\nmax_user = user_counts.idxmax()\ntop_10_users = user_counts.head(10).reset_index()\ntop_10_users.columns = ['Request - User Name', 'count']\nresult = {{'max_user': max_user, 'top_10_users': top_10_users.to_dict('records')}}"
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


Now, provide the JSON response for the given user question. Do not include any text or markdown outside of the JSON object.
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


@sla_router.post("/Explore_sla/")
async def senior_data_analysis_sla(query: str = Form(...), session_id: str = Form(None)):
    try:
        # Session handling
        if not session_id:
            session_id = str(uuid4())

        with SESSION_MEMORY_LOCK:
            # Pass a copy of the history to the LLM
            chat_history_for_llm = list(SESSION_MEMORY.get(session_id, []))

        # Load latest uploaded file
        csv_file_path = os.path.join('uploads_sla/data1.csv')
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=400, detail="No dataset found. Please upload a file first.")

        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        # Step 1: Get AI analysis (intent, code, explanation) using chat history
        analysis_result = get_llm_analysis(query, df, chat_history=chat_history_for_llm)

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

        # Store current exchange in memory
        with SESSION_MEMORY_LOCK:
            SESSION_MEMORY[session_id].append({"role": "user", "content": query})
            SESSION_MEMORY[session_id].append({"role": "bot", "content": bot_response_text})

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
                # Step 3: Execute code
                result = safe_execute_pandas_code(code, df)

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
