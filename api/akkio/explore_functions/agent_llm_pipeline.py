import io
import json
from difflib import get_close_matches
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .llm import get_openai_client, get_language_context
from .formatters import format_result_for_response, format_text_response
from .agent_utils import safe_execute_pandas_code
from .charts import generate_rescue_chart
from universal_prompts import (
    prompt_for_data_analyst,
    Prompt_for_code_execution,
    Visualisation_intelligence_engine,
)


def generate_data_code(prompt_eng: str) -> str:
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt_for_data_analyst},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


def simulate_and_format_with_llm(code_to_simulate: str, dataframe: pd.DataFrame) -> str:
    info_buffer = io.StringIO()
    dataframe.info(buf=info_buffer)
    df_info = info_buffer.getvalue()
    df_head = dataframe.head().to_string()
    user_prompt = f"""
    You are the Universal Code Execution Environment. Your task is to simulate the execution of the following Python code and generate a complete report based on your system instructions.
    ### DATA CONTEXT:
    The code operates on a pandas DataFrame named `df`.you MUST Consider this 'df' throughout the total process and will give the exact and existing results.
    The code operates on a pandas DataFrame named `df`. Here is its metadata and a sample of its first few rows:
    #### DataFrame Info (`df.info()`):
    ```
    {df_info}
    ```
    #### DataFrame Head (`df.head()`):
    ```
    {df_head}
    ```
    Rules for Code generation while working with data:
     - Perform operations directly on the dataset using the full dataframe (df), not just the preview.
     - The preview is for context only - your code should work on the complete dataset.
     - Handle both header-based queries and content-based queries (filtering by specific values in rows).
     - Only return results filtered exactly as per the query.
    ### PYTHON CODE TO SIMULATE:
    You must simulate the execution of this code. Do not just describe it; act as if you have run it and are now reporting the results.
    ```
    {code_to_simulate}
    ```
    You have to use the engines based on the usage.
    If you have the graph related code,then you can use the {Visualisation_intelligence_engine} else {Prompt_for_code_execution}
    ### YOUR TASK:
    1.  **Simulate Execution:** Mentally run the code against the provided data context.
    2.  **Predict Output:** Determine what `print()` statements would produce and what a generated plot would look like.
    3.  **Generate Report:** Produce a single, complete report that STRICTLY follows rules whatever required and formatting defined in your system prompt that should be in the JSON format.You have to follow the required rules wherever necessary.
    ### IMPORTANT:
   - The report should be very informative and Don't include the internal functionings for the generation of the reports,Only the analysis related content and the graph related things and summaries and everything which is not executed internally can be given to the Output.
   - Do not include the internal headings like \"**LANGUAGE:** Python | **MODE:** Simulation | **STATUS:** Success\\n\\n⚡ **EXECUTION OUTPUT:**\\n\".Do not include any of these ,,Just include the headings in the middle of the content only.
   - Report can be present in the markdown format.
   - **If you got the basic code to execute, you MUST execute and give the exact result**. **DO NOT** add all the things regarding visualisation to that.
    """
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": Visualisation_intelligence_engine + Prompt_for_code_execution},
            {"role": "user", "content": user_prompt}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    return all_text


def get_llm_analysis_explore(query: str, df: pd.DataFrame, mode: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, object]:
    num_rows, num_cols = df.shape
    col_names = list(df.columns)
    sample_data = df.head(3).to_dict(orient='records')
    kpi_keywords = ['kpi', 'kpis', 'key performance indicator', 'key performance indicators', 'metrics', 'performance metrics', 'dashboard metrics']
    is_kpi_query = any(keyword in query.lower() for keyword in kpi_keywords)
    legal_keywords = ['document', 'pdf', 'legal', 'contract', 'agreement', 'policy', 'regulation', 'law', 'analysis', 'summary', 'extract', 'content']
    is_legal_query = any(keyword in query.lower() for keyword in legal_keywords)
    dataset_info = f"""
Dataset Information:
- Shape: {num_rows} rows, {num_cols} columns
- Columns: {col_names}
- Sample Data (first 3 rows): {sample_data}
"""
    language, language_instructions = get_language_context(query)
    if language == "arabic":
        instructions_common = f"""
You are a highly intelligent and professional AI assistant specializing in comprehensive data analysis and document processing. Your responses should be structured, professional, and comprehensive like a senior analyst.
{language_instructions}
"""
    else:
        instructions_common = f"""
You are a highly intelligent and professional AI assistant specializing in comprehensive data analysis and document processing. Your responses should be structured, professional, and comprehensive like a senior analyst.
{language_instructions}
"""
    mode_instructions = ""
    if mode == "graph":
        mode_instructions = (
            "Generate high-quality Plotly code that produces the best visualization for the user's question. "
            "CRITICAL CHART GENERATION STEPS: "
            "1) First, compute the required aggregation/grouping in pandas. "
            "2) Ensure axis arrays are non-empty and aligned. "
            "3) Clean data thoroughly and extract numerics from strings with units using regex. "
            "4) Set meaningful titles and labels. "
            "5) If chart generation fails, return a top-10 aggregated DataFrame instead."
        )
    elif mode == "table":
        mode_instructions = "Generate pandas code that returns a tabular result in a DataFrame assigned to result. Do not generate charts."
    else:
        if is_legal_query:
            if language == "arabic":
                mode_instructions = (
                    "This is a legal analysis query in Arabic. Analyze comprehensively and return type='conversational_answer' with structured Arabic HTML payload."
                )
            else:
                mode_instructions = (
                    "This is a legal analysis query. Analyze comprehensively and return type='conversational_answer' with structured HTML payload."
                )
        elif is_kpi_query:
            mode_instructions = "Provide a comprehensive, conversational response about KPIs with proper HTML formatting; return type='conversational_answer'."
        else:
            mode_instructions = (
                "For conversational queries, return type='conversational_answer' with professional HTML formatting. "
                "If it requires data-derived text, generate pandas code that computes the answer and sets result to a concise string."
            )
    conversation_context = ""
    if is_kpi_query:
        if language == "arabic":
            conversation_context = f"""
سياق مؤشرات الأداء الرئيسية:
- لديك إمكانية الوصول إلى مجموعة بيانات تحتوي على {num_cols} أعمدة: {col_names}
- يمكن استخدام هذه البيانات لحساب مؤشرات الأداء الرئيسية المختلفة
"""
        else:
            conversation_context = f"""
KPI Context:
- You have access to a dataset with {num_cols} columns: {col_names}
- This data can be used to calculate various KPIs and performance metrics
"""
    history_context = ""
    if chat_history:
        last_query = chat_history[0]['content'] if chat_history else ""
        if language == "arabic":
            history_context = f"السؤال السابق للسياق: {last_query}"
        else:
            history_context = f"Previous Query for context: {last_query}"
    llm_prompt = f"""
{instructions_common}
{history_context}
Dataset Context:
{dataset_info}
{conversation_context}
User Question: {query}
Mode: {mode}
Specific Instructions: {mode_instructions}

CRITICAL OUTPUT FORMAT:
You MUST return a valid JSON object with this exact structure:

For conversational/text responses:
{{
  "type": "conversational_answer",
  "payload": "<h4>Title</h4><p>Your detailed HTML-formatted response here</p>"
}}

For data analysis with code:
{{
  "type": "data_analysis_answer",
  "payload": {{
    "explanation": "Brief explanation of what the analysis does",
    "code": "import pandas as pd\\nresult = df.head(10)"
  }}
}}

IMPORTANT: 
- Return ONLY the JSON object, no markdown fences, no extra text
- Always include both "type" and "payload" keys
- For text mode, use "conversational_answer" type
- For table/graph mode with code, use "data_analysis_answer" type
"""
    system_message = "You are an expert data analyst AI assistant. You provide helpful, conversational responses with proper HTML formatting and are skilled at Python/Plotly code generation."
    try:
        response = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0.3,
        )
        all_text = ""
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            all_text += chunk_message
        content = all_text.strip()
        
        # Debug logging
        print(f"[DEBUG] LLM raw response: {content[:500]}...")
        
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed_result = json.loads(content)
            # Ensure the result has required keys
            if not isinstance(parsed_result, dict):
                print(f"[WARNING] LLM returned non-dict: {type(parsed_result)}")
                return {"type": "conversational_answer", "payload": str(parsed_result)}
            
            if "type" not in parsed_result or "payload" not in parsed_result:
                print(f"[WARNING] LLM response missing type or payload: {parsed_result.keys()}")
                # Try to fix common issues
                if "type" not in parsed_result:
                    parsed_result["type"] = "conversational_answer"
                if "payload" not in parsed_result:
                    parsed_result["payload"] = content
            
            return parsed_result
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode failed: {e}")
            return {"type": "conversational_answer", "payload": content}
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {e}")
        return {"type": "conversational_answer", "payload": f"I encountered an error analyzing your data: {str(e)}"}


def _extract_code_from_llm_text(all_text: str) -> str:
    try:
        txt = all_text.strip()
        if txt.startswith("```"):
            import re
            txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
            txt = txt.replace("```", "").strip()
        return txt
    except Exception:
        return all_text


def _llm_generate_code(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    response = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    combined = ""
    for ch in response.choices:
        msg = ch.message
        combined += msg.content if msg and msg.content else ""
    return _extract_code_from_llm_text(combined)


def _df_context(df: pd.DataFrame) -> str:
    try:
        info_buf = io.StringIO()
        df.info(buf=info_buf)
        head = df.head(5).to_dict(orient='records')
        return (
            f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
            f"Columns: {list(df.columns)}\n"
            f"Info:\n{info_buf.getvalue()}\n"
            f"Head (5): {head}"
        )
    except Exception as e:
        return f"Could not build df context: {e}"


def _df_schema(df: pd.DataFrame) -> Dict[str, str]:
    try:
        schema = {}
        for c in list(df.columns):
            try:
                schema[c] = str(df[c].dtype)
            except Exception:
                schema[c] = "unknown"
        return schema
    except Exception:
        return {}


def _retry_llm_pandas_extraction(query: str, df: pd.DataFrame, max_retries: int = 3) -> pd.DataFrame:
    system_prompt = (
        "You are a senior data engineer. Generate robust, safe pandas code that operates on a provided DataFrame named df "
        "and assigns the final cleaned and aggregated tabular result to a variable named result (a pandas DataFrame). "
        "Strict requirements: 1) Never read/write files; use df only. 2) Parse numeric values embedded in strings using regex and convert to float. "
        "3) Detect and parse likely date/time columns using pd.to_datetime with errors='coerce' when needed for time grouping. "
        "4) Remove empty-like tokens and drop rows with missing values in the plotted columns. "
        "5) Avoid geospatial lat/lon unless explicitly requested. 6) If filtering leads to empty result, create a sensible top-10 aggregated table related to the question. "
        "7) Return ONLY executable Python code; no markdown; final variable must be named result and must be a pandas DataFrame."
    )
    last_error = None
    for attempt in range(1, max_retries + 1):
        guidance = ""
        if last_error:
            guidance = f"Previous attempt failed with error: {last_error}. Revise to ensure 'result' is a non-empty pandas DataFrame."
        user_prompt = (
            f"User question (graph intent): {query}\n\n"
            f"Dataset context:\n{_df_context(df)}\n\n"
            f"{guidance}\n"
            "Deliver only code that sets a pandas DataFrame named result."
        )
        code = _llm_generate_code(system_prompt, user_prompt)
        try:
            extracted = safe_execute_pandas_code(code, df)
            if isinstance(extracted, pd.DataFrame) and not extracted.empty:
                return extracted
            if not isinstance(extracted, pd.DataFrame):
                last_error = f"expected DataFrame 'result', got {type(extracted).__name__}"
            else:
                last_error = "DataFrame is empty after cleaning/filtering"
        except Exception as e:
            last_error = str(e)
    fallback = df.head(50).copy()
    return fallback


def _retry_llm_plot_from_clean_df(query: str, df_clean: pd.DataFrame, max_retries: int = 3) -> Dict[str, object]:
    # Fast path: if we already have a clear time/numeric pair, build a chart locally without another LLM call
    try:
        date_cols = [c for c in df_clean.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'period'])]
        metric_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
        # Avoid ID-like metrics
        metric_cols = [c for c in metric_cols if not any(tok in c.lower() for tok in ['id', 'device', 'name', 'uuid', 'code'])]
        if date_cols and metric_cols:
            x_col = date_cols[0]
            y_col = metric_cols[0]
            tmp = df_clean[[x_col, y_col]].copy()
            # Ensure datetime and numeric
            tmp[x_col] = pd.to_datetime(tmp[x_col], errors='coerce')
            tmp[y_col] = pd.to_numeric(tmp[y_col], errors='coerce')
            tmp = tmp.dropna(subset=[x_col, y_col]).sort_values(x_col)
            if not tmp.empty:
                # If too many points, downsample using time-based resampling to ~500 points
                n = len(tmp)
                if n > 1000:
                    try:
                        ts = tmp[x_col]
                        total_seconds = max(1.0, (ts.iloc[-1] - ts.iloc[0]).total_seconds())
                        # target ~500 buckets
                        step_seconds = max(1, int(total_seconds // 500))
                        tmp = (
                            tmp.set_index(x_col)
                               .resample(f'{step_seconds}S')
                               .mean(numeric_only=True)
                               .dropna(subset=[y_col])
                               .reset_index()
                        )
                    except Exception:
                        # Fallback: stride sample
                        stride = max(1, n // 500)
                        tmp = tmp.iloc[::stride, :]
                import plotly.graph_objects as go  # type: ignore
                fig = go.Figure(go.Scatter(x=list(tmp[x_col]), y=list(tmp[y_col]), mode='lines+markers', name=str(y_col)))
                fig.update_layout(title=f"{y_col} over time", xaxis_title=str(x_col), yaxis_title=str(y_col))
                return format_result_for_response(fig)
    except Exception:
        pass
    system_prompt = (
        "You are a senior data visualization engineer. Generate robust Plotly code (px/go/ff) that operates on a provided DataFrame named df "
        "and assigns the final figure to a variable named result. "
        "Requirements: 1) Perform any needed aggregation in pandas first. 2) Clean empty-like tokens and drop rows with missing values in the plot columns. "
        "3) If metrics are strings with units, extract numeric with regex and convert to float before plotting. 4) Ensure x/y arrays are aligned and non-empty. "
        "5) Set a descriptive title and axis labels. 6) If a valid chart is impossible, set result to a top-10 aggregated pandas DataFrame instead (not None). "
        "7) Return ONLY executable Python code; no markdown; final variable must be named result."
    )
    last_error = None
    for attempt in range(1, max_retries + 1):
        guidance = ""
        if last_error:
            guidance = f"Prior error: {last_error}. Revise to ensure non-empty aligned arrays or return a top-10 aggregated DataFrame as result."
        user_prompt = (
            f"User question (graph intent): {query}\n\n"
            f"CLEAN DATA schema and sample:\n"
            f"Columns: {list(df_clean.columns)}\n"
            f"Head (5): {df_clean.head(5).to_dict(orient='records')}\n\n"
            f"{guidance}\n"
            "Deliver only code that sets 'result' to either a Plotly figure or a pandas DataFrame fallback."
        )
        code = _llm_generate_code(system_prompt, user_prompt)
        try:
            exec_obj = safe_execute_pandas_code(code, df_clean)
            formatted = format_result_for_response(exec_obj)
            if formatted.get("type") == "plotly":
                return formatted
            if formatted.get("type") == "table":
                payload = formatted.get("payload") or []
                if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
                    try:
                        tmp_df = pd.DataFrame(payload)
                        date_cols = [c for c in tmp_df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'period'])]
                        numeric_cols = [c for c in tmp_df.columns if pd.api.types.is_numeric_dtype(tmp_df[c])]
                        if date_cols and numeric_cols:
                            x_col = date_cols[0]
                            y_col = numeric_cols[0]
                            tmp_df[x_col] = pd.to_datetime(tmp_df[x_col], errors='coerce')
                            tmp_df = tmp_df.dropna(subset=[x_col, y_col])
                            if not tmp_df.empty:
                                import plotly.graph_objects as go  # type: ignore
                                fig = go.Figure(go.Scatter(x=list(tmp_df[x_col]), y=list(tmp_df[y_col]), mode='lines+markers'))
                                fig.update_layout(title=f"{y_col} over time", xaxis_title=str(x_col), yaxis_title=str(y_col))
                                return format_result_for_response(fig)
                        if not numeric_cols:
                            for c in tmp_df.columns:
                                if c not in date_cols:
                                    coerced = pd.to_numeric(tmp_df[c], errors='coerce')
                                    if coerced.notna().sum() > 0:
                                        tmp_df[c] = coerced
                                        numeric_cols.append(c)
                                        break
                        cat_cols = [c for c in tmp_df.columns if c not in numeric_cols]
                        if numeric_cols:
                            x_col = cat_cols[0] if cat_cols else tmp_df.columns[0]
                            y_col = numeric_cols[0]
                            tmp_df[x_col] = tmp_df[x_col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan})
                            plot_df = tmp_df.dropna(subset=[x_col, y_col])
                            if not plot_df.empty:
                                import plotly.graph_objects as go  # type: ignore
                                fig = go.Figure(go.Bar(x=plot_df[x_col].astype(str).tolist(), y=plot_df[y_col].tolist()))
                                fig.update_layout(title=f"{y_col} by {x_col}", xaxis_title=str(x_col), yaxis_title=str(y_col))
                                return format_result_for_response(fig)
                    except Exception:
                        pass
            last_error = "Empty or invalid visualization payload"
        except Exception as e:
            last_error = str(e)
    try:
        rescue_fig, biz_exp = generate_rescue_chart(df_clean, query)
        if rescue_fig is not None:
            formatted = format_result_for_response(rescue_fig)
            if biz_exp:
                formatted["explanation"] = biz_exp
            return formatted
    except Exception:
        pass
    try:
        date_cols = [c for c in df_clean.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'period'])]
        metric_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
        if date_cols and metric_cols:
            x_col = date_cols[0]
            y_col = metric_cols[0]
            tmp = df_clean[[x_col, y_col]].copy()
            tmp[x_col] = pd.to_datetime(tmp[x_col], errors='coerce')
            tmp = tmp.dropna(subset=[x_col, y_col])
            if not tmp.empty:
                daily = tmp.groupby(tmp[x_col].dt.date)[y_col].mean().reset_index()
                import plotly.graph_objects as go  # type: ignore
                fig = go.Figure(go.Scatter(x=list(pd.to_datetime(daily[x_col])), y=list(daily[y_col]), mode='lines+markers'))
                fig.update_layout(title=f"{y_col} over time", xaxis_title="Date", yaxis_title=y_col)
                return format_result_for_response(fig)
    except Exception:
        pass
    return {"type": "text", "payload": "Unable to generate a chart; please refine the query."}


# --------- Lightweight LLM intent parsing for graphs (fast, JSON-only) ----------
def _closest_column(name: str, columns: List[str]) -> Optional[str]:
    try:
        if not name:
            return None
        name_l = name.strip().lower()
        exact = [c for c in columns if c.lower() == name_l]
        if exact:
            return exact[0]
        near = get_close_matches(name_l, [c.lower() for c in columns], n=1, cutoff=0.75)
        if near:
            idx = [c.lower() for c in columns].index(near[0])
            return columns[idx]
    except Exception:
        pass
    return None


def _guess_time_column(df: pd.DataFrame) -> Optional[str]:
    try:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ['date', 'time', 'timestamp', 'period']):
                ser = pd.to_datetime(df[c], errors='coerce')
                if ser.notna().sum() >= max(5, int(0.2 * len(df))):
                    return c
        for c in df.columns:
            try:
                ser = pd.to_datetime(df[c], errors='coerce')
                if ser.notna().sum() >= max(5, int(0.3 * len(df))):
                    return c
            except Exception:
                continue
    except Exception:
        pass
    return None


def _llm_parse_graph_intent(query: str, columns: List[str]) -> Optional[Dict[str, object]]:
    """
    Ask LLM to infer intended chart semantics from the query.
    Returns a small dict: {chart_type, x, y, group_by?}
    """
    try:
        sys = (
            "You are a data viz planner. Given a user query and available column names, "
            "infer a minimal chart plan as strict JSON with keys: "
            "{\"chart_type\": \"line|bar|scatter|histogram\", \"x\": \"col or null\", \"y\": \"col or null\", \"group_by\": \"col or null\"}. "
            "Prefer time-like columns for x when plotting trends. If user says 'each <group> over <metric>', "
            "set group_by to that column, y to the metric, and x to a time-like column if available."
        )
        user = f"Query: {query}\nColumns: {columns}\nReturn ONLY the JSON."
        resp = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        txt = ""
        for ch in resp.choices:
            msg = ch.message
            txt += (msg.content or "")
        txt = txt.strip()
        if txt.startswith("```"):
            import re as _re
            txt = _re.sub(r"^```[a-zA-Z]*\n", "", txt)
            txt = txt.replace("```", "").strip()
        plan = json.loads(txt)
        if isinstance(plan, dict) and "chart_type" in plan:
            return plan
    except Exception:
        return None
    return None


# ---------------- Natural-language date range parsing -----------------
def _parse_date_token(text: str) -> Optional[pd.Timestamp]:
    try:
        # pandas can parse many formats with errors='coerce'
        ts = pd.to_datetime(text, errors='coerce')
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def _parse_date_range_from_query(query: str) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Parse a rough date range from natural-language query.
    Supports:
    - 'on May 4', 'May 4th', '2025-05-04'
    - 'from May 1 to May 7', 'between May 1 and May 7'
    - 'last 7 days', 'past week', 'last week', 'yesterday', 'today'
    """
    try:
        q = (query or "").lower()
        now = pd.Timestamp.utcnow().normalize()
        # last/past N days/weeks/months
        import re as _re
        m = _re.search(r'(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)', q)
        if m:
            n = int(m.group(2))
            unit = m.group(3)
            if 'week' in unit:
                delta = pd.Timedelta(days=7 * n)
            elif 'month' in unit:
                delta = pd.Timedelta(days=30 * n)
            else:
                delta = pd.Timedelta(days=n)
            start = now - delta
            end = pd.Timestamp.utcnow()
            return (start, end)
        if 'last week' in q or 'past week' in q:
            return (now - pd.Timedelta(days=7), pd.Timestamp.utcnow())
        if 'yesterday' in q:
            y = now - pd.Timedelta(days=1)
            return (y, y + pd.Timedelta(days=1))
        if 'today' in q:
            return (now, now + pd.Timedelta(days=1))
        # between / from-to
        m2 = _re.search(r'(between|from)\s+(.+?)\s+(and|to)\s+(.+)', q)
        if m2:
            s_txt = m2.group(2)
            e_txt = m2.group(4)
            s = _parse_date_token(s_txt)
            e = _parse_date_token(e_txt)
            if s is not None and e is not None:
                if e < s:
                    s, e = e, s
                # widen end to end-of-day if time missing
                return (s, e + pd.Timedelta(days=1))
        # single date like 'may 4'
        m3 = _re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{2,4})?', q)
        if m3:
            d = _parse_date_token(m3.group(0))
            if d is not None:
                return (d.normalize(), d.normalize() + pd.Timedelta(days=1))
        # ISO date in text
        m4 = _re.search(r'\d{4}-\d{2}-\d{2}', q)
        if m4:
            d = _parse_date_token(m4.group(0))
            if d is not None:
                return (d.normalize(), d.normalize() + pd.Timedelta(days=1))
    except Exception:
        return None
    return None


def build_graph_two_stage_llm(query: str, df: pd.DataFrame) -> Optional[Dict[str, object]]:
    """
    Two-stage LLM graph pipeline:
    1) Ask LLM for pandas code to transform/filter df into a clean DataFrame named 'result'.
    2) Ask LLM for Plotly code to visualize the cleaned data; return the resulting figure.
    """
    try:
        date_range = _parse_date_range_from_query(query or "")
        # Stage 1: generate pandas code to create cleaned result DataFrame
        schema = _df_schema(df)
        head_preview = df.head(5).to_dict(orient='records')
        sys1 = (
            "You are a senior data engineer. Generate robust, safe pandas code that operates on a provided DataFrame named df "
            "and assigns the final cleaned tabular result to a variable named result (a pandas DataFrame). "
            "Your job is to INTERPRET the user's question and prepare a plotting-ready table with STANDARD COLUMN NAMES:\n"
            "- 'x': the x-axis column (prefer a time-like column if a trend is implied)\n"
            "- 'y': the numeric metric requested by the user\n"
            "- 'group' (optional): a categorical column when the user asks for 'each <category>' / 'by <category>'\n"
            "Strict requirements:\n"
            "1) Never read/write files; operate on the provided df only.\n"
            "2) Parse numeric values embedded in strings using regex and convert to float.\n"
            "3) Detect and parse likely date/time columns using pd.to_datetime with errors='coerce'.\n"
            "4) Remove empty-like tokens and drop rows with missing values in the plotted columns ('x','y', and 'group' if used).\n"
            "5) If a date range is provided, apply it to the time-like 'x' column (inclusive of start, exclusive of end).\n"
            "6) If the query implies grouping (e.g., 'each device id', 'by status'), populate a 'group' column.\n"
            "7) IMPORTANT: Preserve categorical IDs exactly as they appear (e.g., 'TT125'); DO NOT strip prefixes, DO NOT coerce IDs to numbers in 'group'.\n"
            "7) Return ONLY executable Python code; no markdown; the final variable must be a pandas DataFrame named result with columns: ['x','y'] and optional ['group']."
        )
        user1_parts = [
            f"User question: {query}",
            f"DataFrame shape: {df.shape}",
            f"Columns: {list(df.columns)}",
            f"Dtypes: {schema}",
            f"Head(5): {head_preview}",
        ]
        if date_range:
            user1_parts.append(
                f"Apply date filter if appropriate: start={date_range[0].isoformat()}, end={date_range[1].isoformat()} (inclusive of start, exclusive of end)."
            )
        code_stage1 = _llm_generate_code(sys1, "\n".join(user1_parts), temperature=0.0)
        try:
            df_clean = safe_execute_pandas_code(code_stage1, df)
            if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
                df_clean = _retry_llm_pandas_extraction(query, df, max_retries=1)
        except Exception:
            df_clean = _retry_llm_pandas_extraction(query, df, max_retries=1)
        if not isinstance(df_clean, pd.DataFrame) or df_clean.empty:
            return None
        # Optional date filter if stage1 did not apply
        if date_range:
            try:
                x_guess = _guess_time_column(df_clean)
                if x_guess:
                    xser = pd.to_datetime(df_clean[x_guess], errors='coerce')
                    mask = (xser >= date_range[0]) & (xser < date_range[1])
                    df_clean = df_clean[mask]
            except Exception:
                pass
        if df_clean.empty:
            return None
        # Normalize columns to ['x','y','group?'] if LLM returned different names
        cols_lower = {c.lower(): c for c in df_clean.columns}
        has_standard = ('x' in cols_lower) and ('y' in cols_lower)
        if not has_standard:
            try:
                intent = _llm_parse_graph_intent(query, list(df_clean.columns))
            except Exception:
                intent = None
            x_sel = None
            y_sel = None
            g_sel = None
            if intent:
                x_sel = _closest_column(str(intent.get("x") or ""), list(df_clean.columns))
                y_sel = _closest_column(str(intent.get("y") or ""), list(df_clean.columns))
                g_sel = _closest_column(str(intent.get("group_by") or ""), list(df_clean.columns))
            x_sel = x_sel or _guess_time_column(df_clean)
            if y_sel is None:
                try:
                    y_candidates = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
                    if y_candidates:
                        y_sel = y_candidates[0]
                except Exception:
                    y_sel = None
            ren_map = {}
            if x_sel:
                ren_map[x_sel] = 'x'
            if y_sel:
                ren_map[y_sel] = 'y'
            if g_sel:
                ren_map[g_sel] = 'group'
            if ren_map:
                df_clean = df_clean.rename(columns=ren_map)
        # If 'group' exists and looks numeric but the original df has labeled IDs (e.g., 'TT125'),
        # try to rebuild the label using the most common original string for each numeric token.
        try:
            if 'group' in df_clean.columns:
                grp_series = df_clean['group']
                grp_str = grp_series.astype(str).str.strip()
                is_numeric_like = grp_str.dropna().str.fullmatch(r'\d+(\.\d+)?').all()
                if is_numeric_like:
                    # Build candidate mappings from original df
                    candidates = []
                    for col in df.columns:
                        try:
                            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                                s = df[col].astype(str).str.strip()
                                num = s.str.extract(r'(\d+(?:\.\d+)?)')[0]
                                # Build mapping numeric -> most common label
                                tmp_map = {}
                                valid = pd.concat([s, num], axis=1).dropna()
                                if not valid.empty:
                                    counts = valid.groupby(num)[col].agg(lambda x: x.value_counts().idxmax())
                                    tmp_map = counts.to_dict()
                                    # Coverage of current groups
                                    uniq_groups = set(grp_str.unique())
                                    covered = sum(1 for g in uniq_groups if g in tmp_map)
                                    coverage = covered / max(1, len(uniq_groups))
                                    candidates.append((coverage, tmp_map))
                        except Exception:
                            continue
                    if candidates:
                        candidates.sort(key=lambda t: t[0], reverse=True)
                        best_cov, best_map = candidates[0]
                        # Use mapping if reasonably good coverage
                        if best_cov >= 0.5:
                            df_clean['group'] = grp_str.map(lambda v: best_map.get(v, v))
                # Ensure string dtype for legend purposes
                df_clean['group'] = df_clean['group'].astype(str)
        except Exception:
            pass
        # If we now have x,y, try to plot locally for reliability
        if 'x' in df_clean.columns and 'y' in df_clean.columns:
            try:
                tmp = df_clean[['x', 'y'] + (['group'] if 'group' in df_clean.columns else [])].copy()
                tmp['x'] = pd.to_datetime(tmp['x'], errors='coerce')
                tmp['y'] = pd.to_numeric(tmp['y'], errors='coerce')
                if 'group' in tmp.columns:
                    tmp['group'] = tmp['group'].astype(str)
                tmp = tmp.dropna(subset=['x', 'y']).sort_values('x')
                if not tmp.empty:
                    import plotly.graph_objects as go  # type: ignore
                    if 'group' in tmp.columns:
                        fig = go.Figure()
                        top_groups = tmp['group'].value_counts().head(12).index.tolist()
                        for g in top_groups:
                            sub = tmp[tmp['group'] == g]
                            if not sub.empty:
                                fig.add_trace(go.Scatter(x=list(sub['x']), y=list(sub['y']), mode='lines+markers', name=str(g)))
                        fig.update_layout(title="Series over time", xaxis_title="Time", yaxis_title="Value")
                    else:
                        fig = go.Figure(go.Scatter(x=list(tmp['x']), y=list(tmp['y']), mode='lines+markers', name="Series"))
                        fig.update_layout(title="Series over time", xaxis_title="Time", yaxis_title="Value")
                    formatted_local = format_result_for_response(fig)
                    if formatted_local.get("type") == "plotly":
                        return formatted_local
            except Exception:
                pass
        # Reduce to manageable size for context
        if len(df_clean) > 2000:
            df_clean = df_clean.sample(2000, random_state=42)
        # Stage 2: generate Plotly code for this cleaned DataFrame
        schema2 = _df_schema(df_clean)
        head2 = df_clean.head(5).to_dict(orient='records')
        sys2 = (
            "You are a senior data visualization engineer. Generate robust Plotly code (px/go/ff) that operates on a provided DataFrame named df "
            "and assigns the final figure to a variable named result (a Plotly figure). "
            "The DataFrame df MAY ALREADY be standardized to columns: 'x','y' and optional 'group'. Prefer using them directly if present.\n"
            "Requirements: 1) If 'group' exists, create one trace per group over x (multi-series). Otherwise, create a single series. "
            "2) Prefer line charts over time when 'x' is time-like; otherwise consider bar/scatter as appropriate. "
            "3) Clean empty-like tokens and drop rows with missing values in the plot columns. "
            "4) Ensure x/y arrays are aligned and non-empty. 5) Set a descriptive title and axis labels. "
            "6) If a valid chart is impossible, set result to a top-10 aggregated pandas DataFrame instead (not None). "
            "Return ONLY executable Python code; no markdown; final variable must be named result."
        )
        user2 = "\n".join([
            f"User question: {query}",
            f"Cleaned DataFrame schema: {schema2}",
            f"Head(5): {head2}",
            "If df contains 'x','y' (and optionally 'group'), use them directly to build the chart.",
        ])
        code_stage2 = _llm_generate_code(sys2, user2, temperature=0.0)
        exec_obj = safe_execute_pandas_code(code_stage2, df_clean)
        formatted = format_result_for_response(exec_obj)
        if formatted.get("type") == "plotly":
            return formatted
        # If not a chart, try our robust plot fallback on the cleaned df
        return _retry_llm_plot_from_clean_df(query, df_clean, max_retries=1)
    except Exception:
        return None



def _build_chart_from_intent(df: pd.DataFrame, intent: Dict[str, object], query: Optional[str] = None) -> Optional[Dict[str, object]]:
    try:
        cols = list(df.columns)
        x_req = _closest_column(str(intent.get("x") or ""), cols)
        y_req = _closest_column(str(intent.get("y") or ""), cols)
        grp_req = _closest_column(str(intent.get("group_by") or ""), cols)
        chart_type = str(intent.get("chart_type") or "").lower()
        # Guess time column if needed
        if (not x_req) or (x_req and not pd.to_datetime(df[x_req], errors='coerce').notna().any()):
            x_guess = _guess_time_column(df)
            if x_guess:
                x_req = x_req or x_guess
        # Optional date filter from query
        date_range = _parse_date_range_from_query(query or "") if query else None
        # Multi-series over time (common case: 'each device id over speed')
        if chart_type in {"line", ""} and x_req and y_req and grp_req:
            try:
                tmp = df[[x_req, y_req, grp_req]].copy()
                tmp[x_req] = pd.to_datetime(tmp[x_req], errors='coerce')
                tmp[y_req] = pd.to_numeric(tmp[y_req], errors='coerce')
                tmp[grp_req] = tmp[grp_req].astype(str)
                tmp = tmp.dropna(subset=[x_req, y_req, grp_req]).sort_values(x_req)
                if date_range:
                    s, e = date_range
                    tmp = tmp[(tmp[x_req] >= s) & (tmp[x_req] < e)]
                if tmp.empty:
                    return None
                import plotly.graph_objects as go  # type: ignore
                fig = go.Figure()
                top_groups = tmp[grp_req].value_counts().head(12).index.tolist()
                for g in top_groups:
                    sub = tmp[tmp[grp_req] == g]
                    if not sub.empty:
                        fig.add_trace(go.Scatter(x=list(sub[x_req]), y=list(sub[y_req]), mode='lines+markers', name=str(g)))
                fig.update_layout(title=f"{y_req} over time by {grp_req}", xaxis_title=str(x_req), yaxis_title=str(y_req))
                return format_result_for_response(fig)
            except Exception:
                return None
        # Category + metric bar
        if chart_type == "bar" and grp_req and y_req and not x_req:
            try:
                tmp = df[[grp_req, y_req]].copy()
                tmp[y_req] = pd.to_numeric(tmp[y_req], errors='coerce')
                tmp[grp_req] = tmp[grp_req].astype(str)
                tmp = tmp.dropna(subset=[grp_req, y_req])
                if tmp.empty:
                    return None
                agg = tmp.groupby(grp_req)[y_req].mean().sort_values(ascending=False).head(50).reset_index()
                import plotly.graph_objects as go  # type: ignore
                fig = go.Figure(go.Bar(x=agg[grp_req].astype(str).tolist(), y=agg[y_req].tolist(), name=str(y_req)))
                fig.update_layout(title=f"{y_req} by {grp_req}", xaxis_title=str(grp_req), yaxis_title=str(y_req))
                return format_result_for_response(fig)
            except Exception:
                return None
        # Simple time series
        if chart_type == "line" and x_req and y_req and not grp_req:
            try:
                tmp = df[[x_req, y_req]].copy()
                tmp[x_req] = pd.to_datetime(tmp[x_req], errors='coerce')
                tmp[y_req] = pd.to_numeric(tmp[y_req], errors='coerce')
                tmp = tmp.dropna(subset=[x_req, y_req]).sort_values(x_req)
                if date_range:
                    s, e = date_range
                    tmp = tmp[(tmp[x_req] >= s) & (tmp[x_req] < e)]
                if tmp.empty:
                    return None
                import plotly.graph_objects as go  # type: ignore
                fig = go.Figure(go.Scatter(x=list(tmp[x_req]), y=list(tmp[y_req]), mode='lines+markers', name=str(y_req)))
                fig.update_layout(title=f"{y_req} over time", xaxis_title=str(x_req), yaxis_title=str(y_req))
                return format_result_for_response(fig)
            except Exception:
                return None
    except Exception:
        return None
    return None

def handle_graph_agent(query: str, df: pd.DataFrame) -> Optional[Dict[str, object]]:
    try:
        ql = query.lower()
        if 'date' in ql and 'target' in ql:
            date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'period'])]
            target_match = get_close_matches('target', list(df.columns), n=1, cutoff=0.6)
            if date_cols and target_match:
                x_col = date_cols[0]
                y_col = target_match[0]
                try:
                    tmp = df[[x_col, y_col]].copy()
                    tmp[y_col] = tmp[y_col].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
                    tmp[y_col] = pd.to_numeric(tmp[y_col], errors='coerce')
                    tmp[x_col] = pd.to_datetime(tmp[x_col], errors='coerce')
                    tmp = tmp.dropna(subset=[x_col, y_col])
                    if not tmp.empty:
                        daily = tmp.groupby(tmp[x_col].dt.date)[y_col].mean().reset_index()
                        import plotly.graph_objects as go  # type: ignore
                        fig = go.Figure(go.Scatter(x=list(pd.to_datetime(daily[x_col])), y=list(daily[y_col]), mode='lines+markers'))
                        fig.update_layout(title=f"{y_col} over time", xaxis_title="Date", yaxis_title=y_col)
                        return format_result_for_response(fig)
                except Exception:
                    pass
        # Preferred: clean two-stage LLM pipeline (pandas -> plotly)
        try:
            two_stage = build_graph_two_stage_llm(query, df)
            if two_stage and two_stage.get("type") == "plotly":
                return two_stage
        except Exception:
            pass
        # LLM intent parse for graph semantics (captures 'each device id over speed' type)
        try:
            intent = _llm_parse_graph_intent(query, list(df.columns))
            if intent:
                built = _build_chart_from_intent(df, intent, query=query)
                if built and built.get("type") == "plotly":
                    return built
        except Exception:
            pass
        # Explicit column quick path: map tokens to columns and chart directly
        try:
            import re as _re
            from difflib import get_close_matches as _gcm
            tokens = _re.findall(r"[a-zA-Z0-9_]+", ql)
            col_map = {c.lower(): c for c in df.columns}
            matched: list[str] = []
            seen = set()
            for t in tokens:
                if t in seen:
                    continue
                seen.add(t)
                if t in col_map:
                    cand = col_map[t]
                    if cand not in matched:
                        matched.append(cand)
                    continue
                gm = _gcm(t, list(col_map.keys()), n=1, cutoff=0.85)
                if gm:
                    cand = col_map[gm[0]]
                    if cand not in matched:
                        matched.append(cand)
            if len(matched) >= 2:
                x_raw, y_raw = matched[0], matched[1]
                id_like_tokens = {"id", "device", "sensor", "uuid", "code", "name"}
                def is_datetime_col(c: str) -> bool:
                    try:
                        ser = pd.to_datetime(df[c], errors='coerce')
                        return ser.notna().sum() >= max(5, int(0.2 * len(df)))
                    except Exception:
                        return False
                def is_id_like(c: str) -> bool:
                    lc = c.lower()
                    return any(tok in lc for tok in id_like_tokens)
                def is_numeric_col(c: str) -> bool:
                    try:
                        return pd.api.types.is_numeric_dtype(df[c])
                    except Exception:
                        return False
                # Prefer datetime on x if present
                if is_datetime_col(y_raw) and not is_datetime_col(x_raw):
                    x_raw, y_raw = y_raw, x_raw
                if is_datetime_col(x_raw) and is_numeric_col(y_raw):
                    tmp = df[[x_raw, y_raw]].copy()
                    tmp[x_raw] = pd.to_datetime(tmp[x_raw], errors='coerce')
                    tmp[y_raw] = pd.to_numeric(tmp[y_raw], errors='coerce')
                    tmp = tmp.dropna(subset=[x_raw, y_raw]).sort_values(x_raw)
                    # Optional date filter from query
                    dr = _parse_date_range_from_query(query or "")
                    if dr:
                        s, e = dr
                        tmp = tmp[(tmp[x_raw] >= s) & (tmp[x_raw] < e)]
                    if not tmp.empty:
                        import plotly.graph_objects as go  # type: ignore
                        fig = go.Figure(go.Scatter(x=list(tmp[x_raw]), y=list(tmp[y_raw]), mode='lines+markers', name=str(y_raw)))
                        fig.update_layout(title=f"{y_raw} over time", xaxis_title=str(x_raw), yaxis_title=str(y_raw))
                        return format_result_for_response(fig)
                # If x looks categorical (or id-like) and y numeric → bar mean by category
                if is_numeric_col(y_raw) and (not is_numeric_col(x_raw) or is_id_like(x_raw)):
                    tmp = df[[x_raw, y_raw]].copy()
                    tmp[y_raw] = pd.to_numeric(tmp[y_raw], errors='coerce')
                    tmp[x_raw] = tmp[x_raw].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan})
                    tmp = tmp.dropna(subset=[x_raw, y_raw])
                    if not tmp.empty:
                        agg = tmp.groupby(x_raw)[y_raw].mean().sort_values(ascending=False).head(50).reset_index()
                        import plotly.graph_objects as go  # type: ignore
                        fig = go.Figure(go.Bar(x=agg[x_raw].astype(str).tolist(), y=agg[y_raw].tolist(), name=str(y_raw)))
                        fig.update_layout(title=f"{y_raw} by {x_raw}", xaxis_title=str(x_raw), yaxis_title=str(y_raw))
                        return format_result_for_response(fig)
                # If both numeric → scatter
                if is_numeric_col(x_raw) and is_numeric_col(y_raw):
                    tmp = df[[x_raw, y_raw]].copy()
                    tmp[x_raw] = pd.to_numeric(tmp[x_raw], errors='coerce')
                    tmp[y_raw] = pd.to_numeric(tmp[y_raw], errors='coerce')
                    tmp = tmp.dropna(subset=[x_raw, y_raw])
                    if not tmp.empty:
                        import plotly.graph_objects as go  # type: ignore
                        fig = go.Figure(go.Scatter(x=list(tmp[x_raw]), y=list(tmp[y_raw]), mode='markers', name=f"{y_raw} vs {x_raw}"))
                        fig.update_layout(title=f"{y_raw} vs {x_raw}", xaxis_title=str(x_raw), yaxis_title=str(y_raw))
                        return format_result_for_response(fig)
        except Exception:
            pass
        # Faster: only one attempt to derive a clean aggregated DataFrame
        df_clean = _retry_llm_pandas_extraction(query, df, max_retries=1)
        if df_clean is None or df_clean.empty:
            fig, exp = generate_rescue_chart(df, query)
            if fig is None:
                return None
            formatted = format_result_for_response(fig)
            if exp:
                formatted["explanation"] = exp
            return formatted
        # Faster: attempt local or single LLM generation for the figure
        formatted = _retry_llm_plot_from_clean_df(query, df_clean, max_retries=1)
        return formatted
    except Exception:
        return None







