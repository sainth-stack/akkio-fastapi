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
        model="gpt-4.1-mini",
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
        model="gpt-4.1-mini",
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
        model="gpt-4.1-mini",
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
        df_clean = _retry_llm_pandas_extraction(query, df, max_retries=3)
        if df_clean is None or df_clean.empty:
            fig, exp = generate_rescue_chart(df, query)
            if fig is None:
                return None
            formatted = format_result_for_response(fig)
            if exp:
                formatted["explanation"] = exp
            return formatted
        formatted = _retry_llm_plot_from_clean_df(query, df_clean, max_retries=3)
        return formatted
    except Exception:
        return None







