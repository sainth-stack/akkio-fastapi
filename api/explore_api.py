from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
import io
import json
import re
import ast
import textwrap
import datetime as dt
from difflib import get_close_matches
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from uuid import uuid4
from collections import defaultdict
import threading

from openai import OpenAI
from universal_prompts import (
    prompt_for_data_analyst,
    Prompt_for_code_execution,
    Visualisation_intelligence_engine,
)

# Session memory management - only keep last message for accuracy
SESSION_MEMORY = defaultdict(dict)
SESSION_MEMORY_LOCK = threading.Lock()

explore_router = APIRouter()

def manage_session_memory(session_id: str, user_message: str = None, bot_message: str = None, get_history: bool = False):
    """
    Centralized session memory management - keeps conversation history for better context
    """
    with SESSION_MEMORY_LOCK:
        if get_history:
            # Return conversation history for better context
            session_data = SESSION_MEMORY.get(session_id, {})
            conversation_history = session_data.get("conversation_history", [])
            if conversation_history:
                print(f"Retrieved conversation history for session {session_id}: {len(conversation_history)} messages")
                return conversation_history[-3:]  # Return last 3 messages for context
            else:
                print(f"No conversation history found for session {session_id}")
                return []
        
        if user_message is not None or bot_message is not None:
            # Initialize session if it doesn't exist
            if session_id not in SESSION_MEMORY:
                SESSION_MEMORY[session_id] = {"conversation_history": []}
            
            # Add messages to conversation history
        if user_message is not None:
            SESSION_MEMORY[session_id]["conversation_history"].append({"role": "user", "content": user_message})
            print(f"Stored user message for session {session_id}: {user_message[:100]}...")
        
        if bot_message is not None:
            SESSION_MEMORY[session_id]["conversation_history"].append({"role": "assistant", "content": bot_message})
            print(f"Stored bot message for session {session_id}: {bot_message[:100]}...")
        
        # Keep only last 10 messages to prevent memory overflow
        if len(SESSION_MEMORY[session_id]["conversation_history"]) > 10:
            SESSION_MEMORY[session_id]["conversation_history"] = SESSION_MEMORY[session_id]["conversation_history"][-10:]
        
        return None

def detect_arabic_language(text: str) -> bool:
    """
    Detect if the text contains Arabic characters
    """
    if not text:
        return False
    # Check for Arabic Unicode range (U+0600 to U+06FF)
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))

def get_language_context(query: str) -> tuple:
    """
    Detect language and return appropriate language context and instructions
    Returns: (language_code, language_instructions)
    """
    is_arabic = detect_arabic_language(query)
    
    if is_arabic:
        language_code = "arabic"
        language_instructions = """
LANGUAGE CONTEXT: Arabic (العربية)
- User query contains Arabic text
- Provide responses in Arabic with RTL (right-to-left) formatting
- Use Arabic terminology and culturally appropriate explanations
- Format all HTML content with dir="rtl" lang="ar" attributes
- Maintain professional tone suitable for Arabic-speaking audiences
"""
    else:
        language_code = "english"
        language_instructions = """
LANGUAGE CONTEXT: English
- User query is in English
- Provide responses in clear, professional English
- Use standard LTR (left-to-right) formatting
- Maintain professional and accessible language
- Use technical terminology appropriately with explanations when needed
"""
    
    return language_code, language_instructions

def _find_latest_file_in_directory(directory_path: str, extensions: Tuple[str, ...]) -> Optional[str]:
    try:
        if not os.path.isdir(directory_path):
            return None
        candidates = []
        for name in os.listdir(directory_path):
            path = os.path.join(directory_path, name)
            if os.path.isfile(path) and name.lower().endswith(extensions):
                candidates.append((os.path.getmtime(path), path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    except Exception:
        return None

def load_dataset(preferred_path: Optional[str] = None) -> pd.DataFrame:
    """
    Dynamically load a dataset with the following priority:
    1) preferred_path (if provided and exists)
    2) ENV EXPLORE_DATASET_PATH (if set and exists)
    3) Latest file in ./uploads (csv/xlsx/xls)
    4) Fallback: ./data.csv (if exists)
    Raises HTTPException(404) if no dataset found, or 400 if empty.
    """
    # 1) preferred path from request
    search_order: List[str] = []
    if preferred_path:
        search_order.append(preferred_path)

    # 2) environment variable
    env_path = os.getenv('EXPLORE_DATASET_PATH')
    if env_path:
        search_order.append(env_path)

    # 3) uploads directory latest
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    latest = _find_latest_file_in_directory(uploads_dir, ('.csv', '.xlsx', '.xls'))
    if latest:
        search_order.append(latest)

    # 4) fallback data.csv
    fallback = os.path.join(os.getcwd(), 'data.csv')
    if os.path.exists(fallback):
        search_order.append(fallback)

    # Deduplicate order while preserving priority
    seen: set = set()
    ordered_unique_paths: List[str] = []
    for p in search_order:
        try:
            rp = os.path.realpath(p)
        except Exception:
            rp = p
        if rp not in seen:
            seen.add(rp)
            ordered_unique_paths.append(p)

    # Try to load first readable non-empty dataset
    for path in ordered_unique_paths:
        try:
            if not os.path.exists(path):
                continue
            lower = path.lower()
            if lower.endswith('.csv'):
                df = pd.read_csv(path)
            elif lower.endswith('.xlsx') or lower.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                continue
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    raise HTTPException(status_code=404, detail="No valid dataset found to analyze")

def preprocess_dataframe_for_graphing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for robust chart generation:
    - Normalize empty-like tokens to NaN
    - Parse datetime-like columns
    - Extract numeric values from strings with units (e.g., "28.3kmph", "1959 psi", "214°C").
    - Convert common speed units to km/h when unit prevalence is detected.
    Returns a new dataframe copy.
    """
    df_proc = df.copy()

    # Normalize empty-like tokens across all object columns
    empty_tokens = {'', 'nan', 'null', 'none', 'undefined'}
    for col in df_proc.columns:
        try:
            if df_proc[col].dtype == object:
                df_proc[col] = (
                    df_proc[col]
                    .astype(str)
                    .str.strip()
                    .apply(lambda v: np.nan if str(v).strip().lower() in empty_tokens else v)
                )
        except Exception:
            continue

    # Helper for speed unit conversion → km/h
    speed_units = {
        'km/h': 1.0, 'kmhr': 1.0, 'kmh': 1.0, 'kph': 1.0, 'kmph': 1.0,
        'm/s': 3.6, 'mps': 3.6, 'meter/second': 3.6, 'meters/second': 3.6,
        'mph': 1.60934, 'mi/h': 1.60934, 'mile/h': 1.60934, 'miles/hour': 1.60934,
        'knot': 1.852, 'knots': 1.852, 'kt': 1.852, 'kts': 1.852,
        'cm/s': 0.036, 'cms': 0.036, 'mm/s': 0.0036, 'm/h': 0.001,
        'ft/s': 1.09728, 'fps': 1.09728,
        'km/s': 3600.0,
    }

    def extract_number_and_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
        try:
            s = str(text).strip().lower()
            if s == '' or s in empty_tokens:
                return None, None
            m = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z°/%]+)?', s)
            if not m:
                try:
                    return float(s), None
                except Exception:
                    return None, None
            val = float(m.group(1))
            unit = (m.group(2) or '').strip()
            unit = unit.replace('per', '/').replace(' ', '')
            return val, unit
        except Exception:
            return None, None

    # Datetime parsing + numeric extraction from object columns
    for col in df_proc.columns:
        try:
            # Attempt to parse datetimes if column name suggests time or if many values parse
            name_lower = str(col).lower()
            if any(k in name_lower for k in ['time', 'date', 'timestamp', 'period']):
                parsed = pd.to_datetime(df_proc[col], errors='coerce')
                if parsed.notna().sum() >= max(5, int(0.1 * len(df_proc))):
                    df_proc[col] = parsed
                    continue

            # For object columns with potential units, extract numerics
            if df_proc[col].dtype == object:
                series_obj = df_proc[col].astype(str)
                # Sample-based unit prevalence detection
                sample = series_obj.head(200)
                unit_counts: Dict[str, int] = {}
                numeric_vals: List[Optional[float]] = []
                for v in sample:
                    num, unit = extract_number_and_unit(v)
                    numeric_vals.append(num)
                    if unit:
                        unit_counts[unit] = unit_counts.get(unit, 0) + 1

                # If majority values look numeric (with or without unit), convert entire column
                if sum(1 for x in numeric_vals if x is not None) >= max(10, int(0.3 * len(sample))):
                    # Choose most frequent detected unit
                    dominant_unit = None
                    if unit_counts:
                        dominant_unit = max(unit_counts.items(), key=lambda kv: kv[1])[0]

                    def convert_cell(v: object) -> Optional[float]:
                        num, unit = extract_number_and_unit(v)
                        if num is None:
                            return None
                        if unit is None:
                            return num
                        norm = unit
                        # Normalize common variants
                        if norm in {'km/hr', 'km/hrs', 'kmperhour'}:
                            norm = 'km/h'
                        if norm in {'m/ s', 'm/second', 'meter/sec', 'meters/sec'}:
                            norm = 'm/s'
                        # Convert speed-like units if known
                        if norm in speed_units:
                            factor = speed_units[norm]
                            return num * factor
                        # Otherwise just return numeric part
                        return num

                    df_proc[col] = df_proc[col].apply(convert_cell)
        except Exception:
            continue

    return df_proc


def build_entity_metric_line_chart(df: pd.DataFrame, query: str):
    """
    Build a generic line chart for an entity-id column (e.g., sensor id, device id, asset id)
    against a metric column (e.g., speed/velocity/temperature), optionally over a time axis
    if available. Works across arbitrary datasets by:
      - Detecting entity and metric columns from the query and column names (fuzzy).
      - Extracting numeric values from metric strings with units (e.g., '28.3kmph').
      - Using a detected datetime column as x-axis, else uses record index.

    Returns (plotly_figure, explanation) or (None, None) if not applicable.
    """
    ql = query.lower()

    # Tokenize query for fuzzy matching
    tokens = re.findall(r"[a-zA-Z0-9_%-]+", ql)

    # Prepare helpers
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # Synonyms and scoring helpers
    entity_keywords = {
        'sensor', 'sensore', 'sensorid', 'sensor_id', 'sensor id',
        'device', 'deviceid', 'device_id', 'device id',
        'asset', 'assetid', 'asset_id', 'asset id',
        'machine', 'equipment', 'vehicle', 'truck', 'unit', 'id', 'name'
    }
    metric_hint_keywords = {
        'speed', 'velocity', 'rpm', 'temperature', 'temp', 'pressure', 'load', 'usage', 'value',
        'distance', 'duration', 'throughput', 'latency', 'count', 'rate', 'emission', 'co2',
        'current', 'voltage', 'power', 'energy', 'efficiency', 'cost', 'revenue', 'sales'
    }

    def score_entity(col: str) -> int:
        name = col.lower()
        score = 0
        # Prefer non-numeric/string-like identifiers
        if not pd.api.types.is_numeric_dtype(df[col]) or 'id' in name or 'name' in name:
            score += 1
        # Keyword hits
        for kw in entity_keywords:
            if kw in name:
                score += 3
        # Query token overlap
        for t in tokens:
            if t and t in name:
                score += 2
        # Favor moderate cardinality
        try:
            nunq = df[col].nunique(dropna=True)
            if 1 < nunq <= max(50, int(len(df) * 0.5)):
                score += 2
        except Exception:
            pass
        return score

    def score_metric(col: str) -> int:
        name = col.lower()
        score = 0
        # Numeric or convertible preferred
        if pd.api.types.is_numeric_dtype(df[col]):
            score += 3
        # Keyword hits
        for kw in metric_hint_keywords:
            if kw in name:
                score += 3
        # Query token overlap
        for t in tokens:
            if t and t in name:
                score += 2
        return score

    # Pick best entity and metric columns
    entity_col = max(cols, key=score_entity) if cols else None
    metric_col = max(cols, key=score_metric) if cols else None

    # Quick sanity: ensure they are not the same and metric has meaningful values
    if not entity_col or not metric_col or entity_col == metric_col:
        return None, None

    # Build a working frame
    use_cols = [entity_col, metric_col]
    df_local = df[use_cols].copy()
    if df_local.empty:
        return None, None

    # Helper: detect speed context from column/query tokens
    metric_name_lower = str(metric_col).lower()
    speed_context = (
        ('speed' in metric_name_lower) or ('velocity' in metric_name_lower) or
        any(t in {'speed', 'velocity', 'mph', 'kmph', 'kph', 'km/h', 'm/s', 'knots', 'knot'} for t in tokens)
    )

    # Helper: convert string speed values with units to km/h; otherwise extract numerics
    def convert_speed_series_to_kmh(series: pd.Series) -> pd.Series:
        unit_factor_map = {
            'km/h': 1.0, 'kmhr': 1.0, 'kmh': 1.0, 'kph': 1.0, 'kmph': 1.0,
            'm/s': 3.6, 'mps': 3.6, 'meter/second': 3.6, 'meters/second': 3.6,
            'mph': 1.60934, 'mi/h': 1.60934, 'mile/h': 1.60934, 'miles/hour': 1.60934,
            'knot': 1.852, 'knots': 1.852, 'kt': 1.852, 'kts': 1.852,
            'cm/s': 0.036, 'cms': 0.036, 'mm/s': 0.0036, 'm/h': 0.001,
            'ft/s': 1.09728, 'fps': 1.09728,
            'km/s': 3600.0,
        }

        def parse_and_convert(val: object) -> float:
            if pd.isna(val):
                return np.nan  # type: ignore
            text = str(val).strip().lower()
            if text == '' or text in {'nan', 'null', 'none', 'undefined'}:
                return np.nan  # type: ignore
            # Extract number and a trailing unit word/slash pattern if any
            match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z/]+)?', text)
            if not match:
                return pd.to_numeric(text, errors='coerce')
            num_str = match.group(1)
            unit_str = (match.group(2) or '').strip()
            try:
                value = float(num_str)
            except Exception:
                return pd.to_numeric(num_str, errors='coerce')
            if unit_str in unit_factor_map:
                return value * unit_factor_map[unit_str]
            # Normalize common variants
            norm = unit_str.replace('per', '/').replace(' ', '')
            if norm in unit_factor_map:
                return value * unit_factor_map[norm]
            # Heuristics: detect m/s or km/h forms with separators
            if norm in {'km/hr', 'km/hrs', 'kmperhour'}:
                return value * 1.0
            if norm in {'m/ s', 'm/second', 'meter/sec', 'meters/sec'}:
                return value * 3.6
            if norm.endswith('/s') and norm.startswith('m'):
                return value * 3.6
            if norm.endswith('/h') and norm.startswith('km'):
                return value * 1.0
            if 'mph' in norm:
                return value * 1.60934
            if 'knot' in norm or norm.endswith('kt') or norm.endswith('kts'):
                return value * 1.852
            return value  # unknown unit → leave as-is

        return series.apply(parse_and_convert)  # type: ignore

    # Extract or convert numeric values from metric when needed
    yaxis_title = metric_col
    if not pd.api.types.is_numeric_dtype(df_local[metric_col]):
        ser = df_local[metric_col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan})
        if speed_context:
            df_local[metric_col] = convert_speed_series_to_kmh(ser)
            yaxis_title = 'Speed (km/h)'
        else:
            df_local[metric_col] = ser.str.extract(r'([-+]?\d*\.?\d+)')[0]
            df_local[metric_col] = pd.to_numeric(df_local[metric_col], errors='coerce')

    # Detect a datetime column for x-axis
    time_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col = c
            break
    if time_col is None:
        # Try to coerce any likely time column by name
        for c in df.columns:
            name = c.lower()
            if any(k in name for k in ['time', 'date', 'timestamp', 'period']):
                try:
                    coerced = pd.to_datetime(df[c], errors='coerce')
                    if coerced.notna().sum() >= max(5, int(len(df) * 0.2)):
                        time_col = c
                        df_local['__x__'] = coerced
                        break
                except Exception:
                    continue
    if time_col is None:
        # Fallback to index order as x
        df_local['__x__'] = np.arange(len(df_local))
        x_title = 'Record'
    else:
        # align and coerce
        df_local['__x__'] = pd.to_datetime(df[time_col], errors='coerce')
        x_title = time_col

    # Clean rows
    df_local = df_local.dropna(subset=['__x__', metric_col, entity_col])
    if df_local.empty:
        return None, None

    # Limit number of entities to avoid overplotting
    top_entities = (
        df_local[entity_col].astype(str).value_counts().head(12).index.tolist()
    )
    df_local = df_local[df_local[entity_col].astype(str).isin(top_entities)]

    # Build figure: x = time/index, y = metric, color by entity
    import plotly.graph_objects as go
    fig = go.Figure()
    for ent, grp in df_local.sort_values('__x__').groupby(entity_col):
        y_vals = pd.to_numeric(grp[metric_col], errors='coerce').dropna()
        x_vals = grp.loc[y_vals.index, '__x__']
        if len(x_vals) > 0:
            fig.add_trace(
                go.Scatter(x=list(x_vals), y=list(y_vals), mode='lines+markers', name=str(ent))
            )

    fig.update_layout(
        title=f"{entity_col} vs {metric_col}",
        xaxis_title=x_title,
        yaxis_title=yaxis_title,
        legend_title=entity_col
    )

    exp = (
        f"Line chart showing {metric_col} over {x_title} for each {entity_col}. "
        f"Numeric values were extracted dynamically from strings where necessary."
    )
    return fig, exp



# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Optional Plotly availability flag for formatting results
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    go = None  # type: ignore
    PLOTLY_AVAILABLE = False


@explore_router.post("/api/Explore")
async def senior_data_analysis(
    query: str = Form(...),
    dataset_path: Optional[str] = Form(None),
    session_id: str = Form(None),
):
    try:
        # Session handling
        if not session_id:
            session_id = str(uuid4())
            print(f"Generated new session ID: {session_id}")
        
        print(f"Processing query for session {session_id}: {query}")

        # Get chat history for LLM context
        chat_history_for_llm = manage_session_memory(session_id, get_history=True)
        
        # Enhanced logging for follow-up question debugging  
        if chat_history_for_llm:
            print(f"Session {session_id} has previous message for context")
            # Look for potential follow-up indicators
            follow_up_indicators = ['them', 'those', 'these', 'it', 'they']
            if any(indicator in query.lower() for indicator in follow_up_indicators):
                print(f"FOLLOW-UP QUERY DETECTED: '{query}'")
                last_user_query = chat_history_for_llm[0]['content'] if chat_history_for_llm else None
                if last_user_query:
                    print(f"Previous user query for context: '{last_user_query}'")
        else:
            print(f"No previous message found for session {session_id}")

        # Load and validate data dynamically
        df = load_dataset(dataset_path)

        # Preprocess dataframe for robust graphing and analysis
        df = preprocess_dataframe_for_graphing(df)

        # Generate metadata
        metadata_str = ", ".join(df.columns.tolist())

        # Check if query is report-related (Agent 1)
        is_report_query = any(keyword in query.lower() for keyword in
                              ['report', 'summary report', 'analysis report', 'detailed report',
                               'comprehensive report', 'summary_report', 'analysis_report', 'detailed_report',
                               'comprehensive_report'])

        if is_report_query:
            # Generate report-specific prompt
            prompt_eng = (
                f"""
                    You are a Senior data analyst generating a comprehensive report with advanced analytics capabilities. 
                    Always strictly adhere to the following rules: 

                    The metadata required for your analysis: {metadata_str}
                    Consider ALL rows in the currently loaded dataset in memory. No data assumptions can be taken. Consider the entire range from first row to last row. Do not assume any data outside this range.

                    UNIVERSAL DATE HANDLING AND ANALYSIS REQUIREMENTS:
                    1. **Multi-Format Date Detection**: Automatically detect and parse ANY date format:
                    - Timestamp: "01-05-2025 08:00:30", "2025-01-05 08:00:30"
                    - UTC Format: "2025-01-05T08:00:30Z", "2025-01-05T08:00:30.123Z"
                    - ISO Format: "2025-01-05T08:00:30+00:00"
                    - Simple Date: "01-05-2025", "2025-01-05", "05/01/2025"
                    - Any other datetime format

                    2. **Intelligent Data Aggregation Strategy**:
                    - **High-frequency data (seconds/minutes/hours)**: 
                        * Convert all timestamps to ISO format internally
                        * Calculate daily averages from all timestamps within each day
                        * For monthly analysis: Average all daily values within each month
                        * Example: If you have 01-05-2025 08:00, 01-05-2025 09:00, 01-05-2025 10:00
                        → Calculate single daily average for 01-05-2025
                    
                    - **UTC/ISO Timestamps**:
                        * Parse UTC timestamps and convert to local date representation
                        * Apply same daily → monthly aggregation logic
                        * Maintain timezone awareness for accurate date grouping

                    3. **Analysis Logic Based on Data Frequency**:
                    - **Sub-daily data** → Aggregate to daily → Provide daily and monthly insights
                    - **Daily data** → Direct analysis with daily and monthly patterns
                    - **Weekly/Monthly data** → Direct analysis for periodic patterns

                    4. **Date Processing Pipeline**:
                    - Step 1: Detect all possible date/timestamp columns
                    - Step 2: Parse and standardize ALL date formats to ISO internally
                    - Step 3: Determine original data frequency (seconds, minutes, hours, days)
                    - Step 4: Apply appropriate aggregation (if needed)
                    - Step 5: Analyze patterns across the complete date range from {df.head(1)} to {df.tail(1)}
                    - Step 6: Generate insights based on temporal patterns and trends

                    ###IMPORTANT: You MUST return the response in this EXACT JSON format:
                    {{
                        "title": "[Concise, descriptive title for the analysis]",
                        "description": "[Brief one sentence summary of what the analysis covers]",
                        "report": {{
                            "heading": "[Suitable Heading Based on Data Analysis]",
                            "paragraphs": [
                                "First Bullet Point: Analyzed [DATA_FREQUENCY] data from [START_DATE] to [END_DATE] with [TOTAL_RECORDS] records. Data shows [AGGREGATION_METHOD] applied to convert [ORIGINAL_FREQUENCY] timestamps to [FINAL_FREQUENCY] intervals. Quality assessment reveals [DATA_QUALITY_INSIGHTS] with [MISSING_DATA_PERCENTAGE]% missing values. Key patterns include [TREND_DIRECTION] trend with [VOLATILITY_LEVEL] volatility observed across the time series.",
                                "Second Bullet Point: Time series exhibits [STATISTICAL_SUMMARY] with mean value of [MEAN_VALUE] and standard deviation of [STD_DEVIATION]. Distribution analysis shows [DISTRIBUTION_TYPE] pattern with [SKEWNESS_DIRECTION] skewness. Correlation analysis between temporal features reveals [CORRELATION_INSIGHTS]. Outlier detection identified [OUTLIER_COUNT] anomalous points representing [OUTLIER_PERCENTAGE]% of total observations.",
                               ],
                            "table": {{
                                "headers": ["Metric", "Current Value ([LAST_DATE])", "Period Average", "Peak Value", "Growth Rate", "Data Quality"],
                                "rows": [
                                    ["[PRIMARY_METRIC_NAME]", "[CURRENT_VALUE]", "[PERIOD_AVERAGE]", "[PEAK_VALUE]", "[GROWTH_RATE]%", "[QUALITY_SCORE]%"],
                                    ["[SECONDARY_METRIC_NAME]", "[CURRENT_VALUE_2]", "[PERIOD_AVERAGE_2]", "[PEAK_VALUE_2]", "[GROWTH_RATE_2]%", "[QUALITY_SCORE_2]%"],
                                    ["[TERTIARY_METRIC_NAME]", "[CURRENT_VALUE_3]", "[PERIOD_AVERAGE_3]", "[PEAK_VALUE_3]", "[GROWTH_RATE_3]%", "[QUALITY_SCORE_3]%"]
                                ]
                            }},
                            "analysis_charts": [
                                {{
                                    "title": "Data Distribution and Aggregation Analysis",
                                    "plotly": {{
                                        "data": [{{
                                            "x": ["[ACTUAL_CATEGORIES_OR_TIME_BINS]"],
                                            "y": "[AGGREGATED_VALUES_FROM_DATA]",
                                            "type": "bar",
                                            "marker": {{"color": "#3498db"}},
                                            "name": "Aggregated Distribution"
                                        }}],
                                        "layout": {{
                                            "title": "[METRIC_NAME] Distribution After [AGGREGATION_TYPE] Aggregation",
                                            "xaxis": {{"title": "[TIME_PERIOD_OR_CATEGORY]"}},
                                            "yaxis": {{"title": "Aggregated [VALUE_NAME]"}},
                                            "paper_bgcolor": "#fafafa",
                                            "plot_bgcolor": "#ffffff"
                                        }}
                                    }}
                                }},
                                {{
                                    "title": "Historical Trend Analysis",
                                    "plotly": {{
                                        "data": [{{
                                            "x": ["[PROCESSED_DATES_SEQUENCE]"],
                                            "y": "[AGGREGATED_TIME_SERIES_VALUES]",
                                            "type": "scatter",
                                            "mode": "lines+markers",
                                            "marker": {{"color": "#e74c3c"}},
                                            "name": "Historical Trend (Aggregated)"
                                        }}],
                                        "layout": {{
                                            "title": "Time Series Trend from [START_DATE] to [END_DATE] ([AGGREGATION_LEVEL])",
                                            "xaxis": {{"title": "Date ([FINAL_FREQUENCY])"}},
                                            "yaxis": {{"title": "[AGGREGATED_METRIC_NAME]"}},
                                            "paper_bgcolor": "#fafafa",
                                            "plot_bgcolor": "#ffffff"
                                        }}
                                    }}
                                }}
                            ]
                        }}
                    }}

                    CRITICAL IMPLEMENTATION STEPS FOR UNIVERSAL DATE HANDLING:
                    1. **STEP 1**: Load data.csv and scan ALL columns for potential date/timestamp formats from {df.head(1)} to {df.tail(1)}.
                    2. **STEP 2**: Parse ANY date format (timestamp, UTC, ISO, simple date) and standardize internally
                    3. **STEP 3**: Determine original data frequency and decide on aggregation strategy:
                    - Seconds/Minutes/Hours → Aggregate to Daily → Monthly analysis
                    - Daily → Direct daily and monthly analysis  
                    - Weekly/Monthly → Direct period analysis
                    4. **STEP 4**: Apply intelligent aggregation:
                    - For sub-daily: Calculate daily averages from all timestamps within each day
                    - For monthly analysis: Average all daily values within each month
                    5. **STEP 5**: Perform comprehensive statistical analysis on aggregated data
                    6. **STEP 6**: Generate insights about patterns, trends, and anomalies
                    7. **STEP 7**: Create visualizations showing historical patterns and distributions
                    8. **STEP 8**: Replace ALL placeholders with actual aggregated values and insights

                    AGGREGATION EXAMPLES:
                    - **Input**: "01-05-2025 08:00:30", "01-05-2025 14:30:15", "01-05-2025 20:45:00"
                    - **Process**: Average all values for 01-05-2025 → Single daily value
                    - **Output**: One data point for "01-05-2025" representing daily average

                    - **UTC Input**: "2025-01-05T08:00:30Z", "2025-01-05T14:30:15Z" 
                    - **Process**: Convert to local dates, aggregate by day → Daily averages
                    - **Output**: Daily aggregated values for analysis

                    ANALYSIS ACCURACY REQUIREMENTS:
                    - Use actual column names and aggregated values from the CSV
                    - Generate insights based on aggregated historical patterns
                    - Include proper statistical analysis for aggregated data
                    - Provide business insights relevant to the aggregation level chosen
                    - Focus on descriptive and diagnostic analytics (what happened and why)
                    - Include trend analysis, seasonality detection, and anomaly identification

                    Generate a comprehensive report with intelligent date handling and analysis for: {query}
                    The report must include:
                        - 4 Bullet points (2 lines each): current analysis of the data.
                        - 1 summary table  and all other analysis metrics
                        - 2 analysis charts showing current data patterns from the data with main Heading of Analysis.

                    The report must include proper universal date format handling, intelligent aggregation, and realistic insights based entirely on the processed CSV data.
                        """
            )

            code = generate_data_code(prompt_eng)
            result = simulate_and_format_with_llm(code, df)
            cleaned_result = clean_json_response(result)

            # Store report generation in memory
            manage_session_memory(session_id, user_message=query, bot_message="Generated a comprehensive report based on your request.")
            
            return JSONResponse(
                content=jsonable_encoder({
                    "report": cleaned_result["report"],
                    "title": cleaned_result["title"],
                    "description": cleaned_result["description"],
                    "session_id": session_id
                }),
                status_code=200
            )
        # Not a report query → route to one of the three agents (graph/table/text)
        agent = detect_agent(query)

        # Handle general questions (greetings, casual conversation)
        if agent == "general":
            print(f"Processing general question for session {session_id}: {query}")
            general_response = handle_general_question(query)
            
            # Store general response in memory
            manage_session_memory(session_id, user_message=query, bot_message="Responded to general question.")
            
            return JSONResponse(
                content=jsonable_encoder({
                    "type": "conversational_answer",
                    "payload": general_response,
                    "session_id": session_id
                }),
                status_code=200
            )

        # 2,3,4 Agents: Try simple direct handling first for table/text intents
        if agent in ["table", "text"]:
            can_handle_directly, structured = classify_query_complexity(query, list(df.columns))
            if can_handle_directly and structured is not None:
                intent, column = structured
                simple_resp = handle_simple_query(df, intent, column)
                # Store simple query in memory
                manage_session_memory(session_id, user_message=query, bot_message=f"Processed simple query about {column}.")
                simple_resp["session_id"] = session_id
                return JSONResponse(content=jsonable_encoder(simple_resp), status_code=200)

        # GRAPH agent: two-stage LLM flow → (1) extract clean data (2) plot with Plotly, both with retries
        if agent == "graph":
            try:
                print(f"Attempting graph generation for session {session_id}: {query}")
                formatted_graph = handle_graph_agent(query, df)
                if formatted_graph is not None:
                    print(f"Graph generation successful for session {session_id}")
                    # Store graph generation in memory
                    manage_session_memory(session_id, user_message=query, bot_message="Generated a graph visualization based on your request.")
                    formatted_graph["session_id"] = session_id
                    return JSONResponse(content=jsonable_encoder(formatted_graph), status_code=200)
                else:
                    print(f"Graph agent returned None for session {session_id}, trying fallback")
            except Exception as e:
                # proceed to generic LLM path / rescue fallback below
                print(f"Graph agent failed for session {session_id} with error: {str(e)}, falling back to generic LLM")
                pass

        # Use LLM to generate analysis code based on agent intent with chat history
        analysis_result = get_llm_analysis_explore(query, df, mode=agent, chat_history=chat_history_for_llm)

        response_type = analysis_result.get("type")
        payload = analysis_result.get("payload")

        # Determine bot response text for memory
        bot_response_text = "I have processed your request."
        if response_type == "conversational_answer":
            bot_response_text = str(payload) if payload else "I provided a conversational response."
        elif response_type == "data_analysis_answer":
            explanation = payload.get("explanation") if isinstance(payload, dict) else ""
            bot_response_text = explanation or "I have generated the data analysis you requested."
        elif isinstance(payload, str):
            bot_response_text = payload

        # Store current exchange in memory
        manage_session_memory(session_id, user_message=query, bot_message=bot_response_text)

        if not response_type or not payload:
            error_msg = format_text_response("<h4>I'm sorry, I couldn't process that request.</h4><p>Please try rephrasing your question and I'll be happy to help you analyze your data.</p>")
            return JSONResponse(content=jsonable_encoder({"type": "text", "payload": error_msg, "session_id": session_id}), status_code=200)

        # Conversational/text-only answer
        if response_type == "conversational_answer":
            # Ensure HTML formatting for conversational responses
            formatted_payload = format_text_response(str(payload)) if payload else payload
            return JSONResponse(content=jsonable_encoder({"type": "text", "payload": formatted_payload, "session_id": session_id}), status_code=200)

        # Data analysis answer with code to execute
        if response_type == "data_analysis_answer":
            explanation = payload.get("explanation")
            code = payload.get("code")

            if not code:
                fallback_msg = format_text_response(f"<h4>I understood your request</h4><p>{explanation or 'I understood your request but couldn\'t generate the right code. Please try again with more specific details.'}</p>")
                return JSONResponse(content={"type": "text", "payload": fallback_msg, "session_id": session_id}, status_code=200)

            try:
                exec_result = safe_execute_pandas_code(code, df)
                formatted = format_result_for_response(exec_result)

                # If graph agent but no valid chart, attempt a robust server-side rescue chart
                if agent == "graph" and formatted.get("type") != "plotly":
                    print(f"Attempting rescue chart for session {session_id}")
                    rescue_fig, biz_exp = generate_rescue_chart(df, query)
                    if rescue_fig is not None:
                        print(f"Rescue chart successful for session {session_id}")
                        formatted = format_result_for_response(rescue_fig)
                        # Prefer business-style explanation
                        formatted["explanation"] = biz_exp or (explanation or "")
                        formatted["session_id"] = session_id
                        return JSONResponse(content=formatted, status_code=200)
                    else:
                        print(f"Rescue chart also failed for session {session_id}")

                if explanation:
                    formatted["explanation"] = explanation
                formatted["session_id"] = session_id
                return JSONResponse(content=jsonable_encoder(formatted), status_code=200)
            except Exception as e:
                error_msg = format_text_response(f"<h4>Analysis Error</h4><p>There was an error executing the analysis: {str(e)}</p><p>Please try rephrasing your question or provide more specific details.</p>")
                return JSONResponse(content=jsonable_encoder({"type": "text", "payload": error_msg, "explanation": explanation, "session_id": session_id}), status_code=200)

        # Fallback
        fallback_msg = format_text_response(f"<h4>Unrecognized Response</h4><p>I encountered an unexpected response type: {response_type}</p><p>Please try rephrasing your question and I'll provide a better analysis.</p>")
        return JSONResponse(content=jsonable_encoder({"type": "text", "payload": fallback_msg, "session_id": session_id}), status_code=200)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Data file is corrupt"
        )
    except Exception as e:
        # Ensure session_id is available for error responses
        session_id_for_error = session_id if 'session_id' in locals() else None
        print(f"Explore API error for session {session_id_for_error}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


def generate_data_code(prompt_eng: str) -> str:
    response = client.chat.completions.create(
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


def simulate_and_format_with_llm(
    code_to_simulate: str,
    dataframe: pd.DataFrame
) -> str:
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
   - Do not include the internal headings like "**LANGUAGE:** Python | **MODE:** Simulation | **STATUS:** Success\n\n⚡ **EXECUTION OUTPUT:**\n".Do not include any of these ,,Just include the headings in the middle of the content only.
   - Report can be present in the markdown format.
   - **If you got the basic code to execute, you MUST execute and give the exact result**. **DO NOT** add all the things regarding visualisation to that.
    """

    response = client.chat.completions.create(
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


def clean_json_response(response_text: str):
    try:
        # Method 1: Try to extract JSON from markdown code blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)

        if match:
            json_str = match.group(1)
            json_data = json.loads(json_str)
            return json_data

        # Method 2: If no markdown blocks, try to find JSON object directly
        json_pattern2 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern2, response_text, re.DOTALL)

        for match in matches:
            try:
                json_data = json.loads(match)
                return json_data
            except json.JSONDecodeError:
                continue

        return None

    except Exception as e:
        print(f"Error cleaning JSON: {e}")
        return None



# =========================
# New helper functions for Explore agents (graph/table/text)
# =========================

def detect_agent(query: str) -> str:
    q = query.lower().strip()
    
    # Legal keywords - highest priority
    legal_keywords = [
        'law', 'legal', 'penalty', 'fine', 'regulation', 'rule', 'court', 'judge', 'lawyer',
        'contract', 'agreement', 'policy', 'compliance', 'violation', 'offense', 'crime',
        'rights', 'obligation', 'liability', 'jurisdiction', 'legislation', 'statute',
        'administrative', 'federal', 'cabinet', 'resolution', 'decree', 'ordinance',
        'document', 'pdf', 'analysis', 'summary', 'extract', 'content',
        'قانون', 'قانوني', 'غرامة', 'عقوبة', 'محكمة', 'قاضي', 'محامي', 'عقد', 'اتفاقية',
        'سياسة', 'امتثال', 'انتهاك', 'جريمة', 'حقوق', 'التزام', 'مسؤولية', 'اختصاص',
        'تشريع', 'إدارة', 'اتحادي', 'مجلس', 'قرار', 'مرسوم', 'أمر'
    ]
    
    # Check for legal questions first
    if any(keyword in q for keyword in legal_keywords):
        return "text"  # Route legal questions to text agent for legal analysis
    
    # General greeting and casual questions
    general_greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how are you doing', 'what\'s up', 'whats up', 'sup',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'nice to meet you',
        'how can you help', 'what can you do', 'who are you', 'what are you'
    ]
    
    # Check for general greetings
    if any(greeting in q for greeting in general_greetings):
        return "general"
    
    # Non-legal questions - redirect to general handler
    if any(keyword in q for keyword in ['tell me about', 'what is', 'explain', 'describe', 'information about']):
        return "general"
    
    graph_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'visualise', 'trend', 'histogram',
        'scatter', 'line chart', 'line', 'bar chart', 'bar', 'pie', 'box', 'violin', 'heatmap'
    ]
    table_keywords = [
        'table', 'list', 'rows', 'records', 'show rows', 'display table', 'show table', 'top ', 'head', 'tail'
    ]
    
    # Enhanced detection for conversational queries that should stay as text
    conversational_keywords = [
        'what is', 'what are', 'how do', 'how can', 'explain', 'tell me about', 
        'kpi', 'kpis', 'key performance indicator', 'metrics', 'help', 'understand',
        'define', 'meaning', 'purpose', 'importance', 'benefits'
    ]

    # Priority: conversational queries should stay as text for better responses
    if any(k in q for k in conversational_keywords):
        return "text"
    if any(k in q for k in graph_keywords):
        return "graph"
    if any(k in q for k in table_keywords):
        return "table"
    return "text"


def handle_general_question(query: str) -> str:
    """
    Handle general questions and redirect non-legal questions to legal topics
    """
    q = query.lower().strip()
    
    # Detect language
    is_arabic = detect_arabic_language(query)
    
    # Check if it's a legal-related question
    legal_keywords = [
        'law', 'legal', 'penalty', 'fine', 'regulation', 'rule', 'court', 'judge', 'lawyer',
        'contract', 'agreement', 'policy', 'compliance', 'violation', 'offense', 'crime',
        'rights', 'obligation', 'liability', 'jurisdiction', 'legislation', 'statute',
        'administrative', 'federal', 'cabinet', 'resolution', 'decree', 'ordinance',
        'قانون', 'قانوني', 'غرامة', 'عقوبة', 'محكمة', 'قاضي', 'محامي', 'عقد', 'اتفاقية',
        'سياسة', 'امتثال', 'انتهاك', 'جريمة', 'حقوق', 'التزام', 'مسؤولية', 'اختصاص',
        'تشريع', 'إدارة', 'اتحادي', 'مجلس', 'قرار', 'مرسوم', 'أمر'
    ]
    
    is_legal_question = any(keyword in q for keyword in legal_keywords)
    
    if is_arabic:
        # Arabic responses
        if any(greeting in q for greeting in ['hi', 'hello', 'hey', 'مرحبا', 'أهلا', 'السلام عليكم']):
            return '<div dir="rtl" lang="ar"><h3>نظام التحليل القانوني</h3><h4>مرحبا بك!</h4><p>أهلا وسهلا بك في نظام التحليل القانوني المتقدم. أنا مساعدك الذكي المتخصص في تحليل الوثائق القانونية الإماراتية.</p><h4>كيف يمكنني مساعدتك في الشؤون القانونية؟</h4><ul><li><strong>التحليل القانوني:</strong> تحليل الوثائق القانونية والملفات PDF</li><li><strong>الاستفسارات القانونية:</strong> الإجابة على أسئلتك حول القوانين واللوائح الإماراتية</li><li><strong>الغرامات والعقوبات:</strong> توضيح الغرامات والعقوبات المطبقة</li><li><strong>اللوائح الإدارية:</strong> شرح القرارات والمراسيم الحكومية</li></ul><p>ما هو السؤال القانوني الذي تود الاستفسار عنه؟</p></div>'
        elif any(greeting in q for greeting in ['thanks', 'thank you', 'شكرا', 'شكراً']):
            return '<div dir="rtl" lang="ar"><h3>نظام التحليل القانوني</h3><h4>العفو!</h4><p>سعيد لكوني استطعت مساعدتك في الشؤون القانونية. إذا كان لديك أي أسئلة قانونية أخرى، فلا تتردد في السؤال.</p></div>'
        elif any(greeting in q for greeting in ['bye', 'goodbye', 'مع السلامة', 'وداعاً']):
            return '<div dir="rtl" lang="ar"><h3>نظام التحليل القانوني</h3><h4>مع السلامة!</h4><p>أتمنى لك يوماً سعيداً. أراكم قريباً!</p></div>'
        elif not is_legal_question:
            # Non-legal question - redirect to legal topics
            return '<div dir="rtl" lang="ar"><h3>نظام التحليل القانوني</h3><h4>إجابة مباشرة</h4><p>السؤال "{query}" لا يتعلق بالوثائق القانونية الإماراتية المقدمة. تركز الوثائق على قوانين ولوائح دولة الإمارات العربية المتحدة المختلفة، بما في ذلك الغرامات الإدارية والدعم الاجتماعي ومكافآت موظفي المساجد ورسوم الخدمات التحتية.</p><h4>السياق القانوني الإماراتي</h4><ul><li><strong>الغرامات الإدارية لمخالفات الهجرة:</strong> هيئة الهوية والجنسية والجمارك وأمن المنافذ</li><li><strong>الدعم الاجتماعي للأفراد العاطلين:</strong> قرار مجلس الوزراء لعام 2025</li><li><strong>مكافآت موظفي المساجد:</strong> قرار مجلس الوزراء بشأن موظفي المساجد</li><li><strong>رسوم الخدمات التحتية:</strong> قرار مجلس الوزراء لعام 2021</li></ul><h4>الإرشادات العملية</h4><ul><li><strong>للمقيمين والزوار:</strong> تأكد من تجديد التأشيرات وبطاقات الهوية في الوقت المحدد لتجنب الغرامات</li><li><strong>لأصحاب العمل:</strong> كن على دراية بإرشادات المكافآت لموظفي المساجد للامتثال للوائح الوطنية</li><li><strong>لمشاريع البنية التحتية:</strong> احصل على التصاريح اللازمة والتزم بالمواعيد النهائية لتجنب الغرامات الباهظة</li></ul><hr><p><em>✅ تم إكمال التحليل القانوني المهني</em></p><p><em>هذه المعلومات مبنية على الوثائق القانونية الإماراتية وهي للإرشاد فقط. للمسائل القانونية المحددة، يرجى استشارة خبير قانوني مؤهل.</em></p><h4>المراجع القانونية</h4><p><strong>المصدر:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p><p><strong>مرجع إضافي:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p></div>'.format(query=query)
        else:
            return '<div dir="rtl" lang="ar"><h3>نظام التحليل القانوني</h3><h4>مرحبا بك!</h4><p>أهلا وسهلا بك في نظام التحليل القانوني المتقدم. أنا هنا لمساعدتك في الشؤون القانونية الإماراتية.</p><h4>كيف يمكنني مساعدتك؟</h4><ul><li><strong>التحليل القانوني:</strong> تحليل الوثائق القانونية والملفات PDF</li><li><strong>الاستفسارات القانونية:</strong> الإجابة على أسئلتك حول القوانين واللوائح</li><li><strong>الغرامات والعقوبات:</strong> توضيح الغرامات والعقوبات المطبقة</li></ul></div>'
    else:
        # English responses
        if any(greeting in q for greeting in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "<h3>LEGAL ANALYSIS SYSTEM</h3><h4>Hello there!</h4><p>Welcome to our advanced Legal Analysis System! I'm your intelligent assistant specialized in UAE legal document analysis and processing.</p><h4>How can I help you with legal matters today?</h4><ul><li><strong>Legal Analysis:</strong> I can analyze legal documents and PDF files</li><li><strong>Legal Queries:</strong> I can answer questions about UAE laws and regulations</li><li><strong>Penalties & Fines:</strong> I can explain applicable penalties and fines</li><li><strong>Administrative Regulations:</strong> I can explain government decisions and decrees</li></ul><p>What legal question would you like to ask?</p>"
        elif any(greeting in q for greeting in ['thanks', 'thank you']):
            return "<h3>LEGAL ANALYSIS SYSTEM</h3><h4>You're welcome!</h4><p>I'm glad I could help with your legal inquiry! If you have any other legal questions, feel free to ask.</p>"
        elif any(greeting in q for greeting in ['bye', 'goodbye', 'see you']):
            return "<h3>LEGAL ANALYSIS SYSTEM</h3><h4>Goodbye!</h4><p>Have a great day! See you soon!</p>"
        elif not is_legal_question:
            # Non-legal question - redirect to legal topics
            return "<h3>LEGAL ANALYSIS SYSTEM</h3><h4>Direct Answer</h4><p>The question \"{query}\" does not pertain to the UAE legal documents provided. The documents focus on various UAE laws and regulations, including administrative fines, social support, mosque employee remunerations, and infrastructure service fees.</p><h4>UAE Legal Context Overview</h4><ul><li><strong>Administrative Fines for Immigration Violations:</strong> Federal Authority for Identity, Citizenship, Customs, and Port Security</li><li><strong>Social Support for Unemployed Individuals:</strong> Cabinet Resolution of 2025</li><li><strong>Mosque Employee Remunerations:</strong> Cabinet Resolution on Mosque Employees</li><li><strong>Infrastructure Service Fees:</strong> Cabinet Resolution of 2021</li></ul><h4>Practical Guidance</h4><ul><li><strong>For Residents and Visitors:</strong> Ensure timely renewal of visas and identity cards to avoid fines</li><li><strong>For Employers:</strong> Be aware of the remuneration guidelines for mosque employees to comply with national regulations</li><li><strong>For Infrastructure Projects:</strong> Obtain necessary permissions and adhere to timelines to avoid hefty fines</li></ul><hr><p><em>✅ PROFESSIONAL LEGAL ANALYSIS COMPLETED</em></p><p><em>This information is based on UAE legal documents and is for guidance only. For specific legal matters, please consult with a qualified UAE legal professional.</em></p><h4>Legal References</h4><p><strong>SOURCE:</strong> <a href=\"https://uaelegislation.gov.ae\" target=\"_blank\" style=\"color: #3498db;\">https://uaelegislation.gov.ae</a></p><p><strong>Additional Reference:</strong> <a href=\"https://www.moj.gov.ae\" target=\"_blank\" style=\"color: #3498db;\">https://www.moj.gov.ae</a></p>".format(query=query)
        else:
            return "<h3>LEGAL ANALYSIS SYSTEM</h3><h4>Hello!</h4><p>Welcome to our advanced Legal Analysis System! I'm here to help you with UAE legal matters.</p><h4>How can I assist you?</h4><ul><li><strong>Legal Analysis:</strong> Analyze legal documents and PDF files</li><li><strong>Legal Queries:</strong> Answer questions about laws and regulations</li><li><strong>Penalties & Fines:</strong> Explain applicable penalties and fines</li></ul>"


def safe_execute_pandas_code(code: str, df: pd.DataFrame):
    """
    Safely execute sandboxed Python code for data analysis.
    The code should assign the final output to a variable named 'result'.
    """
    # Normalize and sanitize code to avoid indentation and fence issues
    def normalize_code(text: str) -> str:
        # Normalize unicode spaces and invisible chars
        text = text.replace('\u00A0', ' ').replace('\u200b', '').replace('\ufeff', '')
        # If code came as escaped string (e.g., with literal \n), unescape it
        if '\\n' in text and text.count('\n') <= 1:
            try:
                text = bytes(text, 'utf-8').decode('unicode_escape')
            except Exception:
                pass
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Strip code fences if present
        text = text.strip()
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = text.replace('```', '')
        # Expand tabs, dedent, remove leading blank lines
        text = textwrap.dedent(text).expandtabs(4).lstrip('\n')
        # Trim trailing whitespace on each line
        text = '\n'.join([ln.rstrip() for ln in text.split('\n')])
        return text

    code_clean = normalize_code(code)

    # Determine if Plotly is needed
    plotly_keywords = ['px.', 'go.', 'ff.', 'plotly']
    is_plotly_needed = any(keyword in code_clean.lower() for keyword in plotly_keywords)
    px = go_local = ff = None
    np_local = np

    if is_plotly_needed:
        try:
            import plotly.express as px  # type: ignore
            import plotly.graph_objects as go_local  # type: ignore
            import plotly.figure_factory as ff  # type: ignore
            import numpy as np_local  # type: ignore
        except ImportError:
            raise ValueError(
                "Plotly and/or NumPy are required for chart generation but are not installed. "
                "Please install them using: pip install plotly numpy"
            )

    def secure_importer(name, globals=None, locals=None, fromlist=(), level=0):
        allowed_modules = {
            'pandas', 'pd', 'numpy', 'np', 'math', 'random', 'datetime',
            'json', 're', 'collections', 'plotly', 'plotly.express',
            'plotly.graph_objects', 'plotly.figure_factory'
        }
        if name not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is disallowed for security reasons.")
        return __import__(name, globals, locals, fromlist, level)

    safe_globals = {
        'pd': pd,
        'df': df,
        'px': locals().get('px'),
        'go': go_local,
        'ff': ff,
        'np': np_local,
        '__builtins__': {
            '__import__': secure_importer,
            'abs': abs, 'dict': dict, 'enumerate': enumerate, 'float': float,
            'int': int, 'len': len, 'list': list, 'max': max, 'min': min,
            'range': range, 'round': round, 'set': set, 'sorted': sorted,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
        }
    }

    forbidden_patterns = [
        'import os', 'import sys', 'subprocess', 'shutil', 'open', 'eval', 'exec',
        'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr'
    ]
    if any(pattern in code_clean for pattern in forbidden_patterns):
        raise ValueError("Execution failed: Use of a forbidden keyword or function was detected.")

    try:
        try:
            ast.parse(code_clean)
        except (IndentationError, TabError, SyntaxError) as e:
            # Second attempt: aggressively left-strip common indentation
            lines = code_clean.split('\n')
            non_empty = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith('#')]
            if non_empty:
                def leading_spaces(s: str) -> int:
                    return len(s) - len(s.lstrip(' '))
                min_indent = min(leading_spaces(ln) for ln in non_empty)
                if min_indent > 0:
                    lines = [ln[min_indent:] if ln.startswith(' ' * min_indent) else ln for ln in lines]
                    code_clean = '\n'.join(lines)
            # Try parse again
            ast.parse(code_clean)

        local_vars: Dict[str, object] = {}
        exec(code_clean, safe_globals, local_vars)
        if 'result' in local_vars:
            return local_vars['result']

        # Try to evaluate the last line if it's an expression
        lines = code_clean.strip().split('\n')
        last_line = lines[-1].strip() if lines else ''
        if last_line and not last_line.startswith('#'):
            try:
                return eval(last_line, safe_globals, local_vars)
            except Exception:
                return "Code executed successfully, but no output was returned."

        return "Code executed successfully, but no output was returned."
    except Exception as e:
        raise ValueError(f"Code execution failed with error: {e}")


def convert_plotly_arrays(obj):
    """Recursively convert numpy/pandas arrays in Plotly figure dicts to plain lists."""
    # Handle datetime-like scalars first
    try:
        import numpy as _np  # local alias to avoid shadowing
        if obj is None:
            return None
        # pandas/py datetime
        if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
            return obj.isoformat()
        # numpy datetime64
        if isinstance(obj, _np.datetime64):
            try:
                return pd.Timestamp(obj).isoformat()
            except Exception:
                return str(obj)
        # pandas NaT
        if (isinstance(obj, type(pd.NaT)) or (isinstance(obj, float) and _np.isnan(obj))):
            return None
    except Exception:
        pass

    if isinstance(obj, dict):
        if set(obj.keys()) == {'dtype', 'bdata'}:
            return []
        return {k: convert_plotly_arrays(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_plotly_arrays(v) for v in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    else:
        return obj


def _make_json_safe(obj):
    """Recursively convert common non-JSON-serializable objects to safe JSON types."""
    try:
        import numpy as _np
    except Exception:
        _np = None  # type: ignore

    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        # Normalize NaN/Inf
        if isinstance(obj, float):
            if _np is not None and (_np.isnan(obj) or _np.isinf(obj)):
                return None
        return obj

    # Datetime-like
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if _np is not None and isinstance(obj, _np.datetime64):
        try:
            return pd.Timestamp(obj).isoformat()
        except Exception:
            return str(obj)

    # pandas NaT
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # Numpy scalars / arrays
    if _np is not None and isinstance(obj, _np.generic):
        try:
            return _make_json_safe(obj.item())
        except Exception:
            return str(obj)
    if _np is not None and hasattr(obj, 'tolist'):
        try:
            return _make_json_safe(obj.tolist())
        except Exception:
            pass

    # pandas containers
    if isinstance(obj, pd.DataFrame):
        records = obj.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient='records')
        return _make_json_safe(records)
    if isinstance(obj, pd.Series):
        series = obj.replace({np.nan: None, np.inf: None, -np.inf: None})
        try:
            return _make_json_safe(series.to_list())
        except Exception:
            return _make_json_safe(series.to_dict())

    # Collections
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    # Fallback to string
    return str(obj)


def format_text_response(text: str) -> str:
    """
    Ensure text responses have proper HTML formatting without gaps.
    Adds HTML tags if not already present and removes unnecessary spacing.
    """
    if not text:
        return text
        
    # If already contains HTML tags, clean up any gaps and return
    if '<h4>' in text or '<p>' in text:
        # Remove extra newlines between HTML tags to avoid gaps
        text = re.sub(r'>\s*\n\s*<', '><', text)
        # Remove any standalone newlines that might create gaps
        text = re.sub(r'\n+', '', text)
        return text
        
    # Simple text formatting - add basic HTML structure without gaps
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # Skip empty lines to avoid gaps
            continue
            
        # If line looks like a title/heading (short and ends with :)
        if len(line) < 80 and (line.endswith(':') or line.isupper()):
            formatted_lines.append(f'<h4>{line}</h4>')
        else:
            formatted_lines.append(f'<p>{line}</p>')
    
    # Join without extra newlines to avoid gaps
    return ''.join(formatted_lines)


def format_result_for_response(result):
    """Format execution result into response-friendly structure."""
    # Plotly figure
    if PLOTLY_AVAILABLE and go is not None and hasattr(go, 'Figure') and isinstance(result, go.Figure):  # type: ignore
        try:
            if hasattr(result, 'to_json'):
                fig_dict = json.loads(result.to_json())
            else:
                fig_dict = result.to_dict()
            fig_dict = convert_plotly_arrays(fig_dict)
            fig_dict = _make_json_safe(fig_dict)

            # Validate and sanitize traces to avoid empty arrays causing blank charts
            data_traces = fig_dict.get('data', []) or []
            valid_traces = []
            def is_empty_like(val):
                try:
                    if val is None:
                        return True
                    if isinstance(val, float) and np.isnan(val):
                        return True
                    if isinstance(val, str) and val.strip().lower() in {"", "nan", "null", "none", "undefined"}:
                        return True
                except Exception:
                    pass
                return False

            for trace in data_traces:
                # Keys to check for non-empty numerical data
                x_values = trace.get('x')
                y_values = trace.get('y')
                values_values = trace.get('values')
                z_values = trace.get('z')

                # Clean paired x/y lists by removing empty-like pairs
                if isinstance(x_values, list) and isinstance(y_values, list) and len(x_values) == len(y_values):
                    filtered_pairs = [
                        (x, y) for x, y in zip(x_values, y_values)
                        if not (is_empty_like(x) or is_empty_like(y))
                    ]
                    if filtered_pairs:
                        xs, ys = zip(*filtered_pairs)
                        trace['x'] = list(xs)
                        trace['y'] = list(ys)
                    else:
                        trace['x'] = []
                        trace['y'] = []

                # Recompute presence flags after cleaning
                y_values = trace.get('y')
                values_values = trace.get('values')
                z_values = trace.get('z')

                has_y = isinstance(y_values, list) and len(y_values) > 0
                has_values = isinstance(values_values, list) and len(values_values) > 0
                has_z = isinstance(z_values, list) and len(z_values) > 0

                if has_y or has_values or has_z:
                    # Ensure x and y have same length when both present
                    x_values = trace.get('x')
                    if has_y and isinstance(x_values, list):
                        if len(x_values) == len(y_values) and len(y_values) > 0:
                            valid_traces.append(trace)
                        else:
                            continue
                    else:
                        valid_traces.append(trace)

            if not valid_traces:
                print("Chart validation failed: no valid traces found")
                return {"type": "text", "payload": format_text_response("<h4>Chart Generation Issue</h4><p>Chart could not be generated due to insufficient aggregated data. Please try rephrasing your request or ensure your data contains the requested information.</p>")}

            fig_dict['data'] = valid_traces
            print(f"Chart validation successful: {len(valid_traces)} valid traces")
            return {"type": "plotly", "payload": fig_dict}
        except Exception as e:
            # Fall back to text if conversion/validation fails
            print(f"Chart conversion failed: {str(e)}")
            return {"type": "text", "payload": format_text_response("<h4>Chart Generation Error</h4><p>Chart generation failed unexpectedly. Please try a different view or metric, or rephrase your question.</p>")}

    # pandas DataFrame
    if isinstance(result, pd.DataFrame):
        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        if len(result) > 1000:
            result = result.head(1000)
        if result.index.name or all(isinstance(col, str) for col in result.columns):
            result_with_index = result.reset_index()
            payload = result_with_index.to_dict(orient='records')
        else:
            payload = result.to_dict(orient='records')
        return {"type": "table", "payload": _make_json_safe(payload)}

    # pandas Series
    if isinstance(result, pd.Series):
        result = result.replace({np.nan: None, np.inf: None, -np.inf: None})
        if len(result) > 1000:
            result = result.head(1000)
        df_result = result.reset_index()
        if hasattr(result, 'name') and result.name:
            df_result.columns = [result.index.name or 'index', result.name]
        else:
            df_result.columns = [result.index.name or 'index', 'value']
        payload = df_result.to_dict(orient='records')
        return {"type": "table", "payload": _make_json_safe(payload)}

    # dict (try convert to DataFrame)
    if isinstance(result, dict):
        try:
            df_result = pd.DataFrame(result).reset_index()
            payload = df_result.to_dict(orient='records')
            return {"type": "table", "payload": _make_json_safe(payload)}
        except Exception:
            formatted_text = format_text_response(str(result))
            return {"type": "text", "payload": formatted_text}

    # list
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            return {"type": "table", "payload": _make_json_safe(result)}
        else:
            formatted_text = format_text_response(str(result))
            return {"type": "text", "payload": formatted_text}

    # numpy arrays
    if hasattr(result, 'tolist'):
        try:
            list_result = result.tolist()
            formatted_text = format_text_response(str(list_result))
            return {"type": "text", "payload": formatted_text}
        except Exception:
            formatted_text = format_text_response(str(result))
            return {"type": "text", "payload": formatted_text}

    # scalar/string
    formatted_text = format_text_response(str(result))
    return {"type": "text", "payload": formatted_text}


def classify_query_complexity(query: str, col_names: List[str]) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """Classify if a query can be handled directly or needs LLM code generation."""
    query_lower = query.lower()
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

    for pattern, intent in simple_patterns:
        match = re.search(pattern, query_lower)
        if match:
            col_candidate = match.group(1).strip()
            matches = get_close_matches(col_candidate, col_names, n=1, cutoff=0.7)
            if matches:
                return True, (intent, matches[0])

    return False, None


def handle_simple_query(df: pd.DataFrame, intent: str, column: str) -> Dict[str, object]:
    try:
        if intent == "nunique":
            result = int(df[column].nunique())
            payload = format_text_response(f"<h4>Unique Values Analysis</h4><p>There are <strong>{result}</strong> unique values in column '{column}'.</p>")
            return {"type": "text", "payload": payload}
        elif intent == "unique" or intent == "list":
            result = df[column].dropna().unique().tolist()
            result = [str(v) if pd.notna(v) else None for v in result]
            return {"type": "table", "payload": [{column: v} for v in result]}
        elif intent == "sum":
            if df[column].dtype in ['int64', 'float64']:
                result = float(df[column].sum())
                payload = format_text_response(f"<h4>Sum Calculation</h4><p>The sum of column '{column}' is <strong>{result:,.2f}</strong>.</p>")
                return {"type": "text", "payload": payload}
            else:
                payload = format_text_response(f"<h4>Sum Calculation Error</h4><p>Cannot calculate sum for non-numeric column '{column}'. Please ensure the column contains numeric values.</p>")
                return {"type": "text", "payload": payload}
        elif intent == "mean":
            if df[column].dtype in ['int64', 'float64']:
                result = float(df[column].mean())
                payload = format_text_response(f"<h4>Average Calculation</h4><p>The average of column '{column}' is <strong>{result:.2f}</strong>.</p>")
                return {"type": "text", "payload": payload}
            else:
                payload = format_text_response(f"<h4>Average Calculation Error</h4><p>Cannot calculate mean for non-numeric column '{column}'. Please ensure the column contains numeric values.</p>")
                return {"type": "text", "payload": payload}
        elif intent == "min":
            result = df[column].min()
            payload = format_text_response(f"<h4>Minimum Value</h4><p>The minimum value in column '{column}' is <strong>{result}</strong>.</p>")
            return {"type": "text", "payload": payload}
        elif intent == "max":
            result = df[column].max()
            payload = format_text_response(f"<h4>Maximum Value</h4><p>The maximum value in column '{column}' is <strong>{result}</strong>.</p>")
            return {"type": "text", "payload": payload}
        elif intent == "count":
            result = int(df[column].count())
            payload = format_text_response(f"<h4>Record Count</h4><p>There are <strong>{result}</strong> non-null records in column '{column}'.</p>")
            return {"type": "text", "payload": payload}
        elif intent == "groupby":
            group_counts = df.groupby(column).size().reset_index(name='count')
            payload = group_counts.to_dict(orient='records')
            return {"type": "table", "payload": payload}
        else:
            payload = format_text_response(f"<h4>Unsupported Operation</h4><p>Intent '{intent}' is not supported for direct handling. Please try rephrasing your question.</p>")
            return {"type": "text", "payload": payload}
    except Exception as e:
        payload = format_text_response(f"<h4>Query Processing Error</h4><p>Error processing query: {str(e)}</p><p>Please try rephrasing your question with more specific details.</p>")
        return {"type": "text", "payload": payload}


def get_llm_analysis_explore(query: str, df: pd.DataFrame, mode: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, object]:
    """
    Use LLM to analyze user's query and generate code to produce graph/table/text.
    Returns a JSON with keys: type and payload. For data analysis, payload has explanation and code.
    """
    num_rows, num_cols = df.shape
    col_names = list(df.columns)
    sample_data = df.head(3).to_dict(orient='records')
    
    # Detect if it's a KPI-related query
    kpi_keywords = ['kpi', 'kpis', 'key performance indicator', 'key performance indicators', 'metrics', 'performance metrics', 'dashboard metrics']
    is_kpi_query = any(keyword in query.lower() for keyword in kpi_keywords)
    
    dataset_info = f"""
Dataset Information:
- Shape: {num_rows} rows, {num_cols} columns
- Columns: {col_names}
- Sample Data (first 3 rows): {sample_data}
"""

    # Detect language and get appropriate context
    language, language_instructions = get_language_context(query)
    
    # Create language-specific instructions
    if language == "arabic":
        instructions_common = f"""
You are a highly intelligent and professional AI assistant specializing in comprehensive data analysis and document processing. Your responses should be structured, professional, and comprehensive like a senior analyst.

{language_instructions}

PROFESSIONAL RESPONSE FORMATTING REQUIREMENTS FOR ARABIC:
- Always use proper Arabic RTL HTML formatting with dir="rtl" lang="ar"
- Use <div dir="rtl" lang="ar"> wrapper for all content
- Use <h3> for main titles, <h4> for subtitles
- Use <p> for paragraphs, <ul> and <li> for lists
- Use <strong> for emphasis, <em> for italics
- Use <hr> for section separators
- Structure responses with clear sections and subsections

RESPONSE STRUCTURE FOR LEGAL ANALYSIS:
When analyzing legal documents (PDFs, legal documents, contracts, etc.), provide responses in this EXACT format:

For Arabic queries:
```html
<div dir="rtl" lang="ar">
<h3>نظام التحليل القانوني</h3>
<h4>الإجابة المباشرة</h4>
<p>بناءً على الوثائق القانونية لدولة الإمارات العربية المتحدة، يقدم التحليل الشامل التالي معالجة للمسؤوليات والالتزامات القانونية للمحامين في دولة الإمارات العربية المتحدة، مع التركيز بشكل خاص على تمثيل العملاء، تضارب المصالح، والسلوك المهني.</p>
<p>[نظرة شاملة على محتوى الوثيقة القانونية والنتائج الرئيسية]</p>

<h4>المواضيع القانونية الرئيسية</h4>
<ul>
<li><strong>[الموضوع القانوني الأول]:</strong> [شرح مفصل مع مراجع محددة]</li>
<li><strong>[الموضوع القانوني الثاني]:</strong> [شرح مفصل مع مراجع محددة]</li>
<li><strong>[الموضوع القانوني الثالث]:</strong> [شرح مفصل مع مراجع محددة]</li>
</ul>

<h4>الأحكام القانونية/التنظيمية</h4>
<ul>
<li><strong>[اسم الحكم القانوني]:</strong> [تفاصيل محددة وآثار قانونية]</li>
<li><strong>[حكم قانوني آخر]:</strong> [تفاصيل محددة وآثار قانونية]</li>
</ul>

<h4>الإرشادات القانونية العملية</h4>
<ul>
<li><strong>[مجال الإرشاد القانوني الأول]:</strong> [توصيات قانونية قابلة للتنفيذ]</li>
<li><strong>[مجال الإرشاد القانوني الثاني]:</strong> [توصيات قانونية قابلة للتنفيذ]</li>
</ul>

<h4>الملخص القانوني</h4>
<p>[ملخص شامل مع النقاط القانونية الرئيسية والخطوات التالية]</p>

<hr>
<p><em>✅ تم إكمال التحليل القانوني المهني</em></p>
<p><em>هذا التحليل مبني على محتوى الوثيقة القانونية المقدمة وهو للإرشاد فقط. للمسائل القانونية المحددة، يرجى استشارة المحامين المؤهلين.</em></p>

<h4>المراجع القانونية</h4>
<p><strong>المصدر:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
<p><strong>مرجع إضافي:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
</div>
```

For English queries:
```html
<h3>LEGAL ANALYSIS SYSTEM</h3>
<h4>Direct Answer</h4>
<p>Based on the UAE legal documentation, the following comprehensive analysis addresses the legal responsibilities and obligations of lawyers in the UAE, particularly focusing on client representation, conflict of interest, and professional conduct.</p>
<p>[Comprehensive overview of the legal document content and key findings]</p>

<h4>Key Legal Topics Covered</h4>
<ul>
<li><strong>[Legal Topic 1]:</strong> [Detailed explanation with specific references]</li>
<li><strong>[Legal Topic 2]:</strong> [Detailed explanation with specific references]</li>
<li><strong>[Legal Topic 3]:</strong> [Detailed explanation with specific references]</li>
</ul>

<h4>Legal/Regulatory Provisions</h4>
<ul>
<li><strong>[Legal Provision Name]:</strong> [Specific details and legal implications]</li>
<li><strong>[Another Legal Provision]:</strong> [Specific details and legal implications]</li>
</ul>

<h4>Practical Legal Guidance</h4>
<ul>
<li><strong>[Legal Guidance Area 1]:</strong> [Actionable legal recommendations]</li>
<li><strong>[Legal Guidance Area 2]:</strong> [Actionable legal recommendations]</li>
</ul>

<h4>Legal Summary</h4>
<p>[Comprehensive summary with key legal takeaways and next steps]</p>

<hr>
<p><em>✅ PROFESSIONAL LEGAL ANALYSIS COMPLETED</em></p>
<p><em>This legal analysis is based on the provided legal document content and is for guidance only. For specific legal matters, please consult with qualified legal professionals.</em></p>

<h4>Legal References</h4>
<p><strong>SOURCE:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
<p><strong>Additional Reference:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
```

RESPONSE STRUCTURE FOR DATA ANALYSIS:
For data analysis queries, provide responses in this format:

For Arabic queries:
```html
<div dir="rtl" lang="ar">
<h3>نظام تحليل البيانات</h3>
<h4>نظرة عامة على التحليل</h4>
<p>بناءً على الوثائق القانونية لدولة الإمارات العربية المتحدة، يقدم التحليل الشامل التالي معالجة للمسؤوليات والالتزامات القانونية للمحامين في دولة الإمارات العربية المتحدة، مع التركيز بشكل خاص على تمثيل العملاء، تضارب المصالح، والسلوك المهني.</p>
<p>[نظرة شاملة على التحليل المنجز]</p>

<h4>النتائج الرئيسية</h4>
<ul>
<li><strong>[النتيجة الأولى]:</strong> [شرح مفصل مع رؤى البيانات]</li>
<li><strong>[النتيجة الثانية]:</strong> [شرح مفصل مع رؤى البيانات]</li>
</ul>

<h4>الملخص الإحصائي</h4>
<ul>
<li><strong>حجم مجموعة البيانات:</strong> [عدد السجلات والأبعاد]</li>
<li><strong>المقاييس الرئيسية:</strong> [القياسات الإحصائية المهمة]</li>
<li><strong>جودة البيانات:</strong> [تقييم اكتمال ودقة البيانات]</li>
</ul>

<h4>التوصيات</h4>
<ul>
<li><strong>[التوصية الأولى]:</strong> [رؤى قابلة للتنفيذ]</li>
<li><strong>[التوصية الثانية]:</strong> [رؤى قابلة للتنفيذ]</li>
</ul>

<hr>
<p><em>✅ تم إكمال التحليل المهني</em></p>
<p><em>هذا التحليل مبني على البيانات المقدمة وهو للإرشاد فقط. للمسائل المحددة، يرجى استشارة المهنيين المؤهلين.</em></p>

<h4>المراجع القانونية</h4>
<p><strong>المصدر:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
<p><strong>مرجع إضافي:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
</div>
```

For English queries:
```html
<h3>DATA ANALYSIS SYSTEM</h3>
<h4>Analysis Overview</h4>
<p>Based on the UAE legal documentation, the following comprehensive analysis addresses the legal responsibilities and obligations of lawyers in the UAE, particularly focusing on client representation, conflict of interest, and professional conduct.</p>
<p>[Comprehensive overview of the analysis performed]</p>

<h4>Key Findings</h4>
<ul>
<li><strong>[Finding 1]:</strong> [Detailed explanation with data insights]</li>
<li><strong>[Finding 2]:</strong> [Detailed explanation with data insights]</li>
</ul>

<h4>Statistical Summary</h4>
<ul>
<li><strong>Dataset Size:</strong> [Number of records and dimensions]</li>
<li><strong>Key Metrics:</strong> [Important statistical measures]</li>
<li><strong>Data Quality:</strong> [Assessment of data completeness and accuracy]</li>
</ul>

<h4>Recommendations</h4>
<ul>
<li><strong>[Recommendation 1]:</strong> [Actionable insights]</li>
<li><strong>[Recommendation 2]:</strong> [Actionable insights]</li>
</ul>

<hr>
<p><em>✅ PROFESSIONAL ANALYSIS COMPLETED</em></p>
<p><em>This analysis is based on the provided data and is for guidance only. For specific matters, please consult with qualified professionals.</em></p>

<h4>Legal References</h4>
<p><strong>SOURCE:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
<p><strong>Additional Reference:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
```

Response Types:
1. If the response is conversational only: use type="conversational_answer" and payload as an HTML-formatted string
2. If the response requires data processing: use type="data_analysis_answer" and payload as an object with keys "explanation" and "code"

For conversational responses about KPIs or general questions:
- Be comprehensive and informative like ChatGPT
- Provide context and insights
- Use proper HTML formatting (h4, p tags, \n)
- Be helpful and engaging

For any code you generate:
- Use pandas only for data ops; for charts, use Plotly (px/go/ff) and set the final object to a variable named result.
- Always compute/aggregate the data needed before plotting. Avoid empty arrays. Use .dropna()/.fillna(0) appropriately.
- For filtering with boolean masks, ensure masks contain only True/False (never NaN).
- Always set a meaningful chart title and axis labels when plotting.
- Never read/write external files. Operate on the provided DataFrame df only.
- The final line must produce a variable named result.
"""
    else:
        instructions_common = f"""
You are a highly intelligent and professional AI assistant specializing in comprehensive data analysis and document processing. Your responses should be structured, professional, and comprehensive like a senior analyst.

{language_instructions}

PROFESSIONAL RESPONSE FORMATTING REQUIREMENTS FOR ENGLISH:
- Always use proper HTML formatting
- Use <h3> for main titles, <h4> for subtitles
- Use <p> for paragraphs, <ul> and <li> for lists
- Use <strong> for emphasis, <em> for italics
- Use <hr> for section separators
- Structure responses with clear sections and subsections

RESPONSE STRUCTURE FOR LEGAL ANALYSIS:
When analyzing legal documents (PDFs, legal documents, contracts, etc.), provide responses in this EXACT format:

```html
<h3>LEGAL ANALYSIS SYSTEM</h3>
<h4>Direct Answer</h4>
<p>Based on the UAE legal documentation, the following comprehensive analysis addresses the legal responsibilities and obligations of lawyers in the UAE, particularly focusing on client representation, conflict of interest, and professional conduct.</p>
<p>[Comprehensive overview of the legal document content and key findings]</p>

<h4>Key Legal Topics Covered</h4>
<ul>
<li><strong>[Legal Topic 1]:</strong> [Detailed explanation with specific references]</li>
<li><strong>[Legal Topic 2]:</strong> [Detailed explanation with specific references]</li>
<li><strong>[Legal Topic 3]:</strong> [Detailed explanation with specific references]</li>
</ul>

<h4>Legal/Regulatory Provisions</h4>
<ul>
<li><strong>[Legal Provision Name]:</strong> [Specific details and legal implications]</li>
<li><strong>[Another Legal Provision]:</strong> [Specific details and legal implications]</li>
</ul>

<h4>Practical Legal Guidance</h4>
<ul>
<li><strong>[Legal Guidance Area 1]:</strong> [Actionable legal recommendations]</li>
<li><strong>[Legal Guidance Area 2]:</strong> [Actionable legal recommendations]</li>
</ul>

<h4>Legal Summary</h4>
<p>[Comprehensive summary with key legal takeaways and next steps]</p>

<hr>
<p><em>✅ PROFESSIONAL LEGAL ANALYSIS COMPLETED</em></p>
<p><em>This legal analysis is based on the provided legal document content and is for guidance only. For specific legal matters, please consult with qualified legal professionals.</em></p>

<h4>Legal References</h4>
<p><strong>SOURCE:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
<p><strong>Additional Reference:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
```

RESPONSE STRUCTURE FOR DATA ANALYSIS:
For data analysis queries, provide responses in this format:

```html
<h3>DATA ANALYSIS SYSTEM</h3>
<h4>Analysis Overview</h4>
<p>Based on the UAE legal documentation, the following comprehensive analysis addresses the legal responsibilities and obligations of lawyers in the UAE, particularly focusing on client representation, conflict of interest, and professional conduct.</p>
<p>[Comprehensive overview of the analysis performed]</p>

<h4>Key Findings</h4>
<ul>
<li><strong>[Finding 1]:</strong> [Detailed explanation with data insights]</li>
<li><strong>[Finding 2]:</strong> [Detailed explanation with data insights]</li>
</ul>

<h4>Statistical Summary</h4>
<ul>
<li><strong>Dataset Size:</strong> [Number of records and dimensions]</li>
<li><strong>Key Metrics:</strong> [Important statistical measures]</li>
<li><strong>Data Quality:</strong> [Assessment of data completeness and accuracy]</li>
</ul>

<h4>Recommendations</h4>
<ul>
<li><strong>[Recommendation 1]:</strong> [Actionable insights]</li>
<li><strong>[Recommendation 2]:</strong> [Actionable insights]</li>
</ul>

<hr>
<p><em>✅ PROFESSIONAL ANALYSIS COMPLETED</em></p>
<p><em>This analysis is based on the provided data and is for guidance only. For specific matters, please consult with qualified professionals.</em></p>

<h4>Legal References</h4>
<p><strong>SOURCE:</strong> <a href="https://uaelegislation.gov.ae" target="_blank" style="color: #3498db;">https://uaelegislation.gov.ae</a></p>
<p><strong>Additional Reference:</strong> <a href="https://www.moj.gov.ae" target="_blank" style="color: #3498db;">https://www.moj.gov.ae</a></p>
```

Response Types:
1. If the response is conversational only: use type="conversational_answer" and payload as an HTML-formatted string
2. If the response requires data processing: use type="data_analysis_answer" and payload as an object with keys "explanation" and "code"

For conversational responses about KPIs or general questions:
- Be comprehensive and informative like ChatGPT
- Provide context and insights
- Use proper HTML formatting (h4, p tags, \n)
- Be helpful and engaging

For any code you generate:
- Use pandas only for data ops; for charts, use Plotly (px/go/ff) and set the final object to a variable named result.
- Always compute/aggregate the data needed before plotting. Avoid empty arrays. Use .dropna()/.fillna(0) appropriately.
- For filtering with boolean masks, ensure masks contain only True/False (never NaN).
- Always set a meaningful chart title and axis labels when plotting.
- Never read/write external files. Operate on the provided DataFrame df only.
- The final line must produce a variable named result.
"""

    mode_instructions = ""
    if mode == "graph":
        mode_instructions = (
            "Generate high-quality Plotly code that produces the best visualization for the user's question. "
            "CRITICAL CHART GENERATION STEPS: "
            "1) First, compute the required aggregation/grouping in pandas (e.g., value_counts(), groupby(), pivot_table()). "
            "2) Before creating any figure, ensure that axis arrays (x, y, values, etc.) are non-empty and properly aligned. "
            "3) Clean data thoroughly: replace({'': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan}).dropna() on relevant columns. "
            "4) For string columns with numeric values and units (e.g., '28.3kmph', '1959 psi'), extract numeric parts using regex: df[col].astype(str).str.extract(r'([-+]?\\d*\\.?\\d+)')[0].astype(float). "
            "5) Always set meaningful chart titles, axis labels, and legends. Use appropriate chart types (bar, line, pie, scatter, histogram, box, etc.). "
            "6) Validate data before plotting: check len(data) > 0 and data contains valid values. "
            "7) If chart generation fails or data is insufficient, return a meaningful pandas DataFrame with aggregated results as fallback (never return None). "
            "8) Use proper color schemes and ensure charts are visually appealing and informative. "
            "9) For time series data, ensure proper date parsing and chronological ordering. "
            "10) Avoid geospatial columns unless explicitly requested for maps."
        )
    elif mode == "table":
        mode_instructions = (
            "Generate pandas code that returns a tabular result in a DataFrame assigned to result. "
            "Do not generate charts. Include necessary groupby/sort/limit operations based on the question."
        )
    else:  # text
        if is_kpi_query:
            mode_instructions = (
                "This is a KPI (Key Performance Indicator) related query. Provide a comprehensive, conversational response about KPIs like ChatGPT would. "
                "Format your response with proper HTML tags: <h4> for headings, <p> for paragraphs, and \n for line breaks. "
                "Be informative, helpful, and explain what KPIs are, their importance, and how they can be used with this dataset. "
                "Include insights about the available data columns and what KPIs could be derived from them. "
                "Return type='conversational_answer' with an HTML-formatted payload."
            )
        else:
            mode_instructions = (
                "For conversational responses, use proper HTML formatting and be engaging and informative."
            )

    # Enhanced prompt for better conversational responses
    conversation_context = ""
    if is_kpi_query:
        conversation_context = f"""
KPI Context:
- You have access to a dataset with {num_cols} columns: {col_names}
- This data can be used to calculate various KPIs and performance metrics
- Be specific about what KPIs can be derived from the available columns
- Explain the business value and importance of KPIs
"""
    
    # Add chat history context for follow-up questions
    history_context = ""
    if chat_history:
        last_query = chat_history[0]['content'] if chat_history else ""
        history_context = f"""
**Previous Query (for context):**
---
{last_query}
---

**CRITICAL CONTEXT ANALYSIS:**
- If the current user query contains ambiguous references like "how many are", "how many of them", "which ones", "those", etc., you MUST use the previous query context to understand what they're referring to
- When the user says "them", "those", "these", they are referring to the subject/filter from the previous query shown above
- Example: If previous query was "show me high priority tickets" and current query is "how many of them are assigned to John", you should count high priority tickets that are assigned to John
- ALWAYS maintain the same filters/conditions from the previous query when processing follow-up questions
- Pay special attention to names, categories, priorities, or any filters mentioned in the previous query
- If the current query seems like a follow-up, explicitly combine it with the context from the previous query above
"""
    
    llm_prompt = f"""
{instructions_common}

{history_context}

Dataset Context:
{dataset_info}
{conversation_context}

User Question: {query}

Mode: {mode}
Specific Instructions: {mode_instructions}

- For conversational responses: Be comprehensive, helpful, and engaging like ChatGPT
- For data analysis explanations: Write as a concise, business-oriented summary focusing on trend, scale, change, and implications
- Avoid technical implementation details in explanations

IMPORTANT: Return ONLY the JSON object as described above. No markdown fences.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert data analyst AI assistant, similar to ChatGPT. You provide helpful, conversational responses with proper HTML formatting. For text responses, always use <h4> tags for headings and <p> tags for paragraphs. You are also skilled at Python/Plotly code generation for data analysis."},
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
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: conversational text
        return {"type": "conversational_answer", "payload": content}


def generate_rescue_chart(df: pd.DataFrame, query: str):
    """
    Enhanced server-side robust fallback chart when LLM-generated code returns invalid chart data.
    Strategy:
      - Detect likely date column and a primary numeric metric column.
      - Aggregate by day/month depending on granularity and build a clean line chart.
      - Try multiple chart types if the first approach fails.
      - Produce a business-oriented explanation.
    Returns: (plotly.graph_objects.Figure or None, explanation str)
    """
    print(f"Rescue chart generation started for query: {query[:100]}...")
    try:
        ql = query.lower()

        # Utility: score columns by presence and fuzzy match
        def score_column(col: str) -> int:
            name = col.lower()
            score = 0
            # token overlap
            for token in re.findall(r"[a-zA-Z0-9%]+", ql):
                if token and token in name:
                    score += 2
            # special boosts
            boosts = ['achievement', 'target', 'actual', 'value', 'amount', 'sales', 'revenue', 'emission', 'co2', 'count', 'volume', 'cost']
            for b in boosts:
                if b in name and b in ql:
                    score += 3
            return score

        # Candidate date column preference
        date_candidates = []
        for col in df.columns:
            lc = col.lower()
            if 'date' in lc or 'time' in lc or 'period' in lc:
                date_candidates.append((col, score_column(col)+3))
            else:
                # try if parsable
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_candidates.append((col, score_column(col)+1))
                except Exception:
                    pass
        date_col = max(date_candidates, key=lambda x: x[1])[0] if date_candidates else None

        # Candidate metric column (numeric or convertible from strings with units). Avoid geospatial unless explicitly asked
        geospatial_aliases = {"latitude", "longitude", "lat", "lon", "lng"}
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in geospatial_aliases
        ]

        # If no numeric columns, try to synthesize numeric candidates by extracting numbers from strings with units
        synthetic_numeric_map = {}
        if not numeric_cols:
            for c in df.columns:
                if c.lower() in geospatial_aliases:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    continue
                try:
                    ser = df[c].astype(str).str.strip()
                    # Replace empty-like tokens
                    ser = ser.replace({'': np.nan, 'nan': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan})
                    extracted = ser.str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0]
                    numeric_ser = pd.to_numeric(extracted, errors='coerce')
                    # Consider as candidate if we have a reasonable amount of numeric values
                    if numeric_ser.notna().sum() >= max(10, int(0.1 * len(df))):
                        synthetic_numeric_map[c] = numeric_ser
                except Exception:
                    continue
            if synthetic_numeric_map:
                # Prefer columns with higher query score and more numeric entries
                scored = sorted(
                    synthetic_numeric_map.items(),
                    key=lambda kv: (score_column(kv[0]), kv[1].notna().sum()),
                    reverse=True
                )
                best_name, best_series = scored[0]
                numeric_cols = [best_name]
                # Attach synthesized numeric series for use later
                df = df.copy()
                df[best_name] = best_series
        if numeric_cols:
            scored = sorted(numeric_cols, key=lambda c: score_column(c), reverse=True)
            metric_col = scored[0]
        else:
            metric_col = None

        # If the query explicitly names a column (fuzzy), respect it
        tokens = re.findall(r"[a-zA-Z0-9%]+", ql)
        for token in tokens:
            matches = get_close_matches(token, list(df.columns), n=1, cutoff=0.85)
            if matches:
                m = matches[0]
                if m in numeric_cols:
                    metric_col = m
                else:
                    date_col = date_col or m

        # If neither clear time axis nor the query suggests a time trend, try categorical vs metric bar chart
        if date_col is None and metric_col is not None:
            # choose a categorical column with limited unique values and good query score
            categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if categorical_cols:
                cat_scored = sorted(categorical_cols, key=lambda c: (score_column(c), -df[c].nunique()))
                x_col = cat_scored[0]
                # group and take mean, top 20
                df_local = df[[x_col, metric_col]].dropna()
                if df_local.empty:
                    return None, None
                agg = df_local.groupby(x_col)[metric_col].mean().sort_values(ascending=False).head(20).reset_index()
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(x=agg[x_col].astype(str).tolist(), y=agg[metric_col].tolist(), name=metric_col))
                fig.update_layout(title=f"Top {x_col} by average {metric_col}", xaxis_title=x_col, yaxis_title=metric_col)
                biz_exp = (
                    f"Top categories by average {metric_col} highlight where performance concentrates. "
                    f"Use this ranking to focus attention on the highest-impact segments."
                )
                return fig, biz_exp
            # otherwise fail
            return None, None

        # If we have date and metric → time series aggregation
        if date_col is None or metric_col is None:
            return None, None

        df_local = df[[date_col, metric_col]].dropna()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors='coerce')
        df_local = df_local.dropna(subset=[date_col])
        if df_local.empty:
            return None, None

        daily = df_local.groupby(df_local[date_col].dt.date)[metric_col].mean().reset_index()
        daily.columns = ['Date', metric_col]

        # Downsample for long series to monthly
        if len(daily) > 200:
            monthly = df_local.groupby([df_local[date_col].dt.to_period('M')])[metric_col].mean().reset_index()
            monthly[date_col] = monthly[date_col].dt.to_timestamp()
            monthly.columns = ['Period', metric_col]
            x_vals = monthly['Period']
            y_vals = monthly[metric_col]
            x_title = 'Month'
            title = f"{metric_col} Trend by Month"
        else:
            x_vals = pd.to_datetime(daily['Date'])
            y_vals = daily[metric_col]
            x_title = 'Date'
            title = f"{metric_col} Daily Trend"

        import plotly.graph_objects as go  # safe here
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(x_vals), y=list(y_vals), mode='lines', name=metric_col))
        fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=metric_col)

        if len(y_vals) >= 2:
            start_val = float(y_vals.iloc[0])
            end_val = float(y_vals.iloc[-1])
            change = end_val - start_val
            pct = (change / start_val * 100.0) if start_val != 0 else 0.0
            direction = 'increase' if change > 0 else ('decline' if change < 0 else 'stability')
            biz_exp = (
                f"{metric_col} shows a {direction} from {start_val:.1f} to {end_val:.1f}. "
                f"This trajectory indicates a {abs(pct):.1f}% change over the observed period. "
                f"Leverage this trend to steer planning and corrective actions."
            )
        else:
            biz_exp = f"Time-series view of {metric_col} to support quick performance diagnostics."

        print(f"Rescue chart successful: {title}")
        return fig, biz_exp
    except Exception as e:
        print(f"Rescue chart failed: {str(e)}")
        return None, None


# =========================
# Graph Agent: Two-stage robust LLM pipeline
#   1) Clean/aggregate extraction (pandas only) → DataFrame result
#   2) Plot generation (Plotly) → Figure result
# Each stage retries up to 3 times with error feedback
# =========================

def _extract_code_from_llm_text(all_text: str) -> str:
    try:
        txt = all_text.strip()
        if txt.startswith("```"):
            # remove any fenced blocks
            txt = re.sub(r"^```[a-zA-Z]*\n", "", txt)
            txt = txt.replace("```", "").strip()
        return txt
    except Exception:
        return all_text


def _llm_generate_code(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
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
        "Strict requirements: 1) Never read/write files; use df only. 2) Parse numeric values embedded in strings (e.g., '28.3kmph', '1959 psi', '214°C') using regex and convert to float. "
        "3) Detect and parse likely date/time columns using pd.to_datetime with errors='coerce' when needed for time grouping. "
        "4) Remove empty-like tokens: '', 'nan', 'null', 'none', 'undefined' (case-insensitive) and drop rows with missing values in the plotted columns. "
        "5) Avoid geospatial lat/lon unless explicitly requested. 6) If filtering leads to empty result, create a sensible top-10 aggregated table related to the question. "
        "7) Return ONLY executable Python code; no markdown; final variable must be named result and must be a pandas DataFrame."
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        guidance = ""
        if last_error:
            guidance = (
                f"Previous attempt failed with error: {last_error}. "
                "Revise the code to fix the issue. Ensure 'result' is a non-empty pandas DataFrame."
            )
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
            # not a DataFrame or empty → craft error to retry
            if not isinstance(extracted, pd.DataFrame):
                last_error = f"expected DataFrame 'result', got {type(extracted).__name__}"
            else:
                last_error = "DataFrame is empty after cleaning/filtering"
        except Exception as e:
            last_error = str(e)
    # Final fallback: simple preview to avoid None
    fallback = df.head(50).copy()
    return fallback


def _retry_llm_plot_from_clean_df(query: str, df_clean: pd.DataFrame, max_retries: int = 3) -> Dict[str, object]:
    system_prompt = (
        "You are a senior data visualization engineer. Generate robust Plotly code (px/go/ff) that operates on a provided DataFrame named df "
        "and assigns the final figure to a variable named result. "
        "Requirements: 1) Perform any needed aggregation in pandas first. 2) Clean empty-like tokens ('', 'nan', 'null', 'none', 'undefined') and drop rows with missing values in the plot columns. "
        "3) If metrics are strings with units, extract numeric with regex and convert to float before plotting. 4) Ensure x/y arrays are aligned and non-empty. "
        "5) Set a descriptive title and axis labels. 6) If a valid chart is impossible, set result to a top-10 aggregated pandas DataFrame instead (not None). "
        "7) Return ONLY executable Python code; no markdown; final variable must be named result."
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        guidance = ""
        if last_error:
            guidance = (
                f"Prior error: {last_error}. "
                "Revise to ensure non-empty aligned arrays or return a top-10 aggregated DataFrame as result."
            )
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
            # accept plotly first
            if formatted.get("type") == "plotly":
                return formatted
            # If table returned, attempt to convert to a simple chart to honor graph intent
            if formatted.get("type") == "table":
                payload = formatted.get("payload") or []
                if isinstance(payload, list) and len(payload) > 0 and isinstance(payload[0], dict):
                    try:
                        tmp_df = pd.DataFrame(payload)
                        # Prefer a date/time x if available
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
                        # Else fallback to bar using first categorical and numeric columns
                        if not numeric_cols:
                            # attempt to coerce one column to numeric
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
                # if table cannot be converted to chart, keep retrying
            last_error = "Empty or invalid visualization payload"
        except Exception as e:
            last_error = str(e)

    # Final fallback via server-side rescue chart
    try:
        rescue_fig, biz_exp = generate_rescue_chart(df_clean, query)
        if rescue_fig is not None:
            formatted = format_result_for_response(rescue_fig)
            if biz_exp:
                formatted["explanation"] = biz_exp
            return formatted
    except Exception:
        pass
    # Worst case: force a minimal chart if possible
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
        # If the user explicitly asks for date vs target, try deterministic chart first
        ql = query.lower()
        if 'date' in ql and 'target' in ql:
            # Fuzzy match columns
            date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'period'])]
            target_match = get_close_matches('target', list(df.columns), n=1, cutoff=0.6)
            if date_cols and target_match:
                x_col = date_cols[0]
                y_col = target_match[0]
                try:
                    tmp = df[[x_col, y_col]].copy()
                    # Clean and coerce
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

        # Stage 1: robust extraction/cleaning
        df_clean = _retry_llm_pandas_extraction(query, df, max_retries=3)
        if df_clean is None or df_clean.empty:
            # If somehow empty, use rescue directly
            fig, exp = generate_rescue_chart(df, query)
            if fig is None:
                return None
            formatted = format_result_for_response(fig)
            if exp:
                formatted["explanation"] = exp
            return formatted

        # Stage 2: visualization from clean data
        formatted = _retry_llm_plot_from_clean_df(query, df_clean, max_retries=3)
        return formatted
    except Exception:
        return None


# Debug and utility endpoints for session management
@explore_router.get("/api/debug_session_memory/{session_id}")
async def debug_session_memory_explore(session_id: str):
    """
    Debug endpoint to see what's stored in session memory for explore API
    """
    try:
        with SESSION_MEMORY_LOCK:
            session_data = SESSION_MEMORY.get(session_id, {})
            
        return JSONResponse(content={
            "session_id": session_id,
            "last_user_message": session_data.get("last_user_message"),
            "last_bot_message": session_data.get("last_bot_message"),
            "has_previous_context": bool(session_data.get("last_user_message")),
            "total_sessions": len(SESSION_MEMORY),
            "api": "explore"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@explore_router.delete("/api/clear_session_memory/{session_id}")
async def clear_session_memory_explore(session_id: str):
    """
    Clear session memory for testing purposes - explore API
    """
    try:
        with SESSION_MEMORY_LOCK:
            if session_id in SESSION_MEMORY:
                session_data = SESSION_MEMORY[session_id]
                had_user_msg = bool(session_data.get("last_user_message"))
                had_bot_msg = bool(session_data.get("last_bot_message"))
                del SESSION_MEMORY[session_id]
                return JSONResponse(content={
                    "session_id": session_id,
                    "had_user_message": had_user_msg,
                    "had_bot_message": had_bot_msg,
                    "status": "cleared",
                    "api": "explore"
                })
            else:
                return JSONResponse(content={
                    "session_id": session_id,
                    "had_user_message": False,
                    "had_bot_message": False,
                    "status": "not_found",
                    "api": "explore"
                })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")
