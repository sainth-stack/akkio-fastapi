from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import io
import json
import re
import ast
import textwrap
from difflib import get_close_matches
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple

from openai import OpenAI
from universal_prompts import (
    prompt_for_data_analyst,
    Prompt_for_code_execution,
    Visualisation_intelligence_engine,
)

explore_router = APIRouter()

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
    Enhanced preprocessing for robust chart generation:
    - Normalize empty-like tokens to NaN
    - Parse datetime-like columns
    - Extract numeric values from strings with units
    - Handle mixed data types
    - Improve categorical data cleaning
    Returns a new dataframe copy.
    """
    df_proc = df.copy()

    # Enhanced empty token normalization
    empty_tokens = {'', 'nan', 'null', 'none', 'undefined', 'n/a', 'na', '-', '--', '?'}
    for col in df_proc.columns:
        try:
            if df_proc[col].dtype == object:
                df_proc[col] = (
                    df_proc[col]
                    .astype(str)
                    .str.strip()
                    .apply(lambda v: np.nan if str(v).strip().lower() in empty_tokens else v)
                )
                # Remove leading/trailing quotes if present
                df_proc[col] = df_proc[col].str.strip('"\'')
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

    # Enhanced datetime parsing + numeric extraction from object columns
    for col in df_proc.columns:
        try:
            # Skip if already processed as numeric/datetime
            if pd.api.types.is_numeric_dtype(df_proc[col]) or pd.api.types.is_datetime64_any_dtype(df_proc[col]):
                continue
                
            # Attempt to parse datetimes if column name suggests time or if many values parse
            name_lower = str(col).lower()
            if any(k in name_lower for k in ['time', 'date', 'timestamp', 'period', 'created', 'updated', 'modified']):
                parsed = pd.to_datetime(df_proc[col], errors='coerce')
                if parsed.notna().sum() >= max(3, int(0.1 * len(df_proc))):
                    df_proc[col] = parsed
                    continue
                    
            # Try to infer datetime from data content
            if df_proc[col].dtype == object:
                sample = df_proc[col].dropna().head(20)
                if len(sample) > 0:
                    # Try parsing as datetime
                    try:
                        parsed = pd.to_datetime(df_proc[col], errors='coerce')
                        if parsed.notna().sum() >= max(3, int(0.2 * len(df_proc))):
                            df_proc[col] = parsed
                            continue
                    except Exception:
                        pass

            # For object columns with potential units, extract numerics
            if df_proc[col].dtype == object:
                series_obj = df_proc[col].astype(str)
                # Sample-based unit prevalence detection
                sample = series_obj.head(min(200, len(series_obj)))
                unit_counts: Dict[str, int] = {}
                numeric_vals: List[Optional[float]] = []
                for v in sample:
                    num, unit = extract_number_and_unit(v)
                    numeric_vals.append(num)
                    if unit:
                        unit_counts[unit] = unit_counts.get(unit, 0) + 1

                # If majority values look numeric (with or without unit), convert entire column
                valid_numeric_count = sum(1 for x in numeric_vals if x is not None)
                threshold = max(5, int(0.2 * len(sample)))  # Lower threshold, minimum 5 values
                if valid_numeric_count >= threshold:
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

                    # Apply conversion and check if result is meaningful
                    converted_series = df_proc[col].apply(convert_cell)
                    # Only replace if we get a reasonable number of valid values
                    if converted_series.notna().sum() >= threshold:
                        df_proc[col] = converted_series
                        
                # Try simple numeric conversion for columns that failed unit extraction
                elif df_proc[col].dtype == object:
                    # Attempt direct numeric conversion
                    simple_numeric = pd.to_numeric(df_proc[col], errors='coerce')
                    if simple_numeric.notna().sum() >= max(3, int(0.1 * len(df_proc))):
                        df_proc[col] = simple_numeric
        except Exception:
            continue
    
    # Final cleanup: ensure we don't have columns with all NaN after processing
    for col in df_proc.columns:
        if df_proc[col].isna().all():
            # Revert to original if all values became NaN
            df_proc[col] = df[col]

    return df_proc


def build_dynamic_chart(df: pd.DataFrame, query: str):
    """
    Build a dynamic chart based on query analysis and data characteristics.
    Supports multiple chart types: line, bar, scatter, pie, histogram.
    Automatically detects the best chart type and columns to use.

    Returns (plotly_figure, explanation) or (None, None) if not applicable.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return None, None
        
    ql = query.lower()
    tokens = re.findall(r"[a-zA-Z0-9_%-]+", ql)
    cols = list(df.columns)
    
    if df.empty or len(cols) == 0:
        return None, None

    # Determine chart type from query
    chart_type = detect_chart_type_from_query(ql)
    
    # Find the best columns for visualization
    x_col, y_col, color_col = find_best_columns_for_chart(df, ql, tokens, chart_type)
    
    if not x_col and not y_col:
        return None, None

    # Create the chart based on type and data
    fig, explanation = create_chart_by_type(df, chart_type, x_col, y_col, color_col, ql)
    
    return fig, explanation


def detect_chart_type_from_query(query_lower: str) -> str:
    """
    Detect the most appropriate chart type from the query.
    """
    # Line chart indicators
    if any(word in query_lower for word in ['trend', 'over time', 'timeline', 'line', 'time series', 'progression', 'track']):
        return 'line'
    
    # Bar chart indicators
    if any(word in query_lower for word in ['compare', 'comparison', 'bar', 'category', 'group', 'by', 'distribution']):
        return 'bar'
    
    # Pie chart indicators
    if any(word in query_lower for word in ['pie', 'proportion', 'percentage', 'share', 'composition', 'breakdown']):
        return 'pie'
    
    # Scatter plot indicators
    if any(word in query_lower for word in ['scatter', 'correlation', 'relationship', 'vs', 'against']):
        return 'scatter'
    
    # Histogram indicators
    if any(word in query_lower for word in ['histogram', 'frequency', 'distribution of']):
        return 'histogram'
        
    # Default to bar for general queries
    return 'bar'


def find_best_columns_for_chart(df: pd.DataFrame, query_lower: str, tokens: list, chart_type: str) -> tuple:
    """
    Find the best columns for x, y, and color based on data types and query.
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Try to detect datetime columns from non-datetime types
    potential_date_cols = []
    for col in categorical_cols:
        col_lower = col.lower()
        if any(word in col_lower for word in ['date', 'time', 'timestamp', 'period']):
            try:
                sample = df[col].dropna().head(10)
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() >= len(sample) * 0.8:
                    potential_date_cols.append(col)
            except Exception:
                pass
    
    datetime_cols.extend(potential_date_cols)

    def score_column_for_query(col: str, is_numeric: bool = False) -> int:
        """
        Score how well a column matches the query intent.
        """
        col_lower = col.lower()
        score = 0
        
        # Query token matching
        for token in tokens:
            if token and token in col_lower:
                score += 3
                
        # Data type preferences
        if chart_type in ['line', 'scatter', 'histogram'] and is_numeric:
            score += 2
        elif chart_type in ['bar', 'pie'] and not is_numeric:
            score += 1
            
        # Cardinality considerations
        try:
            nunique = df[col].nunique(dropna=True)
            if chart_type == 'pie' and 2 <= nunique <= 10:
                score += 3
            elif chart_type == 'bar' and 2 <= nunique <= 20:
                score += 2
            elif chart_type in ['line', 'scatter'] and nunique > 5:
                score += 1
        except Exception:
            pass
            
        return score

    # Find best columns based on chart type
    x_col = y_col = color_col = None

    if chart_type == 'line':
        # For line charts, prefer datetime for x-axis, numeric for y-axis
        if datetime_cols:
            x_col = max(datetime_cols, key=lambda col: score_column_for_query(col))
        elif categorical_cols:
            x_col = max(categorical_cols, key=lambda col: score_column_for_query(col))
        
        if numeric_cols:
            y_col = max(numeric_cols, key=lambda col: score_column_for_query(col, True))
            
    elif chart_type == 'bar':
        # For bar charts, categorical for x-axis, numeric for y-axis
        if categorical_cols:
            x_col = max(categorical_cols, key=lambda col: score_column_for_query(col))
        if numeric_cols:
            y_col = max(numeric_cols, key=lambda col: score_column_for_query(col, True))
            
    elif chart_type == 'pie':
        # For pie charts, categorical for labels, numeric for values
        if categorical_cols:
            x_col = max(categorical_cols, key=lambda col: score_column_for_query(col))
        if numeric_cols:
            y_col = max(numeric_cols, key=lambda col: score_column_for_query(col, True))
            
    elif chart_type == 'scatter':
        # For scatter plots, numeric for both axes
        if len(numeric_cols) >= 2:
            scored_numeric = sorted(numeric_cols, key=lambda col: score_column_for_query(col, True), reverse=True)
            x_col = scored_numeric[0]
            y_col = scored_numeric[1]
        elif len(numeric_cols) == 1 and categorical_cols:
            y_col = numeric_cols[0]
            x_col = max(categorical_cols, key=lambda col: score_column_for_query(col))
            
    elif chart_type == 'histogram':
        # For histograms, numeric for x-axis
        if numeric_cols:
            x_col = max(numeric_cols, key=lambda col: score_column_for_query(col, True))
    
    # Find color column (categorical with moderate cardinality)
    potential_color_cols = [col for col in categorical_cols 
                           if col != x_col and 2 <= df[col].nunique(dropna=True) <= 10]
    if potential_color_cols:
        color_col = max(potential_color_cols, key=lambda col: score_column_for_query(col))
    
    return x_col, y_col, color_col


def create_chart_by_type(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, color_col: str, query_lower: str):
    """
    Create a chart based on the determined type and columns.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return None, None

    # Clean and prepare data
    df_clean = df.copy()
    
    # Clean columns and extract numeric values from strings if needed
    for col in [x_col, y_col]:
        if col and col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]) and pd.api.types.is_object_dtype(df_clean[col]):
                # Try to extract numeric values from strings
                df_clean[col] = df_clean[col].astype(str).str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0]
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Convert datetime strings if x_col looks like dates
    if x_col and x_col in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean[x_col]):
        col_lower = x_col.lower()
        if any(word in col_lower for word in ['date', 'time', 'timestamp', 'period']):
            try:
                df_clean[x_col] = pd.to_datetime(df_clean[x_col], errors='coerce')
            except Exception:
                pass

    # Remove rows with missing values in key columns
    cols_to_check = [col for col in [x_col, y_col] if col]
    if cols_to_check:
        df_clean = df_clean.dropna(subset=cols_to_check)
    
    if df_clean.empty:
        return None, None
    
    # Limit categorical data to prevent overcrowding
    if x_col and not pd.api.types.is_numeric_dtype(df_clean[x_col]):
        if df_clean[x_col].nunique() > 20:
            top_categories = df_clean[x_col].value_counts().head(20).index
            df_clean = df_clean[df_clean[x_col].isin(top_categories)]

    # Create the chart based on type
    fig = None
    explanation = ""
    
    try:
        if chart_type == 'line' and x_col and y_col:
            if color_col:
                fig = px.line(df_clean, x=x_col, y=y_col, color=color_col, 
                             title=f"{y_col} over {x_col}")
            else:
                fig = px.line(df_clean, x=x_col, y=y_col, 
                             title=f"{y_col} over {x_col}")
            explanation = f"Line chart showing the trend of {y_col} over {x_col}."
            
        elif chart_type == 'bar' and x_col and y_col:
            # Aggregate data for bar chart
            if pd.api.types.is_numeric_dtype(df_clean[y_col]):
                agg_data = df_clean.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg_data, x=x_col, y=y_col, 
                           title=f"Average {y_col} by {x_col}")
                explanation = f"Bar chart showing average {y_col} by {x_col}."
            else:
                fig = px.histogram(df_clean, x=x_col, title=f"Distribution of {x_col}")
                explanation = f"Bar chart showing distribution of {x_col}."
                
        elif chart_type == 'pie' and x_col:
            if y_col and pd.api.types.is_numeric_dtype(df_clean[y_col]):
                pie_data = df_clean.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(pie_data, names=x_col, values=y_col, 
                           title=f"{y_col} distribution by {x_col}")
                explanation = f"Pie chart showing {y_col} distribution across {x_col}."
            else:
                value_counts = df_clean[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'count']
                fig = px.pie(value_counts, names=x_col, values='count', 
                           title=f"Distribution of {x_col}")
                explanation = f"Pie chart showing the distribution of {x_col}."
                
        elif chart_type == 'scatter' and x_col and y_col:
            if color_col:
                fig = px.scatter(df_clean, x=x_col, y=y_col, color=color_col,
                               title=f"{y_col} vs {x_col}")
            else:
                fig = px.scatter(df_clean, x=x_col, y=y_col,
                               title=f"{y_col} vs {x_col}")
            explanation = f"Scatter plot showing the relationship between {x_col} and {y_col}."
            
        elif chart_type == 'histogram' and x_col:
            fig = px.histogram(df_clean, x=x_col, title=f"Distribution of {x_col}")
            explanation = f"Histogram showing the distribution of {x_col}."
            
        else:
            # Fallback: create a simple bar chart with the first categorical and numeric columns
            categorical_cols = [col for col in df_clean.columns if not pd.api.types.is_numeric_dtype(df_clean[col])]
            numeric_cols = [col for col in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[col])]
            
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                agg_data = df_clean.groupby(cat_col)[num_col].mean().reset_index()
                fig = px.bar(agg_data, x=cat_col, y=num_col, 
                           title=f"Average {num_col} by {cat_col}")
                explanation = f"Bar chart showing average {num_col} by {cat_col}."
            elif numeric_cols:
                num_col = numeric_cols[0]
                fig = px.histogram(df_clean, x=num_col, title=f"Distribution of {num_col}")
                explanation = f"Histogram showing the distribution of {num_col}."
                
        if fig:
            fig.update_layout(showlegend=True if color_col else False)
            
    except Exception as e:
        # If chart creation fails, return None
        return None, None
        
    return fig, explanation



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
):
    try:
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

            return JSONResponse(
                content={
                    "report": cleaned_result["report"],
                    "title": cleaned_result["title"],
                    "description": cleaned_result["description"]
                },
                status_code=200
            )
        # Not a report query → route to one of the three agents (graph/table/text)
        agent = detect_agent(query)

        # 2,3,4 Agents: Try simple direct handling first for table/text intents
        if agent in ["table", "text"]:
            can_handle_directly, structured = classify_query_complexity(query, list(df.columns))
            if can_handle_directly and structured is not None:
                intent, column = structured
                simple_resp = handle_simple_query(df, intent, column)
                return JSONResponse(content=simple_resp, status_code=200)

        # If it's a graph request, try dynamic chart generation before using LLM
        if agent == "graph":
            try:
                fig, exp = build_dynamic_chart(df, query)
                if fig is not None:
                    formatted = format_result_for_response(fig)
                    if exp:
                        formatted["explanation"] = exp
                    return JSONResponse(content=formatted, status_code=200)
            except Exception:
                # Fall back to LLM path if direct handling fails silently
                pass

        # Use LLM to generate analysis code based on agent intent
        analysis_result = get_llm_analysis_explore(query, df, mode=agent)

        response_type = analysis_result.get("type")
        payload = analysis_result.get("payload")

        if not response_type or not payload:
            return JSONResponse(content={"type": "text",
                                         "payload": "I'm sorry, I couldn't process that request. Please try rephrasing."},
                                status_code=200)

        # Conversational/text-only answer
        if response_type == "conversational_answer":
            return JSONResponse(content={"type": "text", "payload": payload}, status_code=200)

        # Data analysis answer with code to execute
        if response_type == "data_analysis_answer":
            explanation = payload.get("explanation")
            code = payload.get("code")

            if not code:
                return JSONResponse(content={"type": "text",
                                             "payload": explanation or "I understood your request but couldn't generate the right code. Please try again."},
                                    status_code=200)

            try:
                exec_result = safe_execute_pandas_code(code, df)
                formatted = format_result_for_response(exec_result)

                # If graph agent but no valid chart, attempt a robust server-side rescue chart
                if agent == "graph" and formatted.get("type") != "plotly":
                    rescue_fig, biz_exp = generate_rescue_chart(df, query)
                    if rescue_fig is not None:
                        formatted = format_result_for_response(rescue_fig)
                        # Prefer business-style explanation
                        formatted["explanation"] = biz_exp or (explanation or "")
                        return JSONResponse(content=formatted, status_code=200)

                if explanation:
                    formatted["explanation"] = explanation
                return JSONResponse(content=formatted, status_code=200)
            except Exception as e:
                return JSONResponse(content={"type": "text",
                                             "payload": f"There was an error executing the analysis: {str(e)}",
                                             "explanation": explanation}, status_code=200)

        # Fallback
        return JSONResponse(content={"type": "text", "payload": f"Unrecognized response type: {response_type}"}, status_code=200)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Data file is corrupt"
        )
    except Exception as e:
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
    q = query.lower()
    graph_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'visualise', 'trend', 'histogram',
        'scatter', 'line chart', 'line', 'bar chart', 'bar', 'pie', 'box', 'violin', 'heatmap'
    ]
    table_keywords = [
        'table', 'list', 'rows', 'records', 'show rows', 'display table', 'show table', 'top ', 'head', 'tail'
    ]

    if any(k in q for k in graph_keywords):
        return "graph"
    if any(k in q for k in table_keywords):
        return "table"
    return "text"


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
    if isinstance(obj, dict):
        if set(obj.keys()) == {'dtype', 'bdata'}:
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
    """Format execution result into response-friendly structure."""
    # Plotly figure
    if PLOTLY_AVAILABLE and go is not None and hasattr(go, 'Figure') and isinstance(result, go.Figure):  # type: ignore
        try:
            if hasattr(result, 'to_json'):
                fig_dict = json.loads(result.to_json())
            else:
                fig_dict = result.to_dict()
            fig_dict = convert_plotly_arrays(fig_dict)

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
                return {"type": "text", "payload": "Chart could not be generated due to insufficient aggregated data."}

            fig_dict['data'] = valid_traces
            return {"type": "plotly", "payload": fig_dict}
        except Exception:
            # Fall back to text if conversion/validation fails
            return {"type": "text", "payload": "Chart generation failed unexpectedly. Please try a different view or metric."}

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
        return {"type": "table", "payload": payload}

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
        return {"type": "table", "payload": payload}

    # dict (try convert to DataFrame)
    if isinstance(result, dict):
        try:
            df_result = pd.DataFrame(result).reset_index()
            payload = df_result.to_dict(orient='records')
            return {"type": "table", "payload": payload}
        except Exception:
            return {"type": "text", "payload": str(result)}

    # list
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            return {"type": "table", "payload": result}
        else:
            return {"type": "text", "payload": str(result)}

    # numpy arrays
    if hasattr(result, 'tolist'):
        try:
            list_result = result.tolist()
            return {"type": "text", "payload": str(list_result)}
        except Exception:
            return {"type": "text", "payload": str(result)}

    # scalar/string
    return {"type": "text", "payload": str(result)}


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
            return {"type": "text", "payload": f"There are {result} unique values in column '{column}'."}
        elif intent == "unique" or intent == "list":
            result = df[column].dropna().unique().tolist()
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


def get_llm_analysis_explore(query: str, df: pd.DataFrame, mode: str) -> Dict[str, object]:
    """
    Use LLM to analyze user's query and generate code to produce graph/table/text.
    Returns a JSON with keys: type and payload. For data analysis, payload has explanation and code.
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

    instructions_common = f"""
You are an expert data analyst AI. You must respond with a single JSON object only (no markdown fences).

If the response is conversational only: use type="conversational_answer" and payload as a string.
If the response requires data processing: use type="data_analysis_answer" and payload as an object with keys "explanation" and "code".

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
            "Generate Plotly code that creates the most appropriate visualization for the user's question. "
            "CHART TYPE SELECTION: Choose based on data types and query intent:"
            "- Line charts (px.line): for trends over time, time series data"
            "- Bar charts (px.bar): for comparing categories, showing totals by group"
            "- Scatter plots (px.scatter): for showing relationships between two numeric variables"
            "- Pie charts (px.pie): for showing composition/proportions of a whole"
            "- Histograms (px.histogram): for showing distribution of a single numeric variable"
            "\n"
            "DATA PREPARATION STEPS:"
            "1. Clean data: df = df.dropna(subset=[required_columns]) to remove missing values"
            "2. Handle string numerics: df['col'] = pd.to_numeric(df['col'].str.extract(r'([-+]?\\d*\\.?\\d+)')[0], errors='coerce')"
            "3. Limit categories: For categorical data with >20 unique values, use .value_counts().head(20) to avoid overcrowding"
            "4. Aggregate when needed: Use groupby().agg() for bar charts to get meaningful summaries"
            "\n"
            "CODE STRUCTURE:"
            "- Import: import plotly.express as px"
            "- Always assign the final figure to 'result': result = px.chart_type(...)"
            "- Add clear titles: title='Descriptive Chart Title'"
            "- Use meaningful labels: labels={'x_col': 'X Label', 'y_col': 'Y Label'}"
            "\n"
            "ERROR HANDLING: If chart creation fails, create a fallback table: result = df.groupby('category_col')['numeric_col'].agg(['count', 'mean', 'sum']).reset_index()"
            "Never return None or empty figures. Always ensure result contains valid data."
        )
    elif mode == "table":
        mode_instructions = (
            "Generate pandas code that returns a tabular result in a DataFrame assigned to result. "
            "Do not generate charts. Include necessary groupby/sort/limit operations based on the question."
        )
    else:  # text
        mode_instructions = (
            "If the question is conversational or generic, return a conversational_answer. "
            "If it requires data-derived text, generate pandas code that computes the answer and sets result to a concise human-readable string."
        )

    llm_prompt = f"""
{instructions_common}

Dataset Context:
{dataset_info}

User Question: {query}

Mode: {mode}
Specific Instructions: {mode_instructions}

Explanation Style: Write the explanation as a concise, business-oriented summary focusing on trend, scale, change, and implications. Avoid technical implementation details.

Return ONLY the JSON object as described above.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert data analyst and Python/Plotly code generator."},
            {"role": "user", "content": llm_prompt}
        ],
        temperature=0.1,
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
    Enhanced fallback chart generation with multiple chart type support.
    Attempts to create the most appropriate chart based on data characteristics.
    Returns: (plotly.graph_objects.Figure or None, explanation str)
    """
    try:
        # Use the enhanced dynamic chart function as rescue
        fig, explanation = build_dynamic_chart(df, query)
        if fig is not None:
            return fig, explanation
            
        # If dynamic chart fails, try basic fallback
        ql = query.lower()

        # Find any numeric and categorical columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols and not categorical_cols:
            return None, None

        # Create a simple chart as last resort
        import plotly.express as px
        
        if categorical_cols and numeric_cols:
            # Bar chart: categorical vs numeric
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Limit categories to prevent overcrowding
            if df[cat_col].nunique() > 15:
                top_cats = df[cat_col].value_counts().head(15).index
                df_limited = df[df[cat_col].isin(top_cats)]
        else:
                df_limited = df
                
            agg_data = df_limited.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(agg_data, x=cat_col, y=num_col, 
                        title=f"Average {num_col} by {cat_col}")
            explanation = f"Bar chart showing average {num_col} by {cat_col}."
            return fig, explanation
            
        elif numeric_cols:
            # Histogram for single numeric column
            num_col = numeric_cols[0]
            fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
            explanation = f"Histogram showing the distribution of {num_col}."
            return fig, explanation
            
        elif categorical_cols:
            # Pie chart for categorical data
            cat_col = categorical_cols[0]
            if df[cat_col].nunique() <= 10:
                value_counts = df[cat_col].value_counts().reset_index()
                value_counts.columns = [cat_col, 'count']
                fig = px.pie(value_counts, names=cat_col, values='count',
                           title=f"Distribution of {cat_col}")
                explanation = f"Pie chart showing the distribution of {cat_col}."
                return fig, explanation

        return None, None
    except Exception:
        return None, None

