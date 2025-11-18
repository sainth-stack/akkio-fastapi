import re
import textwrap
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .formatters import format_text_response


def detect_agent(query: str) -> str:
    q = query.lower()
    graph_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'visualise', 'trend', 'histogram',
        'scatter', 'line chart', 'line', 'bar chart', 'bar', 'pie', 'box', 'violin', 'heatmap'
    ]
    table_keywords = [
        'table', 'list', 'rows', 'records', 'show rows', 'display table', 'show table', 'top ', 'head', 'tail'
    ]
    conversational_keywords = [
        'what is', 'what are', 'how do', 'how can', 'explain', 'tell me about', 
        'kpi', 'kpis', 'key performance indicator', 'metrics', 'help', 'understand',
        'define', 'meaning', 'purpose', 'importance', 'benefits'
    ]
    if any(k in q for k in conversational_keywords):
        return "text"
    if any(k in q for k in graph_keywords):
        return "graph"
    if any(k in q for k in table_keywords):
        return "table"
    return "text"


def classify_query_complexity(query: str, col_names: List[str]) -> Tuple[bool, Optional[Tuple[str, str]]]:
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
                payload = format_text_response(f"<h4>Sum Calculation Error</h4><p>Cannot calculate sum for non-numeric column '{column}'.</p>")
                return {"type": "text", "payload": payload}
        elif intent == "mean":
            if df[column].dtype in ['int64', 'float64']:
                result = float(df[column].mean())
                payload = format_text_response(f"<h4>Average Calculation</h4><p>The average of column '{column}' is <strong>{result:.2f}</strong>.</p>")
                return {"type": "text", "payload": payload}
            else:
                payload = format_text_response(f"<h4>Average Calculation Error</h4><p>Cannot calculate mean for non-numeric column '{column}'.</p>")
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
            payload = format_text_response(f"<h4>Unsupported Operation</h4><p>Intent '{intent}' is not supported for direct handling.</p>")
            return {"type": "text", "payload": payload}
    except Exception as e:
        payload = format_text_response(f"<h4>Query Processing Error</h4><p>Error processing query: {str(e)}</p>")
        return {"type": "text", "payload": payload}


def safe_execute_pandas_code(code: str, df: pd.DataFrame):
    """
    Safely execute sandboxed Python code for data analysis.
    The code should assign the final output to a variable named 'result'.
    """
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
            raise ValueError("Plotly and/or NumPy are required for chart generation but are not installed. Please install them using: pip install plotly numpy")

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
            import ast
            ast.parse(code_clean)
        except (IndentationError, TabError, SyntaxError) as e:
            lines = code_clean.split('\n')
            non_empty = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith('#')]
            if non_empty:
                def leading_spaces(s: str) -> int:
                    return len(s) - len(s.lstrip(' '))
                min_indent = min(leading_spaces(ln) for ln in non_empty)
                if min_indent > 0:
                    lines = [ln[min_indent:] if ln.startswith(' ' * min_indent) else ln for ln in lines]
                    code_clean = '\n'.join(lines)
            import ast as _ast
            _ast.parse(code_clean)

        local_vars: Dict[str, object] = {}
        exec(code_clean, safe_globals, local_vars)
        if 'result' in local_vars:
            return local_vars['result']
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








