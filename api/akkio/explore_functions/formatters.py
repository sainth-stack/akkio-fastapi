import json
import datetime as dt
from typing import Any
import numpy as np
import pandas as pd
import re

try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    go = None  # type: ignore
    PLOTLY_AVAILABLE = False


def format_text_response(text: str) -> str:
    """
    Ensure text responses have proper HTML formatting without gaps.
    Adds HTML tags if not already present and removes unnecessary spacing.
    """
    if not text:
        return text
    if '<h4>' in text or '<p>' in text:
        text = re.sub(r'>\s*\n\s*<', '><', text)
        text = re.sub(r'\n+', '', text)
        return text
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 80 and (line.endswith(':') or line.isupper()):
            formatted_lines.append(f'<h4>{line}</h4>')
        else:
            formatted_lines.append(f'<p>{line}</p>')
    return ''.join(formatted_lines)


def convert_plotly_arrays(obj):
    """Recursively convert numpy/pandas arrays in Plotly figure dicts to plain lists."""
    try:
        import numpy as _np  # local alias to avoid shadowing
        if obj is None:
            return None
        if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
            return obj.isoformat()
        if isinstance(obj, _np.datetime64):
            try:
                return pd.Timestamp(obj).isoformat()
            except Exception:
                return str(obj)
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
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float):
            if _np is not None and (_np.isnan(obj) or _np.isinf(obj)):
                return None
        return obj
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    if _np is not None and isinstance(obj, _np.datetime64):
        try:
            return pd.Timestamp(obj).isoformat()
        except Exception:
            return str(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
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
    if isinstance(obj, pd.DataFrame):
        records = obj.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient='records')
        return _make_json_safe(records)
    if isinstance(obj, pd.Series):
        series = obj.replace({np.nan: None, np.inf: None, -np.inf: None})
        try:
            return _make_json_safe(series.to_list())
        except Exception:
            return _make_json_safe(series.to_dict())
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    return str(obj)


def format_result_for_response(result):
    """Format execution result into response-friendly structure."""
    if PLOTLY_AVAILABLE and go is not None and hasattr(go, 'Figure') and isinstance(result, go.Figure):  # type: ignore
        try:
            if hasattr(result, 'to_json'):
                fig_dict = json.loads(result.to_json())
            else:
                fig_dict = result.to_dict()
            fig_dict = convert_plotly_arrays(fig_dict)
            fig_dict = _make_json_safe(fig_dict)
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
                x_values = trace.get('x')
                y_values = trace.get('y')
                values_values = trace.get('values')
                z_values = trace.get('z')
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
                y_values = trace.get('y')
                values_values = trace.get('values')
                z_values = trace.get('z')
                has_y = isinstance(y_values, list) and len(y_values) > 0
                has_values = isinstance(values_values, list) and len(values_values) > 0
                has_z = isinstance(z_values, list) and len(z_values) > 0
                if has_y or has_values or has_z:
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
                return {"type": "text", "payload": format_text_response("<h4>Chart Generation Issue</h4><p>Chart could not be generated due to insufficient aggregated data. Please try again.</p>")}
            fig_dict['data'] = valid_traces
            print(f"Chart validation successful: {len(valid_traces)} valid traces")
            return {"type": "plotly", "payload": fig_dict}
        except Exception as e:
            print(f"Chart conversion failed: {str(e)}")
            return {"type": "text", "payload": format_text_response("<h4>Chart Generation Error</h4><p>Chart generation failed unexpectedly. Please try a different view.</p>")}
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
    if isinstance(result, dict):
        try:
            df_result = pd.DataFrame(result).reset_index()
            payload = df_result.to_dict(orient='records')
            return {"type": "table", "payload": _make_json_safe(payload)}
        except Exception:
            formatted_text = format_text_response(str(result))
            return {"type": "text", "payload": formatted_text}
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], dict):
            return {"type": "table", "payload": _make_json_safe(result)}
        else:
            formatted_text = format_text_response(str(result))
            return {"type": "text", "payload": formatted_text}
    if hasattr(result, 'tolist'):
        try:
            list_result = result.tolist()
            formatted_text = format_text_response(str(list_result))
            return {"type": "text", "payload": formatted_text}
        except Exception:
            formatted_text = format_text_response(str(result))
            return {"type": "text", "payload": formatted_text}
    formatted_text = format_text_response(str(result))
    return {"type": "text", "payload": formatted_text}







