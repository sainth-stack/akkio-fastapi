import re
from difflib import get_close_matches
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd


def build_entity_metric_line_chart(df: pd.DataFrame, query: str):
    ql = query.lower()
    tokens = re.findall(r"[a-zA-Z0-9_%-]+", ql)
    cols = list(df.columns)
    def score_entity(col: str) -> int:
        name = col.lower()
        score = 0
        if not pd.api.types.is_numeric_dtype(df[col]) or 'id' in name or 'name' in name:
            score += 1
        for kw in {'sensor', 'device', 'asset', 'machine', 'equipment', 'vehicle', 'truck', 'unit', 'id', 'name'}:
            if kw in name:
                score += 3
        for t in tokens:
            if t and t in name:
                score += 2
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
        if pd.api.types.is_numeric_dtype(df[col]):
            score += 3
        for kw in {'speed', 'velocity', 'rpm', 'temperature', 'temp', 'pressure', 'load', 'usage', 'value',
                   'distance', 'duration', 'throughput', 'latency', 'count', 'rate', 'emission', 'co2',
                   'current', 'voltage', 'power', 'energy', 'efficiency', 'cost', 'revenue', 'sales'}:
            if kw in name:
                score += 3
        for t in tokens:
            if t and t in name:
                score += 2
        return score
    entity_col = max(cols, key=score_entity) if cols else None
    metric_col = max(cols, key=score_metric) if cols else None
    if not entity_col or not metric_col or entity_col == metric_col:
        return None, None
    use_cols = [entity_col, metric_col]
    df_local = df[use_cols].copy()
    if df_local.empty:
        return None, None
    metric_name_lower = str(metric_col).lower()
    speed_context = (
        ('speed' in metric_name_lower) or ('velocity' in metric_name_lower)
        or any(t in {'speed', 'velocity', 'mph', 'kmph', 'kph', 'km/h', 'm/s', 'knots', 'knot'} for t in tokens)
    )
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
            norm = unit_str.replace('per', '/').replace(' ', '')
            if norm in unit_factor_map:
                return value * unit_factor_map[norm]
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
            return value
        return series.apply(parse_and_convert)  # type: ignore
    yaxis_title = metric_col
    if not pd.api.types.is_numeric_dtype(df_local[metric_col]):
        ser = df_local[metric_col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan})
        if speed_context:
            df_local[metric_col] = convert_speed_series_to_kmh(ser)
            yaxis_title = 'Speed (km/h)'
        else:
            df_local[metric_col] = ser.str.extract(r'([-+]?\d*\.?\d+)')[0]
            df_local[metric_col] = pd.to_numeric(df_local[metric_col], errors='coerce')
    time_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col = c
            break
    if time_col is None:
        for c in df.columns:
            name = c.lower()
            if any(k in name for k in ['time', 'date', 'timestamp', 'period']):
                try:
                    coerced = pd.to_datetime(df[c], errors='coerce')
                    if coerced.notna().sum() >= max(5, int(0.2 * len(df))):
                        time_col = c
                        df_local['__x__'] = coerced
                        break
                except Exception:
                    continue
    if time_col is None:
        df_local['__x__'] = np.arange(len(df_local))
        x_title = 'Record'
    else:
        df_local['__x__'] = pd.to_datetime(df[time_col], errors='coerce')
        x_title = time_col
    df_local = df_local.dropna(subset=['__x__', metric_col, entity_col])
    if df_local.empty:
        return None, None
    top_entities = df_local[entity_col].astype(str).value_counts().head(12).index.tolist()
    df_local = df_local[df_local[entity_col].astype(str).isin(top_entities)]
    import plotly.graph_objects as go
    fig = go.Figure()
    for ent, grp in df_local.sort_values('__x__').groupby(entity_col):
        y_vals = pd.to_numeric(grp[metric_col], errors='coerce').dropna()
        x_vals = grp.loc[y_vals.index, '__x__']
        if len(x_vals) > 0:
            fig.add_trace(go.Scatter(x=list(x_vals), y=list(y_vals), mode='lines+markers', name=str(ent)))
    fig.update_layout(title=f"{entity_col} vs {metric_col}", xaxis_title=x_title, yaxis_title=yaxis_title, legend_title=entity_col)
    exp = f"Line chart showing {metric_col} over {x_title} for each {entity_col}. Numeric values were extracted dynamically from strings where necessary."
    return fig, exp


def generate_rescue_chart(df: pd.DataFrame, query: str):
    print(f"Rescue chart generation started for query: {query[:100]}...")
    try:
        ql = query.lower()
        def score_column(col: str) -> int:
            name = col.lower()
            score = 0
            for token in re.findall(r"[a-zA-Z0-9%]+", ql):
                if token and token in name:
                    score += 2
            boosts = ['achievement', 'target', 'actual', 'value', 'amount', 'sales', 'revenue', 'emission', 'co2', 'count', 'volume', 'cost']
            for b in boosts:
                if b in name and b in ql:
                    score += 3
            return score
        date_candidates = []
        for col in df.columns:
            lc = col.lower()
            if 'date' in lc or 'time' in lc or 'period' in lc:
                date_candidates.append((col, score_column(col)+3))
            else:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_candidates.append((col, score_column(col)+1))
                except Exception:
                    pass
        date_col = max(date_candidates, key=lambda x: x[1])[0] if date_candidates else None
        geospatial_aliases = {"latitude", "longitude", "lat", "lon", "lng"}
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in geospatial_aliases]
        synthetic_numeric_map = {}
        if not numeric_cols:
            for c in df.columns:
                if c.lower() in geospatial_aliases:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    continue
                try:
                    ser = df[c].astype(str).str.strip()
                    ser = ser.replace({'': np.nan, 'nan': np.nan, 'null': np.nan, 'none': np.nan, 'undefined': np.nan})
                    extracted = ser.str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0]
                    numeric_ser = pd.to_numeric(extracted, errors='coerce')
                    if numeric_ser.notna().sum() >= max(10, int(0.1 * len(df))):
                        synthetic_numeric_map[c] = numeric_ser
                except Exception:
                    continue
            if synthetic_numeric_map:
                scored = sorted(synthetic_numeric_map.items(), key=lambda kv: (score_column(kv[0]), kv[1].notna().sum()), reverse=True)
                best_name, best_series = scored[0]
                numeric_cols = [best_name]
                df = df.copy()
                df[best_name] = best_series
        if numeric_cols:
            scored = sorted(numeric_cols, key=lambda c: score_column(c), reverse=True)
            metric_col = scored[0]
        else:
            metric_col = None
        tokens = re.findall(r"[a-zA-Z0-9%]+", ql)
        for token in tokens:
            matches = get_close_matches(token, list(df.columns), n=1, cutoff=0.85)
            if matches:
                m = matches[0]
                if m in numeric_cols:
                    metric_col = m
                else:
                    date_col = date_col or m
        if date_col is None and metric_col is not None:
            categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if categorical_cols:
                cat_scored = sorted(categorical_cols, key=lambda c: (score_column(c), -df[c].nunique()))
                x_col = cat_scored[0]
                df_local = df[[x_col, metric_col]].dropna()
                if df_local.empty:
                    return None, None
                agg = df_local.groupby(x_col)[metric_col].mean().sort_values(ascending=False).head(20).reset_index()
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(x=agg[x_col].astype(str).tolist(), y=agg[metric_col].tolist(), name=metric_col))
                fig.update_layout(title=f"Top {x_col} by average {metric_col}", xaxis_title=x_col, yaxis_title=metric_col)
                biz_exp = f"Top categories by average {metric_col} highlight where performance concentrates."
                return fig, biz_exp
            return None, None
        if date_col is None or metric_col is None:
            return None, None
        df_local = df[[date_col, metric_col]].dropna()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors='coerce')
        df_local = df_local.dropna(subset=[date_col])
        if df_local.empty:
            return None, None
        daily = df_local.groupby(df_local[date_col].dt.date)[metric_col].mean().reset_index()
        daily.columns = ['Date', metric_col]
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
            biz_exp = f"{metric_col} shows a {direction} from {start_val:.1f} to {end_val:.1f}. This indicates a {abs(pct):.1f}% change."
        else:
            biz_exp = f"Time-series view of {metric_col} to support quick performance diagnostics."
        print(f"Rescue chart successful: {title}")
        return fig, biz_exp
    except Exception as e:
        print(f"Rescue chart failed: {str(e)}")
        return None, None








