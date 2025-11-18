import re
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd


def preprocess_dataframe_for_graphing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe for robust chart generation:
    - Normalize empty-like tokens to NaN
    - Parse datetime-like columns
    - Extract numeric values from strings with units
    - Convert common speed units to km/h when unit prevalence is detected.
    Returns a new dataframe copy.
    """
    df_proc = df.copy()
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

    for col in df_proc.columns:
        try:
            name_lower = str(col).lower()
            if any(k in name_lower for k in ['time', 'date', 'timestamp', 'period']):
                parsed = pd.to_datetime(df_proc[col], errors='coerce')
                if parsed.notna().sum() >= max(5, int(0.1 * len(df_proc))):
                    df_proc[col] = parsed
                    continue
            if df_proc[col].dtype == object:
                series_obj = df_proc[col].astype(str)
                sample = series_obj.head(200)
                unit_counts: Dict[str, int] = {}
                numeric_vals: List[Optional[float]] = []
                for v in sample:
                    num, unit = extract_number_and_unit(v)
                    numeric_vals.append(num)
                    if unit:
                        unit_counts[unit] = unit_counts.get(unit, 0) + 1
                if sum(1 for x in numeric_vals if x is not None) >= max(10, int(0.3 * len(sample))):
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
                        if norm in {'km/hr', 'km/hrs', 'kmperhour'}:
                            norm = 'km/h'
                        if norm in {'m/ s', 'm/second', 'meter/sec', 'meters/sec'}:
                            norm = 'm/s'
                        if norm in speed_units:
                            factor = speed_units[norm]
                            return num * factor
                        return num

                    df_proc[col] = df_proc[col].apply(convert_cell)
        except Exception:
            continue
    return df_proc








