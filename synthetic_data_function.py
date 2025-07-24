from typing import Optional

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.utils import resample
from langchain_openai import ChatOpenAI
import re
import dateutil.parser as dp
from collections import Counter
from openai import OpenAIError


def extrapolate_from_seed(seed_df: pd.DataFrame, target_rows: int, datetime_col: str) -> pd.DataFrame:
    if datetime_col not in seed_df.columns:
        raise ValueError(f"{datetime_col} column not found.")

    seed_df[datetime_col] = pd.to_datetime(seed_df[datetime_col])
    seed_df = seed_df.sort_values(by=datetime_col).reset_index(drop=True)

    if len(seed_df) < 2:
        raise ValueError("At least 2 rows required to infer interval.")

    fixed_delta = seed_df[datetime_col].iloc[1] - seed_df[datetime_col].iloc[0]
    if fixed_delta.total_seconds() == 0:
        raise ValueError("Interval between datetime rows is zero.")

    start_time = seed_df[datetime_col].iloc[0]
    new_times = [start_time + i * fixed_delta for i in range(target_rows)]

    sampled_df = resample(seed_df.drop(columns=[datetime_col]), n_samples=target_rows, random_state=42)
    sampled_df.insert(0, datetime_col, new_times)

    return sampled_df.reset_index(drop=True)


def detect_date_format(series: pd.Series) -> Optional[str]:
    sample_dates = series.dropna().astype(str).head(20)
    formats = []
    for val in sample_dates:
        try:
            dt = dp.parse(val, dayfirst=True, fuzzy=True)
            if dt.strftime('%d-%m-%Y') in val:
                formats.append('%d-%m-%Y')
            elif dt.strftime('%m-%d-%Y') in val:
                formats.append('%m-%d-%Y')
            elif dt.strftime('%Y-%m-%d') in val:
                formats.append('%Y-%m-%d')
        except Exception:
            continue
    if formats:
        return Counter(formats).most_common(1)[0][0] + ' %H:%M:%S'
    return '%d-%m-%Y %H:%M:%S'


def generate_synthetic_data(api_key: str, file_path: str, total_rows: int = 1000, llm_seed_rows: int = 100,
                            datetime_col: str = None) -> pd.DataFrame:
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

    if file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Must be .xlsx or .csv")

    data = data.tail(20)  # cap sample to 20 rows only

    date_format = '%d-%m-%Y %H:%M:%S'
    if datetime_col and datetime_col in data.columns:
        data[datetime_col] = pd.to_datetime(data[datetime_col], errors='coerce')
        data = data.sort_values(by=datetime_col)
        date_format = detect_date_format(data[datetime_col])
        data[datetime_col] = data[datetime_col].dt.strftime(date_format)

    sample_str = data.to_csv(index=False, header=False)
    sysp = "You are a synthetic data generator. Your output should only be specified format without any additional text and code fences."

    column_names = list(data.columns)

    prompt = (
        f"Generate {llm_seed_rows} rows of synthetic data based on the structure and distribution of the following sample:\n\n{sample_str}\n"
        "Ensure the new rows are realistic, varied, and maintain the same data types, distribution, and logical relationships. "
        "Format as pipe-separated values ('|') without including column names or old data."
    )

    messages = [
        SystemMessage(content=sysp),
        HumanMessage(content=prompt)
    ]

    try:
        response = llm.invoke(messages)
        rows = [tuple(row.split("|")) for row in response.content.strip().split("\n") if row.strip()]
    except OpenAIError as e:
        raise RuntimeError(f"LLM generation failed: {str(e)}")

    seed_df = pd.DataFrame(rows, columns=column_names)

    for col in column_names:
        if col == datetime_col:
            seed_df[col] = pd.to_datetime(seed_df[col], errors="coerce")
        else:
            seed_df[col] = seed_df[col].astype(data[col].dtype, errors="ignore")

    if datetime_col and datetime_col in seed_df.columns:
        seed_df = seed_df.sort_values(by=datetime_col).reset_index(drop=True)

    final_df = extrapolate_from_seed(seed_df, total_rows, datetime_col=datetime_col)
    return final_df
