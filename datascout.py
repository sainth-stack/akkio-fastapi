import os
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Tuple
import re
from langchain.tools import StructuredTool
from typing import Dict

from pandas import DataFrame
# For PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, Frame, KeepInFrame
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import requests
from typing import List, Dict, Optional, Tuple
import re
import json
from PIL import Image as PILImage
from sklearn.utils import resample


def initialize_llm():
    print("[DEBUG] Initializing OpenAI LLM...")
    return ChatOpenAI(
        model="gpt-4.1-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=4096
    )


# -----------------------------------------------------------------------------------------------------------------
# For DataScout with Excel Generation
# ----------------- Step 1: Enhanced Prompt Parser ---------------------
def parse_prompt_metadata(prompt: str) -> dict[str, int | str | list[Any] | dict[Any, Any]] | None | Any:
    llm = initialize_llm()

    system_instruction = """
    You are a data prompt parser for synthetic time-series generation. Given any natural language prompt, extract structured metadata:

    {
      "num_rows": <int or null>,
      "columns": ["col1", "col2", ...],
      "time_columns": {
        "<col_name>": {
          "type": "timestamp/date/datetime/time",
          "start": "<YYYY-MM-DD HH:MM:SS>",
          "end": "<YYYY-MM-DD HH:MM:SS>",
          "frequency": "<pandas freq string>"
        }
      },
      "default_frequency": "1D",
      "constraints": [
        "<if condition then constraint rule>"
      ]
    }

    Consider all hints in the prompt:
    - Frequency (e.g., every 5 minutes, 15-min interval)
    - Explicit or implicit row count ("create 20,000 rows", or inferred from date range and frequency)
    - Timestamp format inference from examples (e.g., 01/06/2025 8:00)
    - Time columns with names like date, datetime, timestamp, time
    - Multiple time columns (e.g., shift_start_time, shift_end_time)
    - Fallback to date-only column if no time mentioned
    - Use UTC now if no start/end date is given
    - Normalize frequency string to valid pandas format (e.g., min → T, hour → H, daily → D)
    - Infer rules and relationships in columns (e.g., If Obs_Obj is Signal, then Type is Planned)
    - Do NOT return extra text. Output must be JSON only.
    """

    response = llm.invoke(f"{system_instruction}\n\nPrompt: {prompt}")
    content = response.content if hasattr(response, 'content') else str(response)
    print("Content is,,,,",content)
    response = llm.invoke(f"{system_instruction}\n\nPrompt: {prompt}")
    content = response.content if hasattr(response, 'content') else str(response)
    print("Content is,,,,", content)

    try:
        # Extract JSON from markdown code blocks if present
        import re
        json_match = re.search(r'``````', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            json_str = json_match.group(1) if json_match else content

        parsed = json.loads(json_str)

        # Validate required fields
        if not parsed.get("columns") or not isinstance(parsed["columns"], list):

            return print("Warning: No valid columns detected, using fallback")

        # Initialize optional fields
        parsed["time_columns"] = parsed.get("time_columns", {})
        parsed["constraints"] = parsed.get("constraints", [])

        # Rest of your existing logic for timestamp validation and processing
        def is_valid_timestamp(ts: str) -> bool:
            return ts and isinstance(ts, str) and not ts.startswith("<") and not ts.endswith(">")

        import datetime
        import pandas as pd
        now = datetime.datetime.utcnow()

        # Custom date format handling (e.g., 01/06/2025 8:00)
        example_match = re.search(r"(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2})", prompt)
        if example_match:
            example_dt = example_match.group(1)
            try:
                parsed_dt = datetime.datetime.strptime(example_dt, "%d/%m/%Y %H:%M")
                for col, config in parsed["time_columns"].items():
                    config["start"] = parsed_dt.strftime("%Y-%m-%d %H:%M:%S")
                    config["frequency"] = config.get("frequency") or "1D"
                    freq_unit = config["frequency"][0].upper()
                    config["end"] = (
                                parsed_dt + pd.to_timedelta(parsed.get("num_rows", 100) - 1, unit=freq_unit)).strftime(
                        "%Y-%m-%d %H:%M:%S")
                    config["type"] = "datetime"
            except Exception as ex:
                print(f"Failed to parse custom datetime example: {ex}")

        # Process time columns
        parsed_valid = False
        for col, config in parsed["time_columns"].items():
            start, end, freq = config.get("start"), config.get("end"), config.get("frequency")
            freq = freq or parsed.get("default_frequency", "1D")

            freq_unit = freq[0].upper()
            freq_unit = 'T' if freq_unit == 'M' else freq_unit

            if not (is_valid_timestamp(start) and is_valid_timestamp(end)):
                config["start"] = now.strftime("%Y-%m-%d %H:%M:%S")
                config["end"] = (now + pd.to_timedelta(parsed.get("num_rows", 100) - 1, unit=freq_unit)).strftime(
                    "%Y-%m-%d %H:%M:%S")
                config["type"] = "datetime"
                parsed["num_rows"] = parsed.get("num_rows", 100)
                parsed_valid = True
                break
            else:
                try:
                    if not parsed.get("num_rows"):
                        date_range = pd.date_range(start=start, end=end, freq=freq)
                        parsed["num_rows"] = len(date_range)
                    config["type"] = "datetime"
                    parsed_valid = True
                    break
                except Exception as e:
                    print(f"Error processing time column {col}: {e}")
                    pass

        if not parsed_valid and not parsed.get("num_rows"):
            parsed["num_rows"] = 100

        return parsed

    except Exception as e:
        print(f"Error parsing metadata: {str(e)}")
        return {
            "num_rows": 100,
            "columns": [],
            "time_columns": {},
            "default_frequency": "1D",
            "constraints": []
        }




# ----------------- Step 2: Enhanced Seed Data Generator ---------------------
def generate_seed_data(column_names: List[str], seed_limit: int, text_sample: str,
                       time_columns: Dict = {}) -> pd.DataFrame:
    llm = initialize_llm()

    # Prepare column type hints for LLM
    type_hints = []
    for col in column_names:
        if col in time_columns:
            col_type = time_columns[col].get("type", "datetime")
            type_hints.append(f"{col} ({col_type})")
        else:
            type_hints.append(f"{col} (random appropriate data)")

    column_hint_str = ", ".join(type_hints)

    sysp = f"""You are a data generator. Follow these rules:
1. Generate exactly {seed_limit} rows of synthetic data
2. Columns: {column_hint_str}
3. Format: tilde-separated values (no headers)
4. For time columns: use ISO 8601 format (YYYY-MM-DD HH:MM:SS)
5. For date columns: use YYYY-MM-DD
6. For time-only columns: use HH:MM:SS
7. No null values
8. No additional text or markdown"""

    prompt = f"{sysp}\n\nPrompt: {text_sample}"
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)

    # Process the generated data
    generated_rows = []
    for line in content.strip().split('\n'):
        parts = [cell.strip() for cell in line.split('~')]
        if len(parts) == len(column_names):
            generated_rows.append(parts)

    df = pd.DataFrame(generated_rows, columns=column_names)

    # Convert time columns to proper types
    for col, config in time_columns.items():
        if col in df.columns:
            col_type = config.get("type", "datetime")
            if col_type == "date":
                df[col] = pd.to_datetime(df[col]).dt.date
            elif col_type == "time":
                df[col] = pd.to_datetime(df[col]).dt.time
            else:  # datetime/timestamp
                df[col] = pd.to_datetime(df[col])

    return df


# ----------------- Step 3: Enhanced Extrapolation ---------------------
def extrapolate_from_seed(seed_df: pd.DataFrame, target_rows: int, time_columns: Dict = {}) -> pd.DataFrame:
    time_cols = [col for col in time_columns.keys() if col in seed_df.columns]
    non_time_cols = [col for col in seed_df.columns if col not in time_cols]

    if not time_cols:
        factor = (target_rows + len(seed_df) - 1) // len(seed_df)
        repeated_df = pd.concat([seed_df] * factor, ignore_index=True)
        return resample(repeated_df, n_samples=target_rows, random_state=42)

    result_df = pd.DataFrame()

    for col in time_cols:
        config = time_columns[col]
        if "start" in config and "frequency" in config:
            try:
                new_times = pd.date_range(
                    start=config["start"],
                    periods=target_rows,
                    freq=config["frequency"]
                )
            except Exception as e:
                raise ValueError(f"Error generating time range for column '{col}': {e}")

            col_type = config.get("type", "datetime")
            if col_type == "date":
                result_df[col] = new_times.date
            elif col_type == "time":
                result_df[col] = new_times.time
            else:
                result_df[col] = new_times

    for col in non_time_cols:
        factor = (target_rows + len(seed_df) - 1) // len(seed_df)
        repeated = pd.concat([seed_df[col]] * factor, ignore_index=True)
        sampled = resample(repeated, n_samples=target_rows, random_state=42)
        sampled.index = range(target_rows)  # ensure no duplicate index
        result_df[col] = sampled

    result_df.reset_index(drop=True, inplace=True)
    return result_df


# ----------------- Step 4: Robust Final Generator ---------------------
def generate_data_from_text(prompt: str) -> Tuple[str, pd.DataFrame]:
    try:
        # Parse metadata with enhanced parser
        meta = parse_prompt_metadata(prompt)
        print("Parsed Metadata:", meta)

        # Validate
        if not meta.get("columns"):
            raise ValueError("No columns specified in prompt")

        num_rows = meta.get("num_rows", 100)
        time_columns = meta.get("time_columns", {})

        # Generate seed data with time column awareness
        seed_limit = min(150, num_rows)  # Don't generate more seed than needed
        df_seed = generate_seed_data(
            meta["columns"],
            seed_limit,
            prompt,
            time_columns
        )

        # Extrapolate with time series support
        df_final = extrapolate_from_seed(
            df_seed,
            num_rows,
            time_columns
        )

        # Ensure proper column ordering (time columns first)
        time_cols = [col for col in time_columns.keys() if col in df_final.columns]
        other_cols = [col for col in df_final.columns if col not in time_cols]
        df_final = df_final[time_cols + other_cols]

        # Save to Excel
        file_path = "data_output.xlsx"
        df_final.to_excel(file_path, index=False)

        return file_path, df_final

    except Exception as e:
        error_msg = f"Error generating data: {str(e)}"
        print(error_msg)
        return error_msg, pd.DataFrame()

# ----------------------------------------------------------------------------------------------------------------
# For DataScout with PDF Generation
# Data Extraction Tools for PDF
import re
from typing import Optional


# --- Prompt Parsing for PDF (LLM-based) ---
def extract_pdf_prompt_semantics(user_prompt: str) -> Tuple[Optional[int], List[str]]:
    llm = initialize_llm()

    system_instruction = (
        """You are a document request parser. Given a user prompt, extract:
        1. The number of pages requested (as an integer).
        2. The list of section names (as a list of title-case strings).

        Respond with JSON only in this format:
        {"num_pages": <int or null>, "sections": ["Section 1", "Section 2", ...]}.
        No commentary or markdown.
        """
    )

    response = llm.invoke(f"{system_instruction}\n\nPrompt: {user_prompt}")
    content = response.content if hasattr(response, 'content') else str(response)

    try:
        parsed = json.loads(content)
        return parsed.get("num_pages"), parsed.get("sections", [])
    except Exception:
        return None, []


def parse_llm_json_response(response_text: str) -> Optional[dict]:
    if not response_text.strip():
        return None
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except Exception as e:
        print(f"[LLM Parse Error] {e} | Raw: {response_text}")
        return None


# ----------------------------------------
# PDF Generator Core

def generate_structured_content(text_sample: str, sections: List[str], num_pages: int = 15) -> Dict[str, Any]:
    assert isinstance(text_sample, str), "text_sample must be a string"
    assert isinstance(sections, list) and all(
        isinstance(s, str) for s in sections), "sections must be a list of strings"

    llm = initialize_llm()

    try:
        joined_sections = ', '.join(sections)
        analysis_prompt = (
            f"Analyze this document request and return JSON:\n"
            f"{{\n"
            f"    \"title\": \"Document title\",\n"
            f"    \"style\": \"professional/academic\",\n"
            f"    \"sections\": [\n"
            f"        {{\"name\": \"Section name\", \"content_type\": \"text/mixed\", \"needs_visuals\": false}}\n"
            f"    ]\n"
            f"}}\n"
            f"Request: Create a {num_pages}-page document about '{text_sample}' with sections: [{joined_sections}]"
        )

        analysis = llm.invoke(analysis_prompt)
        structure = parse_llm_json_response(analysis.content)

        if not structure:
            print("Analysis failed, using defaults.")
            structure = {
                "title": "Generated Document",
                "style": "professional",
                "sections": [{"name": s, "content_type": "text"} for s in sections]
            }
    except Exception as e:
        print(f"Analysis exception, using defaults: {e}")
        structure = {
            "title": "Generated Document",
            "style": "professional",
            "sections": [{"name": s, "content_type": "text"} for s in sections]
        }

    output = {
        "title": structure["title"],
        "sections": []
    }

    for section in structure["sections"]:
        content_prompt = (
            f"Write professional content for the section: '{section['name']}'\n"
            f"Topic: {text_sample}\n"
            f"Format: Use main heading, and 2-3 subheadings, each with 1-2 detailed paragraphs\n"
            f"Tone: Professional and well-structured\n"
            f"Return as JSON: {{\"heading\": \"...\", \"subsections\": [{{\"subheading\": \"...\", \"content\": \"...\"}}]}}"
        )

        response = llm.invoke(content_prompt)
        structured_section = parse_llm_json_response(response.content)

        if structured_section:
            output["sections"].append(structured_section)
        else:
            print(f"Failed to parse section '{section['name']}'")
            print(f"Raw response: {response.content}")

    return output


# ----------------------------------------------------------------------------------------------------------------
# Tool Wrapping for LangChain For PDF Generation
def pdf_generator_tool(prompt: str, sections: List[str] = None, number_of_pages: int = None) -> Dict[str, Any]:
    if not sections or not isinstance(sections, list):
        sections = ["Introduction", "Content", "Conclusion"]
    if not number_of_pages or number_of_pages <= 0:
        number_of_pages = 1
    return generate_structured_content(prompt, sections, number_of_pages)


def extract_sections_tool(prompt: str) -> List[str]:
    _, sections = extract_pdf_prompt_semantics(prompt)
    return sections


def extract_num_pages_tool(prompt: str) -> int:
    num_pages, _ = extract_pdf_prompt_semantics(prompt)
    return num_pages if num_pages else 1


# -----------------------------------------------------------------------------------------------------------------
def excel_generator_tool(prompt: str) -> tuple[str, DataFrame]:

    return generate_data_from_text(prompt)


# -----------------------------------------------------------------------------------------------------------------
# Agent Setup
def DataScout_agent():
    llm = initialize_llm()
    tools = [
        Tool(
            func=excel_generator_tool,
            name="GenerateExcelFromPrompt",
            description=(
                "Generate a synthetic Excel file from a natural language prompt. "
                "The tool automatically detects whether the data should be time-series or not based on the prompt. "
                "It supports fields like number of records, column names, duration, start time, and frequency (e.g., per minute, monthly, per second). "
                "If no time-series context is provided, it generates generic tabular data."
            ),
            return_direct=True
        )]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


# -----------------------------------------------------------------------------------------------------------------
# Agent Setup for PDF Generation

def DataScout_agent_with_pdf():
    llm = initialize_llm()
    tools = [
        Tool(
            name="ExtractPageCount",
            func=extract_num_pages_tool,
            description="Extracts number of pages from the user's prompt."
        ),
        Tool(
            name="ExtractSectionNames",
            func=extract_sections_tool,
            description="Extracts section names from the user's prompt."
        ),
        StructuredTool.from_function(
            func=pdf_generator_tool,
            name="GeneratePDFFromPrompt",
            description="Generates structured PDF content from the prompt, section list, and number of pages.",
            return_direct=True
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

# pip install langchain-openai pandas openpyxl reportlab


# Testing the pipeline
# if __name__ == "__main__":
#     test_prompt = "Generate a table with 25 rows of realistic synthetic data with field names: product_id, product_name, category, current_stock_quantity, supplier_name, supplier_contact,units_sold,Date"
#     try:
#         agent = DataScout_agent()
#         response = agent.invoke(test_prompt)
#         print(f"Response: {response}")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     test_prompt = "Create a pdf with 3 pages about Artificial Intelligence with sections: Introduction, Methodology, Conclusion in 500 words per each page"
#     try:
#         agent = DataScout_agent_with_pdf()
#         response = agent.invoke(test_prompt)
#         print(f"Response: {response}")
#     except Exception as e:
#         print(f"❌ Error: {e}")
