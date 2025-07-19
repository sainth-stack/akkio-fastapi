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
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=500
    )


# -----------------------------------------------------------------------------------------------------------------
# For DataScout with Excel Generation
# ----------------- Step 1: Parse Prompt ---------------------
def parse_prompt_metadata(prompt: str) -> Dict:
    llm = initialize_llm()

    system_instruction = """
You are a data prompt parser for synthetic time-series generation. Given a natural language prompt, extract:

{
  "num_rows": <int or null>,  # optional if date range is given
  "columns": ["col1", "col2", ...],
  "start_time": "<YYYY-MM-DD HH:MM:SS>",  # always return in this format if present
  "frequency": "<Pandas-style freq string like 1min, 5S, 1MS, 1YS>",
}

Respond strictly with a JSON object and no extra text.
"""

    try:
        response = llm.invoke(f"{system_instruction}\n\nPrompt: {prompt}")
        content = response.content if hasattr(response, 'content') else str(response)

        print("LLM Raw Response:", content)  # For debugging

        parsed = json.loads(content) if isinstance(content, str) else content
        print("parsed_data_is,,,,,,,,,,,,,,,,,,,,,,,",parsed)
        # Validate keys
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a dictionary.")
        if "columns" not in parsed or not isinstance(parsed["columns"], list):
            raise ValueError("Missing or invalid 'columns' field.")

        return parsed
    except Exception as e:
        raise ValueError(f"Failed to parse prompt metadata: {e}")


# ----------------- Step 2: Generate Seed Data ---------------------
def generate_seed_data(column_names: List[str], seed_limit: int, text_sample: str) -> pd.DataFrame:
    llm = initialize_llm()

    sysp = """You are a data generator. Follow these rules:
1. Generate only the requested data format
2. No additional commentary
3. No markdown or code fences
4. No null values generation
5. Strictly follow the output format"""

    column_names_str = ", ".join(column_names)
    generated_rows = []

    prompt = (
        f"{sysp}\n\n"
        f"Description: '{text_sample}'\n"
        f"Generate {seed_limit} rows of synthetic data with columns: {column_names_str}.\n"
        f"Tilde-separated only. No column headers or extra text."
    )

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)

    for line in content.strip().split('\n'):
        parts = [cell.strip() for cell in line.split('~')]
        if len(parts) == len(column_names):
            generated_rows.append(parts)

    return pd.DataFrame(generated_rows, columns=column_names)


# ----------------- Step 3: Extrapolate to Full Dataset ---------------------
def extrapolate_from_seed(seed_df: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    factor = (target_rows + len(seed_df) - 1) // len(seed_df)
    repeated_df = pd.concat([seed_df] * factor, ignore_index=True)
    extrapolated_df = resample(repeated_df, n_samples=target_rows, random_state=42)
    extrapolated_df.reset_index(drop=True, inplace=True)
    return extrapolated_df


# ----------------- Step 4: Generate Final Data ---------------------
def generate_data_from_text(prompt: str) -> Tuple[str, pd.DataFrame]:
    try:
        # Parse metadata from prompt
        meta = parse_prompt_metadata(prompt)
        print(meta)

        num_rows = meta.get("num_rows")
        columns = meta.get("columns", [])
        frequency = meta.get("frequency")
        start_time_str = meta.get("start_time")

        if not columns:
            raise ValueError("No columns extracted from prompt.")

        # Use current time if not provided
        if start_time_str:
            start_time = pd.to_datetime(start_time_str)
        else:
            start_time = pd.Timestamp.now().floor('s')

        # Generate seed and final data
        seed_limit = 150
        df_seed = generate_seed_data(columns, seed_limit, prompt)
        df_final = extrapolate_from_seed(df_seed, num_rows)

        if 'timestamp' in [col.lower() for col in columns] and frequency:
            df_final['timestamp'] = pd.date_range(start=start_time, periods=num_rows, freq=frequency)
            cols = df_final.columns.tolist()
            cols = ['timestamp'] + [col for col in cols if col.lower() != 'timestamp']
            df_final = df_final[cols]

        file_path = "data_output.xlsx"
        df_final.to_excel(file_path, index=False)
        return file_path, df_final

    except Exception as e:
        # Return error message as string, second return is an empty DataFrame
        return f"Error: {str(e)}", pd.DataFrame()


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
    num_rows, columns = parse_prompt_metadata(prompt)
    if not columns:
        raise ValueError("Could not extract column names from prompt.")
    if not num_rows or num_rows <= 0:
        raise ValueError("Could not extract valid number of rows from prompt.")

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
