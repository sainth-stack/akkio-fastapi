import ast
import asyncio
import base64
import io
import os
import re
import smtplib
import sys
import tempfile
import traceback
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from dateutil.parser import parse
import boto3
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
from fastapi.responses import StreamingResponse
import dateutil.parser
import markdown
import plotly.graph_objects as go
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi import UploadFile, File, Form, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from starlette.responses import HTMLResponse, JSONResponse
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from img_datascout import ImageGen_agent
from datascout import DataScout_agent, extract_sections_tool, extract_num_pages_tool, \
    pdf_generator_tool, initialize_llm
from database import PostgresDatabase
from synthetic_data_function import generate_synthetic_data
from models import DBConnectionRequest, GenAIBotRequest, ModelRequest
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from plotly.graph_objs import Figure
import plotly as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import shutil
from sla_apis import sla_router
# Calculate comprehensive metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, \
    mean_squared_error, r2_score
import threading
import uuid
from PyPDF2 import PdfReader
from datetime import datetime, date, time
from decimal import Decimal

load_dotenv()

app = FastAPI()

global connection_obj
# Global variables for chat memory management
CHAT_MEMORY: Dict[str, list] = {}  # In-memory store; replace as needed
CHAT_MEMORY_LOCK = asyncio.Lock()

# Enable CORS if needed (similar to Django's csrf_exempt)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
db = PostgresDatabase()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.include_router(sla_router)


# 1.File upload only-------- It is  useful for uploading the file
@app.post("/api/upload_only")
async def upload_only(
        mail: str = Form(...),
        file: UploadFile = File(...)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        file_name = file.filename
        file_extension = os.path.splitext(file_name)[1].lower()

        # Read file content into a DataFrame
        content = await file.read()
        if file_extension == ".csv":
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        elif file_extension in [".xls", ".xlsx"]:
            # Save to temp file because read_excel reads from file path
            temp_path = f"temp{file_extension}"
            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)
            df = pd.read_excel(temp_path)
            os.remove(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV or Excel allowed")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data")

        # Insert or update in database ONLY (no local save)
        results = db.insert_or_update(mail, df, file_name)

        return JSONResponse(content={
            "message": "File uploaded and data saved to database successfully",
            "db_insert_result": results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# 2.Get user data based on the mail----------get the table data of the user based on the email-------workspace
@app.post("/api/get_user_data")
async def get_user_data(email: str = Form(...)):
    try:
        # Get user tables from database
        table_info = db.get_user_tables(email)
        print(table_info)  # Debug print
        # Return as JSON response
        return {"result": table_info}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving user data: {str(e)}"
        )


@app.post("/api/tabledata")
async def read_data(tablename: str = Form(...)):
    try:
        df = db.get_table_data(tablename)
        if df.empty:
            return JSONResponse(content={"detail": "Table is empty or not found"}, status_code=404)

        # Ultra-safe conversion using pandas built-in JSON handling
        result = ultra_safe_conversion(df)
        
        # Save CSV files (in background)
        threading.Thread(target=save_csv_files, args=(df, tablename)).start()
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            content={"detail": f"Error processing table data: {str(e)}"}, 
            status_code=500
        )


def ultra_safe_conversion(df):
    """
    Uses pandas' built-in JSON conversion which handles most edge cases
    """
    if df.empty:
        return []
    
    try:
        # Let pandas handle the JSON conversion with proper date formatting
        json_str = df.to_json(orient='records', date_format='iso', default_handler=str)
        return json.loads(json_str)
    except Exception as e:
        # Fallback to manual conversion
        return str(e)
    

def save_csv_files(df, tablename):
    """Save CSV files in background"""
    try:
        # Save main CSV
        df.to_csv('data.csv', index=False)
        
        # Create uploads directory and save table-specific CSV
        os.makedirs("uploads", exist_ok=True)
        df.to_csv(os.path.join("uploads", f"{tablename.lower()}.csv"), index=False)
    except Exception as e:
        print(f"Warning: Could not save CSV files: {e}")


# 3.Deleting the user-specific list of tables-----------------Deleting the list of tables corresponding to the specific user
@app.post("/api/delete_selected_tables")
async def delete_selected_tables_by_name(
        email: str = Form(...),
        table_names: List[str] = Form(...)
):
    try:
        print("Parsing request body for table deletion.")  # Debug statement
        print(f"Received email: {email}, table names: {table_names}")  # Debug statement

        if not email or not table_names:
            print("Missing 'email' or 'table_names' in the request.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'email' and 'table_names' are required"
            )

        print(f"Calling delete_selected_user_tables_by_name with email: {email} and table_names: {table_names}")
        deletion_status = db.delete_tables_data(email, table_names)

        if deletion_status:
            print(f"Deleted {len(table_names)} table(s) for email '{email}'.")
            return JSONResponse(
                content={
                    "message": f"{len(table_names)} table(s) associated with email '{email}' have been deleted."
                },
                status_code=status.HTTP_200_OK
            )
        else:
            print(f"No matching tables found for email '{email}' or the provided table names: {table_names}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No matching tables found for email '{email}' or the provided table names."
            )

    except Exception as e:
        print(f"Exception occurred while deleting selected tables: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )


# 4.Get flespi data---------------------------------it is for the flespi data---------- total 4(existing)
@app.post("/api/download_flespi_data")
async def download_flespi_data(
        flespi_URL: str = Form(...),
        flespi_token: str = Form(...)
) -> JSONResponse:
    try:
        # Get current time in IST
        current_datetime = datetime.now(tz=ZoneInfo('Asia/Kolkata'))

        # Calculate start time (1 week ago)
        start_of_day = (current_datetime - timedelta(weeks=1)).replace(
            hour=current_datetime.hour,
            minute=current_datetime.minute,
            second=0,
            microsecond=0
        )

        # Make API request to Flespi
        response = requests.get(
            f'{flespi_URL}?data=%7B%22from%22%3A{start_of_day.timestamp()}%2C%22to%22%3A{datetime.now().timestamp()}%7D',
            headers={
                'Authorization': f'FlespiToken {flespi_token}'
            }
        )
        response.raise_for_status()  # Raises exception for 4XX/5XX responses

        # Process the data
        multi_data = response.json()['result']
        multi_data = pre_process_multi_data(multi_data)
        multi_data = convert_to_hourly(multi_data)

        return JSONResponse(
            content=multi_data.to_dict(orient="records"),
            status_code=200
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,  # Bad Gateway
            detail=f"Flespi API request failed: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=422,  # Unprocessable Entity
            detail=f"Invalid Flespi API response format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing Flespi data: {str(e)}"
        )


# Preprocess multi data
def pre_process_multi_data(multi_data):
    for idx, record in enumerate(multi_data):
        multi_data[idx].update({
            "timestamp": datetime.fromtimestamp(multi_data[idx]["timestamp"],
                                                tz=ZoneInfo('Asia/Kolkata')).strftime("%Y-%m-%d %H-%M-%S"),
            "server.timestamp": datetime.fromtimestamp(multi_data[idx]["server.timestamp"],
                                                       tz=ZoneInfo('Asia/Kolkata')).strftime(
                "%Y-%m-%d %H-%M-%S")
        })
    return multi_data


# conversion of data to hourly.
def convert_to_hourly(data):
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H-%M-%S')

    # Extract the hour for grouping
    df['hour'] = df['timestamp'].dt.floor('H')

    # Define columns for aggregation
    value_columns = ['Current', 'Humidity', 'Power', 'Temperature', 'Voltage']

    # Flatten nested dictionaries for easier aggregation
    for col in value_columns:
        df[f'{col}_value'] = df[col].apply(lambda x: x['value'])

    # Aggregate values by hour
    hourly_data = df.groupby('hour').agg(
        Current_avg=('Current_value', 'mean'),
        Humidity_avg=('Humidity_value', 'mean'),
        Power_avg=('Power_value', 'mean'),
        Temperature_avg=('Temperature_value', 'mean'),
        Voltage_avg=('Voltage_value', 'mean'),
    ).reset_index()

    hourly_data['hour'] = pd.to_datetime(hourly_data['hour'], unit='ms')

    # Now, if you want a specific format (e.g., 'YYYY-MM-DD HH:MM:SS')
    hourly_data['hour'] = hourly_data['hour'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return hourly_data


# 5.Dashboard apis--------------------Dashboard related api for generating the dynamic graphs---------- 5. Have to modify
# Dashboard configuration
CHARTS_DIR = "generated_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Fixed filenames for the charts
FIXED_CHART_FILENAMES = [
    "chart_1.json",
    "chart_2.json",
    "chart_3.json",
    "chart_4.json",
    "chart_5.json",
    "chart_6.json"
]


@app.post("/api/dashboard")
async def gen_plotly_response() -> JSONResponse:
    try:
        # Load and process data
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)

        # Clean column names
        df.columns = df.columns.str.strip()

        num_plots = 6  # Number of plots to generate per topic
        basic_plots=3
        file_path = csv_file_path
        sample_data = df.head(10).to_string()
        data_types_info = df.dtypes.to_string()
        chart_responses = []

        # Initialize all chart files as empty
        for filename in FIXED_CHART_FILENAMES:
            chart_path = os.path.join(CHARTS_DIR, filename)
            with open(chart_path, "w") as f:
                json.dump({}, f)

        # Process each topic to generate charts
        prompt_eng = (f"""
                        You are a data visualization expert and a Python Plotly developer.For Each request you have to generate the graphs dynamically based on the dataset given.

                        I will provide you with a sample dataset.

                        Your task is to:
                        1. Analyze the dataset and identify the top {num_plots} most insightful charts (e.g., trends, distributions, correlations, anomalies).
                        2.You have to generate {basic_plots} basic plots and {num_plots-basic_plots} advanced plots.
                        3. Consider the data source as: {file_path}
                        4. For each chart:
                           - Use a short, meaningful chart title (as the dictionary key).
                           - Write a brief insight about the chart as a Python comment (# insight: ...).
                           - Generate clean Python code that:
                             a. Creates the Plotly chart using the dataset,
                             b. Converts the figure to JSON using fig.to_json(),
                             c. Saves it in a dictionary using chart_dict[<chart_title>] = {{'plot_data': ..., 'description': ...}}
                             d. Wraps the chart generation and JSON conversion in a try-except block using except Exception as e: (capital E).


                        Instructions:
                        - Return *only valid Python code. Do **not* use markdown or bullet points.
                        - Begin with any required imports and initialization of chart_dict.
                        - - Do not use except exception as e:. It is incorrect Python. Always use except Exception as e: (capital E). Any other form is invalid and will cause a runtime error.
                        - All explanations must be in valid Python comments (# ...)
                        - Do not add any extra text outside Python code.
                        - Use a diverse range of charts like: line, bar, scatter, pie, box, heatmap, area, violin, Scatter3d, facet, or animated plots.
                        - Use *aggregations* like .groupby(...).mean(), .count(), .sum() where helpful.
                        - - Apply *filters* when helpful, such as:
                          - Top N categories by value or count,
                          - Recent date ranges,
                          - Removal of nulls or extreme outliers.
                          - Top 5 categories by frequency or value

                        - Explore *advanced Plotly features*, such as:
                          - facet_row, facet_col for comparison grids,
                          - multi-series (e.g. line or scatter with color=column),
                          - combo charts (e.g., bar + line together),
                          - rolling averages or moving means,
                          - violin plots to show distributions,
                          - 3D scatter plots (px.scatter_3d) where 3 numeric dimensions exist,
                          - animations (animation_frame, animation_group) if time-based trends are useful.
                        - Aim for *high-value insights*, like:
                          - Seasonality or cyclic patterns,
                          - Equipment performing worse than average,
                          - Category-wise contribution to deficit or emissions,
                          - Any shocking anomalies or unexpected gaps.

                        - Use this preview of the dataset:
                            {sample_data}

                        - Column names and data types:
                            {data_types_info}

                        IMPORTANT:
                            - If you ever write except exception as e, your answer is wrong and must be corrected before use.
                            - Ensure column names are used *exactly* as they appear in the dataset. *Do not change the case* or formatting of column names.
                            - Always use df.columns = df.columns.str.strip() after loading the dataset to handle unwanted spaces.
                            - After reading the CSV:
                            - Use df.columns = df.columns.str.strip() to remove leading/trailing spaces from column names.
                            - For datetime columns:
                                - Strip values using df[col] = df[col].astype(str).str.strip()
                                - Convert to datetime using pd.to_datetime(df[col], errors='coerce', utc=True)
                                - Drop rows where datetime conversion failed using df.dropna(subset=[col], inplace=True)
                            - Before using .dt, ensure the column is of datetime type using pd.to_datetime().
                            - The basic plots should be simple and straightforward which can be easily understandable by the user, while the advanced plots should be more complex and insightful.
                        """
                      )

        try:
            # Generate code using AI
            generated_code = generate_code4(prompt_eng)
            print(f"Generated code:\n{generated_code}")

            if 'import' not in generated_code.lower():
                raise ValueError("Invalid AI response - missing imports")

            # Execute the generated code
            namespace = {'pd': pd, 'px': px, 'go': go, 'df': df}
            exec(generated_code, namespace)

            # Get the chart dictionary from executed code
            chart_dict = namespace.get("chart_dict", {})

            if not chart_dict:
                raise ValueError("No charts generated - chart_dict is empty")

            # Process each generated chart
            chart_keys = list(chart_dict.keys())[:num_plots]  # Ensure we only take 6 charts

            for i, chart_key in enumerate(chart_keys):
                try:
                    chart_info = chart_dict[chart_key]
                    chart_data = chart_info.get("plot_data")
                    description = chart_info.get("description", "")

                    if not chart_data:
                        raise ValueError(f"No plot data for chart: {chart_key}")

                    # Make data serializable
                    chart_data_serializable = make_serializable(chart_data)
                    chart_filename = FIXED_CHART_FILENAMES[i]
                    chart_path = os.path.join(CHARTS_DIR, chart_filename)

                    # Save individual chart file
                    with open(chart_path, "w", encoding="utf-8") as f:
                        json.dump(chart_data_serializable, f, indent=2, ensure_ascii=False)

                    chart_responses.append({
                        "timestamp": datetime.now().isoformat(),
                        "chart_title": chart_key,
                        "chart_data": chart_data_serializable,
                        "chart_file": chart_filename,
                        "description": description,
                        "status": "success"
                    })

                except Exception as e:
                    print(f"Error processing chart '{chart_key}': {str(e)}")
                    chart_filename = FIXED_CHART_FILENAMES[i] if i < len(
                        FIXED_CHART_FILENAMES) else f"chart_{i + 1}.json"
                    chart_responses.append({
                        "chart_file": chart_filename,
                        "chart_title": chart_key if 'chart_key' in locals() else f"Chart {i + 1}",
                        "status": "failed",
                        "error": str(e)
                    })

        except Exception as e:
            print(f"Error in chart generation: {str(e)}")
            # Create fallback empty responses
            for i in range(num_plots):
                chart_responses.append({
                    "chart_file": FIXED_CHART_FILENAMES[i],
                    "status": "failed",
                    "error": str(e)
                })

            # Prepare final response
        success_count = len([c for c in chart_responses if c.get("status") == "success"])
        response_data = {
            "message": "Chart generation completed",
            "generated_charts": success_count,
            "total_charts": num_plots,
            "chart_files": FIXED_CHART_FILENAMES[:num_plots],
            "charts": chart_responses
        }

        return JSONResponse(content=response_data, status_code=200)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Data file not found"
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chart generation failed: {str(e)}"
        )


# Function to generate code from OpenAI API
def generate_code4(prompt_eng):
    """Generate Python code for creating Plotly charts using AI"""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # Updated model name
        messages=[
            {"role": "system", "content": """
            You are VizCopilot, an expert Python data visualization assistant having 20+ years of experience in specialising in Plotly.

            Your core responsibilities:
            - Generate complete, executable Python code for data visualization
            - Create diverse, insightful charts that reveal different data patterns
            - Use both plotly.express (px) and plotly.graph_objects (go) appropriately
            - Apply data analysis techniques: grouping, filtering, aggregation, transformation

            Code Generation Standards:
            - Always start with necessary imports: pandas, plotly.express, plotly.graph_objects
            - Generate ONLY valid Python code (no markdown, no text outside comments)
            - Use proper exception handling: 'except Exception as e:' (capital E)
            - Create complete, working code blocks with proper indentation
            - Include meaningful chart titles and descriptions
            - Apply best practices for data visualization

            Chart Diversity Requirements:
            - Create different chart types for comprehensive data exploration
            - Use various Plotly features: faceting, animations, multi-series, custom styling
            - Focus on actionable insights: trends, outliers, distributions, correlations
            - Apply appropriate data transformations and filtering

            Technical Requirements:
            - Return charts in a dictionary format: chart_dict[title] = {"plot_data": fig.to_plotly_json(), "description": "insight"}
            - Handle edge cases and data quality issues
            - Use exact column names from provided dataset
            - Ensure all generated code is immediately executable
            - Validate data types and handle datetime conversions properly

            Quality Assurance:
            - Every chart must provide unique insights
            - Code must be syntactically correct and complete
            - No placeholder functions or incomplete logic
            - Proper error handling for robustness
                """},
            {"role": "user", "content": prompt_eng}
        ],
        temperature=0.7,  # Add some randomness for variety in chart generation
        max_tokens=4000
    )

    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message

    print(f"AI Response: {all_text}")

    # Extract Python code from response
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        if code_end == -1:
            code_end = len(all_text)
        code = all_text[code_start:code_end].strip()
    elif "```" in all_text:
        code_start = all_text.find("```") + 3
        code_end = all_text.find("```", code_start)
        if code_end == -1:
            code_end = len(all_text)
        code = all_text[code_start:code_end].strip()
    else:
        code = all_text.strip()

    return code


def make_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable format"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    elif str(type(obj)).startswith("<class 'numpy."):
        return float(obj) if 'float' in str(type(obj)) else int(obj)
    else:
        return obj


# 6..Get summary api-------------getting the summary of the above generated graphs-----------6-----pending.
# In-memory cache for summaries
SUMMARY_CACHE: Dict[str, str] = {}


@app.post("/api/analyze_chart")
async def analyze_chart(
        chart_id: str = Form(...),
        question: Optional[str] = Form(None)
) -> JSONResponse:
    try:
        # 1. Validate Chart ID
        try:
            chart_num = int(chart_id)
            if not 1 <= chart_num <= 6:
                raise ValueError
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid chart ID. Must be an integer between 1-6."
            )

        # 2. Locate and load chart data from file
        filename = f"chart_{chart_id}.json"
        chart_path = os.path.join(CHARTS_DIR, filename)

        if not os.path.exists(chart_path):
            raise HTTPException(status_code=404, detail=f"Chart file '{filename}' not found.")

        with open(chart_path, "r", encoding="utf-8") as f:
            chart_json = json.load(f)

        # 3. Determine action: Summarize or Answer Question
        # --- CASE 1: No question provided -> Generate a detailed summary ---
        if not question or not question.strip():
            # Check cache for existing summary
            if chart_id in SUMMARY_CACHE:
                return JSONResponse(
                    content={
                        "chart_id": chart_id,
                        "response": markdown_to_html(SUMMARY_CACHE[chart_id]),
                        "type": "summary",
                        "cached": True,
                    },
                    status_code=200,
                )

            # Generate and cache a new summary
            prompt = (
                f"You are a data analyst AI. A user selected a chart represented by this Plotly JSON:\n{json.dumps(chart_json)}\n\n"
                f"Analyze and summarize only the insights, patterns, and trends that are directly visible in the chart.\n\n"
                f"Follow this output structure with exactly 4 key observations:\n\n"
                f"Core Insight\n"
                f"• Start with the primary finding from the graph. Bold important terms.\n\n"
                f"Pattern Analysis\n"
                f"• Describe distribution patterns, outliers, clusters, or trends.\n\n"
                f"Give the response in markdown format with proper headings in 'h4' and bullet points."
            )
            summary = generate_text(prompt)
            SUMMARY_CACHE[chart_id] = summary

            return JSONResponse(
                content={
                    "chart_id": chart_id,
                    "response": markdown_to_html(summary),
                    "type": "summary",
                    "cached": False,
                },
                status_code=200,
            )

        # --- CASE 2: Question provided -> Generate a targeted answer ---
        else:
            prompt = (
                f"You are a data analyst AI. A user is asking a question about a chart represented by this Plotly JSON:\n{json.dumps(chart_json)}\n\n"
                f"User Question: '{question}'\n\n"
                f"Follow this output structure with exactly 4 key observations:\n\n"
                f"if the user asks about Core Insight\n"
                f"• Start with the primary finding from the graph. Bold important terms.\n\n"
                f" if the user asks Pattern Analysis\n"
                f"• Describe distribution patterns, outliers, clusters, or trends.\n\n"
                f" if the user asks about Business Context\n"
                f"• Explain what real-world behavior the graph appears to reflect.\n\n"
                f" if the user asks about Action Recommendations\n"
                f"Only describe what you observe. Do not invent data. Use the exact format shown above."
                f"Give the response in markdown format with proper headings in 'h4' format and bullet points."
            )
            answer = generate_text(prompt)

            return JSONResponse(
                content={
                    "chart_id": chart_id,
                    "question": question,
                    "response": markdown_to_html(answer),
                    "type": "answer"
                },
                status_code=200,
            )

    except HTTPException:
        raise  # Re-raise exceptions with specific HTTP status codes
    except Exception as e:
        # Catch-all for any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


def markdown_to_html(md_text):
    html_text = markdown.markdown(md_text)
    return html_text


def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system",
             "content": "You are a helpful data analyst that explains data visualizations and user queries and write insightful summary for the given data.Generate the answers in the plain format with the nice headings and all."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


# 7..Filling missing data------------------Evaluating the missed data in the dataframe----------- 7.
@app.post("/api/fill_missed_data")
async def missing_data() -> JSONResponse:
    try:
        # Load and validate data
        csv_file_path = 'data.csv'
        if not os.path.exists(csv_file_path):
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )

        df = pd.read_csv(csv_file_path)
        print("Original data preview:")
        print(df.head(5))

        # Process missing data
        new_df, html_df, summary = process_missing_data(df.copy())

        # Save processed data
        processed_path = os.path.join('uploads', 'processed_data.csv')
        new_df.to_csv(processed_path, index=False)
        new_df.to_csv("data.csv", index=False)

        # Save HTML representation
        with open(os.path.join('mvt_data.json'), 'w') as fp:
            json.dump({'data': html_df}, fp, indent=4)

        return JSONResponse(
            content={"df": html_df, "summary": summary},
            status_code=200
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Input CSV file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data processing failed: {str(e)}"
        )


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df, summary = handle_missing_data(df)
    return df, html_df, summary


def convert_to_datetime(df):
    """
    Converts object (string) columns containing dates to datetime format.
    """
    for col in df.columns:
        if df[col].dtype == "object":  # Process only string columns
            if df[col].str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", na=False).any():
                df[col] = df[col].apply(detect_and_parse_date)

    return df


import dateutil.parser


def detect_and_parse_date(value):
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return pd.NaT  # Handle missing values safely

    try:
        # Check if it's a date with hyphens or slashes
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", value):
            day_first = False  # Assume MM-DD-YYYY first

            # Check for an ambiguous case (day > 12) → Must be DD-MM-YYYY
            parts = re.split(r"[-/]", value)
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            if day > 12:
                day_first = True  # Switch to DD-MM-YYYY

            # Parse with detected format
            return dateutil.parser.parse(value, dayfirst=day_first)

        # Otherwise, use default dateutil parsing
        return dateutil.parser.parse(value)

    except ValueError:
        return pd.NaT  # Return NaT if parsing fails


from sklearn.impute import KNNImputer


def handle_missing_data(df):
    try:
        ignore_types = ['object', 'string', 'timedelta', 'complex']
        ignored_columns_info = {}

        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_time_cols = df.select_dtypes(include=['datetime64']).columns
        ignored_cols = df.select_dtypes(include=ignore_types).columns
        int_like_cols = [col for col in numeric_cols if is_integer_like(df[col])]

        for col in ignored_cols:
            ignored_columns_info[col] = f"Ignored because of optional data"

        # Impute numeric columns and track which cells were imputed
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols).round(2)

        for col in numeric_cols:
            if col in int_like_cols:
                imputed_numeric_df[col] = imputed_numeric_df[col].round().astype("Int64")

        # Mark imputed cells (True if the original cell was NaN)
        imputed_flags = df[numeric_cols].isnull()
        imputed_flags = imputed_flags.applymap(lambda x: x if x else False)

        # Update DataFrame with imputed values
        df[numeric_cols] = imputed_numeric_df

        for col in df.select_dtypes(include='category').columns:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col].fillna(mode_val, inplace=True)
                imputed_flags[col] = df[col].isnull()
        for col in df.select_dtypes(include='bool').columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode().iloc[0], inplace=True)

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col] = df[col].iloc[i - 1] + avg_diff
                    imputed_flags.loc[i, col] = True

            imputed_flags.fillna(False, inplace=True)

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []

        for _, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = {
                    "value": row[col].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[col], pd.Timestamp) else row[col],
                    "is_imputed": str(imputed_flags[col].get(_, False)) if col in imputed_flags else str(False)
                    # Check if cell was imputed
                }
            data.append(row_data)
        missing_values_summary = summarize_missing_values(imputed_flags)
        missing_values_summary["ignored_columns"] = ignored_columns_info
        return df, data, missing_values_summary
    except Exception as e:
        print(e)


def is_integer_like(series):
    return pd.api.types.is_numeric_dtype(series) and \
        series.dropna().apply(lambda x: float(x).is_integer()).all()


def summarize_missing_values(df):
    try:
        # 1. Total number of missing values
        total_missing = df.sum().sum()

        # 2. Columns with any missing values
        columns_with_missing = df.columns[df.any()].tolist()

        # 3. Count of missing values per column
        missing_count_per_column = df.sum()

        # 4. Percentage of missing values per column (optional)
        missing_percentage = df.mean() * 100

        # Final summary
        summary = {
            "total_missing_values": int(total_missing),
            "columns_with_missing": columns_with_missing,
            "missing_count_per_column": missing_count_per_column.to_dict(),
            "missing_percentage_per_column": missing_percentage.round(2).to_dict()
        }
        return summary
    except Exception as e:
        print(e)
        return {}


def serialize_datetime(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError("Type not serializable")


#Api for generating the column names for the model training
@app.get("/api/get_columns")
async def get_column_names() -> JSONResponse:
    try:
        # Read only the header to get column names
        df = pd.read_csv("data.csv")
        
        return JSONResponse(content={
            "status": "success",
            "columns": df.columns.tolist()
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="data.csv file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# 8.1 Models related apis:
@app.post("/api/models")
async def models(input: ModelRequest):
    try:
        # Load and clean data
        df = pd.read_csv('data.csv')
        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
        df.drop(single_value_columns, axis=1, inplace=True)

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        print(numeric_cols)
        if len(numeric_cols) < 1:
            raise HTTPException(400, "Dataset does not meet modeling requirements")

        # Handle RandomForest
        if input.model == 'RandomForest':
            stat, cols = random_forest(df, input.col)
            return {
                'columns': list(df.columns),
                'rf': True,
                'status': stat,
                'rf_cols': cols
            }

        # Handle ARIMA
        elif input.model == 'Arima':
            if not input.frequency or not input.tenure:
                raise HTTPException(400, "Missing frequency or tenure for ARIMA")
            stat, data, img_data = arima_train(df, input.col, {
                'time_unit': input.frequency,
                'forecast_horizon': input.tenure
            })
            return {
                'columns': list(df.columns),
                'status': stat,
                'arima': True,
                'path': str(img_data),
                'data': json.loads(data.to_json()),
                'description': generate_text_from_json(json.loads(data.to_json()))
            }

        # Handle unsupported models
        else:
            raise HTTPException(400, "Unsupported model type")

    except FileNotFoundError:
        raise HTTPException(404, "Data file not found")
    except Exception as e:
        raise HTTPException(500, str(e))


# Model predict for Random forest
@app.post("/api/model_predict")
async def model_predict(request: Request):
    try:
        form_data = await request.form()
        form_data = dict(form_data)

        # Validate form_name
        if form_data.get('form_name') != 'rf':
            raise HTTPException(
                status_code=400,
                detail="Invalid form type, expected 'rf'"
            )

        # Validate targetColumn
        targetcol = form_data.get('targetColumn')
        if not targetcol:
            raise HTTPException(
                status_code=400,
                detail="Target column (targetColumn) is required"
            )

        # Extract features (exclude form_name and targetColumn)
        features = {k: v for k, v in form_data.items() if k not in ['form_name', 'targetColumn']}
        df_predict = pd.DataFrame([features])

        # Load pipeline and deployment data paths
        model_dir = os.path.join("models", "rf", targetcol)
        pipeline_path = os.path.join(model_dir, "pipeline.pkl")
        deployment_path = os.path.join(model_dir, "deployment.json")

        if not os.path.exists(pipeline_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model pipeline not found for target column '{targetcol}'"
            )

        if not os.path.exists(deployment_path):
            raise HTTPException(
                status_code=404,
                detail=f"Model deployment metadata not found for target column '{targetcol}'"
            )

        # Load model pipeline
        loaded_pipeline = load_pipeline(pipeline_path)

        # Load deployment metadata
        with open(deployment_path, "r") as f:
            deployment_data = json.load(f)
        model_stats = deployment_data.get("stats", {})

        # Make prediction
        predictions = loaded_pipeline.predict(df_predict)

        # Try to get prediction confidence if available
        prediction_proba = None
        if hasattr(loaded_pipeline, 'predict_proba'):
            try:
                prediction_proba = loaded_pipeline.predict_proba(df_predict)
            except Exception as e:
                print(f"Warning: predict_proba failed: {e}")

        # Compute feature importance and impact
        feature_importance = get_feature_importance(loaded_pipeline, features.keys())
        feature_impact = calculate_feature_impact(loaded_pipeline, df_predict, features)

        response = {
            "prediction_result": {
                "predicted_value": round(float(predictions[0]), 2),
                "target_column": targetcol,
                "confidence": get_prediction_confidence(prediction_proba) if prediction_proba is not None else None
            },
            "model_performance": {
                "overall_accuracy": model_stats.get("accuracy", "N/A"),
                "performance_metrics": model_stats.get("metrics", {}),
                "baseline_comparison": model_stats.get("baseline_comparison", "N/A")
            },
            "feature_analysis": {
                "top_fields": feature_importance,
                "input_features": features,
                "feature_impact": feature_impact
            }
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
            headers={"X-Error-Details": str(e)}
        )


# 8.Genai bot plotly visualisation.-----------------Prediction and forecasting related api-------8
@app.post("/api/ai_bot")
async def gen_ai_bot(request_body: GenAIBotRequest):
    try:
        # Load and prepare data
        df = pd.read_csv('data.csv')
        metadata_str = ", ".join(df.columns.tolist())
        sample_data = df.head(2).to_dict(orient='records')

        prompt = request_body.prompt
        session_id = str(uuid.uuid4())

        # Store the new user message in memory
        async with CHAT_MEMORY_LOCK:  # <--- no error now!
            if session_id not in CHAT_MEMORY:
                CHAT_MEMORY[session_id] = []
            CHAT_MEMORY[session_id].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
        # Build LLM messages with full chat history
        messages = []
        for msg in CHAT_MEMORY[session_id]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Use messages as context for the LLM
        # Handle forecasting requests
        if 'forecast' in prompt.lower():
            data = extract_forecast_details_llm(prompt, df.columns)
            print("data printed")
            stat, data, img_data = arima_train(df, data['target_variable'], data)

            # Store bot response
            bot_content = json.dumps({'data': json.loads(data.to_json()), 'plot': make_serializable(img_data)})
            async with CHAT_MEMORY_LOCK:
                CHAT_MEMORY[session_id].append(
                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})

            return JSONResponse({
                'data': json.loads(data.to_json()),
                'plot': make_serializable(img_data),
                'session_id': session_id,
                'description': generate_text_from_json(json.loads(data.to_json()))
            })

        # Handle prediction requests
        elif 'predict' in prompt.lower():
            data = extract_forecast_details_rf(prompt, df.columns)

            if len(data.get('missing_columns', [])) > 0:
                bot_content = json.dumps({'text_pre_code_response': (
                    f'Prediction failed due to missing fields: {data.get("missing_columns")}. '
                    f'Please retry with all required inputs.')})
                async with CHAT_MEMORY_LOCK:
                    CHAT_MEMORY[session_id].append(
                        {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                return JSONResponse({
                    'text_pre_code_response': (
                        f'Prediction failed due to missing fields: {data.get("missing_columns")}. '
                        f'Please retry with all required inputs.'),
                    'session_id': session_id
                })

            model_path = os.path.join("models", "rf", data['target_column'])
            pipeline_path = os.path.join(model_path, "pipeline.pkl")
            deployment_path = os.path.join(model_path, "deployment.json")

            # Check and retrain model if needed
            if not os.path.exists(deployment_path) or not os.path.exists(pipeline_path):
                df = pd.read_csv('data.csv')
                model_stats = random_forest(df, data.get('target_column'))
            else:
                # Load existing model statistics
                with open(deployment_path, 'r') as f:
                    model_stats = json.load(f)

            # Make prediction
            df_predict = pd.DataFrame([data.get('features')])
            loaded_pipeline = load_pipeline(pipeline_path)
            predictions = loaded_pipeline.predict(df_predict)

            # Get prediction probability/confidence if available
            prediction_proba = None
            if hasattr(loaded_pipeline, 'predict_proba'):
                try:
                    prediction_proba = loaded_pipeline.predict_proba(df_predict)
                except:
                    pass

            # Calculate feature importance and top contributing fields
            feature_importance = get_feature_importance(loaded_pipeline, data.get('features').keys())

            bot_content = json.dumps({
                "prediction_result": {
                    "predicted_value": round(predictions[0], 2),
                    "target_column": data.get('target_column'),
                    "confidence": get_prediction_confidence(prediction_proba) if prediction_proba is not None else None
                },
                "model_performance": {
                    "overall_accuracy": model_stats.get('accuracy', 'N/A'),
                    "performance_metrics": model_stats.get('metrics', {}),
                    "baseline_comparison": model_stats.get('baseline_comparison', 'N/A')
                },
                "feature_analysis": {
                    "top_fields": feature_importance,
                    "input_features": data.get('features'),
                    "feature_impact": calculate_feature_impact(loaded_pipeline, df_predict, data.get('features'))
                },
                "text_pre_code_response": f"Predicted {data.get('target_column')} value is {round(predictions[0], 2)}"
            })
            async with CHAT_MEMORY_LOCK:
                CHAT_MEMORY[session_id].append(
                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})

            return JSONResponse({
                "prediction_result": {
                    "predicted_value": round(predictions[0], 2),
                    "target_column": data.get('target_column'),
                    "confidence": get_prediction_confidence(prediction_proba) if prediction_proba is not None else None
                },
                "model_performance": {
                    "overall_accuracy": model_stats.get('accuracy', 'N/A'),
                    "performance_metrics": model_stats.get('metrics', {}),
                    "baseline_comparison": model_stats.get('baseline_comparison', 'N/A')
                },
                "feature_analysis": {
                    "top_fields": feature_importance,
                    "input_features": data.get('features'),
                    "feature_impact": calculate_feature_impact(loaded_pipeline, df_predict, data.get('features'))
                },
                "text_pre_code_response": f"Predicted {data.get('target_column')} value is {round(predictions[0], 2)}",
                'session_id': session_id
            })

        # Handle general data analysis requests
        else:
            # Use full chat history as context for the LLM
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )

            pre_code_text, post_code_text, code = process_genai_response(response)
            result: Dict[str, Any] = {}
            result.update({
                'text_pre_code_response': pre_code_text,
                'code': code,
                'text_post_code_response': post_code_text
            })

            if 'import' in code:
                namespace = {}
                try:
                    exec(code, namespace)
                    result['text_output'] = namespace.get('text_output')

                    fig = namespace.get('fig')
                    if fig and isinstance(fig, Figure):
                        result['chart_response'] = make_serializable(fig.to_plotly_json())
                except Exception as e:
                    bot_content = json.dumps({'error': f"Code execution failed: {str(e)}"})
                    async with CHAT_MEMORY_LOCK:
                        CHAT_MEMORY[session_id].append(
                            {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})
                    raise HTTPException(
                        status_code=500,
                        detail=f"Code execution failed: {str(e)}"
                    )

            # Store bot response
            bot_content = json.dumps(result)
            async with CHAT_MEMORY_LOCK:
                CHAT_MEMORY[session_id].append(
                    {"role": "bot", "content": bot_content, "timestamp": datetime.now().isoformat()})

            result['session_id'] = session_id
            return JSONResponse(result)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Data file not found"
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Input CSV file is empty or corrupt"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


def generate_text_from_json(json_data: dict) -> str:
    description = "You are a helpful data analyst. Given the following analysis output, describe the results to the user in plain English within 2 or 3 lines only.."
    data_summary = json.dumps(json_data, indent=2)
    full_prompt = f"{description}\nHere is the analysis output:\n{data_summary}\n"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system",
             "content": "You are a helpful data analyst who explains model outputs, data visualizations, and user queries clearly and insightfully."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def load_pipeline(save_path="model_pipeline.pkl"):
    # Load the saved pipeline
    pipeline = joblib.load(save_path)
    print(f"Pipeline loaded from: {save_path}")
    return pipeline


def extract_forecast_details_llm(prompt, column_names):
    try:
        system_prompt = f""" You are an AI assistant that extracts forecast details from a user's prompt. Given a 
        natural language input and the following column names from the input data, return the following in JSON format:

            1. "target_variable" - The thing being forecasted (e.g., "sales", "revenue"). - If the target variable is 
            misspelled or ambiguous, try to match it to the closest column name from the list below. 2. 
            "forecast_horizon" - The number of time steps. 3. "time_unit" - The unit of time (days, months, years).

            Available column names: {', '.join(column_names)}

            Example Outputs:
            - Input: "Forecast the sales data for 5 years."
              Output: {{"target_variable": "sales", "forecast_horizon": 5, "time_unit": "years"}}

            - Input: "Can you predict electricity demand for the next 12 months?"
              Output: {{"target_variable": "electricity demand", "forecast_horizon": 12, "time_unit": "months"}}

            - Input: "I want to predict CO2 levels for 7 days."
              Output: {{"target_variable": "CO2 levels", "forecast_horizon": 7, "time_unit": "days"}}

            Ensure that the "target_variable" matches one of the available column names, even if the user misspells it.
            """
        forecast_details = ''
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Make it deterministic
        )
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            forecast_details += chunk_message
        print(forecast_details)

        return eval(forecast_details)

    except Exception as e:
        print(e)


def extract_forecast_details_rf(prompt, column_names):
    try:
        system_prompt = f"""
            You are an AI assistant that extracts machine learning input features and the target variable from a user's natural language prompt.

            You are provided a list of available column names: {', '.join(column_names)}.

            Your job is to:
            1. **Correct Spelling**: If any feature or target column is misspelled, match it to the closest name from the provided column list.
            2. **Extract Features**: Identify which features and their values are mentioned in the input.
            3. **Detect Missing Features**: If some required features are not mentioned, list them under "missing_columns".
            4. **Identify Target Column**: If the user specifies a column as the one to be predicted or forecasted, include it as "target_column".
            5. **Always Return All Three Fields**: Even if one or more are empty, the response **must always** contain "features", "missing_columns", and "target_column".

            ### Expected Output Formats:

            #### a) All features and target column provided:
            Input: "Predict CO2 level. The temperature is 25 and humidity is 45."
            Output:
            {{
              "features": {{
                "temperature": 25,
                "humidity": 45
              }},
              "missing_columns":[],
              "target_column": "CO2 level"
            }}

            #### b) Some features missing:
            Input: "I want to predict pressure. Set humidity to 50."
            Output:
            {{
              "features": {{
                "humidity": 50
              }},
              "missing_columns": ["temperature", "CO2 level"],
              "target_column": "pressure"
            }}

            #### c) Misspelled entries:
            Input: "Predict temprature using humdity = 60 and presure = 1000"
            Output:
            {{
              "features": {{
                "humidity": 60,
                "pressure": 1000
              }},
              "missing_columns": ["CO2 level"],
              "target_column": "temperature"
            }}

            ### Notes:
            - Always correct any misspelled column names to the closest match in the available list.
            - Use numeric types for numeric values, not strings.
            - If the target column is not explicitly provided, leave "target_column" as null or omit it.
        """

        predict_details = ''
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0  # Make it deterministic
        )
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            predict_details += chunk_message
        print(predict_details)

        return eval(predict_details)

    except Exception as e:
        print(e)


def process_genai_response(response):
    all_text = ""
    text_post_code = ''
    code_start = -1
    code_end = -1
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    text_pre_code = all_text[:code_start - 9]
    if code_start != -1:
        text_post_code = all_text[code_end:]
    return text_pre_code, text_post_code, code


def get_feature_importance(pipeline, feature_names):
    """Extract feature importance from the trained model"""
    try:
        # For RandomForest models
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importance = list(zip(feature_names, importances))
            # Sort by importance descending
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            # Format as percentage and return top fields
            top_fields = []
            for feature, importance in feature_importance[:5]:  # Top 5 features
                top_fields.append({
                    "field_name": feature,
                    "importance_percentage": round(importance * 100, 1),
                    "contribution": importance
                })
            return top_fields
    except:
        pass
    return []


def get_prediction_confidence(prediction_proba):
    """Calculate prediction confidence from probability scores"""
    if prediction_proba is not None and len(prediction_proba) > 0:
        max_proba = max(prediction_proba[0])
        return round(max_proba * 100, 1)
    return None


def calculate_feature_impact(pipeline, df_predict, features):
    """Calculate the impact of each feature on the prediction"""
    feature_impacts = {}
    base_prediction = pipeline.predict(df_predict)[0]

    for feature_name, feature_value in features.items():
        # Create a copy with this feature set to mean/mode
        df_modified = df_predict.copy()
        try:
            # For numerical features, use mean; for categorical, use most frequent
            if isinstance(feature_value, (int, float)):
                df_modified[feature_name] = df_modified[feature_name].mean() if not df_modified[
                    feature_name].isna().all() else 0
            else:
                df_modified[feature_name] = "baseline_value"

            modified_prediction = pipeline.predict(df_modified)[0]
            impact = abs(base_prediction - modified_prediction)
            impact_percentage = round((impact / abs(base_prediction) * 100), 1) if base_prediction != 0 else 0

            feature_impacts[feature_name] = {
                "impact_value": round(impact, 2),
                "impact_percentage": f"{impact_percentage}%",
                "direction": "positive" if base_prediction > modified_prediction else "negative"
            }
        except:
            feature_impacts[feature_name] = {"impact_value": 0, "impact_percentage": "0%", "direction": "neutral"}

    return feature_impacts


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):  # ✅ Fix for your issue
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def parse_mixed_dates(date_series):
    """Parse a series containing mixed date formats"""
    parsed_dates = []
    for date_str in date_series:
        try:
            # Try parsing with dateutil's flexible parser
            parsed_date = parse(str(date_str), dayfirst=False, yearfirst=False)
            parsed_dates.append(parsed_date)
        except Exception as e:
            print(f"Could not parse date: {date_str} - {e}")
            parsed_dates.append(pd.NaT)  # Not-a-Time for invalid dates

    return pd.to_datetime(parsed_dates)


def arima_train(data, target_col, bot_query=None):
    try:
        print('ArimaTrain')
        print("Column dtypes:\n", data.dtypes)

        # Identify date column
        date_column = None
        results = {}

        if not os.path.exists(os.path.join("models", 'Arima', target_col)):
            for col in data.columns:
                print(f"Checking column '{col}' for dates")
                if data.dtypes[col] == 'object':
                    try:
                        # First check if it's already datetime
                        if pd.api.types.is_datetime64_any_dtype(data[col]):
                            date_column = col
                            break

                        # Try parsing with mixed formats handler
                        print(f"Attempting to parse mixed formats in column '{col}'")
                        data[col] = parse_mixed_dates(data[col])

                        # Check if we successfully parsed any dates
                        if not data[col].isnull().all():
                            date_column = col
                            print(f"Successfully parsed datetime column: {col}")
                            break

                    except Exception as e:
                        print(f"Error parsing column '{col}': {e}")
                        continue

            if not date_column:
                raise ValueError("No datetime column could be parsed from the dataset")

            # Handle any remaining NaT values (invalid dates)
            if data[date_column].isnull().any():
                print(f"Warning: {data[date_column].isnull().sum()} invalid dates found")
                data = data.dropna(subset=[date_column])

            # Standardize the format and set as index
            print(f"Standardizing datetime format for column '{date_column}'")
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)

            # Sort by datetime index
            data = data.sort_index()

            try:
                data_actual = data[[target_col]]
                data_actual.reset_index(inplace=True)
                data_actual.columns = ["datetime", 'value']
                data_actual.set_index("datetime", inplace=True)

                # Check for frequency and handle any irregularities
                train_frequency = check_data_frequency(data_actual)

                # Ensure no duplicate indices
                if data_actual.index.duplicated().any():
                    print("Warning: Duplicate datetime indices found - aggregating")
                    data_actual = data_actual.groupby(data_actual.index).mean()

                train_models(data_actual, target_col)

                with open(os.path.join("models", 'Arima', target_col, target_col + '_results.json'), 'w') as fp:
                    json.dump({
                        'data_freq': train_frequency,
                        'start_date': str(data_actual.index.min()),
                        'end_date': str(data_actual.index.max())
                    }, fp, indent=4)

            except Exception as e:
                print(f"Error during model training: {e}")
                raise

        # Forecasting logic remains the same
        frequency = bot_query['time_unit']
        periods = bot_query['forecast_horizon']
        model_path = os.path.join(os.getcwd(), 'models', 'Arima', target_col, frequency, "best_model.pkl")
        print("model_path", model_path)

        loaded_model = load_forecast_model(model_path)
        freq_map = {
            'hours': 'H',
            'days': 'D',
            'weeks': 'W',
            'months': 'MS',
            'quarters': 'QS',
            'years': 'YS'
        }

        forecasted_data = forecast(loaded_model, periods, freq_map[frequency])
        print(forecasted_data)

        result_graph = plot_graph(forecasted_data)

        print(f"Results saved to {os.path.join('models', 'arima', target_col, target_col + '_results.json')}")
        return True, forecasted_data, result_graph

    except Exception as e:
        print("ARIMA error:", e)
        return False, pd.DataFrame(), ""


def load_forecast_model(model_path):
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        return joblib.load(model_path)
    else:
        print(f"No model found at {model_path}")
        return None


def plot_graph(data):
    try:

        # Create Plotly figure
        fig = go.Figure()
        try:
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(e)
        # Actual Data Line
        fig.add_trace(go.Scatter(
            x=data['date'], y=data["forecasted_value"],
            mode='lines+markers', name='Forecast',
            line=dict(color='blue'), marker=dict(symbol='circle')
        ))

        # Forecast Data Line
        # fig.add_trace(go.Scatter(
        #     x=forecast_dates, y=forecast_values,
        #     mode='lines+markers', name='Forecast',
        #     line=dict(color='orange', dash='dash'), marker=dict(symbol='x')
        # ))

        # Layout Settings
        fig.update_layout(
            title=f'Forecast Values Over Time',
            xaxis_title='Date',
            yaxis_title='Values',
            xaxis=dict(tickangle=-45, type='category', tickformat='%Y-%m-%d'),
            template="plotly_white",
            width=1000, height=600
        )

        # Convert figure to Base64 Image
        fig.show()
        return make_serializable(fig.to_json())

    except Exception as e:
        print(e)
        return str(e)


def forecast(model, periods, freq):
    future = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq=freq)
    future = future.to_series().dt.date.tolist()

    model_type = str(type(model))
    print(f"Detected model type: {model_type}")

    try:
        if 'Prophet' in model_type:
            future_df = pd.DataFrame({'ds': future})
            # Prophet expects 'ds' column and returns 'yhat'
            forecast = model.predict(future_df)
            if 'yhat' in forecast.columns:
                forecast = forecast[['ds', 'yhat']]
                forecast['yhat'] = forecast['yhat'].round(2)
                forecast.columns = ['date', 'forecasted_value']
                return forecast
            else:
                raise ValueError("Prophet output missing 'yhat' column.")

        else:
            # ARIMA, XGBoost, RandomForest — Expect direct prediction
            future_df = pd.DataFrame({'date': future})  # Rename here
            if 'ARIMA' in model_type:
                forecast = model.forecast(steps=future_df.shape[0])
                future_df['forecasted_value'] = np.round(forecast.values, 2)
            else:
                start_idx = model.last_index_ + 1  # get from model
                end_idx = start_idx + len(future_df)
                X_future = np.arange(start_idx, end_idx).reshape(-1, 1)
                print(X_future)
                forecast = model.predict(X_future)
                future_df['forecasted_value'] = np.round(forecast, 2)

            return future_df[['date', 'forecasted_value']]

    except Exception as e:
        print(f"Prediction Error: {e}")


def check_data_frequency(train):
    data_freq = {'D': 'Days', 'W': 'Weeks', "H": "Hours", "Q": "Quarters", 'A': 'Years'}
    m = pd.infer_freq(train.index)
    if m in ['15T', '30T', "H", "D", "W", "M", "Q", "A"]:
        print("Date Frequency is", m)
        return data_freq[m]
    else:
        print('Unsupported frequency')


def train_models(df, target_col):
    frequencies = ['hours', 'days', 'weeks', 'months', 'years']
    for freq in frequencies:
        print(f"\nTraining {freq} models...")

        # Resample the data for each frequency
        resampled_df = resample_data(df, freq)
        train, test = train_test_split(resampled_df, test_size=0.2, shuffle=False)

        trend = detect_trend(train)
        print("Trend is", trend)
        seasonality = detect_seasonality(train)
        print("Seasonality is", seasonality)

        best_model = None
        best_error = float('inf')
        best_model_name = ""
        scenario = ""

        # Scenario 1: Trend only
        if trend and not seasonality:
            scenario = "Trend only"
            arima_model, arima_error = train_arima(train, test)
            xgb_model, xgb_error = train_xgboost(train, test)

            if arima_error < xgb_error:
                best_model, best_error = arima_model, arima_error
                best_model_name = "ARIMA"
            else:
                best_model, best_error = xgb_model, xgb_error
                best_model_name = "XGBoost"

        # Scenario 2: Seasonality only
        if seasonality and not trend:
            scenario = "Seasonality only"
            prophet_model, prophet_error = train_prophet(train, test)
            arima_model, arima_error = train_arima(train, test)

            if prophet_error < arima_error:
                best_model, best_error = prophet_model, prophet_error
                best_model_name = "Prophet"
            else:
                best_model, best_error = arima_model, arima_error
                best_model_name = "ARIMA"

        # Scenario 3: Trend + Seasonality
        if trend and seasonality:
            scenario = "Trend + Seasonality"
            prophet_model, prophet_error = train_prophet(train, test)
            arima_model, arima_error = train_arima(train, test)

            min_error = min(prophet_error, arima_error)
            if min_error == prophet_error:
                best_model, best_error = prophet_model, prophet_error
                best_model_name = "Prophet"
            elif min_error == arima_error:
                best_model, best_error = arima_model, arima_error
                best_model_name = "ARIMA"

        # Scenario 4: No Trend or Seasonality
        if not trend and not seasonality:
            scenario = "No trend or seasonality"
            xgb_model, xgb_error = train_xgboost(train, test)
            rf_model, rf_error = train_randomforest(train, test)

            if xgb_error < rf_error:
                best_model, best_error = xgb_model, xgb_error
                best_model_name = "XGBoost"
            else:
                best_model, best_error = rf_model, rf_error
                best_model_name = "RandomForest"

        # Save the best model
        if best_model:
            model_dir = f'models/Arima/{target_col}/{freq}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_model.pkl')
            save_best_model(best_model, model_path)

            # Save Scenario with Model Name
            with open(f'scenario_{freq}.json', 'w') as f:
                json.dump({"scenario": scenario, "model_name": best_model_name}, f)

            print(f"\n{freq.capitalize()} Training complete. Scenario: {scenario}, Model: {best_model_name}")


def train_arima(train, test):
    print("Arima Started training")
    try:
        # Fit ARIMA model
        model = ARIMA(train['value'], order=(1, 1, 1)).fit()
        pred = model.predict(start=test.index[0], end=test.index[-1])

        # Calculate RMSE (handle different sklearn versions)
        try:
            # For newer sklearn versions
            error = mean_squared_error(test['value'], pred, squared=False)
        except TypeError:
            # For older sklearn versions
            error = np.sqrt(mean_squared_error(test['value'], pred))

        print("ARIMA training completed successfully")
        return model, error
    except Exception as e:
        print(f"Error in ARIMA training: {str(e)}")
        raise


# Train Prophet Model
def train_prophet(train, test):
    print("Training Prophet...")
    prophet_df = train.reset_index().rename(columns={'datetime': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(prophet_df)

    future = pd.DataFrame({'ds': test.index})
    forecast = model.predict(future)

    # Calculate RMSE (compatible with all sklearn versions)
    mse = mean_squared_error(test['value'], forecast['yhat'])
    error = np.sqrt(mse)  # Calculate RMSE manually

    print("Returning the parameters from the Prophet model")
    return model, error


# Train XGBoost Model
def train_xgboost(train, test):
    print("Training XGBoost...")
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['value'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    model.last_index_ = len(train) + len(test) - 1
    pred = model.predict(X_test)

    # Calculate RMSE (compatible with all sklearn versions)
    mse = mean_squared_error(test['value'], pred)
    error = np.sqrt(mse)  # Calculate RMSE manually

    return model, error


# Train RandomForest Model
def train_randomforest(train, test):
    print("Training RandomForest...")
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train['value'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    model.last_index_ = len(train) + len(test) - 1
    pred = model.predict(X_test)

    # Calculate RMSE (compatible with all sklearn versions)
    mse = mean_squared_error(test['value'], pred)
    error = np.sqrt(mse)  # Calculate RMSE manually

    return model, error


# Save the Best Model
def save_best_model(model, model_path):
    joblib.dump(model, model_path)


def resample_data(df, freq):
    print(f"Resampling data to {freq} frequency")
    if freq == 'hours':
        return df.resample('H').mean().ffill()
    elif freq == 'days':
        return df.resample('D').mean().ffill()
    elif freq == 'weeks':
        return df.resample('W').mean().ffill()
    elif freq == 'months':
        return df.resample('M').mean().ffill()
    elif freq == 'years':
        return df.resample('A').mean().ffill()
    else:
        raise ValueError("Unsupported frequency")


# Check Trend using Augmented Dickey-Fuller Test
def detect_trend(df):
    print('Detecting Trend...')
    result = adfuller(df['value'])
    p_value = result[1]
    return p_value > 0.05  # If p-value > 0.05 → Trend exists


# Check Seasonality using autocorrelation
def detect_seasonality(df):
    print('Detecting Seasonality...')
    autocorr = df['value'].autocorr(lag=1)
    return abs(autocorr) > 0.3  # If autocorr > 0.3 → Seasonality exists


def random_forest(data, target_column):
    try:
        model_dir = os.path.join("models", "rf", target_column)
        deployment_path = os.path.join(model_dir, 'deployment.json')

        if not os.path.exists(deployment_path):
            os.makedirs(model_dir, exist_ok=True)

            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if y.nunique() <= 5:  # Classification
                model_type = 'Classification'
                model = RandomForestClassifier(random_state=42)
                is_classification = True
            else:  # Regression
                model_type = 'Regression'
                model = RandomForestRegressor(random_state=42)
                is_classification = False

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            cv = min(5, len(X_test))

            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv)
            print(f"Model Performance (CV): {scores.mean():.4f} ± {scores.std():.4f}")

            # Get predictions on test set for detailed metrics
            predictions = pipeline.predict(X_test)

            # Initialize metric variables
            accuracy = None
            precision = None
            recall = None
            f1 = None
            mae = None
            rmse = None
            r2 = None

            if is_classification:
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            else:
                r2 = r2_score(y_test, predictions)  # R² for regression
                mae = mean_absolute_error(y_test, predictions)
                rmse = np.sqrt(mean_squared_error(y_test, predictions))

            # Calculate baseline comparison
            def calculate_baseline_comparison(y_true, y_pred, is_classification):
                try:
                    if is_classification:
                        if len(set(y_true)) > 1:
                            # baseline accuracy is accuracy of always predicting most frequent class
                            baseline_accuracy = max(np.bincount(y_true)) / len(y_true)
                            model_accuracy = accuracy_score(y_true, y_pred)
                            improvement = ((model_accuracy - baseline_accuracy) / baseline_accuracy) * 100
                            return f"{round(improvement, 1)}% better than baseline"
                        else:
                            return "Baseline comparison unavailable (only one class)"
                    else:
                        # For regression, compare with mean baseline (mean of targets)
                        baseline_mae = mean_absolute_error(y_true, [np.mean(y_true)] * len(y_true))
                        model_mae = mean_absolute_error(y_true, y_pred)
                        improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
                        return f"{round(improvement, 1)}% better than mean baseline"
                except Exception:
                    return "Baseline comparison unavailable"

            baseline_comparison = calculate_baseline_comparison(y_test, predictions, is_classification)

            # Compose model metrics dictionary according to task
            if is_classification:
                model_metrics = {
                    "accuracy": round(accuracy * 100, 1) if accuracy is not None else 'N/A',
                    "precision": round(precision * 100, 1) if precision is not None else None,
                    "recall": round(recall * 100, 1) if recall is not None else None,
                    "f1_score": round(f1 * 100, 1) if f1 is not None else None,
                }
            else:
                model_metrics = {
                    "r2_score": round(r2 * 100, 1) if r2 is not None else None,
                    "mae": round(mae, 2) if mae is not None else None,
                    "rmse": round(rmse, 2) if rmse is not None else None,
                }

            # Create comprehensive model statistics
            model_stats = {
                "model_type": model_type,
                "total_samples": len(y_test),
                "correct_predictions": int(sum(y_test == predictions)) if is_classification else None,
                "cross_val_mean": round(scores.mean(), 4),
                "cross_val_std": round(scores.std(), 4),
                "baseline_comparison": baseline_comparison,
                "metrics": model_metrics,
            }

            # Save the pipeline
            joblib.dump(pipeline, os.path.join(model_dir, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join(model_dir, "pipeline.pkl")}')

            # Save enhanced deployment information with statistics
            deployment_data = {
                "columns": list(X_train.columns),
                "model_type": model_type,
                "target_column": target_column,
                "stats": model_stats,
                "feature_names": list(X.columns),
                "categorical_features": categorical_cols,
                "numerical_features": numerical_cols
            }

            with open(deployment_path, "w") as fp:
                json.dump(deployment_data, fp, indent=4)

            return model_stats, list(X_train.columns)
        else:
            # Load existing model data and statistics
            with open(deployment_path, "r") as fp:
                deployment_data = json.load(fp)
            return deployment_data.get('stats', {}), deployment_data['columns']

    except Exception as e:
        print(f"Error in random_forest: {e}")
        return False, []


# 9.Data scout api----------------------Data scout api for generating the data------------- 9
@app.post("/api/data_scout")
async def create_data_with_data_scout(
        prompt: str = Form(...),
        data_type: str = Form(...),
) -> JSONResponse:
    if not prompt or not data_type:
        raise HTTPException(
            status_code=400,
            detail="Both prompt and type are required"
        )

    try:
        if data_type == "excel":
            agent1 = DataScout_agent()
            result = agent1.invoke({"input": prompt})

            if isinstance(result, dict) and "output" in result:
                output = result["output"]
                print("output is", output)
                # Handle DataFrame conversion
                if hasattr(output, 'to_dict'):  # Check if it's a DataFrame
                    # Convert DataFrame to dictionary
                    data_dict = output.to_dict('records')  # List of dictionaries
                    return JSONResponse(content={
                        "message": "Data generated successfully",
                        "data": data_dict,
                        "shape": output.shape,
                        "columns": output.columns.tolist()
                    })
                else:
                    # Handle string or other serializable output
                    return JSONResponse(content={"message": str(output)})


        elif data_type == "pdf":
            raw_prompt = prompt
            sections = extract_sections_tool(raw_prompt)
            number_of_pages = extract_num_pages_tool(raw_prompt)

            result = pdf_generator_tool(raw_prompt, sections, number_of_pages)

            if isinstance(result, dict) and "title" in result and "sections" in result:
                return JSONResponse(content=result)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate structured PDF content"
            )

        elif data_type == "image":
            agent1 = ImageGen_agent()
            result = agent1.run(prompt)

            # Handle different return formats
            if isinstance(result, list):
                images_data = []
                for item in result:
                    if isinstance(item, dict) and 'image_path' in item:
                        try:
                            with open(item["image_path"], "rb") as img_file:
                                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                                img_data = {
                                    "path": item["image_path"],
                                    "base64": base64_data,
                                    "thumbnail": item.get("thumbnail_path")
                                }
                                images_data.append(img_data)
                        except Exception as e:
                            print(f"Error processing image {item['image_path']}: {e}")
                            continue
                    elif isinstance(item, str):
                        try:
                            with open(item, "rb") as img_file:
                                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                                images_data.append({
                                    "path": item,
                                    "base64": base64_data
                                })
                        except Exception as e:
                            print(f"Error processing image {item}: {e}")
                            continue

                if images_data:
                    return JSONResponse(content={"images": images_data})
                raise HTTPException(
                    status_code=500,
                    detail="Failed to process generated images"
                )

            elif isinstance(result, str) and result.strip():
                try:
                    with open(result.strip(), "rb") as img_file:
                        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                        return JSONResponse(content={
                            "images": [{
                                "path": result.strip(),
                                "base64": base64_data
                            }]
                        })
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to process image: {str(e)}"
                    )

            raise HTTPException(
                status_code=500,
                detail="Failed to generate images - unexpected result format"
            )

        raise HTTPException(
            status_code=400,
            detail="Invalid data type (must be excel, pdf, or image)"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# 17.Data preprocess api----------------------Data preprocessing for the generation of the statistical analysis.-----
@app.get("/api/processing_for_dashboard")
async def perform_statistical_analysis() -> JSONResponse:
    """
    Perform comprehensive statistical analysis on the dataset

    Returns:
        JSONResponse: Contains various statistical metrics and data characteristics
    """
    try:
        # Check if data file exists
        if not os.path.exists('data.csv'):
            raise HTTPException(
                status_code=404,
                detail="Data file not found. Please upload file first."
            )

        # Load and preprocess data
        df = pd.read_csv('data.csv')
        df = updatedtypes(df)

        if df.shape[0] == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset contains no data"
            )

        # Basic statistics
        nullvalues = df.isnull().sum().to_dict()
        parameters = list(nullvalues.keys())
        count = list(nullvalues.values())
        total_missing = df.isnull().sum().sum()
        nor = df.shape[0]
        nof = df.shape[1]

        # Remove single value columns
        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
        df.drop(single_value_columns, axis=1, inplace=True)

        # Initialize variables for analysis
        timestamp = 'N'
        boolean = 'N'
        categorical_vars = []
        boolean_vars = []
        numeric_vars = {}
        datetime_vars = []
        text_data = []
        td = None
        stationary = "NA"
        duplicate_records = df[df.duplicated(keep='first')].shape[0]

        # Column type analysis
        for col, dtype in df.dtypes.items():
            if str(dtype) in ["float64", "int64"]:
                numeric_vars[col] = df[col].describe().to_dict()
            elif str(dtype) == "object" and col not in ['Remark']:
                categorical_vars.append({col: df[col].nunique()})
            elif str(dtype) == "datetime64[ns]":
                if col.upper() in ['DATE', "TIME", "DATE_TIME"]:
                    td = col
                datetime_vars.append(col)
            elif str(dtype) == "bool":
                boolean_vars.append(col)

        # Additional analyses
        istextdata = 'Y' if len(text_data) > 0 else 'N'
        if len(datetime_vars) > 0:
            timestamp = 'Y'
        if td:
            stationary = adf_test(df, td)
        if len(boolean_vars) > 0:
            boolean = 'Y'

        # Prepare categorical data
        catvalues = [{'Parameter': list(data.keys())[0],
                      'Count': list(data.values())[0]}
                     for data in categorical_vars]
        catdf = pd.DataFrame(catvalues) if catvalues else pd.DataFrame()

        # Prepare numeric data
        numdf = pd.DataFrame(numeric_vars).T
        if not numdf.empty:
            numdf['ColumnName'] = numdf.index

        # Prepare missing values data
        missingvalue = pd.DataFrame({
            "Parameters": parameters,
            'Missing Value Count': count
        })

        # Sentiment analysis
        sentiment = checkSentiment(df, categorical_vars)

        # **** Get the preview data (first 50 rows) ****
        preview = df.head(50)
        preview_data = json.loads(preview.to_json(orient='records'))

        # Build response
        response = {
            'nof_rows': str(nor),
            'nof_columns': str(nof),
            'timestamp': timestamp,
            'Preview_data': preview_data,
            "single_value_columns": ",".join(single_value_columns) if single_value_columns else "NA",
            "sentiment": sentiment,
            "stationary": stationary,
            'catdf': json.loads(catdf.to_json(orient='records')) if not catdf.empty else [],
            'missing_data': str(total_missing),
            'numdf': json.loads(numdf.to_json(orient='records')) if not numdf.empty else [],
            'boolean': boolean,
            'missingvalue': json.loads(missingvalue.to_json(orient='records')),
            'textdata': istextdata,
            'duplicate_records': str(duplicate_records)
        }

        return JSONResponse(content=response)

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="Data file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Statistical analysis failed: {str(e)}"
        )


def updatedtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                pd.to_datetime(df[col])
                df.drop(col, axis=1, inplace=True)
                print(df.columns)
            except Exception as e:
                pass
    return df


def adf_test(df, kpi):
    df_t = df.set_index(kpi)

    for col in df_t.columns:
        # Check if the column name is not in the specified list and is numeric
        if col.upper() not in ['DATE', 'TIME', 'DATE_TIME'] and pd.api.types.is_numeric_dtype(df_t[col]):
            if df_t[col].nunique() > 1:
                dftest = adfuller(df_t[col], autolag='AIC')
                statistic_value = dftest[0]
                p_value = dftest[1]
                if (p_value > 0.5) and all([statistic_value > j for j in dftest[4].values()]):
                    return "Y"
            else:
                break
    return "N"


def checkSentiment(df, categorical):
    sentiment = 'N'
    for i in categorical:
        # print([j for j in df[i]])
        data = ' '.join([str(j) for j in df[list(i.keys())[0]]]).upper()
        if ('GOOD' in data) | ('BAD' in data) | ('Better' in data):
            sentiment = "Y"
    return sentiment


# 10.Extend synthetic data generation api
class ContentType(Enum):
    DATA = "data"
    IMAGES = "images"
    PDF = "pdf"


@app.post("/api/generate_synthetic_data", response_model=None)
async def generate_synthetic_content(
        files: List[UploadFile],
        user_prompt: str = Form(...)
) -> Union[StreamingResponse, JSONResponse]:
    """
    Unified API for generating synthetic content (data, images, or PDFs)

    Args:
        files: List of uploaded files (CSV/Excel for data, images for image generation, PDF for document generation)
        user_prompt: Instructions for content generation

    Returns:
        StreamingResponse for CSV data or JSONResponse for images/PDFs
    """
    print("[DEBUG] Entered generate_synthetic_content")

    # Validate API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key not configured"
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded"
        )

    print(f"[DEBUG] Number of files uploaded: {len(files)}")
    print(f"[DEBUG] User prompt: {user_prompt}")

    # Determine content type based on uploaded files
    content_type = determine_content_type(files)
    print(f"[DEBUG] Detected content type: {content_type}")

    try:
        if content_type == ContentType.DATA:
            return await handle_data_generation(files[0], user_prompt, openai_api_key)
        elif content_type == ContentType.IMAGES:
            return await handle_image_generation(files, user_prompt, openai_api_key)
        elif content_type == ContentType.PDF:
            return await handle_pdf_generation(files[0], user_prompt)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type combination"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content generation failed: {str(e)}"
        )


def determine_content_type(files: List[UploadFile]) -> ContentType:
    """Determine the type of content to generate based on uploaded files"""
    if len(files) == 1:
        file = files[0]
        extension = os.path.splitext(file.filename)[1].lower()

        if extension in ['.csv', '.xlsx']:
            return ContentType.DATA
        elif extension == '.pdf':
            return ContentType.PDF
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return ContentType.IMAGES

    # Multiple files - check if all are images
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    if all(os.path.splitext(f.filename)[1].lower() in image_extensions for f in files):
        return ContentType.IMAGES

    raise HTTPException(
        status_code=400,
        detail="Invalid file combination. Upload CSV/Excel for data, images for image generation, or PDF for document generation."
    )


async def handle_data_generation(
        file: UploadFile,
        user_prompt: str,
        openai_api_key: str
) -> JSONResponse:
    """Handle synthetic data generation"""
    print("[DEBUG] Processing data generation request")
    temp_file_name = None

    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        print(f"[DEBUG] File extension: {file_extension}")

        # Read input file
        if file_extension == ".xlsx":
            df = pd.read_excel(file.file)
        elif file_extension == ".csv":
            df = pd.read_csv(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format for data generation. Please upload Excel or CSV."
            )

        print(f"[DEBUG] Original DataFrame shape: {df.shape}")

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file_name = temp_file.name
            if file_extension == ".xlsx":
                df.to_excel(temp_file_name, index=False)
            else:
                df.to_csv(temp_file_name, index=False)
            print(f"[DEBUG] Temporary file saved at: {temp_file_name}")

        # Process user prompt
        num_rows = extract_num_rows_from_prompt1(user_prompt, openai_api_key)
        print(f"[DEBUG] Extracted number of rows: {num_rows}")

        if num_rows is None:
            raise HTTPException(
                status_code=400,
                detail="Could not determine number of rows from prompt"
            )

        if num_rows > 100_000:
            raise HTTPException(
                status_code=400,
                detail="Too many rows requested. Limit is 100,000."
            )

        # Generate synthetic data
        datetime_col = infer_datetime_column(df)
        print(f"[DEBUG] Inferred datetime column: {datetime_col}")

        generated_df = generate_synthetic_data(
            openai_api_key,
            temp_file_name,
            num_rows,
            datetime_col=datetime_col
        )
        print(f"[DEBUG] Generated synthetic data shape: {generated_df.shape}")

        # Combine and return results
        combined_df = pd.concat([df, generated_df], ignore_index=True)
        print(f"[DEBUG] Combined DataFrame shape: {combined_df.shape}")

        # Convert DataFrame to JSON-serializable format
        # Handle datetime columns and other non-serializable types
        def convert_to_serializable(obj):
            """Convert non-serializable objects to serializable format"""
            if pd.isna(obj):
                return None
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Convert DataFrame to records with proper serialization
        data_records = []
        for _, row in combined_df.iterrows():
            record = {}
            for col, value in row.items():
                record[col] = convert_to_serializable(value)
            data_records.append(record)

        result_data = {
            "status": "success",
            "original_rows": len(df),
            "generated_rows": len(generated_df),
            "total_rows": len(combined_df),
            "data": data_records
        }

        print("[DEBUG] Successfully generated synthetic data response.")
        return JSONResponse(content=result_data)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[ERROR] Error in data generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content generation failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)
            print(f"[DEBUG] Temporary file {temp_file_name} deleted.")


async def handle_image_generation(
        images: List[UploadFile],
        user_prompt: str,
        openai_api_key: str
) -> JSONResponse:
    """Handle synthetic image generation"""
    print("[DEBUG] Processing image generation request")

    # Extract number of images from prompt
    requested_count = extract_num_images_from_prompt(user_prompt, openai_api_key)
    if requested_count is None or requested_count <= 0:
        raise HTTPException(
            status_code=400,
            detail="Could not extract valid image count from prompt. Include a clear number like 'generate 20 images'"
        )

    if requested_count > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum limit is 100 images per request"
        )

    # Ensure images folder exists
    folder_path = ensure_images_folder()

    # Analyze all images for comprehensive style
    print("Analyzing uploaded images for style...")
    comprehensive_style = analyze_all_images_for_style(images, openai_api_key)
    print(f"Extracted style: {comprehensive_style}")

    # Get starting index for new images
    start_index = get_next_image_index(folder_path)
    total_generated = 0
    generated_images_info = []

    while total_generated < requested_count:
        remaining = requested_count - total_generated
        batch_size = min(5, remaining)

        # Construct prompt with style guidance
        full_prompt = (
            f"Generate {batch_size} high-quality, stylistically consistent images.\n"
            f"Visual style from references: {comprehensive_style}\n"
            f"Avoid redundant compositions.\n"
            f"Create unique subjects while maintaining style.\n"
            f"User instructions: {user_prompt.strip()}"
        )

        print(f"Generating batch of {batch_size} images...")
        image_urls = generate_images(client, full_prompt, batch_size)

        for url in image_urls:
            try:
                # Download and save image
                response = requests.get(url)
                response.raise_for_status()

                filename = f"synthetic_{start_index + total_generated}.png"
                file_path = Path(folder_path) / filename

                with open(file_path, 'wb') as f:
                    f.write(response.content)

                # Convert to base64
                with open(file_path, 'rb') as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')

                generated_images_info.append({
                    'filename': filename,
                    'path': str(file_path),
                    'base64': base64_image,
                    'index': start_index + total_generated
                })

                total_generated += 1
                print(f"Generated {total_generated}/{requested_count}")

            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

    return JSONResponse(
        content={
            'success': True,
            'content_type': 'images',
            'message': f'Generated {total_generated} images',
            'total_generated': total_generated,
            'images': generated_images_info
        }
    )


async def handle_pdf_generation(
        pdf: UploadFile,
        user_prompt: str
) -> JSONResponse:
    """Handle synthetic PDF generation"""
    print("[DEBUG] Processing PDF generation request")

    # Validate PDF upload
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted for document generation."
        )

    # Extract target pages from prompt
    target_pages = extract_pdf_prompt_semantics(user_prompt)
    if not target_pages:
        raise HTTPException(
            status_code=400,
            detail="Could not determine target page count from prompt"
        )

    # Process uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await pdf.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        # Extract text from PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        current_pages = len(documents)
        full_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)

    # Generate extended document
    final_document = generate_complete_document(
        full_text,
        current_pages,
        target_pages
    )

    # Check for generation errors
    if isinstance(final_document, dict) and final_document.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Document generation failed: {final_document.get('error')}"
        )

    return JSONResponse(
        content={
            'success': True,
            'content_type': 'pdf',
            'message': f'Generated extended PDF document',
            'structured_document': final_document,
            'original_pages': current_pages,
            'target_pages': target_pages
        },
        status_code=200
    )


def infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.8:
                return col
        except Exception:
            continue
    return None


# Semantic ai related number of rows detection
def extract_num_rows_from_prompt1(prompt: str, api_key: str) -> Optional[int]:
    """
    Extracts number of rows to generate using LLM-based semantic parsing only.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key)
    messages = [
        SystemMessage(content="You extract the number of rows to generate from user input. Return only the integer."),
        HumanMessage(content=prompt)
    ]
    try:
        response = llm.invoke(messages)
        match = re.search(r'\d+', response.content)
        return int(match.group()) if match else None
    except Exception as e:
        print(f"[ERROR] Semantic extraction failed: {e}")
        return None


GENERATED_IMAGES_FOLDER = "generated_synthetic_images"


def ensure_images_folder():
    """Create the images folder if it doesn't exist"""
    if not os.path.exists(GENERATED_IMAGES_FOLDER):
        os.makedirs(GENERATED_IMAGES_FOLDER)
    return GENERATED_IMAGES_FOLDER


def get_next_image_index(folder_path):
    """Get the next image index based on existing files in the folder"""
    existing_files = [f for f in os.listdir(folder_path) if f.startswith('synthetic_') and f.endswith('.png')]
    if not existing_files:
        return 1

    # Extract numbers from filenames and find the maximum
    indices = []
    for filename in existing_files:
        try:
            # Extract number from filename like "synthetic_123.png"
            number = int(filename.replace('synthetic_', '').replace('.png', ''))
            indices.append(number)
        except ValueError:
            continue

    return max(indices) + 1 if indices else 1


def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None


def analyze_all_images_for_style(uploaded_images, openai_api_key):
    """
    Analyze all uploaded images to extract a comprehensive style description.
    """
    individual_descriptions = []

    # Get description for each uploaded image
    for i, image_file in enumerate(uploaded_images):
        try:
            # Read the file content from UploadFile
            image_file.file.seek(0)  # Reset file pointer to beginning
            image_content = image_file.file.read()

            # Create a BytesIO object from the content
            image_bytes = io.BytesIO(image_content)

            # Now open with PIL
            image = Image.open(image_bytes).convert("RGB")
            vision_caption = describe_image(image, openai_api_key)
            individual_descriptions.append({
                'index': i + 1,
                'description': vision_caption
            })
            print(f"Image {i + 1} analysis: {vision_caption}")
        except Exception as e:
            print(f"Error analyzing image {i + 1}: {str(e)}")
            continue

    if not individual_descriptions:
        return "No valid images could be analyzed"
    # Create a comprehensive style analysis prompt
    style_analysis_prompt = f"""
    Analyze the following {len(individual_descriptions)} images and provide a comprehensive style summary that captures the common visual elements, artistic approach, and aesthetic characteristics across all images.

    Individual Image Descriptions:
    """

    for desc in individual_descriptions:
        style_analysis_prompt += f"\nImage {desc['index']}: {desc['description']}"

    style_analysis_prompt += """

    Based on these individual descriptions, provide a unified style analysis that includes:
    1. Common visual elements (colors, lighting, composition patterns)
    2. Consistent artistic style and approach
    3. Technical characteristics (quality, resolution, photographic style)
    4. Subject matter patterns and themes
    5. Overall aesthetic and mood

    Focus on the elements that are consistent across all images to define the core style that should be maintained in generated images. If there are variations, mention the acceptable range of variation within the style.

    Provide a concise but comprehensive style guide that can be used for generating new images in the same style.
    You must provide the final response in 500 characters only.Return the main content only .Don't return headings and unuseful information.
    """

    # Use ChatOpenAI to analyze the combined descriptions
    try:
        analysis_llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=openai_api_key)
        analysis_messages = [
            SystemMessage(
                content="You are an expert visual style analyst. Analyze multiple image descriptions to extract a comprehensive, unified style guide."),
            HumanMessage(content=style_analysis_prompt)
        ]

        response = analysis_llm.invoke(analysis_messages)
        comprehensive_style = response.content.strip()
        print(f"Comprehensive style analysis: {comprehensive_style}")
        return comprehensive_style

    except Exception as e:
        print(f"Error in comprehensive style analysis: {str(e)}")
        # Fallback: combine individual descriptions
        combined_description = " | ".join([desc['description'] for desc in individual_descriptions])
        return f"Combined style elements from {len(individual_descriptions)} images: {combined_description}"


ROBUST_VISION_SYSTEM_PROMPT = """
You are an expert computer vision analyst with exceptional observational skills. Your task is to provide comprehensive, detailed visual analysis of images with scientific precision and artistic sensitivity.

CORE ANALYSIS FRAMEWORK:

1. COMPOSITIONAL STRUCTURE:
   - Analyze the overall layout, framing, and spatial organization
   - Identify foreground, middle ground, and background elements
   - Describe the visual flow and how elements guide the viewer's eye
   - Note any compositional techniques (rule of thirds, symmetry, leading lines)

2. OBJECTS AND SUBJECTS:
   - Catalog ALL visible objects, people, animals, and entities
   - Describe their positions, sizes, and relationships to each other
   - Identify specific details: clothing, expressions, poses, conditions
   - Note any text, signs, logos, or written elements
   - Mention partially visible or obscured objects

3. VISUAL CHARACTERISTICS:
   - Color palette: dominant colors, color harmony, saturation levels
   - Lighting: source, direction, quality (harsh/soft), shadows, highlights
   - Texture and materials: surfaces, fabrics, finishes
   - Depth and dimensionality: perspective, scale relationships
   - Focus and clarity: sharp vs. blurred areas, depth of field

4. STYLE AND AESTHETIC:
   - Artistic style (realistic, abstract, minimalist, ornate, etc.)
   - Genre or category (portrait, landscape, still life, architectural, etc.)
   - Mood and atmosphere conveyed
   - Cultural or historical context if apparent
   - Technical quality and craftsmanship

5. CONTEXTUAL ELEMENTS:
   - Setting and environment (indoor/outdoor, specific location type)
   - Time indicators (lighting suggests time of day, seasonal clues)
   - Weather conditions if visible
   - Social or cultural context
   - Any narrative or story elements

6. TECHNICAL OBSERVATIONS:
   - Image quality, resolution, and clarity
   - Camera angle and perspective
   - Any visible artifacts, distortions, or technical issues
   - Photographic or artistic techniques employed

RESPONSE GUIDELINES:
- Begin with a concise overview sentence
- Organize observations logically from general to specific
- Use precise, descriptive language without unnecessary adjectives
- Quantify when possible (approximate counts, sizes, proportions)
- Be objective while noting subjective elements like mood or style
- Mention what's NOT present if it's notable or expected
- Conclude with the most striking or significant visual element

ACCURACY REQUIREMENTS:
- Never invent details not visible in the image
- Distinguish between what you can see clearly vs. what you infer
- Use conditional language for uncertain observations ("appears to be", "seems to")
- Prioritize factual description over interpretation
- If image quality limits observation, acknowledge this

Your goal is to create a verbal representation so detailed that someone could understand the image's content, composition, and character without seeing it themselves. Just give the final response in 200 characters  only.
"""


# Usage example with your function:
def describe_image(image: Image.Image, api_key: str) -> str:
    vision_llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    base64_image = base64.b64encode(image_bytes.read()).decode('utf-8')

    vision_messages = [
        SystemMessage(content=ROBUST_VISION_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ])
    ]

    try:
        response = vision_llm.invoke(vision_messages)
        return response.content.strip()
    except Exception:
        return "a visually rich and consistent image based on user style"


def generate_images(client: OpenAI, prompt: str, count: int) -> list:
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        n=count,
        size="512x512"
    )
    return [r.url for r in response.data if r.url]


def extract_num_images_from_prompt(prompt: str, api_key: str) -> Optional[int]:
    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", openai_api_key=api_key)
        messages = [
            SystemMessage(
                content="You extract the number of images to generate from user input. Return only the integer."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        match = re.search(r'\d+', response.content)
        if match:
            return int(match.group())
    except Exception as e:
        print(f"[ERROR] LLM semantic extraction failed: {e}")

    try:
        fallback_match = re.search(r'(?:generate|create|make)?\s*(\d{1,3})\s*(?:images|pictures)', prompt,
                                   re.IGNORECASE)
        if fallback_match:
            return int(fallback_match.group(1))
    except Exception as e:
        print(f"[ERROR] Regex fallback extraction failed: {e}")

    return None


# --- NEW, UNIFIED FUNCTION TO HANDLE EVERYTHING ---
def generate_complete_document(
        full_text: str, current_pages: int, target_pages: int
) -> Dict[str, Any]:
    """
    Uses a single, powerful LLM call to structure original text and append new,
    generated sections to meet a target page count.
    """
    pages_to_add = max(0, target_pages - current_pages)
    if pages_to_add == 0:
        # If no pages to add, just structure the existing text
        print("No pages to add. Structuring existing content only.")

    prompt = f"""
You are an expert technical writer and document analyst. Your task is to produce a single, complete JSON document that first structures the original content provided, and then seamlessly extends it with new, original sections to meet a target page count.

**Original Raw Text:**
---
{full_text}
---

**Task Details:**
- Current page count of original text: {current_pages}
- Target page count for the final document: {target_pages}
- New pages worth of content to generate and add: {pages_to_add}

**Your Response MUST be a single, cohesive, valid JSON object.**
The final JSON must contain BOTH the structured original content AND the new sections.

**Required JSON Format:**
{{
  "title": "The document's main title",
  "sections": [
    // 1. First, sections from the original text
    {{
      "heading": "Original Section 1 Heading",
      "subsections": [
        {{
          "subheading": "Original Subsection 1.1 Heading",
          "content": "The full, original text content for this subsection..."
        }}
      ]
    }},
    // 2. Then, the NEWLY GENERATED sections
    {{
      "heading": "New, Logically Following Section Heading",
      "subsections": [
        {{
          "subheading": "New Subsection Heading",
          "content": "Full, detailed, and comprehensive written content for the new subsection. This must be entirely new content you generate..."
        }}
      ]
    }}
  ]
}}

**Crucial Instructions:**
1.  **Integrate, Then Extend:** First, accurately place the provided original text into the JSON structure. Then, create and append new sections that logically follow the original content.
2.  **Generate Substantial New Content:** The new sections must be detailed and comprehensive enough to expand the total document to approximately {target_pages} pages. Do not use placeholders.
3.  **Cohesive Final Document:** The final output must be ONE JSON object containing both original and new content seamlessly integrated. Do NOT output only the new parts.
4.  **Strictly JSON:** Your entire response must be a single, valid JSON object. Do not add any explanations, apologies, or markdown (e.g., ```
"""
    llm = initialize_llm()
    try:
        response = llm.invoke(prompt)
        # Attempt to parse the response directly
        return json.loads(response.content)
    except json.JSONDecodeError:
        # If it fails, use our LLM-based repair function as a fallback
        repaired_json_string = repair_json_with_llm(response.content)
        try:
            return json.loads(repaired_json_string)
        except json.JSONDecodeError as e:
            print(f"FATAL: Could not parse even the repaired JSON. Error: {e}")
            return {"error": "Failed to generate and parse final document", "details": str(e)}


def repair_json_with_llm(broken_json_string: str) -> str:
    print("--- Standard JSON parsing failed. Attempting LLM-based repair. ---")

    # Use a different, more capable model for repair if available, or the same one
    repair_llm = initialize_llm()

    # A highly-constrained prompt focused solely on fixing the JSON
    repair_prompt = f"""
            You are a specialized AI assistant that corrects malformed JSON.
            Your single task is to take the provided text and output a valid, well-formed JSON object.

            CRUCIAL INSTRUCTIONS:
            1.  **CORRECT ALL ERRORS:** Fix syntax errors like missing commas, unclosed brackets or braces, incorrect string escaping, and trailing commas.
            2.  **REMOVE EXTRA TEXT:** Delete any text, explanations, or apologies that are outside of the JSON structure.
            3.  **JSON ONLY:** Your entire response MUST be ONLY the corrected JSON object. Do not wrap it in markdown (e.g., ```

            --- MALFORMED JSON INPUT ---
            {broken_json_string}
            --- END MALFORMED JSON INPUT ---

            Your corrected JSON output:
            """

    try:
        response = repair_llm.invoke(repair_prompt)
        # The response content should be the clean JSON string
        repaired_text = response.content.strip()
        return repaired_text
    except Exception as e:
        print(f"--- LLM-based repair also failed: {e} ---")
        # Return a string that represents a JSON error object
        return f'{{"error": "Failed to repair JSON", "details": "{str(e)}"}}'


def extract_pdf_prompt_semantics(user_prompt: str) -> Optional[int]:
    llm = initialize_llm()

    system_instruction = (
        """You are a document request parser. Given a user prompt, extract:
        1. The number of pages requested (as an integer).

        Respond with JSON only in this format:
        {"num_pages": <int or null>}.
        No commentary or markdown.
        """
    )

    response = llm.invoke(f"{system_instruction}\n\nPrompt: {user_prompt}")
    content = response.content if hasattr(response, 'content') else str(response)

    try:
        parsed = json.loads(content)
        return parsed.get("num_pages")
    except Exception:
        return None

# 11.Explore api
@app.post("/api/Explore")
async def senior_data_analysis(
        query: str = Form(...)
) -> JSONResponse:
    try:
        # Load and validate data
        csv_file_path = 'data.csv'
        if not os.path.exists(csv_file_path):
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )

        df = pd.read_csv(csv_file_path)
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Dataset is empty"
            )

        # Generate metadata
        metadata_str = ", ".join(df.columns.tolist())

        # Check if query is report-related
        is_report_query = any(keyword in query.lower() for keyword in
                              ['report', 'summary report', 'analysis report', 'detailed report',
                               'comprehensive report', 'summary_report', 'analysis_report', 'detailed_report',
                               'Comprehensive_report'])

        if is_report_query:
            # Generate report-specific prompt
            prompt_eng = (
                f"""
                            You are a Senior data analyst generating a comprehensive report with advanced analytics and forecasting capabilities. 
                            Always strictly adhere to the following rules: 

                            The metadata required for your analysis: {metadata_str}
                            Dataset location: data.csv only. No data assumptions can be taken.

                            Forecasting Information:
                            -Take the information for forecasting from the data.csv itself.
                            -Extract the dates very accurately from the data.csv and use that dates for forecasting.
                        
                            ###IMPORTANT: You MUST return the response in this EXACT JSON format with "report","title","Description" as the keys:
                            {{
                                "report": {{
                                    "heading": "[Suitable Heading]",
                                    "paragraphs": [
                                        "First Bullet point: Provide detailed current data analysis insights, trends, and patterns observed in the dataset. Include statistical summaries, key findings, and data quality observations. This should be exactly 4 lines of detailed analysis.",
                                        "Second Bullet point: Present forecasting analysis, methodology used, future predictions, confidence intervals, and business implications. Explain the forecasting approach, seasonal patterns if any, and actionable recommendations. This should be exactly 4 lines of forecasting insights."
                                    ],
                                    "table": {{
                                        "headers": ["Metric", "Current Value", "Forecasted Value", "Change %", "Confidence"],
                                        "rows": [
                                            ["Key Metric 1", "Current", "Predicted", "+X%", "95%"],
                                            ["Key Metric 2", "Current", "Predicted", "+Y%", "90%"],
                                            ["Key Metric 3", "Current", "Predicted", "-Z%", "85%"]
                                        ]
                                    }},
                                    "analysis_charts": [
                                        {{
                                            "title": "[heading]",
                                            "plotly": {{
                                                "data": [{{
                                                    "x": ["category_values_from_actual_data"],
                                                    "y": [actual_numeric_values],
                                                    "type": "bar",
                                                    "marker": {{"color": "#3498db"}},
                                                    "name": "Current Distribution"
                                                }}],
                                                "layout": {{
                                                    "title": "[title]",
                                                    "xaxis": {{"title": "Categories"}},
                                                    "yaxis": {{"title": "Values"}},
                                                    "paper_bgcolor": "#fafafa",
                                                    "plot_bgcolor": "#ffffff"
                                                }}
                                            }}
                                        }},
                                        {{
                                            "title": "[heading]",
                                            "plotly": {{
                                                "data": [{{
                                                    "x": ["time_periods_from_data"],
                                                    "y": [trend_values],
                                                    "type": "scatter",
                                                    "mode": "lines+markers",
                                                    "marker": {{"color": "#e74c3c"}},
                                                    "name": "Historical Trend"
                                                }}],
                                                "layout": {{
                                                    "title": "[title]",
                                                    "xaxis": {{"title": "Time Period"}},
                                                    "yaxis": {{"title": "Values"}},
                                                    "paper_bgcolor": "#fafafa",
                                                    "plot_bgcolor": "#ffffff"
                                                }}
                                            }}
                                        }}
                                    ],
                                    "forecasting_charts": [
                                        {{
                                            "title": "[heading]",
                                            "plotly": {{
                                                "data": [
                                                    {{
                                                        "x": ["historical_dates"],
                                                        "y": [historical_values],
                                                        "type": "scatter",
                                                        "mode": "lines+markers",
                                                        "marker": {{"color": "#2ecc71"}},
                                                        "name": "Historical Data"
                                                    }},
                                                    {{
                                                        "x": ["future_dates"],
                                                        "y": [predicted_values],
                                                        "type": "scatter",
                                                        "mode": "lines+markers",
                                                        "marker": {{"color": "#f39c12", "symbol": "diamond"}},
                                                        "name": "Forecasted Values"
                                                    }},
                                                    {{
                                                        "x": ["future_dates"],
                                                        "y": [upper_confidence_values],
                                                        "type": "scatter",
                                                        "mode": "lines",
                                                        "line": {{"dash": "dash", "color": "#95a5a6"}},
                                                        "name": "Upper Confidence",
                                                        "showlegend": false
                                                    }},
                                                    {{
                                                        "x": ["future_dates"],
                                                        "y": [lower_confidence_values],
                                                        "type": "scatter",
                                                        "mode": "lines",
                                                        "line": {{"dash": "dash", "color": "#95a5a6"}},
                                                        "fill": "tonexty",
                                                        "fillcolor": "rgba(149, 165, 166, 0.2)",
                                                        "name": "Confidence Interval"
                                                    }}
                                                ],
                                                "layout": {{
                                                    "title": "[title]",
                                                    "xaxis": {{"title": "Date"}},
                                                    "yaxis": {{"title": "Predicted Values"}},
                                                    "paper_bgcolor": "#fafafa",
                                                    "plot_bgcolor": "#ffffff"
                                                }}
                                            }}
                                        }},
                                        {{
                                            "title": "[heading]",
                                            "plotly": {{
                                                "data": [
                                                    {{
                                                        "x": ["time_periods"],
                                                        "y": [seasonal_trend],
                                                        "type": "scatter",
                                                        "mode": "lines",
                                                        "marker": {{"color": "#9b59b6"}},
                                                        "name": "Seasonal Trend"
                                                    }},
                                                    {{
                                                        "x": ["time_periods"],
                                                        "y": [forecasted_seasonal],
                                                        "type": "scatter",
                                                        "mode": "lines+markers",
                                                        "marker": {{"color": "#e67e22"}},
                                                        "name": "Forecasted Seasonal Pattern"
                                                    }}
                                                ],
                                                "layout": {{
                                                    "title": "[title]",
                                                    "xaxis": {{"title": "Time Period"}},
                                                    "yaxis": {{"title": "Seasonal Values"}},
                                                    "paper_bgcolor": "#fafafa",
                                                    "plot_bgcolor": "#ffffff"
                                                }}
                                            }}
                                        }}
                                    ]
                                }}
                                "title: "[Suitable Title for the Report]",
                                "Description":"[A brief  one liner description of the report's purpose and scope]"
                            }}

                            FORECASTING REQUIREMENTS:
                            1. Use appropriate time series forecasting methods (ARIMA, Exponential Smoothing, or Seasonal Decomposition)
                            2. Generate predictions for the next 6-12 periods based on available data
                            3. Include confidence intervals (80% and 95%) for predictions
                            4. Identify seasonal patterns if present in the data
                            5. Provide forecast accuracy metrics where possible
                            6. Use actual data values, dates, and column names from the dataset
                            7. Ensure all chart data points are realistic and based on actual data patterns

                            DYNAMIC REPORT STRUCTURE:
                            - Vary the order of analysis vs forecasting charts
                            - Rotate between different visualization types (bar, line, scatter, etc.)
                            - Change color schemes for each request
                            - Alternate table structures and metrics shown
                            - Randomize which forecasting method emphasis to show first

                            Generate a comprehensive report with forecasting for: {query}

                            The report must include:
                            - 4 Bullet points (2 lines each): three for current analysis, one for forecasting insights with related headings and all.
                            - 1 summary table with forecasting metrics and all other analysis metrics
                            - 2 analysis charts showing current data patterns from the data with main Heading of Analysis.
                            - 2 forecasting charts with predictions and confidence intervals with main heading of Forecast.
                            - All visualizations must use actual data from the CSV file
                            - Forecasting should predict realistic future values based on historical patterns
                            """)
        else:
            # Generate regular analysis prompt
            prompt_eng = (
                f"""
                        You are a **Senior Data Analyst**.  
                        Handle **any kind of user query**—from casual questions to advanced statistical analysis—while strictly following the rules below.

                        ────────────────────────────────────────────────────────────────────────
                        GLOBAL RESPONSE FORMAT  
                        • Every reply MUST be valid JSON with exactly one top-level key: **"answer"**.  
                        Example:  
                        {
                            "answer": "Your analysis result here..."
                        }
                        • Never add extra keys, nesting, or comments outside the JSON block.
                        ────────────────────────────────────────────────────────────────────────
                        HOW TO DECIDE WHAT TO RETURN
                        1. Generic queries  
                        • If the user question does not involve the dataset, answer concisely in plain text (inside the JSON).

                        2. Data-related queries  
                        • Assume the data source is **data.csv** and its columns are exactly those listed in **{metadata_str}**.  
                        • Perform whatever processing, statistics, KPIs, or predictions are requested.  
                        • Never invent data; use only what is in data.csv.

                        3. Visualisation requests  
                        • Write clean, self-contained Python code that generates the requested chart.  
                        • Follow the Code-Safety rules (see below).  
                        • End the code with `plt.show()` so the plot is displayed.  
                        • The final printed line must be meaningful (e.g., print the figure object or a confirmation message).

                        4. KPI or summary-table requests  
                        • Produce the KPIs or summary statistics in a **pure HTML table** using `<table>`, `<tr>`, `<td>` only (no `<th>`, no headings).

                        5. Mixed or complex queries  
                        • If a question combines several tasks (e.g., “show a chart and list KPIs”), fulfil ALL parts, preserving the rules above.

                        ────────────────────────────────────────────────────────────────────────
                        CODE-SAFETY & EXECUTION RULES  
                        • Use only standard libraries plus pandas, numpy, matplotlib, seaborn.  
                        • No undefined variables (e.g., boxprops) unless you define them.  
                        • No file I/O except reading **data.csv** (already present).  
                        • Do NOT use `os`, `sys`, `input()`, `open()`, or save/write any files.  
                        • Always call `plt.show()` after plotting.  
                        • Always include a `print()` statement at the end of your code.  
                        • The last line of code must print something meaningful.

                        ────────────────────────────────────────────────────────────────────────
                        STYLE & TONE  
                        • Never reveal internal reasoning or the steps you took—only provide the result (code, table, or plain text).  
                        • Do **not** prepend headings or explanations.  
                        • Do **not** reply with “Understood” or similar confirmations.  
                        • Keep answers succinct and relevant.

                        ────────────────────────────────────────────────────────────────────────
                        PLACEHOLDERS  
                        • `{metadata_str}`: replace with the real column list when rendering the prompt to the analyst.  
                        • `{query}`: the user's question.

                        ────────────────────────────────────────────────────────────────────────
                        EXAMPLES

                        User: “What is 2 + 2?”  
                        Return:  
                        {
                            "answer": "4"
                        }

                        User: “Show a histogram of the Age column.”  
                        Return:  
                        {
                            "answer": "``````"
                        }

                        User: “Give me average salary and headcount.”  
                        Return:  
                        {
                            "answer": "<table><tr><td>Average Salary</td><td>75,230</td></tr><tr><td>Headcount</td><td>1,024</td></tr></table>"
                        }

            """
            )

        # Generate and execute analysis code
        print("Analysed the above things,,,,,,,,,going to generate the code")
        code = generate_data_code(prompt_eng)
        print("code_generated and going for execution,,,,,,,,")
        result = simulate_and_format_with_llm(code, df)
        print("executed_result is", result)

        return JSONResponse(
            content=clean_json_response(result),
            status_code=200
        )

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


from universal_prompts import prompt_for_data_analyst


# Function to generate code from OpenAI API
def generate_data_code(prompt_eng):
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
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


from universal_prompts import Prompt_for_code_execution, Visualisation_intelligence_engine


def simulate_and_format_with_llm(
        code_to_simulate: str,
        dataframe: pd.DataFrame
) -> str:
    info_buffer = io.StringIO()
    dataframe.info(buf=info_buffer)
    df_info = info_buffer.getvalue()
    df_head = dataframe.head().to_string()

    # Step 2: Construct the detailed user prompt.
    # This prompt gives the LLM its task, the code, and the context.
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

    # Step 4: Extract and return the final formatted text.
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message

    return all_text


def clean_json_response(response_text):
    """
    Extract clean JSON from a response that may contain markdown code blocks
    and additional text explanations.
    """
    try:
        # Method 1: Try to extract JSON from markdown code blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)

        if match:
            json_str = match.group(1)
            # Parse and return as clean JSON
            json_data = json.loads(json_str)
            return json.dumps(json_data)

        # Method 2: If no markdown blocks, try to find JSON object directly
        json_pattern2 = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern2, response_text, re.DOTALL)

        for match in matches:
            try:
                json_data = json.loads(match)
                return json.dumps(json_data)
            except json.JSONDecodeError:
                continue

        return None

    except Exception as e:
        print(f"Error cleaning JSON: {e}")
        return None


# Report related apis like sending and saving and sending.
# Save report api

# AWS/S3 configuration with your bucket and region
S3_BUCKET = "akio-report-data"
S3_REGION = "ap-southeast-1"
s3_client = boto3.client("s3",
                         aws_access_key_id=os.getenv("REACT_APP_AWS_ACCESS_KEY_ID"),
                         aws_secret_access_key=os.getenv("REACT_APP_AWS_SECRET_ACCESS_KEY"),
                         region_name=S3_REGION)



def upload_file_to_s3(file: UploadFile, bucket: str, key: str) -> str:
    """
    Uploads file to S3 bucket and returns the public URL.
    """
    contents = file.file.read()
    s3_client.put_object(Bucket=bucket, Key=key, Body=contents, ContentType=file.content_type)
    # Construct the S3 object URL for Singapore region
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"


def serialize_report(report: dict) -> dict:
    """Convert datetime objects to ISO format strings in a report dict."""
    for key in ['created_at', 'updated_at']:
        if key in report and isinstance(report[key], datetime):
            report[key] = report[key].isoformat()
    return report


@app.post("/api/save_report")
async def save_report(
        email: str = Form(...),
        title: str = Form(...),
        description: str = Form(...),
        pdf_file: UploadFile = File(...)
) -> JSONResponse:
    """
    Upload a PDF file to S3, extract title/description using LLM, and save metadata in database.
    """
    try:
        # Validate file content type
        if pdf_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        # Read file contents (we'll need this for both S3 upload and text extraction)
        file_contents = await pdf_file.read()
        
        # Reset file pointer for S3 upload
        pdf_file.file = io.BytesIO(file_contents)

        # Generate unique key for S3
        safe_email = email.replace("@", "_at_").replace(".", "_dot_")
        unique_filename = f"reports/{safe_email}/{uuid.uuid4().hex}.pdf"
        print(f"Generated S3 key: {unique_filename}")

        # Upload file to S3
        try:
            s3_url = upload_file_to_s3(pdf_file, S3_BUCKET, unique_filename)
            print(f"S3_url: {s3_url}")
        except Exception as s3exc:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {s3exc}")

        # Insert record in DB with extracted metadata INCLUDING title and description
        result = db.insert_report(email, s3_url, title, description)
        result = serialize_report(result) if result else {}

        # Enhanced response with extracted information
        return JSONResponse(content={
            "status": "success",
            "result": result,
            "pdf_url": s3_url,
            "title": title,
            "description": description
        })

    except HTTPException as he:
        raise he
    except Exception as exc:
        print(f"Unexpected error in save_report: {exc}")
        raise HTTPException(status_code=500, detail=f"Error in save_report: {exc}")



@app.post("/api/get_reports_by_email")
async def get_reports_with_email(
        email: str = Form(...)
) -> JSONResponse:
    """
    Retrieve all saved reports for a given email with title and description.
    """
    try:
        reports = db.get_report_by_email(email)  # Now includes title and description

        if not reports:
            return JSONResponse(content=[])

        # Serialize datetime fields in all reports
        serialized_reports = [serialize_report(r) for r in reports]

        return JSONResponse(content=serialized_reports)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/delete_report_by_id")
async def delete_report_by_id(
        email: str = Form(...),
        report_id: int = Form(...)
) -> JSONResponse:
    """
    Delete a specific report by its ID and the associated email.
    """
    try:
        result = db.delete_user_report_by_id(email, report_id)
        return JSONResponse(content={"status": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Email report to the user
@app.post("/api/email_report")
async def email_report(
        email: str = Form(...),
        report_ids: Optional[str] = Form(None)  # Accept comma-separated string
) -> JSONResponse:
    try:
        if report_ids:
            # Parse comma-separated string into list of ints
            report_ids = [int(rid.strip()) for rid in report_ids.split(",") if rid.strip()]
            reports = db.get_reports_by_ids_and_email(email, report_ids)
        else:
            reports = db.get_report_by_email(email)

        if not reports:
            raise HTTPException(
                status_code=404,
                detail="No reports found for this email with specified report IDs."
            )

        msg = MIMEMultipart()
        msg['Subject'] = 'Selected Graph Reports'
        msg['From'] = os.getenv("EMAIL_USER")
        msg['To'] = email
        msg.attach(MIMEText("Please find attached the selected PDF report(s).", 'plain'))

        for i, report in enumerate(reports, start=1):
            pdf_url = report.get("url")
            if not pdf_url:
                print(f"Skipped report {i}: no URL found")
                continue

            try:
                response = requests.get(pdf_url)
                response.raise_for_status()
                pdf_content = response.content

                pdf_part = MIMEApplication(pdf_content, _subtype="pdf")
                pdf_part.add_header('Content-Disposition', 'attachment', filename=f"report_{report.get('id', i)}.pdf")
                msg.attach(pdf_part)
            except Exception as download_err:
                print(f"Failed to download or attach report {i} from {pdf_url}: {download_err}")
                continue

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(msg)

        return JSONResponse(content={"status": "Email sent successfully."})

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Database connection creation
# Initialize the multi-database agent
from database_agent import MultiDatabaseAgent, ChatRequest, ChatResponse, logger

multi_db_agent = MultiDatabaseAgent()


@app.post("/api/database_chat", response_model=ChatResponse)
async def database_chat(request: ChatRequest):
    # Generate session ID if not provided, otherwise use the provided one
    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = await multi_db_agent.process_message(session_id, request.message)

        formatted_response=await llm_format_response(user_query=request.message, response=response.response)
        print(f"Formatted response: {formatted_response}")
        return JSONResponse(
            content={
                "session_id": session_id,
                "response": markdown_to_html(formatted_response),
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


async def llm_format_response(user_query: str, response: str) -> str:
    """
    Formats the LLM response to include user query and response in a structured way.
    """
    prompt = f"""
                You are a helpful formatting assistant.
            The user asked the following question:
            "{user_query}"

            Here is the raw response:
            \"\"\"{json.dumps(response)}\"\"\"

            -Just consider the "data" key only in the raw respose.

            Determine if the user expects the response in:
            - bullet points (if asking for "list", "top items", "options", etc.)
            - a table (if asking for "comparison", "tabular", "table", "data", etc.)
            - plain answer otherwise
            
            -You can add a one liner explanation before the response about the response.
            Convert the response to that format using HTML (use <ul>/<li> for lists, <table> for tables, <p> for plain).
            Reply with only the formatted HTML content.
            """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "you are a responsive formatting assistant"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("final_akio_apis:app", host="127.0.0.1", port=8000, reload=True)


