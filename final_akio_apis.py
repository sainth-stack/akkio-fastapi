import ast
import base64
import io
import os
import re
import smtplib
import sys
import tempfile
import traceback
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from img_datascout import ImageGen_agent
from datascout import DataScout_agent, extract_sections_tool, extract_num_pages_tool, \
    pdf_generator_tool,initialize_llm
from database import PostgresDatabase
from synthetic_data_function import generate_synthetic_data
from models import DBConnectionRequest
from typing import List, Dict, Any, Optional
from openai import OpenAI
from plotly.graph_objs import Figure
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

global connection_obj

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


# 1.Create Connection api ---for creating the connection with the database
@app.post("/connection")
async def connection(request: DBConnectionRequest):
    global connection_obj
    connection_obj = db.create_connection(request.username, request.password, request.database, request.host,
                                          request.port)
    print(connection_obj)
    return JSONResponse(content={"tables": str(connection_obj)})


# 2.File upload only-------- It is  useful for uploading the file
@app.post("/upload_and_store_data")
async def upload_and_store_data(
        request: Request,
        mail: str = Form(...),
        file: UploadFile = File(...)
):
    try:
        print("[DEBUG] Received a request with method: POST")  # Debug statement

        if not file:
            raise HTTPException(status_code=400, detail="No files uploaded")

        file_name = file.filename
        file_extension = os.path.splitext(file_name)[1].lower()  # Extract file extension

        try:
            # Create a directory for storing uploaded files
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)

            # Save the uploaded file locally
            local_file_path = os.path.join(upload_dir, file_name)
            with open(local_file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Process the uploaded file based on its extension
            if file_extension == ".csv":
                print("[DEBUG] Processing as CSV file...")  # Debug statement
                file.file.seek(0)  # Reset file pointer
                content = (await file.read()).decode("utf-8")
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)
                print("[DEBUG] CSV parsed successfully. DataFrame shape:", df.shape)  # Debug statement

            elif file_extension in [".xls", ".xlsx"]:
                print("[DEBUG] Processing as Excel file...")  # Debug statement
                df = pd.read_excel(local_file_path)  # Read from the saved local file
                print("[DEBUG] Excel parsed successfully. DataFrame shape:", df.shape)  # Debug statement

            else:
                file.file.seek(0)  # Reset file pointer
                content = (await file.read()).decode("utf-8")
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)

            # Validate the DataFrame
            if df.empty:
                raise HTTPException(status_code=400, detail="Uploaded file contains no data")

            csv_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.csv').lower())
            df.to_csv(csv_file_path, index=False)

            excel_file_path = os.path.join(upload_dir, file_name.replace(file_extension, '.xlsx').lower())
            df.to_excel(excel_file_path, index=False, engine='openpyxl')

            df.to_csv('data.csv', index=False)
            df.to_excel('data1.xlsx', index=False, engine='openpyxl')

            print("upload_and_store_data........", df.head(3))

            # Store data directly into the database
            print("[DEBUG] Storing data into the database...")  # Debug statement
            results = db.insert_or_update(mail, df, file_name)  # Insert into MongoDB
            print("[DEBUG] Database operation results:", results)  # Debug statement

            # Prepare response
            response_data = {
                "message": "File uploaded, stored locally, and data saved to the database successfully",
                "upload_status": results,
                "preview": df.head(10).to_dict(orient="records"),
            }

            return JSONResponse(content=response_data)

        except Exception as e:
            print("[ERROR] Failed to process and store file:", str(e))  # Debug statement
            raise HTTPException(status_code=500, detail=f"Failed to process and store file: {str(e)}")

    except Exception as e:
        print("[ERROR] An error occurred:", str(e))  # Debug statement
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# 3.Table info getting-----------getting the information of the tables
@app.post("/get_tableinfo")
async def get_table_info():
    table_info = db.get_tables_info()
    return JSONResponse(content=jsonable_encoder(table_info))


# 4.Reading table data-------Reading the table data.
@app.post("/read_data")
async def read_data(table_name: str = Form(...)):
    try:
        # Get data from database
        df = db.get_table_data(table_name)

        # Save to CSV files
        df.to_csv('data.csv', index=False)
        os.makedirs("uploads", exist_ok=True)  # Ensure uploads directory exists
        df.to_csv(os.path.join("uploads", f"{table_name.lower()}.csv"), index=False)

        # Return JSON response
        return JSONResponse(content=df.to_dict(orient="records"))

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Table not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading data: {str(e)}"
        )


# 5.Get user data based on the mail----------get the table data of the user based on the email
@app.post("/get_user_data")
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


# 6.Deleting the user-specific list of tables-----------------Deleting the list of tables corresponding to the specific user
@app.post("/delete_selected_tables_by_name/")
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


# 7.Deleting all user tables----------------Deleting all user tables
@app.post("/delete_all_user_tables/")
async def delete_all_user_tables(email: str = Form(...)):
    try:
        print(f"Received email for deletion: {email}")  # Debug statement

        # Call database operation
        print(f"Calling delete_user_tables method with email: {email}")
        deletion_status = db.delete_all_tables_data(email)

        if deletion_status:
            msg = f"All tables associated with email '{email}' have been deleted."
            print(msg)
            return JSONResponse(
                content={"message": msg},
                status_code=status.HTTP_200_OK
            )
        else:
            msg = f"No tables found for email '{email}' or an error occurred."
            print(msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=msg
            )

    except Exception as e:
        error_msg = f"Exception occurred while deleting tables: {str(e)}"
        print(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )


# 8.Generating the test response-----------------Generating the text response for the given user query(txt to sql)
@app.post("/gen_txt_response")
async def gen_txt_response(query: str = Form(...)):
    try:
        # Load CSV data
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        # Create the prompt
        prompt_eng = (
            f"""
                You are an expert Data Analyst AI. Your mission is to answer a user's query about a dataset. Always strictly adhere to the following rules:               
                1. Generic Queries:
                    If the user's query is generic and not related to data, respond with a concise and appropriate print statement. For example:

                    Query: "What is AI?"
                    Response: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."
                2. Data-Related Queries:
                    If the query is about data processing, assume the file data.csv is the data source and contains the following columns: {metadata_str}.

                    For these queries, respond with Python code only, no additional explanations.
                    The code should:

                    Load data.csv using pandas.
                    Perform operations to directly address the query.
                    Exclude plotting, visualization, or other unnecessary steps.
                    Include comments for key steps in the code.
                    Example:

                    Query: "How can I filter rows where 'Column1' > 100?"
                    Response:
                    python
                    Copy code
                    import pandas as pd

                    # Load the dataset
                    data = pd.read_csv('data.csv')

                    # Filter rows where 'Column1' > 100
                    filtered_data = data[data['Column1'] > 100]

                    # Output the result
                    print(filtered_data)

                3. Theoretical Concepts:
                    For theoretical questions, provide a brief explanation as a print statement. Keep the explanation concise and focused.

                    Example:

                    Query: "What is normalization in data preprocessing?"
                    Response:
                    "Normalization is a data preprocessing technique used to scale numeric data within a specific range, typically [0, 1], to ensure all features contribute equally to the model."

                4. For Analytical & Insight Queries:
                    -If the user asks for a summary, aggregation, finding, or "what is/are" type of question, provide a direct and concise **textual answer**.
                    - Your answer must be derived from the data in the `df` DataFrame.
                    - Provide the Python code you used to find the answer.

                Never reply with: "Understood!" or similar confirmations. Always directly respond to the query following the above rules.

                User query is {query}.
                    """
        )

        # Generate and execute code
        code = generate_code(prompt_eng)
        result = execute_py_code(code, df)

        return JSONResponse(content={"answer": result})

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
            detail=f"Error processing request: {str(e)}"
        )


# Function to generate code from OpenAI API
def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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


def execute_py_code(code: str, df: pd.DataFrame) -> str:
    # --- Pre-flight Syntax Check using AST ---
    try:
        ast.parse(code)
    except SyntaxError as e:
        # If ast.parse fails, it's not valid Python code.
        return f"Syntax Error: The provided text is not valid Python code. Details: {e}"

    # --- Execution Logic ---
    buffer = io.StringIO()
    sys.stdout = buffer
    local_vars = {'df': df, 'pd': pd}  # Include pandas for convenience

    try:
        # If syntax is valid, execute the code
        exec(code, globals(), local_vars)
        output = buffer.getvalue().strip()

        # This part is now less critical because the LLM prompt should enforce a print(),
        # but it's good fallback logic.
        if not output:
            try:
                last_line = code.strip().split('\n')[-1]
                if not last_line.startswith(('print', '#')):
                    # eval the last line to get its value
                    eval_output = eval(last_line, globals(), local_vars)
                    output = str(eval_output)
            except Exception:
                # If eval fails, it means there was no returnable expression.
                output = "Code executed successfully with no output."

    except Exception as e:
        output = f"Runtime Error: {str(e)}"
    finally:
        # Always reset stdout
        sys.stdout = sys.__stdout__

    return output


# 9.Get flespi data---------------------------------it is for the flespi data
@app.post("/download_flespi_data/")
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


# Dashboard apis--------------------Dashboard related api for generating the dynamic graphs
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


def make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects to serializable formats."""
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


# 10.Dashboard main api
@app.post("/gen_plotly_response/")
async def gen_plotly_response() -> JSONResponse:
    try:
        # Load and process data
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)

        # Infer metadata and generate topics
        metadata = infer_metadata(df)
        print("Metadata:", metadata)

        topics = generate_topics_llm(metadata)
        print("Generated topics:", topics)

        chart_responses = []

        # Initialize all chart files as empty
        for filename in FIXED_CHART_FILENAMES:
            chart_path = os.path.join(CHARTS_DIR, filename)
            with open(chart_path, "w") as f:
                json.dump({}, f)

        # Process each topic to generate charts
        for i, topic in enumerate(topics[:6]):
            print(f"Processing topic {i + 1}: {topic}")

            prompt_eng = (
                f"You are an AI specialized in data analytics and visualization.\n"
                f"Data used for analysis is stored in 'data.csv' with attributes: {metadata}.\n"
                f"Based on the topic '{topic}', generate Python code using Plotly to create:\n"
                f"- Basic, easily understandable graphs (no scatter plots)\n"
                f"- Rich, attractive colors\n"
                f"- Meaningful visualization for the topic\n"
                f"Output must be a Plotly Figure object named 'fig'."
            )

            try:
                chat = generate_code4(prompt_eng)
                print(f"Generated code:\n{chat}")

                if 'import' not in chat:
                    raise ValueError("Invalid AI response - missing imports")

                namespace = {}
                exec(chat, namespace)
                fig = namespace.get("fig")

                if not fig or not isinstance(fig, Figure):
                    raise ValueError("No valid Plotly figure generated")

                # Process and save chart data
                chart_data = fig.to_plotly_json()
                chart_data_serializable = make_serializable(chart_data)
                chart_filename = FIXED_CHART_FILENAMES[i]
                chart_path = os.path.join(CHARTS_DIR, chart_filename)

                # Save individual chart file
                with open(chart_path, "w", encoding="utf-8") as f:
                    json.dump(chart_data_serializable, f, indent=2, ensure_ascii=False)

                chart_responses.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": topic["type"],
                    "chart_data": chart_data_serializable,
                    "chart_file": chart_filename,
                    "status": "success"
                })

            except Exception as e:
                print(f"Error processing topic '{topic}': {str(e)}")
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
            "total_charts": len(FIXED_CHART_FILENAMES),
            "chart_files": FIXED_CHART_FILENAMES,
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


def infer_metadata(df: pd.DataFrame) -> Dict:
    meta = {
        "columns": [],
        "correlation": df.select_dtypes(include='number').corr().to_dict(),
        "shape": df.shape,
        "total_nulls": int(df.isnull().sum().sum())
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_series = df[col].dropna()
        sample_values = non_null_series.sample(min(3, len(non_null_series)), random_state=1).tolist() if len(
            non_null_series) > 0 else []

        col_meta = {
            "name": col,
            "dtype": dtype,
            "nulls": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "example_values": sample_values
        }

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            col_meta['type'] = 'datetime'
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
            col_meta['type'] = 'categorical'
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_meta['type'] = 'numerical'
        else:
            col_meta['type'] = 'other'

        meta["columns"].append(col_meta)

    return meta


def generate_topics_llm(meta: Dict) -> List[Dict]:
    prompt = f"""
    Given the following metadata about a dataset:
    {json.dumps(meta, indent=2)}

        For each visualization topic, return a dictionary with:
    - `title`: A clear, concise, and human-readable title for the chart (e.g., "Distribution of Price", "Sales over Time").
    - `type`: A basic chart type that best fits the topic, such as  `bar`,`line`, `scatter`, `histogram`, `heatmap`, `pie`, `box`, etc.
    - `columns`: A list of column names from the dataset that are relevant to the chart.

    Your response must be a **valid JSON list of exactly six such dictionaries**. Avoid duplication across the topics.

    In choosing the topics, apply the following reasoning:
    - Identify key metrics for **distribution analysis** (e.g., using histograms, box plots).
    - Use **relationships or comparisons** (e.g., category vs value, correlation between two columns).
    - If there's a time/datetime column, include **time series analysis** (e.g., line plots).
    - Include at least one **summary view**, such as a heatmap of numerical correlations.
    - Choose **basic visualizations only** that are clear and accessible to a general user audience.
    -Do not select the topics which we cannot able to draw the plot.
    - Do not give the topics repeatedly.Give the topics uniquely  for the generation of the graphs.

    Ensure your choices adapt dynamically to each metadata input and avoid repeating the same topic titles across calls.

    Always return output in valid JSON format with no additional text.

    For Each requests,the topics should be dynamically changed leads to the different types of analysis in various scenarios.
    Give the topics in which the user can easily understand with the help of visualisations.Just give the topics for basic analysis only.
    You must give the six insightful data visualization topics.
    Return a valid JSON list of dictionaries only.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a data visualization expert which can give basic topic names for the visualising of the plots.You specialize in helping users generate insightful and beginner-friendly data analysis ideas based on dataset metadata."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    try:
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        return json.loads(content)
    except Exception as e:
        print("Error parsing LLM response:", e)
        return []


# Function to generate code from OpenAI API
def generate_code4(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"""
                                        You are a helpful coding assistant named VizCopilot. You're an expert in Python and specialize in generating interactive visualizations using Plotly (both Plotly Express and Plotly Graph Objects).

                                        Your job is to:
                                        - Understand the user's data context, visualization goals.
                                        - Generate full, working Plotly code snippets using best practices (readable, maintainable, idiomatic).
                                        - Use `plotly.express` for standard charts, and `plotly.graph_objects` when more customization is needed.
                                        - Include layout configuration for titles, labels, tooltips, and themes.
                                        - Always use a fully reproducible Python code block.
                                        - Format outputs in Markdown with proper syntax highlighting.

                                        When generating code (especially Plotly/Python):
                                        - Always produce **fully working, valid Python code**.
                                        - Use **correct imports**, avoid missing modules like `import plotly.express as px`, `import pandas as pd`, etc.
                                        - Never leave incomplete functions or syntax or rising of the serialisable json issues.
                                        - Validate your code logically before outputting it.
                                        - Always close brackets, function calls, and maintain indentation properly.

                                        You aim to make data visualization with Plotly fast, clear, and interactive. Never skip code steps. Always generate complete, working code.Do not give errors while executing the code.
                                          """},
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


# 11.Get summary api-------------getting the summary of the above generated graphs
# In-memory cache for summaries
SUMMARY_CACHE: Dict[str, str] = {}


@app.post("/summarize_chart/")
async def summarize_chart(chart_id: str = Form(...)) -> JSONResponse:
    try:
        # Validate chart ID
        try:
            chart_num = int(chart_id)
            if not 1 <= chart_num <= 6:
                raise ValueError
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid chart ID. Must be integer between 1-6"
            )

        # Check if summary exists in cache
        if chart_id in SUMMARY_CACHE:
            return JSONResponse(
                content={
                    "chart_id": chart_id,
                    "summary": markdown_to_html(SUMMARY_CACHE[chart_id]),
                    "cached": True
                },
                status_code=200
            )

        # Locate chart file
        filename = f"chart_{chart_id}.json"
        chart_path = os.path.join(CHARTS_DIR, filename)

        if not os.path.exists(chart_path):
            raise HTTPException(
                status_code=404,
                detail="Chart not found"
            )

        # Load chart data
        with open(chart_path, "r", encoding="utf-8") as f:
            chart_json = json.load(f)

        # Generate analysis prompt
        prompt = (
            f"You are a data analyst AI. A user selected a chart represented by this Plotly JSON:\n{json.dumps(chart_json)}\n"
            f"Analyze and summarize only the insights, patterns, and trends that are directly visible in the chart.\n\n"
            f"Follow this output structure:\n"
            f"- Start with a core insight derived from the graph. Bold important terms where needed.\n"
            f"- Describe distribution patterns if visible (e.g., skewness, outliers, clusters).\n"
            f"- Explain what real-world behavior the graph appears to reflect (only if clearly supported by the data).\n"
            f"- Mention modeling implications, **only if they are suggested by the visual pattern**.\n"
            f"- If any transformation effect is evident from the graph (e.g., log-scale, smoothing), describe it clearly.\n"
            f"  Use a code block to describe its effect:\n"
            f"```\n"
            f"- Point 1\n"
            f"- Point 2\n"
            f"- Point 3\n"
            f"```\n"
            f"- Mention any business insight that is clearly supported by the visual.\n"
            f"- Suggest focused actions based on the graph's trends (e.g., rising spikes, drop-offs, correlation zones).\n\n"
            f"Only describe what you observe from the chart. Do not invent data or generalize beyond the chart. Use clean bullet points. No section headings. No intro or conclusion."
        )

        # Generate and cache summary
        summary = generate_text(prompt)
        SUMMARY_CACHE[chart_id] = summary

        return JSONResponse(
            content={
                "chart_id": chart_id,
                "summary": markdown_to_html(summary),
                "cached": False
            },
            status_code=200
        )

    except HTTPException:
        raise  # Re-raise already handled exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summary generation failed: {str(e)}"
        )


def markdown_to_html(md_text):
    html_text = markdown.markdown(md_text)
    return html_text


def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are a helpful data analyst that explains data visualizations and user queries and write insightful summary for the given data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


# 12.Ask about chart--------------asking the questions for the above summary
@app.post("/ask_about_chart/")
async def ask_about_chart(
        chart_id: str = Form(...),
        question: str = Form(...)
) -> JSONResponse:
    try:
        # Validate inputs
        if not chart_id or not question:
            raise HTTPException(
                status_code=400,
                detail="Both chart_id and question are required"
            )

        # Check if summary exists in cache
        summary = SUMMARY_CACHE.get(chart_id)
        if not summary:
            raise HTTPException(
                status_code=400,
                detail="No summary available. Call /summarize_chart first."
            )

        # Generate answer prompt with strict requirements
        prompt = (
            f"Chart Summary Reference:\n{summary}\n\n"
            f"User Question: {question}\n\n"
            f"Provide a concise answer with exactly these elements:\n"
            f"1. First bullet point (key insight)\n"
            f"2. Second bullet point (supporting detail)\n"
            f"3. Third bullet point (action/implication)\n"
            f"- Each point must be under 15 words\n"
            f"- Use simple language\n"
            f"- Only include information visible in the chart\n"
            f"- Format strictly as:\n"
            f"• Point 1\n• Point 2\n• Point 3"
        )

        answer = generate_text(prompt)

        # Validate answer format
        bullet_points = [line.strip() for line in answer.split('\n') if line.startswith('•')]
        if len(bullet_points) != 3:
            answer = "• Key insight\n• Supporting detail\n• Recommended action"
            print(f"Invalid answer format from LLM. Using fallback response.")

        return JSONResponse(
            content={
                "chart_id": chart_id,
                "question": question,
                "answer": answer
            },
            status_code=200
        )

    except HTTPException:
        raise  # Re-raise already handled exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )


# 13.Filling missing data------------------Evaluating the missed data in the dataframe
@app.post("/missing_data/")
async def handle_missing_data() -> JSONResponse:
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
        new_df, html_df = process_missing_data(df.copy())

        # Save processed data
        processed_path = os.path.join('uploads', 'processed_data.csv')
        new_df.to_csv(processed_path, index=False)
        new_df.to_csv("data.csv", index=False)

        # Save HTML representation
        with open(os.path.join('mvt_data.json'), 'w') as fp:
            json.dump({'data': html_df}, fp, indent=4)

        return JSONResponse(
            content={"df": html_df},
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
    df, html_df = handle_missing_data(df)
    return df, html_df


def convert_to_datetime(df):
    for col in df.columns:
        if df[col].dtype == "object":  # Process only string columns
            if df[col].str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", na=False).any():
                df[col] = df[col].apply(detect_and_parse_date)

    return df


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


# 14Genai bot plotly visualisation.-----------------Prediction and forecasting related api
@app.post("/gen_ai_bot/")
async def gen_ai_bot(request: Request) -> JSONResponse:
    """
    AI bot endpoint for data analysis, forecasting, and predictions

    Handles:
    - Data analysis queries
    - Forecasting requests
    - Prediction requests
    - Visualization generation

    Returns:
        JSONResponse: Response varies by request type
    """
    try:
        # Load and prepare data
        df = pd.read_csv('data.csv')
        metadata_str = ", ".join(df.columns.tolist())
        sample_data = df.head(2).to_dict(orient='records')

        # Get prompt from request
        content_type = request.headers.get('Content-Type')
        if content_type == "application/json":
            body = await request.json()
            prompt = body.get('prompt')
        else:
            form_data = await request.form()
            prompt = form_data.get('prompt')

        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required"
            )

        # Handle forecasting requests
        if 'forecast' in prompt.lower():
            data = extract_forecast_details_llm(prompt, df.columns)
            stat, data, img_data = arima_train(df, data['target_variable'], data)

            return JSONResponse({
                'data': json.loads(data.to_json()),
                'plot': make_serializable(img_data)
            })

        # Handle prediction requests
        elif 'predict' in prompt.lower():
            data = extract_forecast_details_rf(prompt, df.columns)

            if len(data.get('missing_columns', [])) > 0:
                return JSONResponse({
                    'text_pre_code_response': (
                        f'Prediction failed due to missing fields: {data.get("missing_columns")}. '
                        f'Please retry with all required inputs.')
                })

            model_path = os.path.join("models", "rf", data['target_column'])
            pipeline_path = os.path.join(model_path, "pipeline.pkl")
            deployment_path = os.path.join(model_path, "deployment.json")

            # Check and retrain model if needed
            if not os.path.exists(deployment_path) or not os.path.exists(pipeline_path):
                df = pd.read_csv('data.csv')
                _ = random_forest(df, data.get('target_column'))

            # Make prediction
            df_predict = pd.DataFrame([data.get('features')])
            loaded_pipeline = load_pipeline(pipeline_path)
            predictions = loaded_pipeline.predict(df_predict)

            return JSONResponse({
                "text_pre_code_response": f"Predicted {data.get('target_column')} value is {round(predictions[0], 2)}"
            })

        # Handle general data analysis requests
        else:
            system_prompt = f"""You are an AI specialized in data analytics and visualization. The data for analysis is 
            stored in a CSV file named data.csv, with the following attributes: {metadata_str} and sample data as 
            {sample_data}.

            Follow these rules while responding to user queries:
            1. Strictly use 'data.csv' as the data source
            2. For numerical insights, extract data and provide concise summaries
            3. For visualizations, generate Plotly code with fig as output
            4. For forecasting, use ARIMA and store results
            """

            result: Dict[str, Any] = {}
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            pre_code_text, post_code_text, code = process_genai_response(response)
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
                    raise HTTPException(
                        status_code=500,
                        detail=f"Code execution failed: {str(e)}"
                    )

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
            model="gpt-4o-mini",
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
            model="gpt-4o-mini",
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


def arima_train(data, target_col, bot_query=None):
    try:
        print('ArimaTrain')
        print("Column dtypes:\n", data.dtypes)
        # Identify date column by checking for datetime type
        date_column = None
        results = {}
        if not os.path.exists(os.path.join("models", 'Arima', target_col)):
            for col in data.columns:
                if data.dtypes[col] == 'object':
                    try:
                        # Attempt to convert column to datetime
                        pd.to_datetime(data[col])
                        date_column = col
                        break
                    except (ValueError, TypeError):
                        continue
            if not date_column:
                raise ValueError("No datetime column found in the dataset.")
            print(date_column)
            # Set the date column as index
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)

            try:
                data_actual = data[[target_col]]
                data_actual.reset_index(inplace=True)
                data_actual.columns = ["datetime", 'value']
                data_actual.set_index("datetime", inplace=True)
                train_frequency = check_data_frequency(data_actual)

                train_models(data_actual, target_col)

                with open(os.path.join("models", 'Arima', target_col, target_col + '_results.json'), 'w') as fp:
                    json.dump({'data_freq': train_frequency}, fp, indent=4)
                # result_graph = plot_graph(results, os.path.join('models', 'arima', target_col))
            except Exception as e:
                print(e)

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
        seasonality = detect_seasonality(train)

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
    model = ARIMA(train['value'], order=(1, 1, 1)).fit()
    pred = model.predict(start=test.index[0], end=test.index[-1])
    error = mean_squared_error(test['value'], pred, squared=False)
    return model, error


# Train Prophet Model
def train_prophet(train, test):
    print("Training Prophet...")
    prophet_df = train.reset_index().rename(columns={'datetime': 'ds', 'value': 'y'})
    model = Prophet()
    model.fit(prophet_df)

    future = pd.DataFrame({'ds': test.index})
    forecast = model.predict(future)
    error = mean_squared_error(test['value'], forecast['yhat'], squared=False)
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
    error = mean_squared_error(test['value'], pred, squared=False)
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
    error = mean_squared_error(test['value'], pred, squared=False)
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
        if not os.path.exists(os.path.join("models", "rf", target_column, 'deployment.json')):
            os.makedirs(os.path.join("models", "rf", target_column), exist_ok=True)
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if y.nunique() <= 5:  # Classification for few unique target values
                model_type = 'Classification'
                model = RandomForestClassifier(random_state=42)
            else:  # Regression for continuous target values
                model_type = 'Regression'
                model = RandomForestRegressor(random_state=42)

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

            # Save the pipeline
            joblib.dump(pipeline, os.path.join("models", "rf", target_column, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join("models", "rf", target_column, "pipeline.pkl")}')

            with open(os.path.join("models", "rf", target_column, "deployment.json"), "w") as fp:
                json.dump({"columns": list(X_train.columns), "model_type": model_type, "Target_column": target_column},
                          fp, indent=4)
            return True, list(X_train.columns)
        else:
            with open(os.path.join(os.getcwd(), "models", "rf", target_column, 'deployment.json'), "r") as fp:
                data = json.load(fp)
            return True, data['columns']
    except Exception as e:
        print(e)
        return False, []


# 15.Get Columns Description----------------Getting the columns description from the dataset.
@app.post("/col_description/")
async def get_column_descriptions() -> JSONResponse:
    try:
        # Load and validate data
        csv_file_path = 'data.csv'
        if not os.path.exists(csv_file_path):
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )

        df = pd.read_csv(csv_file_path)
        print("Data preview:")
        print(df.head(5))

        # Generate column descriptions
        prompt_eng = (
            f"You are analytics_bot. Analyze the data: {df.head()} and provide:\n"
            f"1. Column name\n"
            f"2. Description (1-2 sentences)\n"
            f"3. Example values\n"
            f"Format as markdown bullet points:\n"
            f"- **Column1**: Description...\n  Examples: val1, val2\n"
            f"- **Column2**: Description...\n  Examples: val1, val2\n"
            f"Only include columns present in the data."
        )

        column_description = generate_code(prompt_eng)

        return JSONResponse(
            content={"Column_description": markdown_to_html(column_description)},
            status_code=200
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="CSV file is empty or corrupt"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate descriptions: {str(e)}"
        )


# 16.Data scout api----------------------Data scout api for generating the data
@app.post("/create_data/")
async def create_data_with_data_scout(
        prompt: str = Form(...),
        data_type: str = Form(...),
        file: Optional[UploadFile] = File(None)
) -> JSONResponse:
    """
    Create data based on prompt and type (excel, pdf, or image)

    Args:
        prompt: The input prompt for data generation
        data_type: Type of data to create (excel/pdf/image)
        file: Optional file upload for image/pdf generation

    Returns:
        JSONResponse: Generated data in appropriate format
    """
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
                return JSONResponse(content={"message": result["output"]})
            raise HTTPException(
                status_code=500,
                detail="Unexpected result format from agent"
            )

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


# 17.Data preprocess api----------------------Data preprocessing for the generation of the statistical analysis.
@app.get("/data_processing/")
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

        # Build response
        response = {
            'nof_rows': str(nor),
            'nof_columns': str(nof),
            'timestamp': timestamp,
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


# 18.Synthetic Data Extended api
@app.post("/synthetic_data/extended/")
async def handle_synthetic_data_extended(
        file: UploadFile,
        user_prompt: str = Form(...)
) -> StreamingResponse:
    print("[DEBUG] Entered handle_synthetic_data_extended")
    temp_file_name = None

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key not configured"
            )

        print(f"[DEBUG] Uploaded file: {file.filename}")
        print(f"[DEBUG] User prompt: {user_prompt}")

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
                detail="Unsupported file format. Please upload Excel or CSV."
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

        # Stream CSV response
        stream = io.StringIO()
        combined_df.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="synthetic_output.csv"'
            }
        )

        print("[DEBUG] Successfully generated synthetic data response.")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Data generation failed: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)
            print(f"[DEBUG] Temporary file {temp_file_name} deleted.")


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
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
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


# 19.Generating extended images
@app.post("/generate_synthetic_images/")
async def generate_synthetic_images(
        images: List[UploadFile],
        user_prompt: str = Form(...)
) -> JSONResponse:
    """
    Generate synthetic images based on uploaded examples and user prompt

    Args:
        images: List of uploaded reference images
        user_prompt: Instructions for image generation

    Returns:
        JSONResponse: Contains generated images in base64 format and metadata
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing OpenAI API key"
        )

    if not images:
        raise HTTPException(
            status_code=400,
            detail="No images uploaded"
        )

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
    client = OpenAI(api_key=openai_api_key)

    try:
        # Analyze all images for comprehensive style
        print("Analyzing uploaded images for style...")
        comprehensive_style = await analyze_all_images_for_style(images, openai_api_key)
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
                'message': f'Generated {total_generated} images',
                'total_generated': total_generated,
                'images': generated_images_info
            }
        )

    except Exception as e:
        print(f"Image generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )


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
            image = Image.open(image_file).convert("RGB")
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
        analysis_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
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
    vision_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
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
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
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


#20.Synthetic_pdf generation
@app.post("/generate_synthetic_pdfs/")
async def generate_synthetic_pdfs(
        pdf: UploadFile,
        user_prompt: str = Form(...)
) -> JSONResponse:
    """
    Generate extended PDF documents based on uploaded PDF and user prompt

    Args:
        pdf: Uploaded PDF file
        user_prompt: Instructions for document generation

    Returns:
        JSONResponse: Contains structured document with generated content
    """
    # Validate API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing OpenAI API key"
        )

    # Validate PDF upload
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted."
        )

    # Extract target pages from prompt
    target_pages = extract_pdf_prompt_semantics(user_prompt)
    if not target_pages:
        raise HTTPException(
            status_code=400,
            detail="Could not determine target page count from prompt"
        )

    try:
        # 1. Process uploaded PDF
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

        # 2. Generate extended document
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
            content={"structured_document": final_document},
            status_code=200
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
