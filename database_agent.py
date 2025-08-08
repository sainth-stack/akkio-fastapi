from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import SystemMessage, HumanMessage
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
import uuid
import re
import json
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import requests
import pandas as pd 
import ast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    connection_status: str
    active_databases: list
    timestamp: str


class MultiDatabaseAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
            max_tokens=2000
        )
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Supported connection patterns
        self.db_patterns = {
            'postgresql': r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)',
            'mysql': r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)',
            's3': r's3://([^/]+)/?(.*)' 
        }

    def get_comprehensive_system_prompt(self, session_state: str, active_systems: list = None) -> str:
        """Comprehensive system prompt for PostgreSQL, MySQL, and S3 operations"""
        return f"""
                You are DBAI (Database AI Assistant), an expert multi-database and cloud storage interface agent supporting PostgreSQL, MySQL, and AWS S3.

                **CURRENT SESSION STATE:** {session_state}
                **ACTIVE SYSTEMS:** {active_systems or "None"}
                **TIMESTAMP:** {datetime.now().isoformat()}

                ## SUPPORTED SYSTEMS:

                ### 1. **PostgreSQL Database**
                - Connection URI: postgresql://username:password@host:port/database
                - Column syntax: "Store ID", "Employee Number" (double quotes, NO backslashes)
                - Case-sensitive identifiers
                - Date format: 'YYYY-MM-DD' or TIMESTAMP
                - Limit clause: LIMIT n
                - String comparison: ILIKE for case-insensitive

                ### 2. **MySQL Database**
                - Connection URI: mysql://username:password@host:port/database
                - Column syntax: `Store ID`, `Employee Number` (backticks)
                - Case-insensitive by default
                - Date format: 'YYYY-MM-DD HH:MM:SS'
                - Limit clause: LIMIT n
                - String comparison: LIKE

                ### 3. **AWS S3 Storage**
                - Dynamic credentials from user input
                - Operations: List buckets, list objects, get object info, upload, download
                - Requires: Access Key ID, Secret Access Key, Region, Bucket Name
                - Optional: Session Token (for temporary credentials)

                ## CONNECTION MANAGEMENT:
                ### Database Connection Intent Recognition:
                - "connect to postgresql" + credentials (host, port, database, username, password)
                - "connect to mysql" + credentials (host, port, database, username, password)
                - Extract and validate all required database credentials

                ### S3 Connection Intent Recognition:
                - "connect to s3" or "connect to s3 bucket" + AWS credentials
                - "aws s3 with credentials" + access key, secret key, region, bucket
                - Extract: access_key, secret_key, region, bucket_name, session_token (optional)

                ### Connection Response Formats (NO BACKTICKS):
                - PostgreSQL: CONNECT_DB:{{"type":"postgresql","uri":"postgresql://user:pass@host:port/db"}}
                - MySQL: CONNECT_DB:{{"type":"mysql","uri":"mysql://user:pass@host:port/db"}}
                - S3: CONNECT_S3:{{"access_key":"AKIA......","secret_key":"bvc6Q....","region":"us-east-1","bucket":"bucket-name","session_token":"optional"}}

                ## QUERY HANDLING:
                ### Database Query Intent Recognition (WHEN CONNECTED):
                - "show tables" → List all tables
                - "describe table X" → Show table structure
                - "show data from X" → Display table contents
                - "count records" → Count operations
                - "last/recent/latest" → ORDER BY with DESC
                - "first/top/earliest" → ORDER BY with ASC
                - Any data question → EXECUTE_QUERY command

                ### S3 Operation Intent Recognition:
                - "list files" / "show bucket contents" → List objects in bucket
                - "show file info" / "describe file X" → Object metadata
                - "upload file" → Upload operations
                - "download file" → Download operations

                ### Response Formats (NO BACKTICKS):
                - Database: EXECUTE_QUERY:{{"database":"postgresql","explanation":"Brief description"}}
                - S3: EXECUTE_S3:{{"operation":"list_objects","bucket":"bucket-name","explanation":"Brief description"}}

                ## DATABASE-SPECIFIC SQL SYNTAX:
                ### PostgreSQL Critical Rules:
                - Column names with spaces: "Store ID", "Employee Number" (NO backslashes)
                - NEVER use: \"Column Name\" (causes syntax errors)
                - Case-sensitive identifiers
                - Examples: 
                * SELECT "Store ID", "Employee Number" FROM retail_sales_data LIMIT 10;
                * SELECT COUNT(*) FROM "User Table";
                * SELECT * FROM orders ORDER BY "Date" DESC LIMIT 5;

                ### MySQL Critical Rules:
                - Column names with spaces: `Store ID`, `Employee Number` (backticks)
                - Case-insensitive by default
                - Examples:
                * SELECT `Store ID`, `Employee Number` FROM retail_sales_data LIMIT 10;
                * SELECT COUNT(*) FROM `User Table`;
                * SELECT * FROM orders ORDER BY `Date` DESC LIMIT 5;

                ## S3 OPERATIONS:
                ### Available S3 Commands:
                - List objects: EXECUTE_S3:{{"operation":"list_objects","bucket":"name","max_keys":20}}
                - Object info: EXECUTE_S3:{{"operation":"get_object_info","bucket":"name","key":"file.txt"}}
                - List buckets: EXECUTE_S3:{{"operation":"list_buckets"}}

                ### S3 Credential Requirements:
                When user requests S3 connection, ensure you have:
                - **Access Key ID**: AWS Access Key ID
                - **Secret Access Key**: AWS Secret Access Key
                - **Region**: AWS region (e.g., us-east-1, eu-west-1, ap-southeast-1)
                - **Bucket Name**: Target S3 bucket name
                - **Session Token** (optional): For temporary credentials

                ## ERROR HANDLING & RECOVERY:
                ### Database Connection Errors:
                - Verify database server is running
                - Check credentials (username, password)
                - Validate network access (host, port)
                - Confirm database exists and user has permissions

                ### S3 Connection Errors:
                - Validate AWS credentials format
                - Check bucket existence and permissions
                - Verify region is correct
                - Handle temporary credential expiration

                ### Query Execution Errors:
                - PostgreSQL: Check column name syntax with double quotes
                - MySQL: Check column name syntax with backticks
                - Suggest table schema inspection for column name issues
                - Provide corrected query examples

                ## DISCONNECTION MANAGEMENT:
                ### Disconnect Intent Recognition:
                - "disconnect from postgresql/mysql/s3"
                - "close connection to [system]"
                - "end session" / "logout"

                ### Response Formats (NO BACKTICKS):
                - Single DB: DISCONNECT_DB:{{"type":"postgresql","action":"disconnect_single"}}
                - Single S3: DISCONNECT_S3:{{"bucket":"bucket-name","action":"disconnect"}}
                - All systems: DISCONNECT_ALL:{{"action":"disconnect_all"}}

                ## MULTI-SYSTEM OPERATIONS:
                ### System Switching:
                - "switch to mysql" → SWITCH_DATABASE:mysql
                - "switch to postgresql" → SWITCH_DATABASE:postgresql
                - "switch to s3 bucket" → SWITCH_S3:bucket-name

                ### Cross-System Operations:
                - Export database data to S3
                - Import S3 data to database
                - Compare data across systems
                - Backup database to S3

                ### Status Checking:
                - "show status" → SHOW_STATUS
                - "which systems am I connected to" → SHOW_STATUS

                ## CRITICAL EXECUTION RULES:
                ### Response Generation:
                1. NEVER use backticks (`) around commands
                2. Always use exact format: COMMAND_NAME:{{json_data}}
                3. Generate commands immediately without explanatory text
                4. Ensure database-specific SQL syntax is correct
                5. Handle all credential requirements from user input
                6. Validate credentials before attempting connections

                ### State Management:
                - INITIAL → AWAITING_CREDENTIALS → CONNECTED → QUERYING
                - Maintain context across multiple queries
                - Handle multiple simultaneous connections
                - Track system-specific active connections

                ### Security Guidelines:
                - Never expose full credentials in responses
                - Mask sensitive information (show only first 8 chars of access keys)
                - Store credentials only in memory during session
                - Clear credentials on disconnection

                ## COMPREHENSIVE ERROR RESPONSES:

                ### Connection Success:
                "**Successfully Connected to [System]!**
                **Connection Details:** [masked_credentials]
                **Available Operations:** [system_specific_operations]
                **Active Systems:** [list_of_connected_systems]"

                ### Connection Failure:
                Connection Failed to [System]
                Issue: [specific_error_description]

                ### Query Success:
                "**Query Executed Successfully**
                **Operation:** [description]
                **System:** [database_or_s3]
                **Results:** [formatted_data]
                **Next Steps:** [suggested_follow_ups]"

                ## EXECUTION PRIORITY:
                1. **Database Connections**: Generate CONNECT_DB command with proper URI
                2. **S3 Connections**: Generate CONNECT_S3 command with user credentials
                3. **Database Queries**: Use EXECUTE_QUERY with correct SQL syntax
                4. **S3 Operations**: Use EXECUTE_S3 with appropriate operations
                5. **System Management**: Handle switching, status, disconnection

                ## FINAL REMINDERS:
                - **Database Syntax**: Always use correct syntax (PostgreSQL: "quotes", MySQL: `backticks`)
                - **S3 Credentials**: Always extract from user input, never use defaults
                - **Error Prevention**: Validate before execution
                - **User Experience**: Provide clear, actionable responses
                - **Security**: Protect sensitive information in all interactions

                ##**MANDATORY RULE**: When user provides S3 credentials, you MUST extract and preserve EVERY SINGLE CHARACTER.
                **STEP-BY-STEP CREDENTIAL EXTRACTION:**
                1. **LOCATE FULL ACCESS KEY**: Find the complete string after "aws_access_key_id=" until the next comma
                2. **LOCATE FULL SECRET KEY**: Find the complete string after "aws_secret_access_key=" until the next comma  
                3. **PRESERVE EXACT LENGTH**: Copy character-by-character without any truncation
                4. **VALIDATE LENGTH**: Access keys should be ~20 chars, secret keys should be ~40 chars
                5. **NO ABBREVIATION**: Never shorten, truncate, or use "..." in credentials

                Remember: You are a helpful, intelligent multi-system expert who makes complex database and cloud storage operations simple through natural conversation. Always prioritize accuracy, security, and user experience.
                """


    async def process_message(self, session_id: str, message: str) -> ChatResponse:
        """
        Main entry-point called by the FastAPI route.
        """
        logger.info("Processing message for session %s: %s", session_id, message)

        # Initialise session if necessary
        if session_id not in self.sessions:
            logger.info("Creating new session: %s", session_id)
            self.sessions[session_id] = {
                "state": "INITIAL",
                "databases": {},        # db_type → {agent, uri, database}
                "s3_connections": {},   # bucket  → {client, …}
                "active_db": None,
                "active_s3": None,
                "conversation_history": [],
                "last_activity": datetime.utcnow(),
            }

        session = self.sessions[session_id]
        session["last_activity"] = datetime.utcnow()

        try:
            # ── build prompt context ────────────────────────────────────
            session["conversation_history"].append(
                {"user": message, "timestamp": datetime.utcnow()}
            )

            system_prompt = self.get_comprehensive_system_prompt(
                session["state"],
                list(session["databases"].keys()) + list(session["s3_connections"].keys()),
            )

            messages: List[HumanMessage | SystemMessage] = [
                SystemMessage(content=system_prompt)
            ]

            # add last five user / assistant turns
            for entry in session["conversation_history"][-5:-1]:
                messages.append(HumanMessage(content=entry["user"]))
                if "assistant" in entry:
                    messages.append(HumanMessage(content=entry["assistant"]))

            messages.append(HumanMessage(content=message))

            # ── call LLM ────────────────────────────────────────────────
            llm_resp = await self.llm.ainvoke(messages)
            cleaned_resp = self.clean_llm_response(llm_resp.content)
            logger.info("LLM cleaned response: %s", cleaned_resp)

            # ── let the command handler take over ──────────────────────
            final_json = await self.handle_llm_commands(
                session_id=session_id,
                user_message=message,
                llm_response=cleaned_resp,
            )

            # log assistant reply
            session["conversation_history"][-1]["assistant"] = final_json

            return ChatResponse(
                response=final_json,
                session_id=session_id,
                connection_status=session["state"],
                active_databases=list(session["databases"].keys())
                + list(session["s3_connections"].keys()),
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error")
            error_payload = create_standard_response(
                data={"error_details": str(exc)},
                metadata={"session_id": session_id},
            )
            return ChatResponse(
                response=json.dumps(error_payload, indent=2),
                session_id=session_id,
                connection_status="ERROR",
                active_databases=[],
                timestamp=datetime.utcnow().isoformat(),
            )


    def clean_llm_response(self, response: str) -> str:
        """Clean LLM response by removing backticks and markdown formatting"""
        # Remove backticks at the beginning and end
        cleaned = response.strip()

        # Remove markdown code block formatting
        if cleaned.startswith("``````"):
            lines = cleaned.split('\n')
            if len(lines) > 2:
                cleaned = '\n'.join(lines[1:-1])

        # Remove inline backticks around commands
        if cleaned.startswith("`") and cleaned.endswith("`"):
            cleaned = cleaned[1:-1]

        # Remove any remaining backticks around JSON
        cleaned = cleaned.replace("`", "")

        return cleaned.strip()

    async def handle_llm_commands(self, session_id: str, user_message: str, llm_response: str) -> str:
        """Handle all LLM commands for databases and S3 - returns JSON responses"""
        
        session = self.sessions[session_id]
        print("In handle llm commands.....")
        print("session_id:", session_id)
        print("user message:", user_message)
        print("llm_response:", llm_response)

        # Handle database connection (PostgreSQL/MySQL)
        if "CONNECT_DB:" in llm_response:
            try:
                json_start = llm_response.find("CONNECT_DB:") + len("CONNECT_DB:")
                json_data = llm_response[json_start:].strip()

                if json_data.startswith("{"):
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_data):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_data = json_data[:json_end]

                connection_data = json.loads(json_data)
                db_type = connection_data["type"]
                db_uri = connection_data["uri"]

                print(f"Attempting to connect to {db_type} with URI: {db_uri}")

                # Create database connection
                database = SQLDatabase.from_uri(db_uri)

                # Create SQL agent
                agent = create_sql_agent(
                    self.llm,
                    db=database,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=10,
                    handle_parsing_errors=True,
                    return_intermediate_steps=False
                )

                # Store database connection
                session["databases"][db_type] = {
                    "agent": agent,
                    "uri": db_uri,
                    "database": database,
                    "connected": True,
                    "connected_at": datetime.now().isoformat()
                }

                session["active_db"] = db_type
                session["state"] = "CONNECTED"

                # Extract connection details
                uri_parts = self.parse_connection_uri(db_uri)

                # AUTOMATICALLY GET ALL TABLES AFTER CONNECTION
                try:
                    # Get all tables from the database
                    if db_type.lower() == "postgresql":
                        tables_query = """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name;
                        """
                    elif db_type.lower() == "mysql":
                        tables_query = """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = DATABASE()
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name;
                        """
                    else:
                        # Generic query for other databases
                        tables_query = "SELECT name FROM sqlite_master WHERE type='table';"

                    tables_result = database.run(tables_query)
                    print(f"Raw tables result: {tables_result}")
                    
                    
                    tables_data = []

                    # Handle different response formats
                    if isinstance(tables_result, str):
                        try:
                            # Try to parse as python list of tuples
                            parsed = ast.literal_eval(tables_result)
                            tables_data = [tup[0] for tup in parsed if tup and len(tup) > 0]
                        except Exception as e:
                            print(f"Error parsing table string: {e}")
                            # fallback to line splitting (rarely needed)
                            lines = [ln.strip() for ln in tables_result.splitlines() if ln.strip()]
                            for ln in lines:
                                if ln.lower() == "table_name" or ln.startswith("-"):
                                    continue
                                if "|" in ln:
                                    ln = ln.split("|")[1].strip()
                                ln = ln.strip("() ,'")
                                if ln:
                                    tables_data.append(ln)
                    # Case 2: If result is already a tuple/list
                    elif isinstance(tables_result, (list, tuple)):
                        for row in tables_result:
                            if not row:
                                continue
                            tables_data.append(str(row[0]).strip())
                    # Fallback to LangChain helper if still empty
                    if not tables_data:
                        try:
                            names = database.get_usable_table_names()
                            tables_data = [{"table_name": n, "schema": "public"} for n in names]
                        except Exception as helper_err:
                            logger.warning("get_usable_table_names() failed ➜ %s", helper_err)

                    total_tables = len(tables_data)
                    response = create_standard_response(
                        data={
                            "database_type": db_type,
                            "username": uri_parts.get('username', 'N/A'),
                            "host": uri_parts.get('host', 'N/A'),
                            "database_name": uri_parts.get('database', 'N/A'),
                            'password': uri_parts.get('password', 'N/A'),
                            "total_tables": total_tables,
                            "tables": tables_data
                        },
                        metadata={
                            "session_state" : session["state"],
                            "active_systems": list(session["databases"].keys()) +
                                            list(session["s3_connections"].keys()),
                            "auto_loaded"   : "tables"
                        }
                    )

                except Exception as table_error:
                    print(f"Error retrieving tables: {table_error}")
                    # Still return successful connection but without table info
                    response = create_standard_response(
                        data={
                            "database_type": db_type,
                            "host": uri_parts.get('host', 'N/A'),
                            "database_name": uri_parts.get('database', 'N/A'),
                            "connection_status": "active",
                            "tables_error": "Could not retrieve table information",
                            "tables_error_details": str(table_error)
                        },
                        metadata={
                            "session_state": session["state"],
                            "active_systems": list(session["databases"].keys()) + list(session["s3_connections"].keys())
                        }
                    )

                return json.dumps(response, indent=2)

            except Exception as e:
                print(f"Database connection error: {e}")
                response = create_standard_response(
                    data={"error_details": str(e)},
                    metadata={"attempted_connection": db_type if 'db_type' in locals() else "unknown"}
                )
                return json.dumps(response, indent=2)

        # Handle S3 connection
        elif "CONNECT_S3:" in llm_response:
            try:
                json_start = llm_response.find("CONNECT_S3:") + len("CONNECT_S3:")
                json_data = llm_response[json_start:].strip()

                if json_data.startswith("{"):
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_data):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_data = json_data[:json_end]

                s3_data = json.loads(json_data)
                access_key = s3_data.get("access_key")
                secret_key = s3_data.get("secret_key") 
                region = s3_data.get("region", "us-east-1")
                bucket_name = s3_data.get("bucket")
                session_token = s3_data.get("session_token")

                # Validate required credentials
                if not access_key or not secret_key or not bucket_name:
                    response = create_standard_response(
                        data={
                            "required_fields": ["access_key", "secret_key", "bucket"],
                            "missing_fields": [
                                field for field in ["access_key", "secret_key", "bucket"] 
                                if not s3_data.get(field)
                            ]
                        },
                        metadata={
                            "help": "Provide all required S3 credentials to establish connection"
                        }
                    )
                    return json.dumps(response, indent=2)

                # Create S3 client
                try:
                    credentials = {
                        'aws_access_key_id': access_key,
                        'aws_secret_access_key': secret_key,
                        'region_name': region
                    }
                    
                    if session_token:
                        credentials['aws_session_token'] = session_token
                    
                    s3_client = boto3.client('s3', **credentials)
                    s3_client.head_bucket(Bucket=bucket_name)
                    
                    # Store S3 connection
                    session["s3_connections"][bucket_name] = {
                        "client": s3_client,
                        "bucket": bucket_name,
                        "region": region,
                        "access_key": access_key[:8] + "..." + access_key[-4:],
                        "connected": True,
                        "connected_at": datetime.now().isoformat()
                    }
                    
                    session["active_s3"] = bucket_name
                    session["state"] = "CONNECTED"

                    # AUTOMATICALLY LIST ALL OBJECTS AFTER S3 CONNECTION
                    try:
                        # List all objects in the bucket (limited to first 1000 for performance)
                        response_data = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1000)
                        objects = response_data.get('Contents', [])
                        
                        object_list = [
                            {
                                "key": obj['Key'],
                                "size": obj['Size'],
                                "last_modified": obj['LastModified'].isoformat(),
                                "etag": obj.get('ETag', '').replace('"', '')
                            }
                            for obj in objects
                        ]

                        # Calculate total size
                        total_size = sum(obj['Size'] for obj in objects)
                        
                        response = create_standard_response(
                            data={
                                "total_objects": len(objects),
                                "total_size_bytes": total_size,
                                "objects": object_list
                            },
                            metadata={
                                "session_state": session["state"],
                                "active_systems": list(session["databases"].keys()) + list(session["s3_connections"].keys()),
                                "max_objects_shown": 1000,
                                "has_more": response_data.get('IsTruncated', False),
                                "auto_loaded": "objects"
                            }
                        )
                        
                    except Exception as list_error:
                        print(f"Error listing objects: {list_error}")
                        # Still return successful connection but without object list
                        response = create_standard_response(
                            data={
                                "bucket_name": bucket_name,
                                "region": region,
                                "access_key_masked": access_key[:8] + "..." + access_key[-4:],
                                "connection_status": "active",
                                "objects_error": "Could not retrieve object list",
                                "objects_error_details": str(list_error)
                            },
                            metadata={
                                "session_state": session["state"],
                                "active_systems": list(session["databases"].keys()) + list(session["s3_connections"].keys())
                            }
                        )

                    return json.dumps(response, indent=2)

                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    error_messages = {
                        'NoSuchBucket': f"Bucket '{bucket_name}' does not exist or is not accessible",
                        'InvalidAccessKeyId': "Invalid AWS Access Key ID provided",
                        'SignatureDoesNotMatch': "Invalid AWS Secret Access Key provided",
                        'AccessDenied': f"Access denied to bucket '{bucket_name}'"
                    }
                    
                    response = create_standard_response(
                        data={
                            "error_code": error_code,
                            "error_details": error_messages.get(error_code, str(e)),
                            "bucket_name": bucket_name
                        }
                    )
                    return json.dumps(response, indent=2)

            except Exception as e:
                response = create_standard_response(
                    data={"error_details": str(e)}
                )
                return json.dumps(response, indent=2)
       
        # Handle database queries
        elif "EXECUTE_QUERY:" in llm_response:
            try:
                json_start = llm_response.find("EXECUTE_QUERY:") + len("EXECUTE_QUERY:")
                json_data = llm_response[json_start:].strip()

                if json_data.startswith("{"):
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_data):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_data = json_data[:json_end]

                query_data = json.loads(json_data)
                target_db = query_data.get("database", session["active_db"])
                explanation = query_data.get("explanation", "Executing database query")

                if not target_db or target_db not in session["databases"]:
                    response = create_standard_response(
                        data={"available_databases": list(session["databases"].keys())},
                        metadata={"suggestion": "Connect to PostgreSQL or MySQL first"}
                    )
                    return json.dumps(response, indent=2)

                # Generate and execute SQL query
                db_info = session["databases"][target_db]
                database = db_info["database"]
                
                sql_query = await self.generate_sql_query(user_message, database, target_db)
                
                if not sql_query:
                    response = create_standard_response(
                        data={"user_request": user_message},
                        metadata={"suggestion": "Try rephrasing your request"}
                    )
                    return json.dumps(response, indent=2)

                try:
                    result = database.run(sql_query)
                    
                    response = create_standard_response(
                        data={
                            "result": result
                        },
                        metadata={
                            "execution_time": datetime.now().isoformat(),
                            "database": target_db
                        }
                    )
                    return json.dumps(response, indent=2)
                
                except Exception as query_error:
                    response = create_standard_response(
                        data={
                            "error_details": str(query_error),
                            "database_type": target_db
                        }
                    )
                    return json.dumps(response, indent=2)

            except Exception as e:
                response = create_standard_response(
                    data={"error_details": str(e)}
                )
                return json.dumps(response, indent=2)

        # Handle S3 operations
        elif "EXECUTE_S3:" in llm_response:
            try:
                json_start = llm_response.find("EXECUTE_S3:") + len("EXECUTE_S3:")
                json_data = llm_response[json_start:].strip()

                if json_data.startswith("{"):
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_data):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_data = json_data[:json_end]

                s3_data = json.loads(json_data)
                operation = s3_data.get("operation")
                bucket = s3_data.get("bucket", session.get("active_s3"))

                if not bucket or bucket not in session["s3_connections"]:
                    response = create_standard_response(
                        data={"available_buckets": list(session["s3_connections"].keys())},
                        metadata={"suggestion": "Connect to an S3 bucket first"}
                    )
                    return json.dumps(response, indent=2)

                s3_connection = session["s3_connections"][bucket]
                s3_client = s3_connection["client"]

                if operation == "list_objects":
                    try:
                        response_data = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=20)
                        objects = response_data.get('Contents', [])
                        
                        object_list = [
                            {
                                "key": obj['Key'],
                                "size": obj['Size'],
                                "last_modified": obj['LastModified'].isoformat(),
                                "etag": obj.get('ETag', '')
                            }
                            for obj in objects
                        ]

                        response = create_standard_response(
                            data={
                                "bucket_name": bucket,
                                "object_count": len(objects),
                                "objects": object_list,
                                "operation": "list_objects"
                            },
                            metadata={
                                "max_keys": 20,
                                "has_more": response_data.get('IsTruncated', False)
                            }
                        )
                        return json.dumps(response, indent=2)

                    except Exception as s3_error:
                        response = create_standard_response(
                            data={
                                "bucket_name": bucket,
                                "operation": "list_objects",
                                "error_details": str(s3_error)
                            }
                        )
                        return json.dumps(response, indent=2)

                elif operation == "list_buckets":
                    try:
                        response_data = s3_client.list_buckets()
                        buckets = response_data.get('Buckets', [])
                        
                        bucket_list = [
                            {
                                "name": bucket_info['Name'],
                                "creation_date": bucket_info['CreationDate'].isoformat()
                            }
                            for bucket_info in buckets
                        ]

                        response = create_standard_response(
                            data={
                                "bucket_count": len(buckets),
                                "buckets": bucket_list,
                                "operation": "list_buckets"
                            }
                        )
                        return json.dumps(response, indent=2)

                    except Exception as s3_error:
                        response = create_standard_response(
                            data={
                                "operation": "list_buckets",
                                "error_details": str(s3_error)
                            }
                        )
                        return json.dumps(response, indent=2)

            except Exception as e:
                response = create_standard_response(
                    data={"error_details": str(e)}
                )
                return json.dumps(response, indent=2)

        # Handle disconnection commands
        elif "DISCONNECT_DB:" in llm_response:
            try:
                json_start = llm_response.find("DISCONNECT_DB:") + len("DISCONNECT_DB:")
                json_data = llm_response[json_start:].strip()

                if json_data.startswith("{"):
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_data):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_data = json_data[:json_end]

                disconnect_data = json.loads(json_data)
                disconnect_type = disconnect_data["type"]
                action = disconnect_data["action"]

                if action == "disconnect_single" and disconnect_type in session["databases"]:
                    db_info = session["databases"][disconnect_type]
                    
                    try:
                        if "database" in db_info and hasattr(db_info["database"], "_engine"):
                            db_info["database"]._engine.dispose()
                    except Exception as cleanup_error:
                        logger.warning(f"Error during cleanup: {cleanup_error}")

                    del session["databases"][disconnect_type]

                    if session["active_db"] == disconnect_type:
                        remaining_dbs = list(session["databases"].keys())
                        session["active_db"] = remaining_dbs[0] if remaining_dbs else None
                        if not remaining_dbs and not session["s3_connections"]:
                            session["state"] = "INITIAL"

                    response = create_standard_response(
                        data={
                            "disconnected_system": disconnect_type,
                            "remaining_databases": list(session["databases"].keys()),
                            "remaining_s3_connections": list(session["s3_connections"].keys())
                        },
                        metadata={
                            "session_state": session["state"],
                            "can_reconnect": True
                        }
                    )
                    return json.dumps(response, indent=2)

            except Exception as e:
                response = create_standard_response(
                    data={"error_details": str(e)}
                )
                return json.dumps(response, indent=2)

        # Handle S3 disconnection
        elif "DISCONNECT_S3:" in llm_response:
            try:
                json_start = llm_response.find("DISCONNECT_S3:") + len("DISCONNECT_S3:")
                json_data = llm_response[json_start:].strip()

                if json_data.startswith("{"):
                    brace_count = 0
                    json_end = 0
                    for i, char in enumerate(json_data):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    json_data = json_data[:json_end]

                s3_data = json.loads(json_data)
                bucket = s3_data.get("bucket", session.get("active_s3"))

                if bucket and bucket in session["s3_connections"]:
                    del session["s3_connections"][bucket]
                    
                    if session["active_s3"] == bucket:
                        remaining_buckets = list(session["s3_connections"].keys())
                        session["active_s3"] = remaining_buckets[0] if remaining_buckets else None
                        if not remaining_buckets and not session["databases"]:
                            session["state"] = "INITIAL"

                    response = create_standard_response(
                        data={
                            "disconnected_bucket": bucket,
                            "remaining_databases": list(session["databases"].keys()),
                            "remaining_s3_connections": list(session["s3_connections"].keys())
                        },
                        metadata={
                            "session_state": session["state"],
                            "can_reconnect": True
                        }
                    )
                    return json.dumps(response, indent=2)

            except Exception as e:
                response = create_standard_response(
                    data={"error_details": str(e)}
                )
                return json.dumps(response, indent=2)

        # Handle status requests
        elif "SHOW_STATUS" in llm_response:
            if not session["databases"] and not session["s3_connections"]:
                response = create_standard_response(
                    data={
                        "databases": {},
                        "s3_connections": {},
                        "total_connections": 0
                    },
                    metadata={
                        "suggestion": "Connect to PostgreSQL, MySQL, or S3 to get started"
                    }
                )
                return json.dumps(response, indent=2)

            # Prepare database status
            db_status = {}
            for db_type, db_info in session["databases"].items():
                db_status[db_type] = {
                    "status": "connected",
                    "connected_since": db_info['connected_at'],
                    "is_active": db_type == session["active_db"],
                    "type": "database"
                }

            # Prepare S3 status
            s3_status = {}
            for bucket, s3_info in session["s3_connections"].items():
                s3_status[bucket] = {
                    "status": "connected",
                    "connected_since": s3_info['connected_at'],
                    "is_active": bucket == session["active_s3"],
                    "region": s3_info.get('region', 'unknown'),
                    "type": "s3"
                }

            response = create_standard_response(
                data={
                    "databases": db_status,
                    "s3_connections": s3_status,
                    "total_connections": len(db_status) + len(s3_status),
                    "session_state": session["state"]
                },
                metadata={
                    "available_actions": [
                        "Switch systems: 'switch to [system]'",
                        "Query data: 'show tables' or 'list files'",
                        "Disconnect: 'disconnect from [system]'"
                    ]
                }
            )
            return json.dumps(response, indent=2)

        # Regular LLM response
        else:
            if any(word in user_message.lower() for word in ["connect", "database", "postgresql", "mysql", "s3", "aws"]):
                if session["state"] == "INITIAL":
                    session["state"] = "AWAITING_CREDENTIALS"

            response = create_standard_response(
                data={"content": llm_response},
                metadata={
                    "session_state": session["state"],
                    "is_llm_response": True
                }
            )
            return json.dumps(response, indent=2)
        

    def parse_connection_uri(self, uri: str) -> Dict[str, str]:
        """Parse connection URI to extract components"""
        for db_type, pattern in self.db_patterns.items():
            match = re.match(pattern, uri)
            if match:
                if db_type == 's3':
                    return {"bucket": match.group(1), "path": match.group(2), "type": db_type}
                else:
                    return {
                        "type": db_type,
                        "username": match.group(1),
                        "password": match.group(2),
                        "host": match.group(3),
                        "port": match.group(4),
                        "database": match.group(5)
                    }
        return {"type": "unknown"}

    async def generate_sql_query(self, user_message: str, database, db_type: str) -> str:
        """Generate SQL query with correct syntax for PostgreSQL/MySQL"""
        
        try:
            tables = database.get_usable_table_names()
            schema_info = ""

            # Get schema for relevant tables
            for table in tables[:3]:  # Limit to 3 tables for performance
                try:
                    table_info = database.get_table_info([table])
                    schema_info += f"\n{table_info}"
                except:
                    continue

        except Exception as e:
            schema_info = "Schema information not available"
            tables = []

        # Database-specific SQL prompt
        if db_type == "postgresql":
            syntax_rules = '''
                        **PostgreSQL Syntax Rules:**
                        - Column names with spaces: "Store ID", "Employee Number" (NO backslashes)
                        - NEVER use: \"Column Name\" (causes syntax errors)
                        - Case-sensitive identifiers
                        - Examples: SELECT "Store ID", "Employee Number" FROM retail_sales_data LIMIT 10;
                                    '''
        else:  # MySQL
            syntax_rules = '''
                        **MySQL Syntax Rules:**
                        - Column names with spaces: `Store ID`, `Employee Number`
                        - Use backticks for identifiers with spaces
                        - Examples: SELECT `Store ID`, `Employee Number` FROM retail_sales_data LIMIT 10;
                                    '''

        sql_prompt = f"""
                        Generate a single, executable {db_type.upper()} query for this request.

                        **User Request**: {user_message}
                        **Available Tables**: {', '.join(tables) if tables else 'Unknown'}

                        **Schema Information**:
                        {schema_info}

                        {syntax_rules}

                        **IMPORTANT**: Return ONLY the SQL query, no explanations or formatting.
                        """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=sql_prompt)])
            sql_query = response.content.strip()
            
            # Clean the SQL query
            sql_query = sql_query.replace("``````", "").strip()
            sql_query = sql_query.replace('\\"', '"')
            
            return sql_query

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None


def create_standard_response( data: Optional[Dict[str, Any]] = None,metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "data": data or {},
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat()
    }


