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
                "**Connection Failed to [System]**
                **Issue:** [specific_error_description]
                **Solutions:** [step_by_step_troubleshooting]
                **Try:** [alternative_approaches]"

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
        """Main conversation processing logic"""
        print(f"Processing message for session {session_id}: {message}")

        # Initialize session if new
        if session_id not in self.sessions:
            print(f"Creating new session: {session_id}")
            self.sessions[session_id] = {
                "state": "INITIAL",
                "databases": {},  # db_type -> {agent, uri, connected}
                "s3_connections": {},  # bucket -> {client, connection_info}
                "active_db": None,
                "active_s3": None,
                "conversation_history": [],
                "last_activity": datetime.now()
            }

        session = self.sessions[session_id]
        session["last_activity"] = datetime.now()

        print(f"Current session state: {session['state']}")
        print(f"Active databases: {list(session['databases'].keys())}")
        print(f"Active S3 connections: {list(session['s3_connections'].keys())}")

        try:
            # Add to conversation history
            session["conversation_history"].append({"user": message, "timestamp": datetime.now()})

            # Get system prompt with current context
            active_systems = list(session["databases"].keys()) + list(session["s3_connections"].keys())
            system_prompt = self.get_comprehensive_system_prompt(session["state"], active_systems)

            # Build conversation context
            messages = [SystemMessage(content=system_prompt)]

            # Add recent conversation history (last 5 exchanges for performance)
            recent_history = session["conversation_history"][-5:]
            for entry in recent_history[:-1]:
                messages.append(HumanMessage(content=entry["user"]))
                if "assistant" in entry:
                    messages.append(HumanMessage(content=entry["assistant"]))

            # Add current message
            messages.append(HumanMessage(content=message))

            # Get LLM response
            llm_response = await self.llm.ainvoke(messages)
            response_content = llm_response.content
            print("llm_response:", response_content)

            # Clean the response content
            cleaned_response = self.clean_llm_response(response_content)
            print("cleaned_response:", cleaned_response)

            # Process LLM commands
            final_response = await self.handle_llm_commands(session_id, message, cleaned_response)
            print("final_response after handling llm commands:", final_response)

            # Add assistant response to history
            session["conversation_history"][-1]["assistant"] = final_response

            return ChatResponse(
                response=final_response,
                session_id=session_id,
                connection_status=session["state"],
                active_databases=list(session["databases"].keys()) + list(session["s3_connections"].keys()),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {str(e)}")
            return ChatResponse(
                response=f"❌ **System Error**: {str(e)}\n\nPlease try again or start a new session.",
                session_id=session_id,
                connection_status="ERROR",
                active_databases=[],
                timestamp=datetime.now().isoformat()
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
        """Handle all LLM commands for databases and S3"""

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

                # Create SQL agent with improved settings
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

                print(f"Successfully connected! Session state: {session['state']}")
                print(f"Active databases: {list(session['databases'].keys())}")

                # Extract connection details for display
                uri_parts = self.parse_connection_uri(db_uri)

                return f"""
                      **Successfully Connected to {db_type.title()} at {uri_parts.get('host', 'N/A')}!**
                        What would you like to explore?
                        """

            except Exception as e:
                print(f"Database connection error: {e}")
                return f"**Database Connection Failed**: {str(e)}"

        # Handle S3 connection with user-provided credentials
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
                
                # Extract user-provided credentials
                access_key = s3_data.get("access_key")
                secret_key = s3_data.get("secret_key") 
                region = s3_data.get("region", "us-east-1")
                bucket_name = s3_data.get("bucket")
                session_token = s3_data.get("session_token")  # Optional

                # Validate required credentials
                if not access_key or not secret_key or not bucket_name:
                    return """**Missing S3 Credentials**
                            Please provide all required S3 credentials:
                            - **Access Key ID**: Your AWS Access Key ID (starts with AKIA...)
                            - **Secret Access Key**: Your AWS Secret Access Key
                            - **Bucket Name**: S3 bucket name
                            - **Region**: AWS region (optional, defaults to us-east-1)

                            **Example**: "Connect to S3 with access_key=AKIA123..., secret_key=xyz789..., region=us-east-1, bucket=my-bucket"
                            """

                # Create S3 client with user-provided credentials
                try:
                    # Build credentials dictionary
                    credentials = {
                        'aws_access_key_id': access_key,
                        'aws_secret_access_key': secret_key,
                        'region_name': region
                    }
                    
                    # Add session token if provided (for temporary credentials)
                    if session_token:
                        credentials['aws_session_token'] = session_token
                    
                    # Create S3 client with user credentials
                    s3_client = boto3.client('s3', **credentials)
                    
                    # Test connection by checking if bucket exists and is accessible
                    s3_client.head_bucket(Bucket=bucket_name)
                    
                    # Store S3 connection info
                    session["s3_connections"][bucket_name] = {
                        "client": s3_client,
                        "bucket": bucket_name,
                        "region": region,
                        "access_key": access_key[:8] + "..." + access_key[-4:],  # Mask middle part
                        "connected": True,
                        "connected_at": datetime.now().isoformat()
                    }
                    
                    session["active_s3"] = bucket_name
                    session["state"] = "CONNECTED"

                    return f"""**Successfully Connected to S3!**
                            - Bucket: {bucket_name}
                            What would you like to do with S3?"""

                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'NoSuchBucket':
                        return f" **S3 Bucket Not Found**: Bucket '{bucket_name}' does not exist or you don't have access to it."
                    elif error_code == 'InvalidAccessKeyId':
                        return " **Invalid Access Key**: The AWS Access Key ID you provided is not valid."
                    elif error_code == 'SignatureDoesNotMatch':
                        return " **Invalid Secret Key**: The AWS Secret Access Key you provided is not valid."
                    elif error_code == 'AccessDenied':
                        return f" **Access Denied**: You don't have permission to access bucket '{bucket_name}'."
                    else:
                        return f" **S3 Connection Failed**: {str(e)}"
                        
                except NoCredentialsError:
                    return " **No AWS Credentials**: Please provide valid AWS Access Key ID and Secret Access Key."
                    
                except Exception as s3_error:
                    return f" **S3 Connection Error**: {str(s3_error)}"

            except json.JSONDecodeError as je:
                return " **S3 Credential Parsing Error**: Could not parse S3 connection details."
            except Exception as e:
                return f" **S3 Setup Error**: {str(e)}"

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
                    return " **No Database Connection**\nPlease connect to PostgreSQL or MySQL first."

                # Generate and execute SQL query
                db_info = session["databases"][target_db]
                database = db_info["database"]
                
                sql_query = await self.generate_sql_query(user_message, database, target_db)
                
                if not sql_query:
                    return " **Could not generate SQL query**\nPlease try rephrasing your request."

                try:
                    result = database.run(sql_query)
                    return result
                
                except Exception as query_error:
                    return str(query_error)

            except Exception as e:
                return f"**Query Processing Error**: {str(e)}"

        # Handle S3 operations using stored client
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
                    return " **No S3 Connection**: Please connect to an S3 bucket first."

                # Get the user-created S3 client
                s3_connection = session["s3_connections"][bucket]
                s3_client = s3_connection["client"]

                if operation == "list_objects":
                    try:
                        response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=20)
                        objects = response.get('Contents', [])
                        
                        if objects:
                            object_list = "\n".join([
                                f"📄 {obj['Key']} ({obj['Size']} bytes, {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')})" 
                                for obj in objects
                            ])
                        else:
                            object_list = "No objects found in bucket"

                        return f"""
                                    **S3 Objects Listed**
                                     **Bucket**: {bucket}
                                     **Objects** ({len(objects)} items):
                                    {object_list}
                                    """
                    except Exception as s3_error:
                        return f" **S3 List Error**: {str(s3_error)}"

                elif operation == "list_buckets":
                    try:
                        response = s3_client.list_buckets()
                        buckets = response.get('Buckets', [])
                        
                        if buckets:
                            bucket_list = "\n".join([
                                f"🪣 {bucket['Name']} (created: {bucket['CreationDate'].strftime('%Y-%m-%d')})" 
                                for bucket in buckets
                            ])
                        else:
                            bucket_list = "No buckets found"

                        return f""" **S3 Buckets Listed**
                                    **Your S3 Buckets** ({len(buckets)} items):
                                    {bucket_list}
                                    """
                    except Exception as s3_error:
                        return f" **S3 Bucket List Error**: {str(s3_error)}"

            except Exception as e:
                return f" **S3 Operation Error**: {str(e)}"

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

                    return f"""**Disconnected from {disconnect_type.title()}**
                                 Connection closed successfully
                                 Remaining systems: {', '.join(list(session['databases'].keys()) + list(session['s3_connections'].keys())) if (session['databases'] or session['s3_connections']) else 'None'}
                                You can reconnect anytime"""

            except Exception as e:
                return f" **Disconnect Error**: {str(e)}"

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

                    return f"""**Disconnected from S3 Bucket: {bucket}**
                                 S3 connection closed successfully
                                 Remaining systems: {', '.join(list(session['databases'].keys()) + list(session['s3_connections'].keys())) if (session['databases'] or session['s3_connections']) else 'None'}
                                You can reconnect anytime"""

            except Exception as e:
                return f" **S3 Disconnect Error**: {str(e)}"

        # Handle status requests
        elif "SHOW_STATUS" in llm_response:
            if not session["databases"] and not session["s3_connections"]:
                return "📭 **No Active Connections**\n\nConnect to PostgreSQL, MySQL, or S3 to get started!"

            status_info = "🗄️ **Active Connections:**\n\n"
            
            # Show database connections
            for db_type, db_info in session["databases"].items():
                active_indicator = "🔴" if db_type == session["active_db"] else "🟢"
                status_info += f"{active_indicator} **{db_type.title()}**: Connected since {db_info['connected_at'][:19]}\n"
            
            # Show S3 connections
            for bucket, s3_info in session["s3_connections"].items():
                active_indicator = "🔴" if bucket == session["active_s3"] else "☁️"
                status_info += f"{active_indicator} **S3/{bucket}**: Connected since {s3_info['connected_at'][:19]}\n"

            status_info += f"\n**Available Actions:**\n- Switch systems: 'switch to [system]'\n- Query data: 'show tables' or 'list files'\n- Disconnect: 'disconnect from [system]'"
            return status_info

        # Regular LLM response
        else:
            if any(word in user_message.lower() for word in ["connect", "database", "postgresql", "mysql", "s3", "aws"]):
                if session["state"] == "INITIAL":
                    session["state"] = "AWAITING_CREDENTIALS"

            return llm_response

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
                        "password": "***",
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

