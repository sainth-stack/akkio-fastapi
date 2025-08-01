from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import SystemMessage, HumanMessage
import os
import uuid
import re
import json
from typing import Optional, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Make session_id optional


class ChatResponse(BaseModel):
    response: str
    session_id: str
    connection_status: str
    active_databases: list
    timestamp: str


class MultiDatabaseAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fixed the model name
            temperature=0,
            max_tokens=2000
        )
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Supported database connection patterns
        self.db_patterns = {
            'postgresql': r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)',
            'mysql': r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)',
            'sqlite': r'sqlite:///(.+\.db)',
            'oracle': r'oracle://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)',
            'sqlserver': r'mssql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)'
        }

    def get_comprehensive_system_prompt(self, session_state: str, active_dbs: list = None) -> str:
        """Most robust system prompt for multi-database operations"""
        return f"""
                You are DBAI (Database AI Assistant), an expert multi-database interface agent supporting PostgreSQL, MySQL, SQLite, Oracle, and SQL Server.
            
                **CURRENT SESSION STATE:** {session_state}
                **ACTIVE DATABASES:** {active_dbs or "None"}
                **TIMESTAMP:** {datetime.now().isoformat()}
            
                ## CORE RESPONSIBILITIES:
            
                ### 1. CONNECTION MANAGEMENT
            
                **Connection Intent Recognition:**
                - "connect to [database_type]" or "[database_type] with credentials"
                - "I want to connect to my database"
                - Extract credentials: host, port, database, username, password
            
                **Response Format (NO BACKTICKS):**
                CONNECT_DB:{{"type":"postgresql","uri":"postgresql://user:pass@host:port/db"}}
            
                **Database-Specific URI Formats:**
                - PostgreSQL: postgresql://username:password@host:port/database
                - MySQL: mysql://username:password@host:port/database
                - SQLite: sqlite:///path/to/database.db
                - Oracle: oracle://username:password@host:port/database
                - SQL Server: mssql://username:password@host:port/database
            
                ### 2. QUERY HANDLING
            
                **Query Intent Recognition (WHEN STATE = CONNECTED):**
                - "show tables" → List all tables
                - "describe table X" → Show table structure
                - "show data from X" → Display table contents
                - "count records" → Count operations
                - "last/recent/latest" → ORDER BY with DESC
                - "first/top/earliest" → ORDER BY with ASC
                - Any data question → EXECUTE_QUERY command
            
                **Response Format (NO BACKTICKS):**
                EXECUTE_QUERY:{{"database":"postgresql","explanation":"Brief description of operation"}}
            
                ### 3. DATABASE-SPECIFIC SQL SYNTAX RULES:
            
                **PostgreSQL:**
                - Column names with spaces: "Store ID", "Employee Number" (NO backslashes)
                - Case-sensitive identifiers: "Date", "Sales"
                - NEVER use: \"Column Name\" (causes syntax errors)
                - Date format: 'YYYY-MM-DD' or TIMESTAMP
                - Limit clause: LIMIT n
                - String comparison: ILIKE for case-insensitive
            
                **MySQL:**
                - Backticks for identifiers: `Store ID`, `Employee Number`
                - Case-insensitive by default
                - Date format: 'YYYY-MM-DD HH:MM:SS'
                - Limit clause: LIMIT n
                - String comparison: LIKE
            
                **SQLite:**
                - Double quotes or brackets: "Store ID" or [Store ID]
                - Case-insensitive COLLATE NOCASE
                - Date as TEXT: 'YYYY-MM-DD HH:MM:SS'
                - Limit clause: LIMIT n
            
                **SQL Server:**
                - Square brackets: [Store ID], [Employee Number]
                - Case-insensitive by default
                - Date format: 'YYYY-MM-DD HH:MM:SS'
                - Limit clause: TOP n (before SELECT) or OFFSET/FETCH
            
                **Oracle:**
                - Double quotes for case-sensitive: "Store_ID"
                - Uppercase by default unless quoted
                - Date format: TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')
                - Limit clause: ROWNUM <= n or FETCH FIRST n ROWS ONLY
            
                ### 4. ROBUST QUERY PATTERNS:
            
                **Data Retrieval Patterns:**
                - Show all data: SELECT * FROM table_name LIMIT 10;
                - Specific columns: SELECT "col1", "col2" FROM table_name LIMIT 20;
                - Latest records: SELECT * FROM table_name ORDER BY "Date" DESC LIMIT 10;
                - Count operations: SELECT COUNT(*) as total_records FROM table_name;
                - Unique values: SELECT DISTINCT "column_name" FROM table_name;
            
                **Schema Exploration:**
                - List tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
                - Table structure: SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'table_name';
                - Table indexes: SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'table_name';
            
                **Analytics Patterns:**
                - Aggregations: SELECT "category", COUNT(*), AVG("amount") FROM table_name GROUP BY "category";
                - Date-based: SELECT DATE("date_column"), SUM("sales") FROM table_name GROUP BY DATE("date_column");
                - Top N: SELECT * FROM table_name ORDER BY "sales" DESC LIMIT 5;
            
                ### 5. ERROR HANDLING & RECOVERY:
            
                **Common Error Scenarios:**
                - Table doesn't exist → Suggest listing tables first
                - Column name errors → Suggest describing table structure
                - Permission errors → Check user access rights
                - Connection timeout → Retry with simpler query
            
                **Query Optimization Guidelines:**
                - Always use LIMIT for large datasets (default: 20 rows)
                - Use appropriate ORDER BY for meaningful results
                - Handle NULL values appropriately
                - Optimize JOIN operations for performance
            
                ### 6. DISCONNECTION MANAGEMENT
            
                **Disconnect Intent Recognition:**
                - "disconnect from [database_type]"
                - "close connection"
                - "end session"
                - "logout"
            
                **Response Formats (NO BACKTICKS):**
                - Single: DISCONNECT_DB:{{"type":"postgresql","action":"disconnect_single"}}
                - All: DISCONNECT_DB:{{"type":"all","action":"disconnect_all"}}
                - Session: DISCONNECT_DB:{{"type":"session","action":"clear_session"}}
            
                ### 7. MULTI-DATABASE OPERATIONS
            
                **Database Switching:**
                - "switch to mysql" → SWITCH_DATABASE:mysql
                - "use postgresql database" → SWITCH_DATABASE:postgresql
            
                **Cross-Database Queries:**
                - "compare data between databases"
                - "show counts from all connected databases"
            
                **Status Checking:**
                - "show status" → SHOW_STATUS
                - "which databases am I connected to" → SHOW_STATUS
            
                ### 8. ADVANCED FEATURES:
            
                **Data Insights:**
                - Provide data summaries and insights
                - Suggest related queries
                - Identify data patterns and anomalies
            
                **Query Suggestions:**
                - Based on table structure, suggest meaningful queries
                - Recommend JOIN operations when multiple tables exist
                - Propose filtering and grouping options
            
                **Performance Optimization:**
                - Monitor query execution times
                - Suggest indexing for slow queries
                - Recommend query improvements
            
                ### 9. CRITICAL EXECUTION RULES:
            
                **Response Generation:**
                1. NEVER use backticks (`) around commands
                2. Always use exact format: COMMAND_NAME:{{json_data}}
                3. Generate commands immediately without explanatory text
                4. Ensure database-specific SQL syntax is correct
                5. Handle edge cases gracefully
            
                **State Management:**
                - INITIAL → AWAITING_CREDENTIALS → CONNECTED → QUERYING
                - Maintain context across multiple queries
                - Remember user preferences and query history
                - Handle session timeouts gracefully
            
                **SQL Query Validation:**
                - Double-check column name syntax for target database
                - Ensure proper date/time handling
                - Validate data types and constraints
                - Use appropriate aggregation functions
            
                ### 10. COMPREHENSIVE ERROR RESPONSES:
            
                **Connection Errors:**
                "❌ **Connection Failed**
                - **Issue**: [Specific error description]
                - **Solutions**: [Step-by-step troubleshooting]
                - **Alternatives**: [Alternative connection methods]
                - **Help**: [Specific guidance for the database type]"
            
                **Query Errors:**
                "❌ **Query Execution Error**
                - **Database**: [Database type]
                - **Query**: [The attempted query]
                - **Issue**: [Error explanation in plain English]
                - **Fix**: [Corrected query or approach]
                - **Prevention**: [How to avoid this error]"
            
                **Success Responses:**
                "✅ **[Operation] Successful**
                - **Database**: [Database type]
                - **Operation**: [What was performed]
                - **Results**: [Formatted data/information]
                - **Insights**: [Data observations]
                - **Next Steps**: [Suggested follow-up queries]"
            
                ### 11. CONTEXT AWARENESS:
            
                **Conversation Memory:**
                - Remember previous queries and results
                - Understand references like "that table" or "the last query"
                - Maintain continuity across database operations
                - Adapt responses based on user expertise level
            
                **Intelligent Suggestions:**
                - Based on data structure, suggest relevant analyses
                - Recommend data exploration paths
                - Identify potential data quality issues
                - Propose business intelligence queries
            
                ## EXECUTION PRIORITY:
            
                1. **Connection Requests**: Immediately generate CONNECT_DB command
                2. **Data Queries (when connected)**: Always use EXECUTE_QUERY command
                3. **Schema Questions**: Use appropriate information_schema queries
                4. **Disconnection**: Generate DISCONNECT_DB command
                5. **Status/Help**: Provide comprehensive information
            
                ## FINAL REMINDERS:
            
                - **Database-Specific Syntax**: Always use correct syntax for the target database
                - **Error Prevention**: Validate queries before suggesting execution
                - **User Experience**: Provide clear, actionable responses
                - **Performance**: Optimize queries for speed and resource usage
                - **Security**: Never expose sensitive information in logs or responses
            
                Remember: You are a helpful, intelligent database expert who makes complex database operations simple through natural conversation. Always prioritize data accuracy, query performance, and user experience.
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
                "active_db": None,
                "conversation_history": [],
                "last_activity": datetime.now()  # Track last activity
            }

        session = self.sessions[session_id]
        session["last_activity"] = datetime.now()  # Update last activity

        print(f"Current session state: {session['state']}")
        print(f"Active databases: {list(session['databases'].keys())}")

        try:
            # Add to conversation history
            session["conversation_history"].append({"user": message, "timestamp": datetime.now()})

            # Get system prompt with current context
            active_dbs = list(session["databases"].keys())
            system_prompt = self.get_comprehensive_system_prompt(session["state"], active_dbs)

            # Build conversation context
            messages = [SystemMessage(content=system_prompt)]

            # Add recent conversation history (last 10 exchanges)
            recent_history = session["conversation_history"][-10:]
            for entry in recent_history[:-1]:  # Exclude current message
                messages.append(HumanMessage(content=entry["user"]))
                if "assistant" in entry:
                    messages.append(HumanMessage(content=entry["assistant"]))

            # Add current message
            messages.append(HumanMessage(content=message))

            # Get LLM response
            llm_response = await self.llm.ainvoke(messages)
            response_content = llm_response.content
            print("llm_response:", response_content)

            # Clean the response content - remove backticks and markdown formatting
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
                active_databases=list(session["databases"].keys()),
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {str(e)}")
            return ChatResponse(
                response=f"**System Error**: {str(e)}\n\nPlease try again or start a new session.",
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
        """Handle special LLM commands and database operations"""

        session = self.sessions[session_id]
        print("In handle llm commands.....")
        print("session_id:", session_id)
        print("user message:", user_message)
        print("llm_response:", llm_response)

        # Handle database connection command
        if "CONNECT_DB:" in llm_response:
            try:
                # Extract JSON data after CONNECT_DB:
                json_start = llm_response.find("CONNECT_DB:") + len("CONNECT_DB:")
                json_data = llm_response[json_start:].strip()

                # Find the JSON object
                if json_data.startswith("{"):
                    # Find the end of JSON object
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

                print("Extracted JSON data:", json_data)
                connection_data = json.loads(json_data)
                db_type = connection_data["type"]
                db_uri = connection_data["uri"]

                print(f"Attempting to connect to {db_type} with URI: {db_uri}")

                # Create database connection
                database = SQLDatabase.from_uri(db_uri)

                # Create SQL agent for this database
                # Create SQL agent with improved settings
                agent = create_sql_agent(
                    self.llm,
                    db=database,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=10,  # Increased iterations
                    handle_parsing_errors=True,
                    return_intermediate_steps=False  # Reduce output verbosity
                )

                # Store database connection
                session["databases"][db_type] = {
                    "agent": agent,
                    "uri": db_uri,
                    "database": database,
                    "connected": True,
                    "connected_at": datetime.now().isoformat()
                }

                # Set as active database
                session["active_db"] = db_type
                session["state"] = "CONNECTED"

                print(f"Successfully connected! Session state: {session['state']}")
                print(f"Active databases: {list(session['databases'].keys())}")

                # Extract connection details for display
                uri_parts = self.parse_connection_uri(db_uri)

                return f"""Successfully Connected to {db_type.title()}!
                        Connection Details:
                        - Database Type: {db_type.upper()}
                        - Host: {uri_parts.get('host', 'N/A')}
                        - Port: {uri_parts.get('port', 'N/A')}
                        - Database: {uri_parts.get('database', 'N/A')}
                        - User: {uri_parts.get('username', 'N/A')}
                        
                        **Active Databases:** {', '.join(session['databases'].keys())}
                        What would you like to explore first?"""

            except json.JSONDecodeError as je:
                print(f"JSON decode error: {je}")
                return "**Connection Setup Error**: Could not process connection details. Please provide database credentials again."
            except Exception as e:
                print(f"Connection error: {e}")
                return f"""**Connection Failed**
                        **Error**: {str(e)}
                        Please verify your details and try connecting again."""

        # Handle disconnect commands
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
                    # Disconnect specific database
                    db_info = session["databases"][disconnect_type]

                    # Close the database connection safely
                    try:
                        if "database" in db_info and hasattr(db_info["database"], "_engine"):
                            db_info["database"]._engine.dispose()
                    except Exception as cleanup_error:
                        logger.warning(f"Error during cleanup for {disconnect_type}: {cleanup_error}")

                    # Remove from session
                    del session["databases"][disconnect_type]

                    # Update active database if needed
                    if session["active_db"] == disconnect_type:
                        remaining_dbs = list(session["databases"].keys())
                        session["active_db"] = remaining_dbs[0] if remaining_dbs else None
                        if not remaining_dbs:
                            session["state"] = "INITIAL"

                    return f"""Disconnected from {disconnect_type.title()}
                             Connection closed successfully
                             Remaining connections: {', '.join(session['databases'].keys()) if session['databases'] else 'None'}
                             You can reconnect anytime by saying 'connect to {disconnect_type}'
                            
                            {f"**Current Active Database:** {session['active_db'].title()}" if session['active_db'] else "**Status:** No active database connections"}"""

                elif action == "disconnect_all":
                    # Disconnect all databases
                    disconnected_dbs = list(session["databases"].keys())

                    for db_type, db_info in session["databases"].items():
                        try:
                            if "database" in db_info and hasattr(db_info["database"], "_engine"):
                                db_info["database"]._engine.dispose()
                        except Exception as cleanup_error:
                            logger.warning(f"Error during cleanup for {db_type}: {cleanup_error}")

                    # Clear all connections
                    session["databases"] = {}
                    session["active_db"] = None
                    session["state"] = "INITIAL"

                    return f"""Disconnected from All Databases
                             All connections closed successfully
                            Disconnected from: {', '.join(disconnected_dbs)}
                             Say 'connect to my database' to start a new connection"""

                else:
                    return f"**Disconnect Error**: Database '{disconnect_type}' not found or invalid action '{action}'"

            except Exception as e:
                return f"**Disconnect Error**: {str(e)}"

        # Handle query execution command
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
                    return f"**No Active Database Connection**\n\nPlease connect to a database first."

                # Get the database connection directly
                db_info = session["databases"][target_db]
                database = db_info["database"]

                # Generate SQL query using LLM for specific user request
                sql_query = await self.generate_sql_query(user_message, database, target_db)

                if not sql_query:
                    return "**Could not generate SQL query**\n\nPlease try rephrasing your request."

                # Execute query directly on database
                start_time = datetime.now()
                try:
                    print(f"Executing SQL query: {sql_query}")

                    # Execute query directly using SQLDatabase
                    result = database.run(sql_query)
                    execution_time = (datetime.now() - start_time).total_seconds()

                    return f"""✅ **Query Executed Successfully**
                            **Operation**: {explanation}
                            **Database**: {target_db.title()}
                            **SQL Query**: {sql_query}
                        
                             **Results**:
                            {result}
                            """

                except Exception as query_error:
                    print(f"Direct query execution error: {query_error}")
                    return f"""❌ **Query Execution Error**
                            🗄️ **Database**: {target_db.title()}
                            📝 **SQL Query**: {sql_query}
                            🔍 **Issue**: {str(query_error)}
                            Would you like me to help troubleshoot this query?"""

            except Exception as e:
                return f"❌ **Query Error**: {str(e)}"


        # Handle database management commands
        elif "SWITCH_DATABASE:" in llm_response:
            db_type = llm_response.split("SWITCH_DATABASE:", 1)[1].strip()
            if db_type in session["databases"]:
                session["active_db"] = db_type
                return f" **Switched to {db_type.title()} database**\n\nYou can now query the {db_type} database directly. What would you like to know?"
            else:
                return f" **Database Not Connected**: {db_type} is not connected.\n\nAvailable databases: {', '.join(session['databases'].keys())}"

        # Handle database status requests
        elif "SHOW_STATUS" in llm_response:
            if not session["databases"]:
                return " **No Database Connections**\n\nYou haven't connected to any databases yet. Say 'connect to my database' to get started!"

            status_info = "🗄️ **Active Database Connections:**\n\n"
            for db_type, db_info in session["databases"].items():
                active_indicator = "🔴" if db_type == session["active_db"] else "🟢"
                status_info += f"{active_indicator} **{db_type.title()}**: Connected since {db_info['connected_at'][:19]}\n"

            status_info += f"\n**Current Active Database**: {session['active_db'].title() if session['active_db'] else 'None'}"
            status_info += f"\n\n**Available Actions:**\n- Switch: 'use [database] database'\n- Disconnect: 'disconnect from [database]'\n- Disconnect all: 'close all connections'"
            return status_info

        # Regular LLM response - update state if needed
        else:
            # Check if user is trying to connect
            if any(word in user_message.lower() for word in
                   ["connect", "database", "postgresql", "mysql", "sqlite", "oracle", "sqlserver"]):
                if session["state"] == "INITIAL":
                    session["state"] = "AWAITING_CREDENTIALS"

            return llm_response

    def parse_connection_uri(self, uri: str) -> Dict[str, str]:
        """Parse connection URI to extract components"""
        for db_type, pattern in self.db_patterns.items():
            match = re.match(pattern, uri)
            if match:
                if db_type == 'sqlite':
                    return {"database": match.group(1), "type": db_type}
                else:
                    return {
                        "type": db_type,
                        "username": match.group(1),
                        "password": "***",  # Don't expose password
                        "host": match.group(3),
                        "port": match.group(4),
                        "database": match.group(5)
                    }
        return {"type": "unknown"}

    async def generate_sql_query(self, user_message: str, database, db_type: str) -> str:
        """Generate SQL query using LLM with proper syntax guidance"""

        # Get table schema for context
        try:
            tables = database.get_usable_table_names()
            schema_info = ""

            # Get schema for relevant tables (limit to avoid token overflow)
            for table in tables[:5]:  # Limit to first 5 tables
                try:
                    table_info = database.get_table_info([table])
                    schema_info += f"\n{table_info}"
                except:
                    continue

        except Exception as e:
            schema_info = "Schema information not available"
            tables = []

        sql_prompt = f"""
                You are a PostgreSQL SQL expert. Generate a single, executable SQL query for this request.
            
                **User Request**: {user_message}
            
                **Available Tables**: {', '.join(tables) if tables else 'Unknown'}
            
                **Schema Information**:
                {schema_info}
            
                **CRITICAL SQL RULES FOR POSTGRESQL:**
                1. Column names with spaces: "Store ID", "Employee Number" (NO backslashes)
                2. Use double quotes for identifiers: "Column Name"
                3. Proper PostgreSQL syntax only
                4. Limit results to 10-20 rows for large datasets
                5. Use ORDER BY for "last/recent" requests
            
                **Examples of CORRECT PostgreSQL syntax:**
                - SELECT "Store ID", "Employee Number" FROM retail_sales_data LIMIT 10;
                - SELECT COUNT(*) FROM retail_sales_data;
                - SELECT * FROM retail_sales_data ORDER BY "Date" DESC LIMIT 10;
            
                **IMPORTANT**: Return ONLY the SQL query, no explanations or formatting.
                """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=sql_prompt)])
            sql_query = response.content.strip()

            # Clean the SQL query
            sql_query = sql_query.replace("``````", "").strip()

            # Remove any backslashes before quotes
            sql_query = sql_query.replace('\\"', '"')

            print(f"Generated SQL query: {sql_query}")
            return sql_query

        except Exception as e:
            return f"Error generating SQL query: {e}"