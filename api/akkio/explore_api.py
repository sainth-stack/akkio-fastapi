from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
from uuid import uuid4
import pandas as pd
import os
import glob
from pathlib import Path
 

from .explore_functions import (
    SESSION_MEMORY,
    SESSION_MEMORY_LOCK,
    manage_session_memory,
    load_dataset,
    preprocess_dataframe_for_graphing,
    detect_agent,
    classify_query_complexity,
    handle_simple_query,
    format_text_response,
    format_result_for_response,
    generate_data_code,
    simulate_and_format_with_llm,
    get_llm_analysis_explore,
    handle_graph_agent,
    analyze_legal_content,
)

explore_router = APIRouter()


@explore_router.post("/api/Explore")
async def senior_data_analysis(
    query: str = Form(...),
    dataset_path: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    session_id: str = Form(None),
):
    try:
        if not session_id:
            session_id = str(uuid4())
            print(f"Generated new session ID: {session_id}")
        print(f"Processing query for session {session_id}: {query}")
        chat_history_for_llm = manage_session_memory(session_id, get_history=True)
        if chat_history_for_llm:
            print(f"Session {session_id} has previous message for context")
            follow_up_indicators = ['them', 'those', 'these', 'it', 'they']
            if any(indicator in query.lower() for indicator in follow_up_indicators):
                print(f"FOLLOW-UP QUERY DETECTED: '{query}'")
                last_user_query = chat_history_for_llm[0]['content'] if chat_history_for_llm else None
                if last_user_query:
                    print(f"Previous user query for context: '{last_user_query}'")
        else:
            print(f"No previous message found for session {session_id}")
        # (No custom greeting fallback; rely on core pipeline)

        # Resolve dataset_path from filename if provided
        if not dataset_path and filename:
            try:
                # Prefer project-root uploads over CWD to avoid uvicorn cwd issues
                project_root = Path(__file__).resolve().parents[2]
                uploads_dir_primary = project_root / "uploads"
                uploads_dir_fallback = Path(os.getcwd()) / "uploads"
                search_dirs = [uploads_dir_primary, uploads_dir_fallback] if uploads_dir_primary != uploads_dir_fallback else [uploads_dir_primary]
                candidates = []
                for d in search_dirs:
                    if d.exists():
                        pattern = str(d / f"{filename.strip().lower()}*.csv")
                        candidates.extend(glob.glob(pattern))
                if candidates:
                    # Choose the most recent file
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    dataset_path = candidates[0]
            except Exception:
                pass

        df = load_dataset(dataset_path)
        legal_keywords = ['document', 'pdf', 'legal', 'contract', 'agreement', 'policy', 'regulation', 'law', 'analysis', 'summary', 'extract', 'content']
        is_legal_query = any(keyword in query.lower() for keyword in legal_keywords)
        is_legal_data = 'text_content' in df.columns and 'document_type' in df.columns
        if is_legal_query and is_legal_data:
            print(f"Processing legal analysis for session {session_id}")
            professional_analysis = analyze_legal_content(df, query)
            manage_session_memory(session_id, user_message=query, bot_message="Generated comprehensive legal analysis.")
            return JSONResponse(
                content=jsonable_encoder({
                    "type": "text",
                    "payload": professional_analysis,
                    "session_id": session_id
                }),
                status_code=200
            )
        df = preprocess_dataframe_for_graphing(df)
        metadata_str = ", ".join(df.columns.tolist())
        is_report_query = any(keyword in query.lower() for keyword in
                              ['report', 'summary report', 'analysis report', 'detailed report',
                               'comprehensive report', 'summary_report', 'analysis_report', 'detailed_report',
                               'comprehensive_report'])
        if is_report_query:
            prompt_eng = (
                f"""
                    You are a Senior data analyst generating a comprehensive report with advanced analytics capabilities. 
                    Always strictly adhere to the following rules: 
                    The metadata required for your analysis: {metadata_str}
                    Consider ALL rows in the currently loaded dataset in memory. No data assumptions can be taken. Consider the entire range from first row to last row. Do not assume any data outside this range.
                    Generate a comprehensive report with intelligent date handling and analysis for: {query}
                    The report must include:
                        - 4 Bullet points (2 lines each): current analysis of the data.
                        - 1 summary table  and all other analysis metrics
                        - 2 analysis charts showing current data patterns from the data with main Heading of Analysis.
                """
            )
            code = generate_data_code(prompt_eng)
            result = simulate_and_format_with_llm(code, df)
            cleaned_result = None
            try:
                import re, json
                match = re.search(r'```json\s*(\{.*?\})\s*```', result, re.DOTALL)
                if match:
                    cleaned_result = json.loads(match.group(1))
                else:
                    objs = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
                    for m in objs:
                        try:
                            cleaned_result = json.loads(m)
                            break
                        except Exception:
                            continue
            except Exception:
                cleaned_result = None
            if not cleaned_result:
                cleaned_result = {"report": {}, "title": "", "description": ""}
            manage_session_memory(session_id, user_message=query, bot_message="Generated a comprehensive report based on your request.")
            return JSONResponse(
                content=jsonable_encoder({
                    "report": cleaned_result.get("report"),
                    "title": cleaned_result.get("title"),
                    "description": cleaned_result.get("description"),
                    "session_id": session_id
                }),
                status_code=200
            )
        agent = detect_agent(query)
        if agent in ["table", "text"]:
            can_handle_directly, structured = classify_query_complexity(query, list(df.columns))
            if can_handle_directly and structured is not None:
                intent, column = structured
                simple_resp = handle_simple_query(df, intent, column)
                manage_session_memory(session_id, user_message=query, bot_message=f"Processed simple query about {column}.")
                simple_resp["session_id"] = session_id
                return JSONResponse(content=jsonable_encoder(simple_resp), status_code=200)
        if agent == "graph":
            try:
                print(f"Attempting graph generation for session {session_id}: {query}")
                formatted_graph = handle_graph_agent(query, df)
                if formatted_graph is not None:
                    print(f"Graph generation successful for session {session_id}")
                    manage_session_memory(session_id, user_message=query, bot_message="Generated a graph visualization based on your request.")
                    formatted_graph["session_id"] = session_id
                    return JSONResponse(content=jsonable_encoder(formatted_graph), status_code=200)
                else:
                    print(f"Graph agent returned None for session {session_id}, trying fallback")
            except Exception as e:
                print(f"Graph agent failed for session {session_id} with error: {str(e)}, falling back to generic LLM")
                pass
        analysis_result = get_llm_analysis_explore(query, df, mode=agent, chat_history=chat_history_for_llm)
        
        # Debug logging for LLM response
        print(f"[DEBUG] LLM analysis_result for session {session_id}: {analysis_result}")
        
        response_type = analysis_result.get("type")
        payload = analysis_result.get("payload")
        
        # Debug logging for extracted values
        print(f"[DEBUG] response_type: {response_type}, payload type: {type(payload)}")
        
        bot_response_text = "I have processed your request."
        if response_type == "conversational_answer":
            bot_response_text = str(payload) if payload else "I provided a conversational response."
        elif response_type == "data_analysis_answer":
            explanation = payload.get("explanation") if isinstance(payload, dict) else ""
            bot_response_text = explanation or "I have generated the data analysis you requested."
        elif isinstance(payload, str):
            bot_response_text = payload
        manage_session_memory(session_id, user_message=query, bot_message=bot_response_text)
        if not response_type or not payload:
            error_msg = format_text_response("<h4>I'm sorry, I couldn't process that request.</h4><p>Please try rephrasing your question and I'll be happy to help you analyze your data.</p>")
            return JSONResponse(content=jsonable_encoder({"type": "text", "payload": error_msg, "session_id": session_id}), status_code=200)
        if response_type == "conversational_answer":
            formatted_payload = format_text_response(str(payload)) if payload else payload
            return JSONResponse(content=jsonable_encoder({"type": "text", "payload": formatted_payload, "session_id": session_id}), status_code=200)
        if response_type == "data_analysis_answer":
            explanation = payload.get("explanation")
            code = payload.get("code")
            if not code:
                default_msg = "I understood your request but couldn't generate the right code. Please try again with more specific details."
                message_text = explanation or default_msg
                fallback_msg = format_text_response(f"<h4>I understood your request</h4><p>{message_text}</p>")
                return JSONResponse(content={"type": "text", "payload": fallback_msg, "session_id": session_id}, status_code=200)
            try:
                from .explore_functions.agent_utils import safe_execute_pandas_code
                exec_result = safe_execute_pandas_code(code, df)
                formatted = format_result_for_response(exec_result)
                if agent == "graph" and formatted.get("type") != "plotly":
                    print(f"Attempting rescue chart for session {session_id}")
                    from .explore_functions.charts import generate_rescue_chart
                    rescue_fig, biz_exp = generate_rescue_chart(df, query)
                    if rescue_fig is not None:
                        print(f"Rescue chart successful for session {session_id}")
                        formatted = format_result_for_response(rescue_fig)
                        formatted["explanation"] = biz_exp or (explanation or "")
                        formatted["session_id"] = session_id
                        return JSONResponse(content=formatted, status_code=200)
                    else:
                        print(f"Rescue chart also failed for session {session_id}")
                if explanation:
                    formatted["explanation"] = explanation
                formatted["session_id"] = session_id
                return JSONResponse(content=jsonable_encoder(formatted), status_code=200)
            except Exception as e:
                error_msg = format_text_response(f"<h4>Analysis Error</h4><p>There was an error executing the analysis: {str(e)}</p><p>Please try rephrasing your question or provide more specific details.</p>")
                return JSONResponse(content=jsonable_encoder({"type": "text", "payload": error_msg, "explanation": explanation, "session_id": session_id}), status_code=200)
        fallback_msg = format_text_response(f"<h4>Unrecognized Response</h4><p>I encountered an unexpected response type: {response_type}</p><p>Please try rephrasing your question and I'll provide a better analysis.</p>")
        return JSONResponse(content=jsonable_encoder({"type": "text", "payload": fallback_msg, "session_id": session_id}), status_code=200)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Data file is corrupt")
    except Exception as e:
        session_id_for_error = session_id if 'session_id' in locals() else None
        print(f"Explore API error for session {session_id_for_error}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@explore_router.get("/api/debug_session_memory/{session_id}")
async def debug_session_memory_explore(session_id: str):
    try:
        with SESSION_MEMORY_LOCK:
            session_data = SESSION_MEMORY.get(session_id, {})
        return JSONResponse(content={
            "session_id": session_id,
            "last_user_message": session_data.get("last_user_message"),
            "last_bot_message": session_data.get("last_bot_message"),
            "has_previous_context": bool(session_data.get("last_user_message")),
            "total_sessions": len(SESSION_MEMORY),
            "api": "explore"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


@explore_router.delete("/api/clear_session_memory/{session_id}")
async def clear_session_memory_explore(session_id: str):
    try:
        with SESSION_MEMORY_LOCK:
            if session_id in SESSION_MEMORY:
                session_data = SESSION_MEMORY[session_id]
                had_user_msg = bool(session_data.get("last_user_message"))
                had_bot_msg = bool(session_data.get("last_bot_message"))
                del SESSION_MEMORY[session_id]
                return JSONResponse(content={
                    "session_id": session_id,
                    "had_user_message": had_user_msg,
                    "had_bot_message": had_bot_msg,
                    "status": "cleared",
                    "api": "explore"
                })
            else:
                return JSONResponse(content={
                    "session_id": session_id,
                    "had_user_message": False,
                    "had_bot_message": False,
                    "status": "not_found",
                    "api": "explore"
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


