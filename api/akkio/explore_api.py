from fastapi import APIRouter, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
from uuid import uuid4
import pandas as pd
import os
import glob
import json
import asyncio
from pathlib import Path
from langchain_community.vectorstores import Chroma
try:
    from langchain_openai import OpenAIEmbeddings  # preferred
except Exception:
    from langchain_community.embeddings import OpenAIEmbeddings  # fallback
from database import PostgresDatabase

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
from .explore_functions.llm import get_openai_client, call_llm_with_usage, stream_llm_with_usage
from .usage_tracking import set_current_email, reset_current_email

import base64

explore_router = APIRouter()

def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (s or ""))

def _resolve_collection_name(persist_dir: Path, email: Optional[str], filename: str) -> Optional[str]:
    """
    Resolve a Chroma collection name for a given email+filename, allowing minor
    mismatches like missing trailing tokens (e.g., '__2') in filename stem.
    Preference order:
    1) Exact '<safe_email>__<safe_name>'
    2) Startswith '<safe_email>__<safe_name>'
    3) Endswith '__<safe_name>'
    4) Contains '__<safe_name>__'
    Picks the longest match to be safe.
    """
    try:
        import chromadb
    except Exception:
        return None
    safe_name = _safe(Path(filename).stem or filename)
    safe_email = _safe(email) if email else None
    exact = f"{safe_email}__{safe_name}" if safe_email else None
    try:
        cclient = chromadb.PersistentClient(path=str(persist_dir))
        collections = cclient.list_collections()
    except Exception:
        return None
    try:
        print(f"[VECTOR][RESOLVE] email='{email}', filename='{filename}', safe_email='{safe_email}', safe_name='{safe_name}', total_collections={len(collections)}")
    except Exception:
        pass
    # Build candidates with scores
    scored: list[tuple[int, str]] = []
    for coll in collections:
        name = coll.name
        score = -1
        if exact and name == exact:
            score = 400  # best
        elif safe_email and name.startswith(f"{safe_email}__{safe_name}"):
            score = 300 + len(name)
        elif name.endswith(f"__{safe_name}"):
            score = 200 + len(name)
        elif f"__{safe_name}__" in name:
            score = 100 + len(name)
        if score >= 0:
            scored.append((score, name))
    if not scored:
        try:
            print(f"[VECTOR][RESOLVE] No matching collection for safe_name='{safe_name}'")
        except Exception:
            pass
        return None
    scored.sort(reverse=True)
    try:
        print(f"[VECTOR][RESOLVE] candidates={scored[:5]}")
    except Exception:
        pass
    return scored[0][1]

def _is_generic_summary_query(q: str) -> bool:
    ql = (q or "").strip().lower()
    generic_markers = [
        "explain", "summary", "summarize", "summarise", "about", "overview",
        "describe", "details", "in detailed", "in detail",
        "what is this doc", "what is this pdf", "what is this document",
        "explain about the pdf", "explain about doc", "explain pdf", "explain document",
        "what the uploaded document contains", "what the document contains",
        "what does the document contain", "what does this document contain",
        "what does the pdf contain", "content of the document", "document content",
        "contents of the document", "contains", "content",
        "what is this audio", "explain about audio", "explain audio", "audio content",
        "what does the audio contain", "transcription", "what was said", "what is said"
    ]
    return any(m in ql for m in generic_markers)

def _retrieve_docs(vectordb: Chroma, query: str, k: int):
    try:
        # Prefer diverse retrieval if available
        if hasattr(vectordb, "max_marginal_relevance_search"):
            return vectordb.max_marginal_relevance_search(query, k=max(3, k), fetch_k=max(10, k*2))
    except Exception:
        pass
    try:
        return vectordb.similarity_search(query, k=max(3, k))
    except Exception:
        return []

@explore_router.post("/Explore")
async def senior_data_analysis(
    query: str = Form(...),
    dataset_path: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    session_id: str = Form(None),
    email: Optional[str] = Form(None),
    file_type: Optional[str] = Form(None),
):
    email_token = set_current_email(email)
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
                        base = filename.strip().lower()
                        for ext in ("csv", "xlsx", "xls", "pdf", "docx"):
                            pattern = str(d / f"{base}*.{ext}")
                            candidates.extend(glob.glob(pattern))
                if candidates:
                    # Choose the most recent file
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    dataset_path = candidates[0]
            except Exception:
                pass
        # If still no dataset_path, but filename provided, try vector or raw-image fallback (pdf/word/images)
        if not dataset_path and filename:
            # PRIORITY: If file_type is "image", try vision model first with raw image
            if (file_type or "").lower() == "image":
                try:
                    print(f"[IMAGE][EXPLORE] Attempting vision model analysis for filename='{filename}'")
                    db = PostgresDatabase()
                    raw = db.get_raw_file(Path(filename).stem)
                    if raw:
                        client = get_openai_client()
                        b64 = base64.b64encode(raw).decode("utf-8")
                        
                        # Determine if it's a generic summary query or specific question
                        if _is_generic_summary_query(query):
                            system_prompt = (
                                "You are a helpful assistant analyzing images. Provide a detailed description of the image, "
                                "including what objects, people, text, colors, and visual elements are present.\n"
                                "Format your response in clean HTML with proper structure:\n"
                                "- Use <h4> for main headings\n"
                                "- Use <p> for paragraphs\n"
                                "- Use <ul> and <li> for bullet lists\n"
                                "- Use <strong> for emphasis\n"
                                "Be specific and descriptive."
                            )
                            user_prompt = "Please provide a detailed description and analysis of this image."
                        else:
                            system_prompt = (
                                "You are a helpful assistant answering questions about images using visual analysis.\n"
                                "Format your response in clean HTML with proper structure:\n"
                                "- Use <h4> for main headings\n"
                                "- Use <p> for paragraphs\n"
                                "- Use <ul> and <li> for lists\n"
                                "- Use <strong> for emphasis\n"
                                "Be specific and refer to what you actually see in the image."
                            )
                            user_prompt = query
                        
                        chat = call_llm_with_usage(
                            model=None,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": [
                                    {"type": "text", "text": user_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                                ]}
                            ],
                            temperature=0.0,
                            email=email
                        )

                        answer = chat.choices[0].message.content if chat and chat.choices else "No answer generated."
                        
                        # Clean up the answer
                        answer = answer.strip()
                        if answer.startswith("```html"):
                            answer = answer[7:]
                        if answer.startswith("```"):
                            answer = answer[3:]
                        if answer.endswith("```"):
                            answer = answer[:-3]
                        answer = answer.strip()
                        answer = answer.replace("\\n", "").replace("\n", "").replace("\r", "")
                        
                        manage_session_memory(session_id, user_message=query, bot_message=answer)
                        print(f"[IMAGE][EXPLORE] Successfully analyzed image with vision model")
                        return JSONResponse(content=jsonable_encoder({
                            "type": "text",
                            "payload": answer,
                            "session_id": session_id,
                        }), status_code=200)
                    else:
                        print(f"[IMAGE][EXPLORE] No raw image data found in database for '{filename}'")
                except Exception as e:
                    print(f"[IMAGE][EXPLORE] Vision model failed: {str(e)}, falling back to vector store")
                    pass
            
            # PRIORITY: If file_type is "audio", handle transcription queries
            if (file_type or "").lower() == "audio":
                try:
                    print(f"[AUDIO][EXPLORE] ═══════════════════════════════════════════════")
                    print(f"[AUDIO][EXPLORE] Audio query detected")
                    print(f"[AUDIO][EXPLORE] Filename: {filename}")
                    print(f"[AUDIO][EXPLORE] Query: {query}")
                    print(f"[AUDIO][EXPLORE] Email: {email}")
                    print(f"[AUDIO][EXPLORE] Session ID: {session_id}")
                    print(f"[AUDIO][EXPLORE] ═══════════════════════════════════════════════")
                    
                    project_root = Path(__file__).resolve().parents[2]
                    persist_dir = project_root / "chroma_store"
                    print(f"[AUDIO][EXPLORE] Vector store directory: {persist_dir}")
                    
                    if persist_dir.exists():
                        print(f"[AUDIO][EXPLORE] ✓ Vector store directory exists")
                        print(f"[AUDIO][EXPLORE] Initializing OpenAI embeddings...")
                        embeddings = OpenAIEmbeddings()
                        print(f"[AUDIO][EXPLORE] ✓ Embeddings initialized")
                        
                        print(f"[AUDIO][EXPLORE] Resolving collection name...")
                        coll_name = _resolve_collection_name(persist_dir, email, filename)
                        print(f"[AUDIO][EXPLORE] Collection name resolved: {coll_name}")
                        
                        if coll_name:
                            print(f"[AUDIO][EXPLORE] Loading Chroma vector database...")
                            vectordb = Chroma(collection_name=coll_name, persist_directory=str(persist_dir), embedding_function=embeddings)
                            print(f"[AUDIO][EXPLORE] ✓ Vector database loaded")
                            
                            # Get database for metadata
                            print(f"[AUDIO][EXPLORE] Fetching audio metadata from database...")
                            db = PostgresDatabase()
                            metadata_info = ""
                            try:
                                # Try to get audio metadata from database
                                df_meta = db.get_table_data(Path(filename).stem)
                                if df_meta is not None and not df_meta.empty:
                                    print(f"[AUDIO][EXPLORE] ✓ Found metadata in database: {len(df_meta)} rows")
                                    # Extract metadata from first row
                                    if 'duration_seconds' in df_meta.columns:
                                        duration = df_meta['duration_seconds'].iloc[0]
                                        sample_rate = df_meta.get('sample_rate_hz', pd.Series([None])).iloc[0]
                                        channels = df_meta.get('channel_names', pd.Series([None])).iloc[0]
                                        quality = df_meta.get('audio_quality', pd.Series([None])).iloc[0]
                                        
                                        print(f"[AUDIO][EXPLORE] Metadata: duration={duration}s, sample_rate={sample_rate}Hz, channels={channels}, quality={quality}")
                                        
                                        metadata_info = f"\n\n<strong>Audio File Properties:</strong><ul>"
                                        if duration:
                                            metadata_info += f"<li>Duration: {duration} seconds</li>"
                                        if sample_rate:
                                            metadata_info += f"<li>Sample Rate: {sample_rate} Hz</li>"
                                        if channels:
                                            metadata_info += f"<li>Channels: {channels}</li>"
                                        if quality:
                                            metadata_info += f"<li>Quality: {quality}</li>"
                                        metadata_info += "</ul>"
                                        print(f"[AUDIO][EXPLORE] ✓ Metadata info formatted for response")
                                else:
                                    print(f"[AUDIO][EXPLORE] No metadata found in database")
                            except Exception as me:
                                print(f"[AUDIO][EXPLORE] ✗ Could not extract metadata: {me}")
                                import traceback
                                traceback.print_exc()
                            
                            # Determine if it's a generic summary query
                            is_summary = _is_generic_summary_query(query)
                            print(f"[AUDIO][EXPLORE] Query type: {'SUMMARY' if is_summary else 'SPECIFIC'}")
                            
                            if is_summary:
                                # Get all content for summary
                                print(f"[AUDIO][EXPLORE] Retrieving documents for summary (k=30)...")
                                docs = _retrieve_docs(vectordb, "audio transcription summary", k=30)
                                print(f"[AUDIO][EXPLORE] ✓ Retrieved {len(docs)} documents")
                                
                                full_text = "\n".join([d.page_content for d in docs])
                                print(f"[AUDIO][EXPLORE] Combined text length: {len(full_text)} characters")
                                
                                print(f"[AUDIO][EXPLORE] Calling OpenAI for summary generation...")
                                client = get_openai_client()
                                system_prompt = (
                                    "You are analyzing an audio file transcription. Provide a comprehensive summary including:\n"
                                    "1. Main topics discussed\n"
                                    "2. Key points and insights\n"
                                    "3. Any important information or conclusions\n"
                                    "Format your response in clean HTML with:\n"
                                    "- Use <h4> for section headings\n"
                                    "- Use <p> for paragraphs\n"
                                    "- Use <ul> and <li> for bullet points\n"
                                    "- Use <strong> for emphasis"
                                )
                                user_prompt = f"Audio transcription:\n\n{full_text}\n\nProvide a detailed summary."
                                
                                chat = call_llm_with_usage(
                                    model=None,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}
                                    ],
                                    temperature=0.0,
                                    email=email
                                )

                                answer = chat.choices[0].message.content if chat and chat.choices else "No answer generated."
                                print(f"[AUDIO][EXPLORE] ✓ Summary generated: {len(answer)} characters")
                            else:
                                # Specific query - use semantic search
                                print(f"[AUDIO][EXPLORE] Performing semantic search for query (k=10)...")
                                docs = _retrieve_docs(vectordb, query, k=10)
                                print(f"[AUDIO][EXPLORE] ✓ Retrieved {len(docs)} relevant documents")
                                
                                context = "\n".join([d.page_content for d in docs])
                                print(f"[AUDIO][EXPLORE] Context length: {len(context)} characters")
                                
                                print(f"[AUDIO][EXPLORE] Calling OpenAI for Q&A...")
                                client = get_openai_client()
                                system_prompt = (
                                    "You are answering questions about an audio transcription. "
                                    "Use the provided context to answer accurately.\n"
                                    "Format your response in clean HTML with:\n"
                                    "- Use <h4> for headings if needed\n"
                                    "- Use <p> for paragraphs\n"
                                    "- Use <ul> and <li> for lists\n"
                                    "- Use <strong> for emphasis\n"
                                    "If the context doesn't contain relevant information, say so politely."
                                )
                                user_prompt = f"Context from audio transcription:\n\n{context}\n\nQuestion: {query}"
                                
                                chat = call_llm_with_usage(
                                    model=None,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}
                                    ],
                                    temperature=0.0,
                                    email=email
                                )

                                answer = chat.choices[0].message.content if chat and chat.choices else "No answer generated."
                                print(f"[AUDIO][EXPLORE] ✓ Answer generated: {len(answer)} characters")
                            
                            # Add metadata to answer if available
                            if metadata_info:
                                print(f"[AUDIO][EXPLORE] Appending metadata to answer...")
                                answer = answer + metadata_info
                                print(f"[AUDIO][EXPLORE] ✓ Metadata appended")
                            
                            # Clean up the answer
                            print(f"[AUDIO][EXPLORE] Cleaning up answer formatting...")
                            answer = answer.strip()
                            if answer.startswith("```html"):
                                answer = answer[7:]
                            if answer.startswith("```"):
                                answer = answer[3:]
                            if answer.endswith("```"):
                                answer = answer[:-3]
                            answer = answer.strip()
                            answer = answer.replace("\\n", "").replace("\n", "").replace("\r", "")
                            print(f"[AUDIO][EXPLORE] ✓ Answer cleaned, final length: {len(answer)} characters")
                            
                            print(f"[AUDIO][EXPLORE] Saving to session memory...")
                            manage_session_memory(session_id, user_message=query, bot_message=answer)
                            print(f"[AUDIO][EXPLORE] ✓ Session memory updated")
                            
                            print(f"[AUDIO][EXPLORE] ═══════════════════════════════════════════════")
                            print(f"[AUDIO][EXPLORE] ✓✓✓ QUERY PROCESSED SUCCESSFULLY ✓✓✓")
                            print(f"[AUDIO][EXPLORE] Query: {query[:50]}{'...' if len(query) > 50 else ''}")
                            print(f"[AUDIO][EXPLORE] Answer length: {len(answer)} characters")
                            print(f"[AUDIO][EXPLORE] Metadata included: {'Yes' if metadata_info else 'No'}")
                            print(f"[AUDIO][EXPLORE] ═══════════════════════════════════════════════")
                            
                            return JSONResponse(content=jsonable_encoder({
                                "type": "text",
                                "payload": answer,
                                "session_id": session_id,
                            }), status_code=200)
                        else:
                            print(f"[AUDIO][EXPLORE] ✗ No vector collection found for '{filename}'")
                    else:
                        print(f"[AUDIO][EXPLORE] ✗ Vector store directory does not exist: {persist_dir}")
                except Exception as e:
                    print(f"[AUDIO][EXPLORE] ═══════════════════════════════════════════════")
                    print(f"[AUDIO][EXPLORE] ✗✗✗ AUDIO PROCESSING FAILED ✗✗✗")
                    print(f"[AUDIO][EXPLORE] Error: {str(e)}")
                    print(f"[AUDIO][EXPLORE] ═══════════════════════════════════════════════")
                    import traceback
                    traceback.print_exc()
                    pass
            
            # Fallback: Try vector store for PDFs, Word docs, or if image vision fails
            try:
                project_root = Path(__file__).resolve().parents[2]
                persist_dir = project_root / "chroma_store"
                if persist_dir.exists():
                    coll_name = _resolve_collection_name(persist_dir, email, filename)
                    try:
                        print(f"[VECTOR][EXPLORE] resolved collection='{coll_name}' for filename='{filename}', email='{email}'")
                    except Exception:
                        pass
                    if coll_name:
                        try:
                            embeddings = OpenAIEmbeddings()
                            vectordb = Chroma(collection_name=coll_name, persist_directory=str(persist_dir), embedding_function=embeddings)
                            docs = _retrieve_docs(vectordb, query, k=7)
                            try:
                                print(f"[VECTOR][EXPLORE] retrieved_docs={len(docs)} for query='{query}'")
                            except Exception:
                                pass
                            if docs:
                                context = "\n\n---\n\n".join([d.page_content for d in docs if d and d.page_content])
                                try:
                                    print(f"[VECTOR][EXPLORE] context_chars={len(context)} sample='{context[:200].replace(chr(10),' ')}...'")
                                    # Debug: Print all retrieved chunks to see what's being used
                                    for i, d in enumerate(docs):
                                        print(f"[VECTOR][DEBUG] Chunk {i}: {d.page_content[:100]}... (Source: {d.metadata})")
                                except Exception:
                                    pass
                                client = get_openai_client()
                                if _is_generic_summary_query(query):
                                    system_prompt = (
                                        "You are a helpful assistant. Create a clear, self-contained summary of the document "
                                        "STRICTLY from the provided context. Include key sections, main points, and any important numbers/dates.\n"
                                        "Format your response in clean HTML with proper structure:\n"
                                        "- Use <h4> for main headings\n"
                                        "- Use <p> for paragraphs\n"
                                        "- Use <ul> and <li> for bullet lists\n"
                                        "- Use <strong> for emphasis\n"
                                        "Make it user-friendly and well-formatted."
                                    )
                                    user_payload = f"Context:\n{context}\n\nTask:\nProvide a concise, detailed, well-formatted HTML summary of this document."
                                else:
                                    system_prompt = (
                                        "You are a helpful assistant answering questions strictly from the provided context.\n"
                                        "Format your response in clean HTML with proper structure:\n"
                                        "- Use <h4> for main headings\n"
                                        "- Use <p> for paragraphs\n"
                                        "- Use <ul> and <li> for lists\n"
                                        "- Use <strong> for emphasis\n"
                                        "If the answer is not present in the context, say you cannot find it in a friendly HTML format."
                                    )
                                    user_payload = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer using only the context, formatted in clean HTML."
                                chat = call_llm_with_usage(
                                    model=None,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_payload},
                                    ],
                                    temperature=0.0,
                                    email=email
                                )

                                answer = chat.choices[0].message.content if chat and chat.choices else "No answer generated."
                                # Clean up markdown code fences and escape sequences
                                answer = answer.strip()
                                if answer.startswith("```html"):
                                    answer = answer[7:]
                                if answer.startswith("```"):
                                    answer = answer[3:]
                                if answer.endswith("```"):
                                    answer = answer[:-3]
                                answer = answer.strip()
                                # Remove all newlines (both escaped and literal)
                                answer = answer.replace("\\n", "").replace("\n", "").replace("\r", "")
                                manage_session_memory(session_id, user_message=query, bot_message=answer)
                                return JSONResponse(content=jsonable_encoder({
                                    "type": "text",
                                    "payload": answer,
                                    "session_id": session_id,
                                }), status_code=200)
                        except Exception:
                            pass
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
            code = generate_data_code(prompt_eng, email=email)
            result = simulate_and_format_with_llm(code, df, email=email)
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
        analysis_result = get_llm_analysis_explore(query, df, mode=agent, chat_history=chat_history_for_llm, email=email)
        
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
    finally:
        reset_current_email(email_token)


@explore_router.get("/debug_session_memory/{session_id}")
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


@explore_router.delete("/clear_session_memory/{session_id}")
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

def _extract_texts_for_embedding_from_df(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    try:
        if 'text_content' in df.columns:
            texts = [str(t).strip() for t in df['text_content'].dropna().astype(str).tolist() if str(t).strip()]
        else:
            # Fallback: serialize first N rows
            records = df.fillna("").astype(str).to_dict(orient="records")
            texts = [str(rec) for rec in records[:1000]]
    except Exception:
        texts = []
    # Deduplicate
    seen = set()
    uniq = []
    for t in texts:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ============== Vector chat over uploaded docs (pdf/word/images) ==============
@explore_router.post("/vector_chat")
async def vector_chat(
    query: str = Form(...),
    filename: str = Form(...),
    email: Optional[str] = Form(None),
    file_type: Optional[str] = Form(None),
    top_k: int = Form(5),
    session_id: str = Form(None),
):
    """
    Answer questions over previously uploaded non-tabular documents (pdf/word/images).
    - Looks up ChromaDB collection by email+filename (or by filename suffix).
    - Retrieves top_k similar chunks and asks LLM to answer using ONLY that context.
    """
    try:
        # Resolve Chroma persistence dir (project root)
        project_root = Path(__file__).resolve().parents[2]
        persist_dir = project_root / "chroma_store"
        if not persist_dir.exists():
            return JSONResponse(content={"detail": "No vector store available yet. Upload a file first."}, status_code=404)

        # Determine collection name robustly
        coll_name = _resolve_collection_name(persist_dir, email, filename)

        if not coll_name:
            return JSONResponse(content={"detail": f"No vector collection found for '{filename}'. Try re-uploading."}, status_code=404)

        # If explicit image type, try answering directly with raw image first
        if (file_type or "").lower() == "image":
            try:
                db = PostgresDatabase()
                raw = db.get_raw_file(Path(filename).stem)
                if raw:
                    client = get_openai_client()
                    b64 = base64.b64encode(raw).decode("utf-8")
                    system_prompt = "Answer the user's question about the provided image. Use visual reasoning."
                    chat = call_llm_with_usage(
                        model=None,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": query},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                            ]}
                        ],
                        temperature=0.0,
                        email=email
                    )

                    answer = chat.choices[0].message.content if chat and chat.choices else "No answer generated."
                    return JSONResponse(content=jsonable_encoder({
                        "type": "text",
                        "payload": answer,
                        "session_id": session_id,
                    }), status_code=200)
            except Exception:
                pass

        # Query vector store
        try:
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            return JSONResponse(content={"detail": f"Embeddings unavailable: {e}"}, status_code=500)

        vectordb = Chroma(collection_name=coll_name, persist_directory=str(persist_dir), embedding_function=embeddings)
        docs = _retrieve_docs(vectordb, query, k=max(3, min(15, top_k)))
        try:
            print(f"[VECTOR][CHAT] resolved collection='{coll_name}', retrieved_docs={len(docs)} for query='{query}'")
        except Exception:
            pass
        # If nothing found, try to build vectors from DB-stored dataframe for this filename
        if not docs:
            try:
                db = PostgresDatabase()
                df = db.get_table_data(Path(filename).stem)
                texts = _extract_texts_for_embedding_from_df(df)
                try:
                    print(f"[VECTOR][CHAT] docs empty; building from DB rows={len(df) if isinstance(df, pd.DataFrame) else 'NA'}, texts={len(texts)}")
                except Exception:
                    pass
                if texts:
                    ids = [f"{coll_name}_{i}" for i in range(len(texts))]
                    vectordb.add_texts(texts=texts, metadatas=[{"email": email, "name": Path(filename).stem}] * len(texts), ids=ids)
                    docs = _retrieve_docs(vectordb, query, k=max(3, min(15, top_k)))
            except Exception:
                pass
        if not docs:
            try:
                print(f"[VECTOR][CHAT] No relevant context found after rebuild for collection='{coll_name}'")
            except Exception:
                pass
            return JSONResponse(content={"detail": "No relevant context found in vector store"}, status_code=404)

        # Build context for LLM
        context = "\n\n---\n\n".join([d.page_content for d in docs if d and d.page_content])
        try:
            print(f"[VECTOR][CHAT] context_chars={len(context)} sample='{context[:200].replace(chr(10),' ')}...'")
        except Exception:
            pass

        # Ask LLM, restricted to context
        try:
            client = get_openai_client()
        except Exception as e:
            return JSONResponse(content={"detail": f"LLM unavailable: {e}"}, status_code=500)

        if _is_generic_summary_query(query):
            system_prompt = (
                "You are a helpful assistant. Create a clear, self-contained summary of the document "
                "STRICTLY from the provided context. Include key sections, main points, and any important numbers/dates.\n"
                "Format your response in clean HTML with proper structure:\n"
                "- Use <h4> for main headings\n"
                "- Use <p> for paragraphs\n"
                "- Use <ul> and <li> for bullet lists\n"
                "- Use <strong> for emphasis\n"
                "Make it user-friendly and well-formatted."
            )
            user_payload = f"Context:\n{context}\n\nTask:\nProvide a concise, detailed, well-formatted HTML summary of this document."
        else:
            system_prompt = (
                "You are a helpful assistant answering questions strictly from the provided context.\n"
                "Format your response in clean HTML with proper structure:\n"
                "- Use <h4> for main headings\n"
                "- Use <p> for paragraphs\n"
                "- Use <ul> and <li> for lists\n"
                "- Use <strong> for emphasis\n"
                "If the answer is not present in the context, say you cannot find it in a friendly HTML format."
            )
            user_payload = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer using only the context, formatted in clean HTML."
        chat = call_llm_with_usage(
            model=None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            email=email
        )

        answer = chat.choices[0].message.content if chat and chat.choices else "No answer generated."
        
        # Clean up markdown code fences and escape sequences
        answer = answer.strip()
        if answer.startswith("```html"):
            answer = answer[7:]
        if answer.startswith("```"):
            answer = answer[3:]
        if answer.endswith("```"):
            answer = answer[:-3]
        answer = answer.strip()
        # Remove all newlines (both escaped and literal)
        answer = answer.replace("\\n", "").replace("\n", "").replace("\r", "")

        # Track session memory lightly
        try:
            if not session_id:
                session_id = str(uuid4())
            manage_session_memory(session_id, user_message=query, bot_message=answer)
        except Exception:
            pass

        return JSONResponse(content=jsonable_encoder({
            "type": "text",
            "payload": answer,
            "session_id": session_id,
            "source_docs": [getattr(d, 'metadata', {}) for d in docs]
        }), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector chat failed: {str(e)}")


async def _send_websocket_message(websocket: WebSocket, message_type: str, data: dict):
    """Helper to send JSON messages through WebSocket with proper serialization"""
    try:
        # Ensure all data is JSON-serializable
        serialized_data = jsonable_encoder(data)
        # If data already has a "type" key, merge it properly
        message = {"type": message_type}
        message.update(serialized_data)
        await websocket.send_json(message)
    except Exception as e:
        print(f"Error sending WebSocket message: {e}")
        import traceback
        traceback.print_exc()


async def _stream_text_response(websocket: WebSocket, text: str, message_type: str = "text_chunk"):
    """Stream text character by character or in chunks"""
    # Stream in chunks of reasonable size for better UX
    chunk_size = 10
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        await _send_websocket_message(websocket, message_type, {"chunk": chunk})
        await asyncio.sleep(0.01)  # Small delay to make streaming visible


@explore_router.websocket("/Explore/ws")
async def explore_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming explore API responses.
    Expected message format: {
        "query": str,
        "filename": Optional[str],
        "session_id": Optional[str],
        "email": Optional[str],
        "file_type": Optional[str],
        "dataset_path": Optional[str]
    }
    """
    await websocket.accept()
    email_token = None
    
    try:
        # Receive initial message with query parameters
        data = await websocket.receive_json()
        query = data.get("query", "")
        filename = data.get("filename")
        session_id = data.get("session_id")
        email = data.get("email")
        file_type = data.get("file_type")
        dataset_path = data.get("dataset_path")
        
        if not query:
            await _send_websocket_message(websocket, "error", {"message": "Query is required"})
            await websocket.close()
            return
        
        email_token = set_current_email(email)
        
        if not session_id:
            session_id = str(uuid4())
            await _send_websocket_message(websocket, "session_id", {"session_id": session_id})
        
        # Get chat history
        chat_history_for_llm = manage_session_memory(session_id, get_history=True)
        
        # Resolve dataset_path from filename if provided
        if not dataset_path and filename:
            try:
                project_root = Path(__file__).resolve().parents[2]
                uploads_dir_primary = project_root / "uploads"
                uploads_dir_fallback = Path(os.getcwd()) / "uploads"
                search_dirs = [uploads_dir_primary, uploads_dir_fallback] if uploads_dir_primary != uploads_dir_fallback else [uploads_dir_primary]
                candidates = []
                for d in search_dirs:
                    if d.exists():
                        base = filename.strip().lower()
                        for ext in ("csv", "xlsx", "xls", "pdf", "docx"):
                            pattern = str(d / f"{base}*.{ext}")
                            candidates.extend(glob.glob(pattern))
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    dataset_path = candidates[0]
            except Exception:
                pass
        
        # Handle image type with vision model
        if not dataset_path and filename and (file_type or "").lower() == "image":
            try:
                db = PostgresDatabase()
                raw = db.get_raw_file(Path(filename).stem)
                if raw:
                    client = get_openai_client()
                    b64 = base64.b64encode(raw).decode("utf-8")
                    
                    if _is_generic_summary_query(query):
                        system_prompt = (
                            "You are a helpful assistant analyzing images. Provide a detailed description of the image, "
                            "including what objects, people, text, colors, and visual elements are present.\n"
                            "Format your response in clean HTML with proper structure:\n"
                            "- Use <h4> for main headings\n"
                            "- Use <p> for paragraphs\n"
                            "- Use <ul> and <li> for bullet lists\n"
                            "- Use <strong> for emphasis\n"
                            "Be specific and descriptive."
                        )
                        user_prompt = "Please provide a detailed description and analysis of this image."
                    else:
                        system_prompt = (
                            "You are a helpful assistant answering questions about images using visual analysis.\n"
                            "Format your response in clean HTML with proper structure:\n"
                            "- Use <h4> for main headings\n"
                            "- Use <p> for paragraphs\n"
                            "- Use <ul> and <li> for lists\n"
                            "- Use <strong> for emphasis\n"
                            "Be specific and refer to what you actually see in the image."
                        )
                        user_prompt = query
                    
                    await _send_websocket_message(websocket, "response_type", {"type": "text"})
                    
                    full_answer = ""
                    async for chunk in stream_llm_with_usage(
                        model=None,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                            ]}
                        ],
                        temperature=0.0,
                        email=email
                    ):
                        full_answer += chunk
                        await _send_websocket_message(websocket, "text_chunk", {"chunk": chunk})
                    
                    # Clean up answer
                    answer = full_answer.strip()
                    if answer.startswith("```html"):
                        answer = answer[7:]
                    if answer.startswith("```"):
                        answer = answer[3:]
                    if answer.endswith("```"):
                        answer = answer[:-3]
                    answer = answer.strip().replace("\\n", "").replace("\n", "").replace("\r", "")
                    
                    manage_session_memory(session_id, user_message=query, bot_message=answer)
                    await _send_websocket_message(websocket, "complete", {"session_id": session_id})
                    await websocket.close()
                    return
            except Exception as e:
                print(f"[IMAGE][EXPLORE] Vision model failed: {str(e)}")
                pass
        
        # Handle audio type
        if not dataset_path and filename and (file_type or "").lower() == "audio":
            try:
                project_root = Path(__file__).resolve().parents[2]
                persist_dir = project_root / "chroma_store"
                
                if persist_dir.exists():
                    embeddings = OpenAIEmbeddings()
                    coll_name = _resolve_collection_name(persist_dir, email, filename)
                    
                    if coll_name:
                        vectordb = Chroma(collection_name=coll_name, persist_directory=str(persist_dir), embedding_function=embeddings)
                        
                        # Get metadata
                        db = PostgresDatabase()
                        metadata_info = ""
                        try:
                            df_meta = db.get_table_data(Path(filename).stem)
                            if df_meta is not None and not df_meta.empty:
                                if 'duration_seconds' in df_meta.columns:
                                    duration = df_meta['duration_seconds'].iloc[0]
                                    sample_rate = df_meta.get('sample_rate_hz', pd.Series([None])).iloc[0]
                                    channels = df_meta.get('channel_names', pd.Series([None])).iloc[0]
                                    quality = df_meta.get('audio_quality', pd.Series([None])).iloc[0]
                                    
                                    metadata_info = f"\n\n<strong>Audio File Properties:</strong><ul>"
                                    if duration:
                                        metadata_info += f"<li>Duration: {duration} seconds</li>"
                                    if sample_rate:
                                        metadata_info += f"<li>Sample Rate: {sample_rate} Hz</li>"
                                    if channels:
                                        metadata_info += f"<li>Channels: {channels}</li>"
                                    if quality:
                                        metadata_info += f"<li>Quality: {quality}</li>"
                                    metadata_info += "</ul>"
                        except Exception:
                            pass
                        
                        is_summary = _is_generic_summary_query(query)
                        
                        await _send_websocket_message(websocket, "response_type", {"type": "text"})
                        
                        if is_summary:
                            docs = _retrieve_docs(vectordb, "audio transcription summary", k=30)
                            full_text = "\n".join([d.page_content for d in docs])
                            
                            system_prompt = (
                                "You are analyzing an audio file transcription. Provide a comprehensive summary including:\n"
                                "1. Main topics discussed\n"
                                "2. Key points and insights\n"
                                "3. Any important information or conclusions\n"
                                "Format your response in clean HTML with:\n"
                                "- Use <h4> for section headings\n"
                                "- Use <p> for paragraphs\n"
                                "- Use <ul> and <li> for bullet points\n"
                                "- Use <strong> for emphasis"
                            )
                            user_prompt = f"Audio transcription:\n\n{full_text}\n\nProvide a detailed summary."
                        else:
                            docs = _retrieve_docs(vectordb, query, k=10)
                            context = "\n".join([d.page_content for d in docs])
                            
                            system_prompt = (
                                "You are answering questions about an audio transcription. "
                                "Use the provided context to answer accurately.\n"
                                "Format your response in clean HTML with:\n"
                                "- Use <h4> for headings if needed\n"
                                "- Use <p> for paragraphs\n"
                                "- Use <ul> and <li> for lists\n"
                                "- Use <strong> for emphasis\n"
                                "If the context doesn't contain relevant information, say so politely."
                            )
                            user_prompt = f"Context from audio transcription:\n\n{context}\n\nQuestion: {query}"
                        
                        full_answer = ""
                        async for chunk in stream_llm_with_usage(
                            model=None,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.0,
                            email=email
                        ):
                            full_answer += chunk
                            await _send_websocket_message(websocket, "text_chunk", {"chunk": chunk})
                        
                        if metadata_info:
                            full_answer += metadata_info
                            await _stream_text_response(websocket, metadata_info, "text_chunk")
                        
                        answer = full_answer.strip()
                        if answer.startswith("```html"):
                            answer = answer[7:]
                        if answer.startswith("```"):
                            answer = answer[3:]
                        if answer.endswith("```"):
                            answer = answer[:-3]
                        answer = answer.strip().replace("\\n", "").replace("\n", "").replace("\r", "")
                        
                        manage_session_memory(session_id, user_message=query, bot_message=answer)
                        await _send_websocket_message(websocket, "complete", {"session_id": session_id})
                        await websocket.close()
                        return
            except Exception as e:
                print(f"[AUDIO][EXPLORE] Audio processing failed: {str(e)}")
                pass
        
        # Handle vector store for PDFs/Word docs
        if not dataset_path and filename:
            try:
                project_root = Path(__file__).resolve().parents[2]
                persist_dir = project_root / "chroma_store"
                if persist_dir.exists():
                    coll_name = _resolve_collection_name(persist_dir, email, filename)
                    if coll_name:
                        embeddings = OpenAIEmbeddings()
                        vectordb = Chroma(collection_name=coll_name, persist_directory=str(persist_dir), embedding_function=embeddings)
                        docs = _retrieve_docs(vectordb, query, k=7)
                        if docs:
                            context = "\n\n---\n\n".join([d.page_content for d in docs if d and d.page_content])
                            client = get_openai_client()
                            
                            await _send_websocket_message(websocket, "response_type", {"type": "text"})
                            
                            if _is_generic_summary_query(query):
                                system_prompt = (
                                    "You are a helpful assistant. Create a clear, self-contained summary of the document "
                                    "STRICTLY from the provided context. Include key sections, main points, and any important numbers/dates.\n"
                                    "Format your response in clean HTML with proper structure:\n"
                                    "- Use <h4> for main headings\n"
                                    "- Use <p> for paragraphs\n"
                                    "- Use <ul> and <li> for bullet lists\n"
                                    "- Use <strong> for emphasis\n"
                                    "Make it user-friendly and well-formatted."
                                )
                                user_payload = f"Context:\n{context}\n\nTask:\nProvide a concise, detailed, well-formatted HTML summary of this document."
                            else:
                                system_prompt = (
                                    "You are a helpful assistant answering questions strictly from the provided context.\n"
                                    "Format your response in clean HTML with proper structure:\n"
                                    "- Use <h4> for main headings\n"
                                    "- Use <p> for paragraphs\n"
                                    "- Use <ul> and <li> for lists\n"
                                    "- Use <strong> for emphasis\n"
                                    "If the answer is not present in the context, say you cannot find it in a friendly HTML format."
                                )
                                user_payload = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer using only the context, formatted in clean HTML."
                            
                            full_answer = ""
                            async for chunk in stream_llm_with_usage(
                                model=None,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_payload},
                                ],
                                temperature=0.0,
                                email=email
                            ):
                                full_answer += chunk
                                await _send_websocket_message(websocket, "text_chunk", {"chunk": chunk})
                            
                            answer = full_answer.strip()
                            if answer.startswith("```html"):
                                answer = answer[7:]
                            if answer.startswith("```"):
                                answer = answer[3:]
                            if answer.endswith("```"):
                                answer = answer[:-3]
                            answer = answer.strip().replace("\\n", "").replace("\n", "").replace("\r", "")
                            
                            manage_session_memory(session_id, user_message=query, bot_message=answer)
                            await _send_websocket_message(websocket, "complete", {"session_id": session_id})
                            await websocket.close()
                            return
            except Exception:
                pass
        
        # Handle tabular data analysis
        df = load_dataset(dataset_path)
        
        # Check for legal analysis
        legal_keywords = ['document', 'pdf', 'legal', 'contract', 'agreement', 'policy', 'regulation', 'law', 'analysis', 'summary', 'extract', 'content']
        is_legal_query = any(keyword in query.lower() for keyword in legal_keywords)
        is_legal_data = 'text_content' in df.columns and 'document_type' in df.columns
        if is_legal_query and is_legal_data:
            professional_analysis = analyze_legal_content(df, query)
            manage_session_memory(session_id, user_message=query, bot_message="Generated comprehensive legal analysis.")
            await _send_websocket_message(websocket, "response_type", {"type": "text"})
            await _stream_text_response(websocket, professional_analysis, "text_chunk")
            await _send_websocket_message(websocket, "complete", {"session_id": session_id})
            await websocket.close()
            return
        
        df = preprocess_dataframe_for_graphing(df)
        metadata_str = ", ".join(df.columns.tolist())
        
        # Check for report query
        is_report_query = any(keyword in query.lower() for keyword in
                              ['report', 'summary report', 'analysis report', 'detailed report',
                               'comprehensive report', 'summary_report', 'analysis_report', 'detailed_report',
                               'comprehensive_report'])
        if is_report_query:
            # For reports, we'll send the complete response (not ideal for streaming, but maintains compatibility)
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
            code = generate_data_code(prompt_eng, email=email)
            result = simulate_and_format_with_llm(code, df, email=email)
            cleaned_result = None
            try:
                import re
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
            await _send_websocket_message(websocket, "response_type", {"type": "report"})
            await _send_websocket_message(websocket, "report_data", {
                "report": cleaned_result.get("report"),
                "title": cleaned_result.get("title"),
                "description": cleaned_result.get("description")
            })
            await _send_websocket_message(websocket, "complete", {"session_id": session_id})
            await websocket.close()
            return
        
        # Handle simple queries
        agent = detect_agent(query)
        if agent in ["table", "text"]:
            can_handle_directly, structured = classify_query_complexity(query, list(df.columns))
            if can_handle_directly and structured is not None:
                intent, column = structured
                simple_resp = handle_simple_query(df, intent, column)
                manage_session_memory(session_id, user_message=query, bot_message=f"Processed simple query about {column}.")
                await _send_websocket_message(websocket, "response_type", {"type": simple_resp.get("type", "text")})
                if simple_resp.get("type") == "table":
                    table_payload = simple_resp.get("payload", [])
                    # Ensure table payload is a list of dicts
                    if not isinstance(table_payload, list):
                        table_payload = []
                    # Ensure each row is a dict
                    table_payload = [row if isinstance(row, dict) else {} for row in table_payload]
                    await _send_websocket_message(websocket, "table_data", {"rows": table_payload})
                elif simple_resp.get("type") == "text":
                    await _stream_text_response(websocket, str(simple_resp.get("payload", "")), "text_chunk")
                await _send_websocket_message(websocket, "complete", {"session_id": session_id})
                await websocket.close()
                return
        
        # Handle graph agent
        if agent == "graph":
            try:
                formatted_graph = handle_graph_agent(query, df)
                if formatted_graph is not None:
                    manage_session_memory(session_id, user_message=query, bot_message="Generated a graph visualization based on your request.")
                    await _send_websocket_message(websocket, "response_type", {"type": formatted_graph.get("type", "plotly")})
                    plotly_payload = formatted_graph.get("payload")
                    # Ensure plotly figure has proper structure
                    if isinstance(plotly_payload, dict):
                        if "data" not in plotly_payload:
                            plotly_payload["data"] = []
                        if "layout" not in plotly_payload:
                            plotly_payload["layout"] = {}
                    await _send_websocket_message(websocket, "plotly_data", {"figure": plotly_payload})
                    await _send_websocket_message(websocket, "complete", {"session_id": session_id})
                    await websocket.close()
                    return
            except Exception:
                pass
        
        # Default LLM analysis with streaming
        await _send_websocket_message(websocket, "response_type", {"type": "text"})
        
        # For streaming, we'll need to modify get_llm_analysis_explore to support streaming
        # For now, let's use a simplified streaming approach
        analysis_result = get_llm_analysis_explore(query, df, mode=agent, chat_history=chat_history_for_llm, email=email)
        
        response_type = analysis_result.get("type")
        payload = analysis_result.get("payload")
        
        if response_type == "conversational_answer":
            formatted_payload = format_text_response(str(payload)) if payload else payload
            await _stream_text_response(websocket, formatted_payload, "text_chunk")
            manage_session_memory(session_id, user_message=query, bot_message=str(payload) if payload else "I provided a conversational response.")
        elif response_type == "data_analysis_answer":
            explanation = payload.get("explanation") if isinstance(payload, dict) else ""
            code = payload.get("code") if isinstance(payload, dict) else None
            
            if explanation:
                await _stream_text_response(websocket, explanation, "explanation_chunk")
            
            if code:
                try:
                    from .explore_functions.agent_utils import safe_execute_pandas_code
                    exec_result = safe_execute_pandas_code(code, df)
                    formatted = format_result_for_response(exec_result)
                    
                    if formatted.get("type") == "plotly":
                        plotly_payload = formatted.get("payload")
                        # Ensure plotly figure has proper structure
                        if isinstance(plotly_payload, dict):
                            # Ensure data and layout exist
                            if "data" not in plotly_payload:
                                plotly_payload["data"] = []
                            if "layout" not in plotly_payload:
                                plotly_payload["layout"] = {}
                        await _send_websocket_message(websocket, "plotly_data", {"figure": plotly_payload})
                    elif formatted.get("type") == "table":
                        table_payload = formatted.get("payload", [])
                        # Ensure table payload is a list of dicts
                        if not isinstance(table_payload, list):
                            table_payload = []
                        # Ensure each row is a dict
                        table_payload = [row if isinstance(row, dict) else {} for row in table_payload]
                        await _send_websocket_message(websocket, "table_data", {"rows": table_payload})
                    else:
                        await _stream_text_response(websocket, str(formatted.get("payload", "")), "text_chunk")
                    
                    manage_session_memory(session_id, user_message=query, bot_message=explanation or "I have generated the data analysis you requested.")
                except Exception as e:
                    error_msg = f"<h4>Analysis Error</h4><p>There was an error executing the analysis: {str(e)}</p>"
                    await _stream_text_response(websocket, error_msg, "text_chunk")
            else:
                default_msg = explanation or "I understood your request but couldn't generate the right code."
                await _stream_text_response(websocket, default_msg, "text_chunk")
        else:
            error_msg = format_text_response("<h4>I'm sorry, I couldn't process that request.</h4><p>Please try rephrasing your question.</p>")
            await _stream_text_response(websocket, error_msg, "text_chunk")
        
        await _send_websocket_message(websocket, "complete", {"session_id": session_id})
        await websocket.close()
        
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await _send_websocket_message(websocket, "error", {"message": str(e)})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if email_token:
            reset_current_email(email_token)

