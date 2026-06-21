"""
Multi-Model Training API
Handles training, progress tracking, and querying for multi-domain AI models
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from typing import List, Optional, Callable

from api.auth.dependencies import CurrentUser
from api.auth.request_auth import resolve_user, resolve_user_flexible, user_email_from
from api.auth.ws_auth import authenticate_websocket
import uuid
import os
import json
import pandas as pd
from datetime import datetime
import shutil
import requests
import base64
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from db import PostgresDatabase
from .multi_model_agents import MultiModelAgentSystem
from .usage_tracking import set_current_email, reset_current_email
from .usage_tracking import record_llm_usage
from .explore_functions.file_loaders import process_pdf, process_docx, process_txt

try:
    from llm_config import get_llm_config, get_api_key, get_model_name
    LLM_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import llm_config: {e}")
    LLM_CONFIG_AVAILABLE = False

router = APIRouter()

db = PostgresDatabase()

_chroma_client = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        chroma_path = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_store")
        os.makedirs(chroma_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'uploads_multi_model')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_tabular_file(file_path: str, file_name: str, user_email: str, session_id: str) -> dict:
    """Process CSV/Excel files and store in database"""
    try:
        # Read file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular format: {file_path}")
        
        print(f"Fetched tabular data from {file_name}: {len(df)} rows, {len(df.columns)} columns")
        
        # Clean table name
        table_name = f"{session_id}_{file_name.split('.')[0]}"
        table_name = table_name.replace('-', '_').replace(' ', '_')[:50]
        
        # Store in database
        db.insert_or_update(
            email=user_email,
            data=df,
            tb_name=table_name,
            data_type='tabular',
            data_subtype='multi_model'
        )
        
        return {
            'status': 'success',
            'db_table_name': table_name,
            'rows': len(df),
            'columns': len(df.columns)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def process_document_file(file_path: str, file_name: str, session_id: str) -> dict:
    """Process PDF/DOC files and store in vector database"""
    try:
        # Extract text based on file type
        if file_path.endswith('.pdf'):
            text = process_pdf(file_path)
        elif file_path.endswith(('.doc', '.docx')):
            text = process_docx(file_path)
        elif file_path.endswith('.txt'):
            text = process_txt(file_path)
        else:
            raise ValueError(f"Unsupported document format: {file_path}")
        
        print(f"Fetched document text from {file_name}: {len(text)} characters")
        
        # Split into chunks
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Create vector collection
        clean_filename = file_name.split('.')[0]
        clean_filename = clean_filename.replace('-', '_').replace(' ', '_')
        clean_filename = clean_filename.rstrip('_').lstrip('_')
        
        collection_name = f"multi_model_{session_id}_{clean_filename}"[:63]
        collection_name = collection_name.strip('_')
        
        try:
            collection = get_chroma_client().get_or_create_collection(collection_name)
        except:
            collection = get_chroma_client().create_collection(collection_name)
        
        # Add documents to collection
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{'source': file_name, 'chunk_id': i} for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        return {
            'status': 'success',
            'vector_collection_id': collection_name,
            'chunks': len(chunks),
            'text_length': len(text)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def process_image_file(file_path: str, file_name: str, session_id: str) -> dict:
    """Process image files and store in vector database"""
    try:
        collection_name = f"multi_model_{session_id}_images"[:63]
        collection_name = collection_name.replace('-', '_').replace(' ', '_')
        
        try:
            collection = get_chroma_client().get_or_create_collection(collection_name)
        except:
            collection = get_chroma_client().create_collection(collection_name)
            
        print(f"Fetched image data from {file_name}")
        
        # Store image metadata
        collection.add(
            documents=[f"Image file: {file_name}"],
            ids=[f"{session_id}_{file_name}"],
            metadatas=[{
                'source': file_name,
                'type': 'image',
                'path': file_path
            }]
        )
        
        return {
            'status': 'success',
            'vector_collection_id': collection_name,
            'storage_path': file_path
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def train_multi_model_background(
    session_id: str,
    model_name: str,
    user_email: str,
    system_prompt: str,
    files: List[tuple],
    temperature: float = 0.0,
    workflow: str = None,
    output_format: str = None,
    run_smoke_test: bool = True,
):
    """Background task for training multi-model"""
    try:
        # Update progress: Starting
        db.update_multi_model_progress(session_id, 10, "Processing uploaded files", "training")
        
        # Process each file
        total_files = len(files)
        for idx, (file_path, file_name, file_type) in enumerate(files):
            progress = 10 + int((idx / total_files) * 60)
            db.update_multi_model_progress(
                session_id, 
                progress, 
                f"Processing {file_name} ({idx+1}/{total_files})",
                "training"
            )
            
            try:
                if file_type == 'tabular':
                    result = process_tabular_file(file_path, file_name, user_email, session_id)
                    if result['status'] == 'success':
                        db.add_multi_model_file(
                            session_id=session_id,
                            file_name=file_name,
                            file_type=file_type,
                            db_table_name=result['db_table_name'],
                            storage_path=file_path
                        )
                    else:
                        db.add_multi_model_file(
                            session_id=session_id,
                            file_name=file_name,
                            file_type=file_type,
                            storage_path=file_path,
                            error_message=result.get('message', 'Processing failed')
                        )
                
                elif file_type == 'document':
                    result = process_document_file(file_path, file_name, session_id)
                    if result['status'] == 'success':
                        db.add_multi_model_file(
                            session_id=session_id,
                            file_name=file_name,
                            file_type=file_type,
                            vector_collection_id=result['vector_collection_id'],
                            storage_path=file_path
                        )
                    else:
                        db.add_multi_model_file(
                            session_id=session_id,
                            file_name=file_name,
                            file_type=file_type,
                            storage_path=file_path,
                            error_message=result.get('message', 'Processing failed')
                        )
                
                elif file_type == 'image':
                    result = process_image_file(file_path, file_name, session_id)
                    if result['status'] == 'success':
                        db.add_multi_model_file(
                            session_id=session_id,
                            file_name=file_name,
                            file_type=file_type,
                            vector_collection_id=result.get('vector_collection_id'),
                            storage_path=result.get('storage_path')
                        )
                    else:
                         db.add_multi_model_file(
                            session_id=session_id,
                            file_name=file_name,
                            file_type=file_type,
                            storage_path=file_path,
                            error_message=result.get('message', 'Processing failed')
                        )
                else: 
                    # Audio/Other
                    db.add_multi_model_file(
                        session_id=session_id,
                        file_name=file_name,
                        file_type=file_type,
                        storage_path=file_path,
                        processed=True
                    )

                if 'result' in locals() and result.get('status') == 'success':
                    db.mark_file_processed(session_id, file_name)
                elif file_type not in ['tabular', 'document', 'image']:
                     db.mark_file_processed(session_id, file_name)

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                db.add_multi_model_file(
                    session_id=session_id,
                    file_name=file_name,
                    file_type=file_type,
                    storage_path=file_path,
                    error_message=str(e)
                )
        
        if not files:
            db.update_multi_model_progress(session_id, 100, "Training completed (no files)", "completed")
            db.complete_multi_model_training(session_id)
            return

        db.update_multi_model_progress(session_id, 75, "Initializing AI agents", "processing")
        
        # Initialize agent system
        agent_system = MultiModelAgentSystem(
            session_id=session_id,
            system_prompt=system_prompt,
            db_connection=db,
            chroma_client=get_chroma_client(),
            temperature=temperature,
            workflow=workflow,
            output_format=output_format,
        )
        
        if run_smoke_test:
            db.update_multi_model_progress(session_id, 90, "Testing agent system", "training")
            agent_system.query("What data sources are available?")
        
        db.update_multi_model_progress(session_id, 100, "Training completed successfully", "completed")
        db.complete_multi_model_training(session_id)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        db.fail_multi_model_training(session_id, error_msg)
        db.update_multi_model_progress(session_id, 0, error_msg, "failed")


def write_file_sync(file_path: str, content: bytes):
    with open(file_path, 'wb') as f:
        f.write(content)


@router.post("/multi-model/create")
async def create_multi_model(
    request: Request,
    model_name: str = Form(...),
    system_prompt: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    file_types: str = Form(...),
    temperature: float = Form(0.0),
    workflow: str = Form(None),
    output_format: str = Form(None),
):
    current = await resolve_user_flexible(request)
    user_email = user_email_from(current)
    email_token = set_current_email(user_email)
    try:
        has_files = bool(files)
        final_workflow = workflow
        final_output_format = output_format

        # Generate workflow and output_format using LLM if not provided
        if not final_workflow or not final_output_format:
            gen_result = await _generate_agent_config(model_name, system_prompt)
            if gen_result.get("status") == "success":
                if not final_workflow:
                    final_workflow = gen_result.get("workflow") or gen_result.get("background")
                if not final_output_format:
                    final_output_format = gen_result.get("output_format")

        file_types_dict = json.loads(file_types)
        session_id = str(uuid.uuid4())

        await run_in_threadpool(
            db.create_multi_model_session,
            session_id=session_id,
            model_name=model_name,
            user_email=user_email,
            system_prompt=system_prompt,
            temperature=temperature,
            workflow=final_workflow,
            output_format=final_output_format,
        )

        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        saved_files = []
        if has_files:
            for file in files:
                file_path = os.path.join(session_dir, file.filename)
                content = await file.read()
                await run_in_threadpool(write_file_sync, file_path, content)
                file_type = file_types_dict.get(file.filename, "other")
                saved_files.append((file_path, file.filename, file_type))

        if not has_files:
            await run_in_threadpool(db.update_multi_model_progress, session_id, 100, "Training completed (no files)", "completed")
            await run_in_threadpool(db.complete_multi_model_training, session_id)
        else:
            await run_in_threadpool(
                train_multi_model_background,
                session_id,
                model_name,
                user_email,
                system_prompt,
                saved_files,
                temperature,
                final_workflow,
                final_output_format,
                False,
            )

        session = db.get_multi_model_session(session_id=session_id)
        return {
            "status": "success",
            "session_id": session_id,
            "model_name": model_name,
            "message": "Model created successfully",
            "session": session,
            "files_count": len(files),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_email(email_token)


@router.get("/multi-model/progress")
async def get_training_progress(
    session_id: str,
    current: CurrentUser = Depends(resolve_user),
):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=session_id)
        if not session: raise HTTPException(status_code=404, detail="Session not found")
        if session['user_email'] != user_email: raise HTTPException(status_code=403, detail="Unauthorized")
        return {
            'session_id': session['session_id'],
            'model_name': session['model_name'],
            'status': session['status'],
            'progress': session['progress'],
            'stage': session['stage'],
            'error_message': session['error_message'],
            'created_at': session['created_at'].isoformat() if session['created_at'] else None,
            'updated_at': session['updated_at'].isoformat() if session['updated_at'] else None
        }
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


class GenerateBackgroundRequest(BaseModel):
    model_name: str
    system_prompt: str


async def _generate_agent_config(model_name: str, system_prompt: str):
    try:
        # Get LLM configuration
        if LLM_CONFIG_AVAILABLE:
            config = get_llm_config(user_email=None)
            llm_model = config["model"]
            llm_api_key = config.get("api_key")
            llm_provider = config.get("provider", "openai")
        else:
            try:
                from llm_config import DEFAULT_MODEL
                llm_model = DEFAULT_MODEL
            except Exception:
                llm_model = "gpt-4o-mini"
            llm_api_key = os.getenv("OPENAI_API_KEY")
            llm_provider = "openai"
        
        # Mask API key for security (show only last 4 characters)
        masked_api_key = None
        if llm_api_key:
            if len(llm_api_key) > 4:
                masked_api_key = f"{'*' * (len(llm_api_key) - 4)}{llm_api_key[-4:]}"
            else:
                masked_api_key = "*" * len(llm_api_key)
        
        # Use ChatOpenAI directly
        model = ChatOpenAI(
            model=llm_model,
            openai_api_key=llm_api_key,
            temperature=0.7
        )
        
        prompt = f"""You are an AI assistant helping to configure a specialized AI agent.

Agent Name: {model_name}
System Prompt/Purpose: {system_prompt}

Please generate comprehensive configuration for this agent:

1. **background**: Write 2-4 paragraphs describing the agent's context, domain expertise, capabilities, and role. Include:
   - What domain or specialty this agent focuses on
   - What types of tasks or questions it handles
   - What knowledge or expertise it brings
   - How it approaches problem-solving

2. **output_format**: Specify how the agent should structure and format its responses. Include:
   - The style of communication (formal, conversational, technical, etc.)
   - How to organize information (sections, bullet points, step-by-step, etc.)
   - Any specific formatting requirements
   - How to handle different types of queries (analysis, recommendations, explanations, etc.)

Return ONLY a valid JSON object with these two keys: "background" and "output_format".
Make the content detailed, professional, and tailored to the agent's specific purpose.

Example format:
{{
  "background": "Detailed background text here...",
  "output_format": "Detailed output format instructions here..."
}}
"""
        response = await model.ainvoke([HumanMessage(content=prompt)])
        # Note: record_llm_usage is not called here because LangChain response format differs from OpenAI
        # If usage tracking is needed, it should be implemented separately for LangChain
        
        content = response.content.strip()
        # Remove markdown code fences if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
            
        try: 
            result = json.loads(content)
        except Exception as parse_error:
            print(f"JSON parse error: {parse_error}")
            # Fallback with generic but useful defaults
            result = {
                "background": f"This is a specialized AI agent named '{model_name}' designed to help with: {system_prompt}. It provides expert assistance by analyzing queries, accessing relevant knowledge, and delivering accurate, helpful responses.",
                "output_format": "Responses should be clear, well-structured, and professional. Use paragraphs for explanations, bullet points for lists, and step-by-step formatting for procedures. Adapt the tone and detail level based on the query complexity."
            }
            
        return {
            "status": "success",
            "background": result.get("background", ""),
            "workflow": result.get("background", ""),
            "output_format": result.get("output_format", ""),
            "model_name": llm_model,
            "api_key": masked_api_key,
            "provider": llm_provider
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error generating agent config: {e}")
        print(f"Full traceback:\n{error_trace}")
        
        # Try to get LLM config even in error case
        try:
            if LLM_CONFIG_AVAILABLE:
                config = get_llm_config(user_email=None)
                llm_model = config["model"]
                llm_api_key = config.get("api_key")
                llm_provider = config.get("provider", "openai")
            else:
                try:
                    from llm_config import DEFAULT_MODEL
                    llm_model = DEFAULT_MODEL
                except Exception:
                    llm_model = "gpt-4o-mini"
                llm_api_key = os.getenv("OPENAI_API_KEY")
                llm_provider = "openai"
            
            # Mask API key for security
            masked_api_key = None
            if llm_api_key:
                if len(llm_api_key) > 4:
                    masked_api_key = f"{'*' * (len(llm_api_key) - 4)}{llm_api_key[-4:]}"
                else:
                    masked_api_key = "*" * len(llm_api_key)
        except:
            llm_model = "unknown"
            masked_api_key = "unknown"
            llm_provider = "unknown"
        
        # Return fallback values instead of empty strings
        return {
            "status": "error", 
            "background": f"AI agent specialized in: {system_prompt}",
            "workflow": f"AI agent specialized in: {system_prompt}",
            "output_format": "Provide clear, well-structured responses appropriate to the query.",
            "error_detail": str(e),
            "error_trace": error_trace,
            "model_name": llm_model,
            "api_key": masked_api_key,
            "provider": llm_provider
        }


@router.post("/multi-model/generate_background")
async def generate_background(
    request: GenerateBackgroundRequest,
    current: CurrentUser = Depends(resolve_user),
):
    try:
        result = await _generate_agent_config(request.model_name, request.system_prompt)
        if result['status'] == 'error': 
            error_msg = f"Failed to generate content: {result.get('error_detail', 'Unknown error')}"
            print(f"Error in generate_background: {error_msg}")
            if 'error_trace' in result:
                print(f"Error trace: {result['error_trace']}")
            raise HTTPException(status_code=500, detail=error_msg)
        return result
    except HTTPException:
        raise
    except Exception as e: 
        import traceback
        print(f"Unexpected error in generate_background: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


class UpdateMultiModelConfigRequest(BaseModel):
    session_id: str
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    workflow: Optional[str] = None
    output_format: Optional[str] = None


@router.post("/multi-model/update-config")
async def update_multi_model_config(
    request: UpdateMultiModelConfigRequest,
    current: CurrentUser = Depends(resolve_user),
):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=request.session_id)
        if not session: raise HTTPException(status_code=404, detail="Session not found")
        if session["user_email"] != user_email:
            raise HTTPException(status_code=403, detail="Unauthorized")

        updated = await run_in_threadpool(
            db.update_multi_model_session_config,
            request.session_id,
            user_email,
            request.model_name,
            request.system_prompt,
            request.temperature,
            request.workflow,
            request.output_format,
        )
        if not updated: raise HTTPException(status_code=400, detail="No changes applied")
        return {"status": "success", "session": db.get_multi_model_session(session_id=request.session_id)}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))


async def _send_websocket_message(websocket: WebSocket, message_type: str, data: dict):
    try:
        serialized_data = jsonable_encoder(data)
        message = {"type": message_type}
        message.update(serialized_data)
        await websocket.send_json(message)
    except Exception as e: print(f"Error sending WebSocket message: {e}")


@router.websocket("/multi-model/query/ws")
async def query_multi_model_websocket(websocket: WebSocket):
    await websocket.accept()
    email_token = None
    try:
        data = await websocket.receive_json()
        current, data = await authenticate_websocket(websocket, first_message=data)
        user_email = user_email_from(current)
        model_name = data.get("model_name", "")
        query = data.get("query", "")
        messages_list = data.get("messages", [])

        if not query or not model_name:
            await _send_websocket_message(websocket, "error", {"message": "Missing params"})
            await websocket.close()
            return
        
        email_token = set_current_email(user_email)

        if not os.getenv("OPENAI_API_KEY"):
            await _send_websocket_message(
                websocket,
                "error",
                {"message": "OpenAI API key is not configured. Set OPENAI_API_KEY in .env and restart the server."},
            )
            await websocket.close()
            return

        session = db.get_multi_model_session(user_email=user_email, model_name=model_name)
        if not session:
            await _send_websocket_message(websocket, "error", {"message": "Model not found"})
            await websocket.close()
            return
        
        session_id = session['session_id']
        await _send_websocket_message(websocket, "session_id", {"session_id": session_id})
        
        agent_system = MultiModelAgentSystem(
            session_id=session_id,
            system_prompt=session['system_prompt'],
            db_connection=db,
            chroma_client=get_chroma_client(),
            temperature=session.get('temperature', 0.0),
            workflow=session.get('workflow'),
            output_format=session.get('output_format')
        )
        agent_system.email = user_email
        
        async def stream_callback(event_type: str, data: dict):
            await _send_websocket_message(websocket, event_type, data)
        
        result = await agent_system.query_async(query, messages=messages_list, stream_callback=stream_callback)
        
        await _send_websocket_message(websocket, "metadata", {
            'sources': result.get('sources', []),
            'reasoning': result.get('reasoning', ''),
            'agents_used': result.get('agents_used', []),
            'model_name': model_name,
            'session_id': session_id
        })
        await _send_websocket_message(websocket, "complete", {"session_id": session_id})
        await websocket.close()
        
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await _send_websocket_message(websocket, "error", {"message": str(e)})
        except: pass
        try: await websocket.close()
        except: pass
    finally:
        if email_token: reset_current_email(email_token)


@router.get("/multi-model/list")
async def list_multi_models(current: CurrentUser = Depends(resolve_user)):
    try:
        models = db.get_user_multi_models(user_email_from(current))
        return {'status': 'success', 'models': models}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/multi-model/files")
async def get_model_files(session_id: str, current: CurrentUser = Depends(resolve_user)):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=session_id)
        if not session or session['user_email'] != user_email: raise HTTPException(status_code=403, detail="Unauthorized")
        files = db.get_multi_model_files(session_id)
        return {'status': 'success', 'files': files}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.delete("/multi-model/delete")
async def delete_multi_model(session_id: str, current: CurrentUser = Depends(resolve_user)):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=session_id)
        if not session or session['user_email'] != user_email: raise HTTPException(status_code=403, detail="Unauthorized")
        
        files = db.get_multi_model_files(session_id)
        for f in files:
            if f.get('vector_collection_id'):
                try: get_chroma_client().delete_collection(f['vector_collection_id'])
                except: pass
            if f.get('db_table_name'):
                try: db.delete_table(f['db_table_name'])
                except: pass
        
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if os.path.exists(session_dir): shutil.rmtree(session_dir)
        
        db.delete_multi_model_session(session_id)
        return {'status': 'success', 'message': 'Deleted successfully'}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# Restored Utility Endpoints
@router.get("/multi-model/file-preview")
async def get_file_preview(
    session_id: str,
    file_name: str,
    current: CurrentUser = Depends(resolve_user),
):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=session_id)
        if not session or session['user_email'] != user_email: raise HTTPException(status_code=403, detail="Unauthorized")
        files = db.get_multi_model_files(session_id)
        file_info = next((f for f in files if f['file_name'] == file_name), None)
        if not file_info: raise HTTPException(status_code=404, detail="File not found")
        
        file_type = file_info['file_type']
        storage_path = file_info.get('storage_path')
        if not storage_path:
             storage_path = os.path.join(UPLOAD_DIR, session_id, file_name)
        
        response_data = {"name": file_name, "type": file_type, "processed": file_info.get('processed', False)}
        
        if file_type == 'tabular':
             df = None
             if storage_path and os.path.exists(storage_path):
                 try: 
                    if storage_path.endswith('.csv'): df = pd.read_csv(storage_path)
                    else: df = pd.read_excel(storage_path)
                 except: pass
             if (df is None or df.empty) and file_info.get('db_table_name'):
                 df = db.get_table_data(file_info['db_table_name'])
             
             if df is not None:
                 preview = df.head(100)
                 import numpy as np
                 preview = preview.replace([np.nan, np.inf, -np.inf], None)
                 response_data["preview_data"] = {
                     "columns": list(df.columns),
                     "rows": preview.to_dict(orient="records")
                 }
        
        elif file_type == 'document' and storage_path and os.path.exists(storage_path):
            with open(storage_path, 'rb') as f:
                content = f.read()
                response_data["file_data"] = f"data:application/pdf;base64,{base64.b64encode(content).decode('utf-8')}" if storage_path.endswith('.pdf') else ""
        
        elif file_type == 'image' and storage_path and os.path.exists(storage_path):
             with open(storage_path, 'rb') as f:
                response_data["file_data"] = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                
        return response_data
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

class PublishMultiModelRequest(BaseModel):
    session_id: str
    published: bool = True

@router.post("/multi-model/publish")
async def publish_multi_model(body: PublishMultiModelRequest, current: CurrentUser = Depends(resolve_user)):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=body.session_id)
        if not session: raise HTTPException(status_code=404, detail="Session not found")
        if session["user_email"] != user_email: raise HTTPException(status_code=403, detail="Unauthorized")
        
        public_id = session.get("public_id")
        if body.published and not public_id: public_id = uuid.uuid4().hex[:12]
        
        db.set_multi_model_published(body.session_id, user_email, body.published, public_id if body.published else None)
        return {"status": "success", "session": db.get_multi_model_session(session_id=body.session_id)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/multi-model/share-info")
async def get_share_info(session_id: str, request: Request, current: CurrentUser = Depends(resolve_user)):
    user_email = user_email_from(current)
    try:
        session = db.get_multi_model_session(session_id=session_id)
        if not session or session['user_email'] != user_email:
            raise HTTPException(status_code=403, detail="Unauthorized")
        if not session.get("published"):
            return {"status": "success", "published": False}
        base = str(request.base_url).rstrip('/')
        public_id = session.get('public_id')
        return {
            "status": "success",
            "published": True,
            "share_url": f"{base}/chatbot/{public_id}",
            "public_id": public_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PublicQueryRequest(BaseModel):
    query: str
    messages: Optional[list] = None

@router.get("/public/chatbot/{public_id}/config")
async def public_chatbot_config(public_id: str):
    try:
        session = db.get_multi_model_session_by_public_id(public_id)
        if not session or not session.get("published"): raise HTTPException(status_code=404)
        return {"status": "success", "model_name": session.get("model_name")}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.post("/public/chatbot/{public_id}/query")
async def public_chatbot_query(public_id: str, body: PublicQueryRequest):
    email_token = set_current_email(None)
    try:
        session = db.get_multi_model_session_by_public_id(public_id)
        if not session or not session.get("published"): raise HTTPException(status_code=404)
        
        agent_system = MultiModelAgentSystem(
            session_id=session['session_id'],
            system_prompt=session['system_prompt'],
            db_connection=db,
            chroma_client=get_chroma_client(),
            temperature=session.get('temperature', 0.0),
            workflow=session.get('workflow'),
            output_format=session.get('output_format')
        )
        
        result = await agent_system.query_async(body.query, messages=body.messages)
        return {
            "answer": result.get("answer", ""),
            "multi_model_metadata": {}
        }
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    finally: reset_current_email(email_token)
