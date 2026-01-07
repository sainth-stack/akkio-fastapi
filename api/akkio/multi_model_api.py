"""
Multi-Model Training API
Handles training, progress tracking, and querying for multi-domain AI models
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional
import uuid
import os
import json
import pandas as pd
from datetime import datetime
import chromadb
from chromadb.config import Settings
import shutil
from pathlib import Path

# Import existing utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from database import PostgresDatabase
from .multi_model_agents import MultiModelAgentSystem

# Import document processing utilities
try:
    from .explore_functions.file_loaders import process_pdf, process_docx, process_txt
    from .explore_functions.vector_chat.vector_store import create_vector_store, query_vector_store
except:
    pass

router = APIRouter()

# Initialize database
db = PostgresDatabase()
db.create_connection(
    user=os.getenv('PGUSER', 'test_owner'),
    password=os.getenv('PGPASSWORD', 'tcWI7unQ6REA'),
    database=os.getenv('PGDATABASE', 'test'),
    host=os.getenv('PGHOST', 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech')
)
db.create_table()
db.create_training_tables()

# Initialize ChromaDB
CHROMA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'chroma_store')
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

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
            try:
                text = process_pdf(file_path)
            except:
                # Fallback PDF processing
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith(('.doc', '.docx')):
            try:
                text = process_docx(file_path)
            except:
                # Fallback DOCX processing
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported document format: {file_path}")
        
        # Split into chunks
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Create vector collection
        collection_name = f"multi_model_{session_id}_{file_name.split('.')[0]}"[:63]
        collection_name = collection_name.replace('-', '_').replace(' ', '_')
        
        try:
            collection = chroma_client.get_or_create_collection(collection_name)
        except:
            collection = chroma_client.create_collection(collection_name)
        
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
    """Process image files and store in vector database with descriptions"""
    try:
        # For now, store image path and basic metadata
        # In production, you'd use vision models to generate descriptions
        collection_name = f"multi_model_{session_id}_images"[:63]
        collection_name = collection_name.replace('-', '_').replace(' ', '_')
        
        try:
            collection = chroma_client.get_or_create_collection(collection_name)
        except:
            collection = chroma_client.create_collection(collection_name)
        
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
    files: List[tuple],  # [(file_path, file_name, file_type)]
):
    """Background task for training multi-model"""
    try:
        # Update progress: Starting
        db.update_multi_model_progress(session_id, 10, "Processing uploaded files", "training")
        
        # Process each file
        total_files = len(files)
        for idx, (file_path, file_name, file_type) in enumerate(files):
            progress = 10 + int((idx / total_files) * 60)  # 10-70% for file processing
            db.update_multi_model_progress(
                session_id, 
                progress, 
                f"Processing {file_name} ({idx+1}/{total_files})",
                "training"
            )
            
            # Process based on file type
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
            
            # Mark as processed
            db.mark_file_processed(session_id, file_name)
        
        # Update progress: Initializing agents
        db.update_multi_model_progress(session_id, 75, "Initializing AI agents", "training")
        
        # Initialize agent system (this validates the setup)
        agent_system = MultiModelAgentSystem(
            session_id=session_id,
            system_prompt=system_prompt,
            db_connection=db,
            chroma_client=chroma_client
        )
        
        # Update progress: Testing system
        db.update_multi_model_progress(session_id, 90, "Testing agent system", "training")
        
        # Run a test query to validate
        test_result = agent_system.query("What data sources are available?")
        
        # Complete training
        db.update_multi_model_progress(session_id, 100, "Training completed successfully", "completed")
        db.complete_multi_model_training(session_id)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        db.fail_multi_model_training(session_id, error_msg)
        db.update_multi_model_progress(session_id, 0, error_msg, "failed")


def write_file_sync(file_path: str, content: bytes):
    with open(file_path, 'wb') as f:
        f.write(content)


@router.post("/multi-model/train")
async def train_multi_model(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    system_prompt: str = Form(...),
    user_email: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    file_types: str = Form(...)  # JSON string of {filename: type}
):
    """
    Start multi-model training
    """
    try:
        # Parse file types
        file_types_dict = json.loads(file_types)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        

        # Create session in database - offload to threadpool
        await run_in_threadpool(
            db.create_multi_model_session,
            session_id=session_id,
            model_name=model_name,
            user_email=user_email,
            system_prompt=system_prompt
        )

        
        # Save uploaded files
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        saved_files = []
        if files:

            for file in files:
                file_path = os.path.join(session_dir, file.filename)
                
                # Read async
                content = await file.read()
                
                # Write sync - offload to threadpool
                await run_in_threadpool(write_file_sync, file_path, content)
                
                file_type = file_types_dict.get(file.filename, 'other')
                saved_files.append((file_path, file.filename, file_type))

            
        # Start background training
        background_tasks.add_task(
            train_multi_model_background,
            session_id,
            model_name,
            user_email,
            system_prompt,
            saved_files
        )
        print(f"Background task added. Returning.")
        
        return {
            'status': 'success',
            'session_id': session_id,
            'message': 'Training started in background',
            'files_count': len(files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-model/progress")
async def get_training_progress(session_id: str, user_email: str):
    """
    Get training progress for a session
    """
    try:
        session = db.get_multi_model_session(session_id=session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session['user_email'] != user_email:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-model/query")
async def query_multi_model(
    model_name: str = Form(...),
    user_email: str = Form(...),
    query: str = Form(...),
    messages: Optional[str] = Form(None)
):
    """
    Query a trained multi-model
    Returns response in same format as Explore API for consistency
    """
    try:
        # Get session for this model
        session = db.get_multi_model_session(user_email=user_email, model_name=model_name)
        
        if not session:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if session['status'] != 'completed':
            raise HTTPException(status_code=400, detail=f"Model is not ready. Status: {session['status']}")
        
        # Get session_id for response
        session_id = session['session_id']
        
        # Initialize agent system
        agent_system = MultiModelAgentSystem(
            session_id=session_id,
            system_prompt=session['system_prompt'],
            db_connection=db,
            chroma_client=chroma_client
        )
        
        # Execute query
        # Extract messages from form if present
        messages_list = []
        if messages:
            try:
                messages_list = json.loads(messages)
            except:
                pass
                
        result = agent_system.query(query, messages=messages_list)
        
        # Format response similar to Explore API
        # Answer comes as clean HTML/text, metadata comes separately
        return {
            'answer': result.get('answer', ''),  # Clean HTML/text answer
            'multi_model_metadata': {  # Metadata in separate key
                'sources': result.get('sources', []),
                'reasoning': result.get('reasoning', ''),
                'agents_used': result.get('agents_used', []),
                'model_name': model_name,
                'session_id': session_id,
                'validation_notes': result.get('validation_notes', ''),
                'report': result.get('report'),
                'generation_format': result.get('generation_format')
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-model/list")
async def list_multi_models(user_email: str):
    """
    List all multi-models for a user
    """
    try:
        models = db.get_user_multi_models(user_email)
        
        return {
            'status': 'success',
            'models': [
                {
                    'session_id': m['session_id'],
                    'model_name': m['model_name'],
                    'status': m['status'],
                    'progress': m['progress'],
                    'stage': m['stage'],
                    'created_at': m['created_at'].isoformat() if m['created_at'] else None,
                    'updated_at': m['updated_at'].isoformat() if m['updated_at'] else None,
                    'completed_at': m['completed_at'].isoformat() if m['completed_at'] else None
                }
                for m in models
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-model/files")
async def get_model_files(session_id: str, user_email: str):
    """
    Get all files for a multi-model
    """
    try:
        # Verify ownership
        session = db.get_multi_model_session(session_id=session_id)
        if not session or session['user_email'] != user_email:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        files = db.get_multi_model_files(session_id)
        
        return {
            'status': 'success',
            'files': [
                {
                    'file_name': f['file_name'],
                    'file_type': f['file_type'],
                    'processed': f['processed'],
                    'created_at': f['created_at'].isoformat() if f['created_at'] else None
                }
                for f in files
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-model/file-preview")
async def get_file_preview(session_id: str, file_name: str, user_email: str):
    """
    Get file preview data for a specific multi-model file
    Similar to /get_file but for multi-model files
    """
    try:
        # Verify ownership
        session = db.get_multi_model_session(session_id=session_id)
        if not session or session['user_email'] != user_email:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Get file info from database
        files = db.get_multi_model_files(session_id)
        file_info = next((f for f in files if f['file_name'] == file_name), None)
        
        if not file_info:
            raise HTTPException(status_code=404, detail=f"File '{file_name}' not found")
        
        file_type = file_info['file_type']
        storage_path = file_info.get('storage_path')
        
        # If storage_path is not in database, reconstruct it from session_id and file_name
        if not storage_path:
            session_dir = os.path.join(UPLOAD_DIR, session_id)
            potential_path = os.path.join(session_dir, file_name)
            if os.path.exists(potential_path):
                storage_path = potential_path
        
        response_data = {
            "name": file_name,
            "type": file_type,
            "filename": file_name,
            "processed": file_info.get('processed', False)
        }
        
        # Handle different file types
        if file_type == 'tabular':
            # Try reading from file path first, then fallback to database table
            df = None
            try:
                if storage_path and os.path.exists(storage_path):
                    if storage_path.endswith('.csv'):
                        df = pd.read_csv(storage_path)
                    elif storage_path.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(storage_path)
                
                # Fallback: Try to read from database table
                if df is None or df.empty:
                    table_name = file_info.get('db_table_name')
                    if table_name:
                        df = db.get_table_data(table_name)
                
                if df is not None and not df.empty:
                    # Limit rows for preview (first 100 rows)
                    preview_df = df.head(100)
                    
                    # Replace NaN, inf, -inf with None for JSON compatibility
                    import numpy as np
                    preview_df = preview_df.replace([np.nan, np.inf, -np.inf], None)
                    
                    response_data["preview_data"] = {
                        "columns": list(df.columns),
                        "rows": preview_df.to_dict(orient="records"),
                        "total_rows": len(df),
                        "preview_rows": len(preview_df)
                    }
                else:
                    response_data["error"] = "No data available for preview"
            except Exception as e:
                print(f"[ERROR] Failed to load tabular data: {e}")
                import traceback
                traceback.print_exc()
                response_data["error"] = f"Failed to load data: {str(e)}"
        
        elif file_type == 'document' and storage_path and os.path.exists(storage_path):
            # Read document file (PDF, Word, etc.)
            import base64
            try:
                with open(storage_path, 'rb') as f:
                    file_content = f.read()
                
                if storage_path.endswith('.pdf'):
                    # Return PDF as base64
                    pdf_base64 = base64.b64encode(file_content).decode('utf-8')
                    response_data["file_data"] = f"data:application/pdf;base64,{pdf_base64}"
                    
                    # Try to extract text content from vector store or database
                    try:
                        # Try to get from database table if it exists
                        table_name = file_info.get('db_table_name')
                        if table_name:
                            text_df = db.get_table_data(table_name)
                            if text_df is not None and 'text_content' in text_df.columns:
                                text_lines = text_df['text_content'].astype(str).tolist()
                                text_lines = [line.strip() for line in text_lines if line.strip()]
                                text_content = '\n\n'.join(text_lines)
                                response_data["text_content"] = text_content[:10000]
                    except Exception as e:
                        print(f"[ERROR] Failed to extract PDF text: {e}")
                
                elif storage_path.endswith(('.docx', '.doc')):
                    # Return Word doc as base64
                    docx_base64 = base64.b64encode(file_content).decode('utf-8')
                    response_data["file_data"] = f"data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{docx_base64}"
                    
                    # Try to extract text content
                    try:
                        table_name = file_info.get('db_table_name')
                        if table_name:
                            text_df = db.get_table_data(table_name)
                            if text_df is not None and 'text_content' in text_df.columns:
                                text_lines = text_df['text_content'].astype(str).tolist()
                                text_lines = [line.strip() for line in text_lines if line.strip()]
                                text_content = '\n\n'.join(text_lines)
                                response_data["text_content"] = text_content[:10000]
                    except Exception as e:
                        print(f"[ERROR] Failed to extract Word text: {e}")
                
                elif storage_path.endswith('.txt'):
                    # Return text file content
                    text_content = file_content.decode('utf-8')
                    response_data["text_content"] = text_content[:10000]
                
            except Exception as e:
                print(f"[ERROR] Failed to read document: {e}")
                response_data["error"] = f"Failed to load document: {str(e)}"
        
        elif file_type == 'image' and storage_path and os.path.exists(storage_path):
            # Read image file
            import base64
            try:
                with open(storage_path, 'rb') as f:
                    file_content = f.read()
                
                # Determine mime type
                from mimetypes import guess_type
                mime_type, _ = guess_type(storage_path)
                if not mime_type:
                    mime_type = 'image/png'
                
                image_base64 = base64.b64encode(file_content).decode('utf-8')
                response_data["file_data"] = f"data:{mime_type};base64,{image_base64}"
                
            except Exception as e:
                print(f"[ERROR] Failed to read image: {e}")
                response_data["error"] = f"Failed to load image: {str(e)}"
        
        else:
            response_data["error"] = "File not found or unsupported file type"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/multi-model/delete")
async def delete_multi_model(session_id: str, user_email: str):
    """
    Delete a multi-model and all associated data
    """
    try:
        # Verify ownership
        session = db.get_multi_model_session(session_id=session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if session['user_email'] != user_email:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Get all files to clean up vector collections and database tables
        files = db.get_multi_model_files(session_id)
        
        # Delete vector collections
        for file_info in files:
            if file_info.get('vector_collection_id'):
                try:
                    chroma_client.delete_collection(file_info['vector_collection_id'])
                except Exception as e:
                    print(f"Error deleting vector collection {file_info['vector_collection_id']}: {e}")
            
            # Delete database tables for tabular data
            if file_info.get('db_table_name'):
                try:
                    db.delete_table(file_info['db_table_name'])
                except Exception as e:
                    print(f"Error deleting table {file_info['db_table_name']}: {e}")
        
        # Delete uploaded files
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if os.path.exists(session_dir):
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                print(f"Error deleting upload directory {session_dir}: {e}")
        
        # Delete session and files from database
        db.delete_multi_model_session(session_id)
        
        return {
            'status': 'success',
            'message': f'Multi-model "{session.get("model_name")}" deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

