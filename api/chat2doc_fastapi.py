import os
import faiss
import pickle
import numpy as np
import tempfile
import shutil
import hashlib
from fastapi import FastAPI, File, UploadFile, Form,APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
import markdown

load_dotenv()

chat2doc = APIRouter()

# Config
VECTOR_STORE_DIR = "./vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS index setup
embedding_dim = 1536
index = None
doc_store = []

# ---------------------------
# Utility Functions
# ---------------------------
def file_hash(file_path):
    """Generate a hash for the file content to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_cache_paths(file_hash_str):
    """Return paths for cached FAISS index and doc store for a file."""
    index_path = os.path.join(VECTOR_STORE_DIR, f"{file_hash_str}_index.bin")
    docstore_path = os.path.join(VECTOR_STORE_DIR, f"{file_hash_str}_doc.pkl")
    return index_path, docstore_path

def load_cached_embeddings(file_hash_str):
    """Load FAISS index & doc store for a file if cache exists."""
    index_path, docstore_path = get_cache_paths(file_hash_str)
    if os.path.exists(index_path) and os.path.exists(docstore_path):
        file_index = faiss.read_index(index_path)
        with open(docstore_path, "rb") as f:
            file_doc_store = pickle.load(f)
        return file_index, file_doc_store
    return None, None

def save_embeddings(file_hash_str, file_index, file_doc_store):
    """Save FAISS index & doc store for a file."""
    index_path, docstore_path = get_cache_paths(file_hash_str)
    faiss.write_index(file_index, index_path)
    with open(docstore_path, "wb") as f:
        pickle.dump(file_doc_store, f)

# ---------------------------
# Document Chunking
# ---------------------------
def chunk_document(file_path, chunk_size=1500):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    text = text.strip().replace("\n", " ")
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

# ---------------------------
# OpenAI Embeddings
# ---------------------------
def get_openai_embedding(text: str):
    """Get normalized OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector

# ---------------------------
# Main Processing
# ---------------------------
def process_file(file_path, chunk_size=1500):
    """Load cached embeddings if available, else create and store."""
    fhash = file_hash(file_path)
    cached_index, cached_docs = load_cached_embeddings(fhash)
    
    if cached_index is not None and cached_docs is not None:
        return cached_index, cached_docs  # Return cached
    
    # Else process new file
    chunks = chunk_document(file_path, chunk_size)
    file_index = faiss.IndexFlatIP(embedding_dim)
    embeddings = [get_openai_embedding(chunk) for chunk in chunks]
    embeddings = np.vstack(embeddings)
    file_index.add(embeddings)
    file_doc_store = chunks
    
    save_embeddings(fhash, file_index, file_doc_store)
    return file_index, file_doc_store

# ---------------------------
# Query Function
# ---------------------------
def query_all_indexes(user_prompt, indexes_with_docs, top_k=5):
    """Search across multiple FAISS indexes from different files."""
    query_vec = get_openai_embedding(user_prompt).reshape(1, -1)
    all_results = []
    for file_index, file_doc_store in indexes_with_docs:
        D, I = file_index.search(query_vec, top_k)
        for idx, score in zip(I[0], D[0]):
            if idx != -1:
                all_results.append((file_doc_store[idx], score))
    # Sort by score and take top_k
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:top_k]

# ---------------------------
# API Endpoint
# ---------------------------
@chat2doc.post("/chat_with_documents")
async def chat_with_documents(
    files: list[UploadFile] = File(...),
    user_prompt: str = Form(...)
):
    try:
        indexes_with_docs = []

        # Process each file (use cache if available)
        for file in files:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            file_index, file_doc_store = process_file(file_path, chunk_size=1500)
            indexes_with_docs.append((file_index, file_doc_store))

            shutil.rmtree(os.path.dirname(file_path))  # Clean temp file

        # Query combined indexes
        results = query_all_indexes(user_prompt, indexes_with_docs, top_k=5)
        context_docs = "\n\n".join([doc for doc, _ in results])

        # Improved prompt
        prompt = f"""
            You are a knowledgeable assistant. 
            Answer the user's question using only the provided context from uploaded documents. 
            If the answer is not explicitly in the context, make the best possible inference based on related information, 
            but never invent false facts.

            Context:
            {context_docs}

            Question:
            {user_prompt}

            Answer:
"""

        # Get LLM response
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers based on document content."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = chat_response.choices[0].message.content.strip()
        print(f"Generated answer: {answer}")
        print(f"Context used: {context_docs}")
        return JSONResponse(content={
            "response": markdown_to_html(answer)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def markdown_to_html(md_text):
    html_text = markdown.markdown(md_text)
    return html_text

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chat2doc:app", host="127.0.0.1", port=8001, reload=True)
