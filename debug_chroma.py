import chromadb
from pathlib import Path
import os

def debug_chroma():
    project_root = Path(os.getcwd())
    persist_dir = project_root / "chroma_store"
    print(f"Checking chroma store at: {persist_dir}")
    
    if not persist_dir.exists():
        print("Chroma store not found!")
        return

    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        for col in collections:
            print(f" - Name: {col.name}, Count: {col.count()}")
            
        # Simulate resolution
        filename = "LEGAL AI for UAE Legislation"
        email = "admin@gmail.com"
        
        def _safe(s: str) -> str:
            return "".join(ch if ch.isalnum() else "_" for ch in (s or ""))

        safe_name = _safe(Path(filename).stem or filename)
        safe_email = _safe(email) if email else None
        exact = f"{safe_email}__{safe_name}" if safe_email else None
        
        print(f"\nResolving for filename='{filename}', email='{email}'")
        print(f"safe_name: {safe_name}")
        print(f"safe_email: {safe_email}")
        print(f"exact: {exact}")
        
        scored = []
        for coll in collections:
            name = coll.name
            score = -1
            if exact and name == exact:
                score = 400
            elif safe_email and name.startswith(f"{safe_email}__{safe_name}"):
                score = 300 + len(name)
            elif name.endswith(f"__{safe_name}"):
                score = 200 + len(name)
            elif f"__{safe_name}__" in name:
                score = 100 + len(name)
            
            if score >= 0:
                scored.append((score, name))
                
        scored.sort(reverse=True)
        print(f"Candidates: {scored}")
        
        if scored:
            best_coll_name = scored[0][1]
            print(f"Selected collection: {best_coll_name}")
            
            # Try to peek at content
            coll = client.get_collection(best_coll_name)
            peek = coll.peek(limit=3)
            print(f"\nPeek at collection '{best_coll_name}':")
            if peek and 'documents' in peek:
                for i, doc in enumerate(peek['documents']):
                    print(f"Doc {i}: {doc[:100]}...")
                    if 'metadatas' in peek and peek['metadatas']:
                         print(f"Metadata: {peek['metadatas'][i]}")
        else:
            print("No matching collection found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_chroma()
