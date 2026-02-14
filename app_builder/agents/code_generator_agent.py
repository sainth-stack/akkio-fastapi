from typing import Dict
from ..schemas.architecture import ArchitectureDecision

def code_generator_agent(architecture: ArchitectureDecision) -> Dict[str, str]:
    """
    Generates the code for the project based on the architecture.
    Keeps file count minimal: single App.js for frontend (no per-entity components);
    only create separate components when the requirement explicitly needs them.
    For Day 1, this uses a fixed template for an 'Items' CRUD app.
    """
    files = {}

    # Backend - FastAPI + SQLite (Python 3.9+, minimal deps only)
    files["backend/requirements.txt"] = (
        "# Python 3.9+\n"
        "fastapi\n"
        "uvicorn\n"
        "sqlalchemy\n"
        "pydantic\n"
    )
    
    # SQLite by default - no external credentials needed, runs out of the box
    files["backend/database.py"] = """
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
"""

    files["backend/models.py"] = """
from sqlalchemy import Column, Integer, String
from database import Base

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, index=True)
"""

    files["backend/schemas.py"] = """
from pydantic import BaseModel
from typing import Optional

class ItemBase(BaseModel):
    name: str
    description: Optional[str] = None

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int

    class Config:
        from_attributes = True
"""

    files["backend/main.py"] = """
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import models, schemas, database
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/items/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    item_data = item.model_dump() if hasattr(item, "model_dump") else item.dict()
    db_item = models.Item(**item_data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/", response_model=List[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = db.query(models.Item).offset(skip).limit(limit).all()
    return items
"""

    # Frontend - React with styles.css (minimal, works with Node 20+)
    files["frontend/package.json"] = """
{
  "name": "frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "NODE_OPTIONS=--openssl-legacy-provider react-scripts start",
    "build": "NODE_OPTIONS=--openssl-legacy-provider react-scripts build"
  },
  "engines": { "node": ">=20" },
  "eslintConfig": { "extends": ["react-app"] },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  }
}
"""

    files["frontend/public/index.html"] = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <title>React App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
"""

    files["frontend/src/index.js"] = """
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""

    files["frontend/src/styles.css"] = """
/* App styles - fonts, colors, layout */
:root {
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-primary: #4f46e5;
  --color-primary-hover: #4338ca;
  --color-border: #e2e8f0;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
}

* { box-sizing: border-box; }
body { margin: 0; font-family: var(--font-sans); background: var(--color-bg); color: var(--color-text); -webkit-font-smoothing: antialiased; }

.app { min-height: 100vh; padding: 2rem; }
.app-container { max-width: 42rem; margin: 0 auto; }
.app-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }
.app-subtitle { color: var(--color-text-muted); font-size: 0.875rem; margin-bottom: 1.5rem; }

.form-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 2rem; }
.input { flex: 1; min-width: 120px; padding: 0.5rem 1rem; border: 1px solid var(--color-border); border-radius: 0.5rem; font-family: inherit; font-size: 1rem; }
.input:focus { outline: none; border-color: var(--color-primary); box-shadow: 0 0 0 2px rgba(79,70,229,0.2); }

.btn { padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: 500; font-family: inherit; cursor: pointer; border: none; transition: background 0.2s; }
.btn-primary { background: var(--color-primary); color: white; }
.btn-primary:hover:not(:disabled) { background: var(--color-primary-hover); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.list { list-style: none; padding: 0; margin: 0; }
.list-item { padding: 1rem; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 0.75rem; margin-bottom: 0.75rem; box-shadow: var(--shadow-sm); transition: box-shadow 0.2s; }
.list-item:hover { box-shadow: var(--shadow-md); }
.list-item strong { font-weight: 500; }
.list-item .muted { color: var(--color-text-muted); margin-left: 0.5rem; }
"""

    files["frontend/src/App.js"] = """
import React, { useState, useEffect } from 'react';

function App() {
  const [items, setItems] = useState([]);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const backendUrl = (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || '').trim();

  const fetchItems = async () => {
    if (!backendUrl) return;
    try {
      const response = await fetch(`${backendUrl}/items/`);
      const data = await response.json().catch(() => []);
      setItems(Array.isArray(data) ? data : []);
    } catch (error) { console.error('Error fetching items:', error); setItems([]); }
  };

  useEffect(() => { fetchItems(); }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!backendUrl) return;
    try {
      await fetch(`${backendUrl}/items/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description }),
      });
      setName("");
      setDescription("");
      fetchItems();
    } catch (error) { console.error('Error creating item:', error); }
  };

  return (
    <div className="app">
      <div className="app-container">
        <h1 className="app-title">Items</h1>
        {!backendUrl && (
          <p className="app-subtitle">Backend not configured. Set REACT_APP_BACKEND_URL or VITE_BACKEND_URL when running with a backend.</p>
        )}
        <form onSubmit={handleSubmit} className="form-row">
          <input
            type="text"
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="input"
          />
          <input
            type="text"
            placeholder="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="input"
          />
          <button type="submit" disabled={!backendUrl} className="btn btn-primary">
            Add Item
          </button>
        </form>
        <ul className="list">
          {(items || []).map((item) => (
            <li key={item.id} className="list-item">
              <strong>{item.name}</strong>
              {item.description && <span className="muted">— {item.description}</span>}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
"""

    files["README.md"] = """
# Generated Project

**Requirements:** Node.js 20+, Python 3.9+

## Backend (SQLite - no setup)
1. cd backend && pip install -r requirements.txt
2. uvicorn main:app --reload --port 8001
   (Tables are created automatically on startup)

## Frontend
1. cd frontend && npm install
2. REACT_APP_BACKEND_URL=http://localhost:8001 npm start
"""

    return files
