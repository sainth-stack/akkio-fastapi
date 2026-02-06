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

    # Backend - FastAPI (pinned versions to avoid install/version issues)
    files["backend/requirements.txt"] = (
        "fastapi==0.109.2\n"
        "uvicorn==0.27.1\n"
        "sqlalchemy==2.0.25\n"
        "psycopg2-binary==2.9.9\n"
        "pydantic==1.10.19\n"
    )
    
    files["backend/database.py"] = """
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
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
        orm_mode = True
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/items/", response_model=schemas.Item)
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/", response_model=List[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = db.query(models.Item).offset(skip).limit(limit).all()
    return items
"""

    # Frontend - React (Simplified)
    files["frontend/package.json"] = """
{
  "name": "frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "eslintConfig": {
    "extends": [
      "react-app"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
"""

    files["frontend/public/index.html"] = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
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

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
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
      const data = await response.json();
      setItems(data);
    } catch (error) {
      console.error('Error fetching items:', error);
    }
  };

  useEffect(() => {
    fetchItems();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!backendUrl) return;
    try {
      await fetch(`${backendUrl}/items/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, description }),
      });
      setName("");
      setDescription("");
      fetchItems();
    } catch (error) {
      console.error('Error creating item:', error);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Items</h1>
      {!backendUrl && (
        <p style={{ color: "#666", marginBottom: "20px" }}>Backend not configured. Set REACT_APP_BACKEND_URL or VITE_BACKEND_URL when running with a backend.</p>
      )}
      <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
        <input
          type="text"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ marginRight: "10px" }}
        />
        <input
          type="text"
          placeholder="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          style={{ marginRight: "10px" }}
        />
        <button type="submit" disabled={!backendUrl}>Add Item</button>
      </form>
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <strong>{item.name}</strong>: {item.description}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
"""

    files["README.md"] = """
# Generated Project

## Backend
1. cd backend
2. pip install -r requirements.txt --force-reinstall
3. uvicorn main:app --reload --port 8001

## Frontend
1. cd frontend
2. npm install --force
3. REACT_APP_BACKEND_URL=http://localhost:8001 npm start
"""

    return files
