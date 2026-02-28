import json
from typing import Dict, Any, List
from ..schemas.architecture import ArchitectureDecision

def code_generator_agent(architecture: ArchitectureDecision) -> Dict[str, str]:
    """
    Generates the code for the project based on the architecture.
    """
    files = {}
    arch_dict = architecture.model_dump()
    
    # Extract structural info
    backend_struct = arch_dict.get("backend_structure", {})
    frontend_struct = arch_dict.get("frontend_structure", {})
    db_schema = arch_dict.get("database_schema", {})
    tables = db_schema.get("tables", [])
    
    # Generic type mapping
    sa_type_map = {
        "Integer": "Integer",
        "String": "String",
        "Boolean": "Boolean",
        "DateTime": "DateTime",
        "Float": "Float",
        "Text": "Text"
    }
    py_type_map = {
        "Integer": "int",
        "String": "str",
        "Boolean": "bool",
        "DateTime": "datetime",
        "Float": "float",
        "Text": "str"
    }

    # Backend: requirements.txt
    files["backend/requirements.txt"] = (
        "fastapi\n"
        "uvicorn\n"
        "sqlalchemy\n"
        "pydantic\n"
        "python-dateutil\n"
    )

    # Backend: database.py
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

    # Backend: models.py
    models_content = ["from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text", "from sqlalchemy.orm import relationship", "from database import Base", "import datetime", ""]
    for table in tables:
        table_name = table["name"]
        class_name = "".join([part.capitalize() for part in table_name.split("_")])
        if class_name.endswith("s"): class_name = class_name[:-1] # simple singularization
        
        models_content.append(f"class {class_name}(Base):")
        models_content.append(f"    __tablename__ = \"{table_name}\"")
        models_content.append("")
        for column in table["columns"]:
            sa_type = sa_type_map.get(column["type"], "String")
            args = []
            if column.get("primary_key"):
                args.append("primary_key=True")
                args.append("index=True")
            
            if column.get("nullable") is False:
                pass # default is nullable=True
            
            args_str = ", " + ", ".join(args) if args else ""
            models_content.append(f"    {column['name']} = Column({sa_type}{args_str})")
        models_content.append("")
    files["backend/models.py"] = "\n".join(models_content)

    # Backend: schemas.py
    schemas_content = ["from pydantic import BaseModel", "from typing import Optional, List", "from datetime import datetime", ""]
    for table in tables:
        table_name = table["name"]
        class_name = "".join([part.capitalize() for part in table_name.split("_")])
        if class_name.endswith("s"): class_name = class_name[:-1]
        
        # Base schema
        schemas_content.append(f"class {class_name}Base(BaseModel):")
        for column in table["columns"]:
            if column.get("primary_key"): continue
            py_type = py_type_map.get(column["type"], "str")
            if column.get("nullable") is not False:
                schemas_content.append(f"    {column['name']}: Optional[{py_type}] = None")
            else:
                schemas_content.append(f"    {column['name']}: {py_type}")
        schemas_content.append("")
        
        # Create schema
        schemas_content.append(f"class {class_name}Create({class_name}Base):")
        schemas_content.append("    pass")
        schemas_content.append("")
        
        # Response schema
        schemas_content.append(f"class {class_name}({class_name}Base):")
        for column in table["columns"]:
            if column.get("primary_key"):
                py_type = py_type_map.get(column["type"], "int")
                schemas_content.append(f"    {column['name']}: {py_type}")
        schemas_content.append("")
        schemas_content.append("    model_config = ConfigDict(from_attributes=True)")
        schemas_content.append("")
    files["backend/schemas.py"] = "\n".join(schemas_content)

    # Backend: main.py
    main_content = [
        "from fastapi import FastAPI, Depends, HTTPException",
        "from sqlalchemy.orm import Session",
        "from typing import List",
        "import models, schemas, database",
        "from fastapi.middleware.cors import CORSMiddleware",
        "",
        "models.Base.metadata.create_all(bind=database.engine)",
        "",
        "app = FastAPI()",
        "",
        "app.add_middleware(",
        "    CORSMiddleware,",
        "    allow_origins=[\"*\"],",
        "    allow_credentials=False,",
        "    allow_methods=[\"*\"],",
        "    allow_headers=[\"*\"],",
        "    expose_headers=[\"*\"],",
        ")",
        "",
        "def get_db():",
        "    db = database.SessionLocal()",
        "    try:",
        "        yield db",
        "    finally:",
        "        db.close()",
        ""
    ]
    
    for table in tables:
        table_name = table["name"]
        class_name = "".join([part.capitalize() for part in table_name.split("_")])
        if class_name.endswith("s"): class_name = class_name[:-1]
        
        # List
        main_content.append(f"@app.get(\"/{table_name}\", response_model=List[schemas.{class_name}])")
        main_content.append(f"def list_{table_name}(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):")
        main_content.append(f"    return db.query(models.{class_name}).offset(skip).limit(limit).all()")
        main_content.append("")
        
        # Create
        main_content.append(f"@app.post(\"/{table_name}\", response_model=schemas.{class_name})")
        main_content.append(f"def create_{table_name}(item: schemas.{class_name}Create, db: Session = Depends(get_db)):")
        main_content.append(f"    db_item = models.{class_name}(**item.model_dump())")
        main_content.append(f"    db.add(db_item)")
        main_content.append(f"    db.commit()")
        main_content.append(f"    db.refresh(db_item)")
        main_content.append(f"    return db_item")
        main_content.append("")
    
    files["backend/main.py"] = "\n".join(main_content)

    # Frontend: Styles
    files["frontend/src/styles.css"] = """
:root {
  --font-sans: 'Inter', system-ui, sans-serif;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-primary: #4f46e5;
  --color-primary-hover: #4338ca;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-border: #e2e8f0;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
}
body { margin: 0; font-family: var(--font-sans); background: var(--color-bg); color: var(--color-text); line-height: 1.5; }
.app { padding: 3rem 1.5rem; max-width: 900px; margin: 0 auto; }
.app-header { margin-bottom: 2.5rem; text-align: center; }
.section { margin-bottom: 3rem; }
.section-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 1.5rem; border-bottom: 2px solid var(--color-border); padding-bottom: 0.5rem; }
.card { background: var(--color-surface); padding: 1.5rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-md); border: 1px solid var(--color-border); margin-bottom: 1.5rem; }
.form-grid { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem; }
@media (min-width: 640px) { .form-grid { grid-template-columns: repeat(2, 1fr); } }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; }
.input-label { font-size: 0.875rem; font-weight: 500; color: var(--color-text-muted); }
.input { padding: 0.625rem 1rem; border: 1px solid var(--color-border); border-radius: var(--radius-md); font-family: inherit; font-size: 1rem; transition: all 0.2s; box-shadow: var(--shadow-sm); }
.input:focus { outline: none; border-color: var(--color-primary); box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2); }
.btn { display: inline-flex; align-items: center; justify-content: center; padding: 0.625rem 1.25rem; border-radius: var(--radius-md); font-weight: 600; cursor: pointer; border: none; transition: all 0.2s; }
.btn-primary { background: var(--color-primary); color: white; }
.btn-primary:hover { background: var(--color-primary-hover); transform: translateY(-1px); }
.list { display: flex; flex-direction: column; gap: 1rem; }
.list-item { display: flex; align-items: center; justify-content: space-between; padding: 1.25rem; background: white; border: 1px solid var(--color-border); border-radius: var(--radius-md); transition: all 0.2s; }
.list-item:hover { transform: translateX(4px); border-color: var(--color-primary); }
"""

    # Frontend: App.js
    app_js = ["import React, { useState, useEffect } from 'react';", ""]
    for table in tables:
        table_name = table["name"]
        class_name = "".join([part.capitalize() for part in table_name.split("_")])
        if class_name.endswith("s"): class_name = class_name[:-1]
        fields = [c["name"] for c in table["columns"] if not c.get("primary_key")]
        
        app_js.extend([
            f"function {class_name}Manager({{ backendUrl }}) {{",
            f"  const STORAGE_KEY = 'items_{table_name}';",
            f"  const [items, setItems] = useState(() => {{ try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }} catch {{ return []; }} }});",
            f"  const [formData, setFormData] = useState({{{', '.join([f'{f}: \"\"' for f in fields])}}});",
            "",
            "  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(items)); }, [items]);",
            "",
            f"  const fetchData = async () => {{",
            f"    try {{ const r = await fetch(`${{backendUrl}}/{table_name}`); if(r.ok) {{ const d = await r.json(); setItems(Array.isArray(d) ? d : []); }} }} catch(e) {{}}",
            "  };",
            "",
            "  useEffect(() => { fetchData(); }, []);",
            "",
            "  const handleSubmit = async (e) => {",
            "    e.preventDefault();",
            f"    const newItem = {{ ...formData, id: Date.now() }};",
            f"    setItems([...items, newItem]);",
            f"    try {{ await fetch(`${{backendUrl}}/{table_name}`, {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(formData) }}); fetchData(); }} catch(e) {{}}",
            f"    setFormData({{{', '.join([f'{f}: \"\"' for f in fields])}}});",
            "  };",
            "",
            "  return (",
            f"    <div className=\"section\">",
            f"      <h2 className=\"section-title\">Manage {table_name.replace('_', ' ').title()}</h2>",
            "      <div className=\"card\">",
            "        <form onSubmit={handleSubmit}>",
            "          <div className=\"form-grid\">",
        ])
        for field in fields:
            label = field.replace("_", " ").title()
            app_js.append(f"            <div className=\"input-group\"><label className=\"input-label\">{label}</label><input className=\"input\" value={{formData.{field}}} onChange={{e => setFormData({{...formData, {field}: e.target.value}})}} /></div>")
        
        app_js.extend([
            "          </div>",
            f"          <button className=\"btn btn-primary\" type=\"submit\">Add {class_name}</button>",
            "        </form>",
            "      </div>",
            "      <div className=\"list\">",
            "        {items.map(item => (",
            "          <div key={item.id} className=\"list-item\">",
            "            <div>"
        ])
        for field in fields[:2]:
            app_js.append(f"              <div style={{{{ fontWeight: 600 }}}}>{{item.{field}}}</div>")
        app_js.extend([
            "            </div>",
            "          </div>",
            "        ))}",
            "      </div>",
            "    </div>",
            "  );",
            "}",
            ""
        ])

    app_js.extend([
        "function App() {",
        "  const backendUrl = ((typeof window !== 'undefined' && window.location.pathname.startsWith('/app/')) ? (window.location.origin + '/api/apps/' + window.location.pathname.split('/')[2]) : (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || window.location?.origin || '')).trim();",
        "  return (",
        "    <div className=\"app\">",
        f"      <header className=\"app-header\"><h1 className=\"app-title\">{arch_dict.get('rationale', 'Generated App').split(' ')[0]} Management</h1></header>",
    ])
    for table in tables:
        table_name = table["name"]
        class_name = "".join([part.capitalize() for part in table_name.split("_")])
        if class_name.endswith("s"): class_name = class_name[:-1]
        app_js.append(f"      <{class_name}Manager backendUrl={{backendUrl}} />")
    app_js.extend([
        "    </div>",
        "  );",
        "}",
        "export default App;"
    ])
    files["frontend/src/App.js"] = "\n".join(app_js)
    
    files["frontend/package.json"] = json.dumps({
        "name": "frontend",
        "version": "0.1.0",
        "private": True,
        "dependencies": { "react": "^18.2.0", "react-dom": "^18.2.0", "react-scripts": "5.0.1" },
        "scripts": { "start": "NODE_OPTIONS=--openssl-legacy-provider react-scripts start", "build": "NODE_OPTIONS=--openssl-legacy-provider react-scripts build" },
        "engines": { "node": ">=20" }
    }, indent=2)
    files["frontend/public/index.html"] = "<!DOCTYPE html><html><head><meta charset=\"utf-8\" /><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" /><title>App</title><script src=\"https://cdn.tailwindcss.com\"></script><link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap\" rel=\"stylesheet\" /></head><body class=\"bg-slate-50 text-slate-900\"><div id=\"root\"></div></body></html>"
    files["frontend/src/index.js"] = "import React from 'react';\nimport ReactDOM from 'react-dom/client';\nimport App from './App';\nimport './styles.css';\nconst root = ReactDOM.createRoot(document.getElementById('root'));\nroot.render(<React.StrictMode><App /></React.StrictMode>);"

    files["README.md"] = "# Generated App\\n\\n## Backend\\ncd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 5001\\n\\n## Frontend\\ncd frontend && npm install && npm start"
    return files
    return files

def generate_from_template(template: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates code based on a Planet/Akkio JSON template.
    """
    files = {}
    app_name = template.get("app_name", "app")
    entities = template.get("entities", [])
    
    # Generic type mapping
    sa_type_map = {
        "integer": "Integer",
        "string": "String",
        "boolean": "Boolean",
        "datetime": "DateTime",
        "float": "Float"
    }
    py_type_map = {
        "integer": "int",
        "string": "str",
        "boolean": "bool",
        "datetime": "datetime",
        "float": "float"
    }

    # Backend: requirements.txt
    files["backend/requirements.txt"] = (
        "fastapi\n"
        "uvicorn\n"
        "sqlalchemy\n"
        "pydantic\n"
        "python-dateutil\n"
    )

    # Backend: database.py
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

    # Backend: models.py
    models_content = ["from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey", "from sqlalchemy.orm import relationship", "from database import Base", "import datetime", ""]
    for entity in entities:
        models_content.append(f"class {entity['name']}(Base):")
        models_content.append(f"    __tablename__ = \"{entity['table_name']}\"")
        models_content.append("")
        for field in entity["fields"]:
            sa_type = sa_type_map.get(field["type"], "String")
            args = []
            if field.get("primary_key"):
                args.append("primary_key=True")
                if field.get("auto_increment"):
                    args.append("index=True")
            
            if field.get("foreign_key"):
                fk = field["foreign_key"]
                args.append(f"ForeignKey('{fk['table']}.{fk['field']}')")
            
            if sa_type == "String" and field.get("max_length"):
                sa_type = f"String({field['max_length']})"
            
            if field.get("default") is not None:
                default = field["default"]
                if isinstance(default, bool):
                    args.append(f"default={default}")
                elif isinstance(default, str):
                    args.append(f"default='{default}'")
                else:
                    args.append(f"default={default}")
            
            if field.get("auto_now_add"):
                args.append("default=datetime.datetime.utcnow")
            
            models_content.append(f"    {field['name']} = Column({sa_type}, {', '.join(args)})")
        models_content.append("")
    files["backend/models.py"] = "\n".join(models_content)

    # Backend: schemas.py
    schemas_content = ["from pydantic import BaseModel", "from typing import Optional, List", "from datetime import datetime", ""]
    for entity in entities:
        # Base schema
        schemas_content.append(f"class {entity['name']}Base(BaseModel):")
        for field in entity["fields"]:
            if field.get("primary_key"): continue
            py_type = py_type_map.get(field["type"], "str")
            if not field.get("required"):
                schemas_content.append(f"    {field['name']}: Optional[{py_type}] = None")
            else:
                schemas_content.append(f"    {field['name']}: {py_type}")
        schemas_content.append("")
        
        # Create schema
        schemas_content.append(f"class {entity['name']}Create({entity['name']}Base):")
        schemas_content.append("    pass")
        schemas_content.append("")
        
        # Response schema
        schemas_content.append(f"class {entity['name']}({entity['name']}Base):")
        for field in entity["fields"]:
            if field.get("primary_key"):
                py_type = py_type_map.get(field["type"], "int")
                schemas_content.append(f"    {field['name']}: {py_type}")
        schemas_content.append("")
        schemas_content.append("    model_config = ConfigDict(from_attributes=True)")
        schemas_content.append("")
    files["backend/schemas.py"] = "\n".join(schemas_content)

    # Backend: main.py
    main_content = [
        "from fastapi import FastAPI, Depends, HTTPException",
        "from sqlalchemy.orm import Session",
        "from typing import List",
        "import models, schemas, database",
        "from fastapi.middleware.cors import CORSMiddleware",
        "",
        "models.Base.metadata.create_all(bind=database.engine)",
        "",
        "app = FastAPI()",
        "",
        "app.add_middleware(",
        "    CORSMiddleware,",
        "    allow_origins=[\"*\"],",
        "    allow_credentials=False,",
        "    allow_methods=[\"*\"],",
        "    allow_headers=[\"*\"],",
        "    expose_headers=[\"*\"],",
        ")",
        "",
        "def get_db():",
        "    db = database.SessionLocal()",
        "    try:",
        "        yield db",
        "    finally:",
        "        db.close()",
        ""
    ]
    
    for entity in entities:
        name = entity["name"]
        table = entity["table_name"]
        path = entity["endpoints"].get("list", {}).get("path", f"/{table}")
        
        # List
        main_content.append(f"@app.get(\"{path}\", response_model=List[schemas.{name}])")
        main_content.append(f"def list_{table}(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):")
        main_content.append(f"    return db.query(models.{name}).offset(skip).limit(limit).all()")
        main_content.append("")
        
        # Create
        create_path = entity["endpoints"].get("create", {}).get("path", path)
        main_content.append(f"@app.post(\"{create_path}\", response_model=schemas.{name})")
        main_content.append(f"def create_{table}(item: schemas.{name}Create, db: Session = Depends(get_db)):")
        main_content.append(f"    db_item = models.{name}(**item.model_dump())")
        main_content.append(f"    db.add(db_item)")
        main_content.append(f"    db.commit()")
        main_content.append(f"    db.refresh(db_item)")
        main_content.append(f"    return db_item")
        main_content.append("")
        
        # Delete
        delete_path = entity["endpoints"].get("delete", {}).get("path", f"{path}/{{id}}")
        if "{id}" in delete_path:
            main_content.append(f"@app.delete(\"{delete_path}\")")
            main_content.append(f"def delete_{table}(id: int, db: Session = Depends(get_db)):")
            main_content.append(f"    db_item = db.query(models.{name}).filter(models.{name}.id == id).first()")
            main_content.append(f"    if not db_item: raise HTTPException(status_code=404)")
            main_content.append(f"    db.delete(db_item)")
            main_content.append(f"    db.commit()")
            main_content.append(f"    return {{\"ok\": True}}")
            main_content.append("")

    files["backend/main.py"] = "\n".join(main_content)

    # Frontend: Styles with premium defaults if applicable
    is_todo = any(kw in app_name.lower() or kw in str(template).lower() for kw in ["todo", "task"])
    
    style_content = """
:root {
  --font-sans: 'Inter', system-ui, sans-serif;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-primary: #4f46e5;
  --color-primary-hover: #4338ca;
  --color-text: #1e293b;
  --color-text-muted: #64748b;
  --color-border: #e2e8f0;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
}
body { margin: 0; font-family: var(--font-sans); background: var(--color-bg); color: var(--color-text); line-height: 1.5; }
.app { padding: 3rem 1.5rem; max-width: 900px; margin: 0 auto; }
.app-header { margin-bottom: 2.5rem; text-align: center; }
.app-title { font-size: 2.25rem; font-weight: 700; margin-bottom: 0.5rem; }
.section { margin-bottom: 3rem; }
.section-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--color-border); }
.card { background: var(--color-surface); padding: 1.5rem; border-radius: var(--radius-lg); box-shadow: var(--shadow-md); border: 1px solid var(--color-border); margin-bottom: 1.5rem; }
.form-grid { display: grid; grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem; }
@media (min-width: 640px) { .form-grid { grid-template-columns: repeat(2, 1fr); } }
.input-group { display: flex; flex-direction: column; gap: 0.5rem; }
.input-label { font-size: 0.875rem; font-weight: 500; color: var(--color-text-muted); }
.input { padding: 0.625rem 1rem; border: 1px solid var(--color-border); border-radius: var(--radius-md); font-family: inherit; font-size: 1rem; transition: all 0.2s; box-shadow: var(--shadow-sm); }
.input:focus { outline: none; border-color: var(--color-primary); ring: 2px solid rgba(79, 70, 229, 0.2); }
.btn { display: inline-flex; align-items: center; justify-content: center; padding: 0.625rem 1.25rem; border-radius: var(--radius-md); font-weight: 600; cursor: pointer; border: none; transition: all 0.2s; font-size: 0.875rem; }
.btn-primary { background: var(--color-primary); color: white; }
.btn-primary:hover { background: var(--color-primary-hover); transform: translateY(-1px); }
.btn-danger { background: #fee2e2; color: #dc2626; }
.btn-danger:hover { background: #fecaca; }
.list { display: flex; flex-direction: column; gap: 1rem; }
.list-item { display: flex; align-items: center; justify-content: space-between; padding: 1.25rem; background: white; border: 1px solid var(--color-border); border-radius: var(--radius-md); transition: all 0.2s; }
.list-item:hover { transform: translateX(4px); border-color: var(--color-primary); }
"""
    if is_todo:
        style_content += """
/* Premium Todo Glassmorphism */
.todo-card { border: 1px solid rgba(255, 255, 255, 0.3); background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); }
.todo-item-completed { opacity: 0.6; text-decoration: line-through; }
"""
    files["frontend/src/styles.css"] = style_content

    # Frontend: App.js (Full multi-entity management)
    app_js = ["import React, { useState, useEffect } from 'react';", ""]
    
    # Generate sub-components for each entity
    for entity in entities:
        name = entity["name"]
        table = entity["table_name"]
        fields = [f for f in entity["fields"] if not f.get("primary_key")]
        list_path = entity["endpoints"]["list"]["path"]
        create_path = entity["endpoints"]["create"]["path"]
        delete_path_template = entity["endpoints"].get("delete", {}).get("path", f"/{table}/{{id}}")
        
        app_js.extend([
            f"function {name}Manager({{ backendUrl }}) {{",
            f"  const STORAGE_KEY = 'items_{table}';",
            f"  const [items, setItems] = useState(() => {{ try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }} catch {{ return []; }} }});",
            f"  const [formData, setFormData] = useState({{{', '.join([f'{f["name"]}: \"\"' for f in fields])}}});",
            "",
            "  useEffect(() => { localStorage.setItem(STORAGE_KEY, JSON.stringify(items)); }, [items]);",
            "",
            f"  const fetchData = async () => {{",
            f"    try {{ const r = await fetch(`${{backendUrl}}{list_path}`); if(r.ok) {{ const d = await r.json(); setItems(Array.isArray(d) ? d : []); }} }} catch(e) {{}}",
            "  };",
            "",
            "  useEffect(() => { fetchData(); }, []);",
            "",
            "  const handleSubmit = async (e) => {",
            "    e.preventDefault();",
            f"    const newItem = {{ ...formData, id: Date.now() }};",
            f"    setItems([...items, newItem]);",
            f"    try {{ await fetch(`${{backendUrl}}{create_path}`, {{ method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(formData) }}); fetchData(); }} catch(e) {{}}",
            f"    setFormData({{{', '.join([f'{f["name"]}: \"\"' for f in fields])}}});",
            "  };",
            "",
            "  return (",
            f"    <div className=\"section\">",
            f"      <h2 className=\"section-title\">Manage {name}s</h2>",
            "      <div className=\"card\">",
            "        <form onSubmit={handleSubmit}>",
            "          <div className=\"form-grid\">",
        ])
        
        for field in fields:
            label = field["name"].replace("_", " ").title()
            app_js.append(f"            <div className=\"input-group\"><label className=\"input-label\">{label}</label><input className=\"input\" value={{formData.{field['name']}}} onChange={{e => setFormData({{...formData, {field['name']}: e.target.value}})}} /></div>")
            
        app_js.extend([
            "          </div>",
            "          <button className=\"btn btn-primary\" type=\"submit\">Add {name}</button>",
            "        </form>",
            "      </div>",
            "      <div className=\"list\">",
            "        {items.map(item => (",
            "          <div key={item.id} className=\"list-item\">",
            "            <div>"
        ])
        
        # Display first few fields
        display_fields = fields[:2]
        for df in display_fields:
            app_js.append(f"              <div style={{{{ fontWeight: 600 }}}}>{{item.{df['name']}}}</div>")
            
        app_js.extend([
            "            </div>",
            f"            <button className=\"btn btn-danger\" onClick={{async () => {{ await fetch(`${{backendUrl}}{delete_path_template.replace('{id}', '${item.id}')}`, {{method: 'DELETE'}}); fetchData(); }}}}>Delete</button>",
            "          </div>",
            "        ))}",
            "      </div>",
            "    </div>",
            "  );",
            "}",
            ""
        ])

    # Main App Component
    app_js.extend([
        "function App() {",
        "  const backendUrl = ((typeof window !== 'undefined' && window.location.pathname.startsWith('/app/')) ? (window.location.origin + '/api/apps/' + window.location.pathname.split('/')[2]) : (process.env.REACT_APP_BACKEND_URL || process.env.VITE_BACKEND_URL || window.location?.origin || '')).trim();",
        "  return (",
        "    <div className=\"app\">",
        f"      <header className=\"app-header\"><h1 className=\"app-title\">{app_name.replace('_', ' ').title()}</h1></header>",
    ])
    
    for entity in entities:
        app_js.append(f"      <{entity['name']}Manager backendUrl={{backendUrl}} />")
        
    app_js.extend([
        "    </div>",
        "  );",
        "}",
        "export default App;"
    ])
    files["frontend/src/App.js"] = "\n".join(app_js)

    files["backend/requirements.txt"] = "fastapi\nuvicorn\nsqlalchemy\npydantic\npython-dateutil\n"
    files["frontend/package.json"] = json.dumps({
        "name": "frontend",
        "version": "0.1.0",
        "private": True,
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "react-scripts": "5.0.1"
        },
        "scripts": {
            "start": "NODE_OPTIONS=--openssl-legacy-provider react-scripts start",
            "build": "NODE_OPTIONS=--openssl-legacy-provider react-scripts build"
        },
        "engines": { "node": ">=20" }
    }, indent=2)

    files["frontend/public/index.html"] = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <script src="https://cdn.tailwindcss.com"></script>
    <title>Template App</title>
  </head>
  <body>
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
root.render(<React.StrictMode><App /></React.StrictMode>);
"""

    files["README.md"] = f"# {app_name}\\nGenerated from template.\\n\\n## Backend\\ncd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 5001\\n\\n## Frontend\\ncd frontend && npm install && npm start"
    return files
