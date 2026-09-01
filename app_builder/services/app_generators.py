"""
Deterministic full-stack app generators — ensures every app kind ships a working baseline
when LLM output fails validation.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from app_builder.services.app_spec_service import singularize


# ---------------------------------------------------------------------------
# Static quiz (frontend-only)
# ---------------------------------------------------------------------------

def _parse_static_questions(uiux: str, prd: str) -> List[Dict[str, Any]]:
    text = f"{uiux}\n{prd}"
    questions: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*(?:\d+[\).:]\s*Question\s*\d+|\d+[\).:]\s*)", text, flags=re.I)
    if len(blocks) <= 1:
        blocks = re.split(r"\n\s*\d+[\).:]\s+", text)
    for block in blocks:
        block = block.strip()
        if len(block) < 20:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        q_text = lines[0]
        if q_text.lower().startswith("question"):
            q_text = lines[1] if len(lines) > 1 else q_text
        options: List[str] = []
        correct_idx = 0
        for ln in lines[1:]:
            m_opt = re.match(r"^[-•]?\s*[A-Da-d][\).:]\s*(.+)$", ln)
            if m_opt:
                options.append(m_opt.group(1).strip())
                continue
            m_corr = re.search(r"Correct:\s*([A-Da-d])", ln, re.I)
            if m_corr:
                correct_idx = ord(m_corr.group(1).upper()) - ord("A")
        if q_text and len(options) >= 2:
            questions.append({
                "text": q_text.lstrip("- ").strip(),
                "options": options[:4],
                "correctIndex": min(correct_idx, len(options[:4]) - 1),
            })
    return questions[:10]


_DEFAULT_QUESTIONS = [
    {"text": "What is the capital of France?", "options": ["Berlin", "Madrid", "Paris", "Rome"], "correctIndex": 2},
    {"text": "Which planet is known as the Red Planet?", "options": ["Earth", "Mars", "Jupiter", "Venus"], "correctIndex": 1},
    {"text": "What is the largest ocean on Earth?", "options": ["Atlantic", "Indian", "Arctic", "Pacific"], "correctIndex": 3},
    {"text": "Primary Android dev language?", "options": ["Swift", "Kotlin", "JavaScript", "Ruby"], "correctIndex": 1},
    {"text": "Who wrote Hamlet?", "options": ["Dickens", "Shakespeare", "Twain", "Austen"], "correctIndex": 1},
]


def generate_static_quiz_app_jsx(uiux: str = "", prd: str = "", title: str = "Quiz") -> str:
    questions = _parse_static_questions(uiux, prd) or _DEFAULT_QUESTIONS
    q_json = json.dumps(questions, indent=2)
    return f'''import {{ useState }} from 'react';

const QUESTIONS = {q_json};

export default function App() {{
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [locked, setLocked] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const current = QUESTIONS[index];
  const total = QUESTIONS.length;

  const pickOption = (optIdx) => {{
    if (locked || showResults) return;
    setSelected(optIdx);
    setLocked(true);
    setAnswers((prev) => {{ const n = [...prev]; n[index] = optIdx; return n; }});
  }};

  const goNext = () => {{
    if (index + 1 >= total) {{ setShowResults(true); return; }}
    setIndex((i) => i + 1); setSelected(null); setLocked(false);
  }};

  const restart = () => {{
    setIndex(0); setAnswers([]); setSelected(null); setLocked(false); setShowResults(false);
  }};

  const score = answers.reduce((a, ans, i) => a + (ans === QUESTIONS[i]?.correctIndex ? 1 : 0), 0);
  const pct = total ? Math.round((score / total) * 100) : 0;

  if (showResults) {{
    return (
      <div className="app"><div className="app-container">
        <header className="app-header"><h1 className="app-title">Results</h1>
          <p className="app-subtitle">Score: {{score}} / {{total}} ({{pct}}%)</p></header>
        <div className="card"><ul className="list">
          {{QUESTIONS.map((q, i) => (
            <li key={{i}} className="list-item">{{i + 1}}. {{q.text}} — {{answers[i] === q.correctIndex ? 'Correct' : 'Wrong'}}</li>
          ))}}
        </ul><button type="button" className="btn btn-primary" onClick={{restart}}>Restart</button></div>
      </div></div>
    );
  }}

  return (
    <div className="app"><div className="app-container">
      <header className="app-header"><h1 className="app-title">{title}</h1>
        <p className="app-subtitle">Question {{index + 1}} of {{total}}</p></header>
      <div className="card"><h2 className="question-text">{{current.text}}</h2>
        <div className="options">
          {{current.options.map((opt, i) => (
            <button key={{i}} type="button" className="option-row" disabled={{locked}}
              onClick={{() => pickOption(i)}} role="radio" aria-checked={{selected === i}}>
              {{String.fromCharCode(65 + i)}}) {{opt}}
            </button>
          ))}}
        </div>
        {{locked && <div className="feedback">{{selected === current.correctIndex ? 'Correct!' : 'Incorrect.'}} Answer: {{current.options[current.correctIndex]}}</div>}}
        <button type="button" className="btn btn-primary" disabled={{!locked}} onClick={{goNext}}>
          {{index + 1 >= total ? 'View Results' : 'Next'}}
        </button>
      </div>
    </div></div>
  );
}}
'''


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------


def _primary_hover(primary: str) -> str:
    """Simple darker hover — hex only."""
    primary = (primary or "#4f46e5").strip()
    if not primary.startswith("#") or len(primary) < 7:
        return "#4338ca"
    try:
        r, g, b = int(primary[1:3], 16), int(primary[3:5], 16), int(primary[5:7], 16)
        return f"#{max(0, r - 20):02x}{max(0, g - 20):02x}{max(0, b - 20):02x}"
    except ValueError:
        return "#4338ca"


def build_saas_app_css(tokens: Dict[str, Any] | None = None, primary: str = "#4f46e5") -> str:
    """
    Production SaaS stylesheet — design tokens + layout primitives + JSX fallbacks.
    Used for every generated app so colors/backgrounds always apply.
    """
    colors = (tokens or {}).get("colors") if isinstance(tokens, dict) else {}
    if not isinstance(colors, dict):
        colors = tokens if isinstance(tokens, dict) else {}

    primary = str(colors.get("primary") or primary)
    bg = str(colors.get("background") or colors.get("bg") or "#f8fafc")
    surface = str(colors.get("surface") or "#ffffff")
    text = str(colors.get("text") or "#0f172a")
    muted = str(colors.get("muted") or "#64748b")
    border = str(colors.get("border") or "#e2e8f0")
    danger = str(colors.get("danger") or "#ef4444")
    hover = _primary_hover(primary)
    font = "Inter, system-ui, -apple-system, sans-serif"
    if isinstance(tokens, dict) and isinstance(tokens.get("typography"), dict):
        font = tokens["typography"].get("fontFamily") or font
    max_w = "48rem"
    if isinstance(tokens, dict) and isinstance(tokens.get("layout"), dict):
        max_w = tokens["layout"].get("maxContentWidth") or max_w

    return f"""/* Akkio SaaS design system — auto-generated */
:root {{
  --color-primary: {primary};
  --color-primary-hover: {hover};
  --color-bg: {bg};
  --color-background: {bg};
  --color-surface: {surface};
  --color-text: {text};
  --color-text-muted: {muted};
  --color-muted: {muted};
  --color-border: {border};
  --color-danger: {danger};
  --font-sans: {font};
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --shadow-sm: 0 1px 2px rgb(0 0 0 / 0.06);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.08);
  --max-content-width: {max_w};
}}

* {{ box-sizing: border-box; -webkit-font-smoothing: antialiased; }}

html, body {{
  margin: 0;
  padding: 0;
  min-height: 100%;
  font-family: var(--font-sans);
  background: var(--color-bg);
  color: var(--color-text);
  line-height: 1.5;
}}

#root {{ min-height: 100vh; }}

.app {{
  min-height: 100vh;
  background: var(--color-bg);
  color: var(--color-text);
}}

.app-container {{
  max-width: var(--max-content-width);
  margin: 0 auto;
  padding: 2rem 1.25rem;
}}

.app-header {{
  margin-bottom: 2rem;
  text-align: center;
}}

.app-title {{
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.025em;
  margin: 0 0 0.5rem;
  color: var(--color-text);
}}

.app-subtitle {{
  color: var(--color-text-muted);
  font-size: 1.05rem;
  margin: 0;
}}

.navbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  box-shadow: var(--shadow-sm);
}}

.navbar h1 {{
  margin: 0;
  font-size: 1.375rem;
  font-weight: 700;
  color: var(--color-text);
}}

.sidebar {{
  width: 220px;
  min-height: 200px;
  padding: 1rem;
  background: var(--color-bg);
  border-right: 1px solid var(--color-border);
}}

.main-content {{
  flex: 1;
  padding: 1.5rem;
  max-width: var(--max-content-width);
  margin: 0 auto;
  width: 100%;
}}

.card {{
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 1.5rem;
  box-shadow: var(--shadow-md);
  margin-bottom: 1.5rem;
}}

.form-row, .task-form, .todo-form {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
  align-items: center;
}}

.input, .textarea,
.task-form input, .todo-form input,
.form-row input, .card input,
.app input[type="text"], .app input[type="email"],
.app input[type="password"], .app input[type="search"],
.app input[type="date"], .app input[type="number"],
.app input:not([type]), .app select, .app textarea {{
  flex: 1;
  min-width: 140px;
  padding: 0.625rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font: inherit;
  background: var(--color-surface);
  color: var(--color-text);
  transition: border-color 0.15s, box-shadow 0.15s;
}}

.textarea {{ min-height: 100px; resize: vertical; width: 100%; }}

.input:focus, .textarea:focus,
.task-form input:focus, .form-row input:focus,
.app input:focus, .app select:focus, .app textarea:focus {{
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--color-primary) 20%, transparent);
}}

.btn,
.task-form button, .todo-form button,
.card button, .task-card button, .todo-card button,
.app button {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.625rem 1.25rem;
  border-radius: var(--radius-md);
  font-weight: 600;
  font: inherit;
  cursor: pointer;
  border: none;
  transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
}}

.btn-primary,
.task-form button:first-of-type,
.app button:first-of-type {{
  background: var(--color-primary);
  color: #fff;
  box-shadow: var(--shadow-sm);
}}

.btn-primary:hover:not(:disabled),
.task-form button:first-of-type:hover,
.app button:first-of-type:hover {{
  background: var(--color-primary-hover);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}}

.btn-primary:disabled {{ opacity: 0.55; cursor: not-allowed; }}

.btn-ghost {{
  background: transparent;
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
}}

.btn-ghost:hover {{ border-color: var(--color-primary); color: var(--color-primary); }}

.list, .task-list, .todo-list {{
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}}

.list-item, .task-card, .todo-card {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  transition: border-color 0.15s, box-shadow 0.15s;
}}

.task-card, .todo-card {{
  flex-direction: column;
  align-items: stretch;
}}

.task-card:hover, .todo-card:hover, .list-item:hover {{
  border-color: color-mix(in srgb, var(--color-primary) 40%, var(--color-border));
  box-shadow: var(--shadow-md);
}}

.task-card h2, .todo-card h2 {{
  margin: 0 0 0.35rem;
  font-size: 1.125rem;
  font-weight: 600;
}}

.task-card p, .todo-card p {{
  margin: 0 0 0.75rem;
  color: var(--color-text-muted);
  font-size: 0.875rem;
}}

.list-item.done span, .task-card.completed h2, .todo-card.completed h2,
.completed.task-card h2 {{
  text-decoration: line-through;
  color: var(--color-text-muted);
}}

.task-card.completed, .todo-card.completed, .completed {{
  opacity: 0.85;
  background: color-mix(in srgb, var(--color-bg) 70%, var(--color-surface));
}}

.filters {{
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}}

.filters .btn-ghost.active {{
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}}

.empty {{
  text-align: center;
  color: var(--color-text-muted);
  padding: 2.5rem 1rem;
}}

.error-banner {{
  background: #fef2f2;
  color: #991b1b;
  border: 1px solid #fecaca;
  border-radius: var(--radius-md);
  padding: 0.75rem 1rem;
  margin-bottom: 1rem;
}}

.options {{ display: flex; flex-direction: column; gap: 0.5rem; margin: 1rem 0; }}

.option-row {{
  text-align: left;
  padding: 0.875rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  cursor: pointer;
  font: inherit;
  width: 100%;
}}

.option-row:hover:not(:disabled) {{
  border-color: var(--color-primary);
  background: color-mix(in srgb, var(--color-primary) 8%, var(--color-surface));
}}

.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}}

.metric-card {{
  padding: 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  text-align: center;
  background: var(--color-surface);
}}

.metric-value {{
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--color-primary);
}}

@media (max-width: 640px) {{
  .app-container, .main-content {{ padding: 1rem; }}
  .form-row, .task-form {{ flex-direction: column; align-items: stretch; }}
  .sidebar {{ width: 100%; border-right: none; border-bottom: 1px solid var(--color-border); }}
}}
"""


def generate_app_css(primary: str = "#4f46e5") -> str:
    """Backward-compatible wrapper — returns full SaaS stylesheet."""
    return build_saas_app_css({"colors": {"primary": primary}}, primary=primary)


# ---------------------------------------------------------------------------
# CRUD backend + frontend
# ---------------------------------------------------------------------------

def _entity_class(spec: Dict[str, Any]) -> str:
    return singularize(spec.get("primary_table", "items")).title()


def generate_crud_backend(spec: Dict[str, Any]) -> Dict[str, str]:
    table = spec.get("primary_table", "items")
    entity = _entity_class(spec)
    prefix = spec.get("api_prefix", f"/{table}")
    has_completed = table in ("tasks", "todos", "items") or "task" in table or "todo" in table

    model_extra = "\n    completed = Column(Boolean, default=False, nullable=False)" if has_completed else ""

    models = f'''from sqlalchemy import Column, Integer, String, Boolean
from database import Base


class {entity}(Base):
    __tablename__ = "{table}"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False){model_extra}
'''

    schemas = f'''from pydantic import BaseModel, ConfigDict


class {entity}Base(BaseModel):
    title: str
'''
    if has_completed:
        schemas += f"    completed: bool = False\n"
    schemas += f'''

class {entity}Create({entity}Base):
    pass


class {entity}Update(BaseModel):
    title: str | None = None
'''
    if has_completed:
        schemas += "    completed: bool | None = None\n"
    schemas += f'''

class {entity}Out({entity}Base):
    id: int
    model_config = ConfigDict(from_attributes=True)
'''

    routes = f'''from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import {entity}
from schemas import {entity}Create, {entity}Out, {entity}Update

router = APIRouter(prefix="{prefix}", tags=["{table}"])


@router.get("", response_model=list[{entity}Out])
def list_items(db: Session = Depends(get_db)):
    return db.query({entity}).order_by({entity}.id.desc()).all()


@router.post("", response_model={entity}Out, status_code=201)
def create_item(payload: {entity}Create, db: Session = Depends(get_db)):
    row = {entity}(**payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.patch("/{{item_id}}", response_model={entity}Out)
def update_item(item_id: int, payload: {entity}Update, db: Session = Depends(get_db)):
    row = db.query({entity}).filter({entity}.id == item_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(row, k, v)
    db.commit()
    db.refresh(row)
    return row


@router.delete("/{{item_id}}", status_code=204)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    row = db.query({entity}).filter({entity}.id == item_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(row)
    db.commit()
    return None
'''
    return {
        "backend/models.py": models,
        "backend/schemas.py": schemas,
        "backend/routes.py": routes,
    }


def generate_crud_app_jsx(spec: Dict[str, Any], title: str = "My App") -> str:
    table = spec.get("primary_table", "items")
    # Dynamic runtime API uses collection name: /api/apps/{project}/{table}
    api_path = f"/{table.lstrip('/')}"
    has_completed = table in ("tasks", "todos", "items") or "task" in table

    toggle_block = ""
    filter_block = ""
    if has_completed:
        toggle_block = """
  const toggleItem = async (id) => {
    const item = items.find((i) => i.id === id);
    if (!item) return;
    const next = !item.completed;
    setItems((prev) => prev.map((i) => (i.id === id ? { ...i, completed: next } : i)));
    try {
      await apiFetch(`${API}/${id}`, { method: 'PUT', body: JSON.stringify({ completed: next }) });
    } catch { /* keep local */ }
  };"""
        filter_block = """
          <div className="filters">
            {['all', 'active', 'completed'].map((f) => (
              <button key={f} type="button" className={`btn btn-ghost${filter === f ? ' active' : ''}`} onClick={() => setFilter(f)}>
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>"""
        list_render = """
              {visible.map((item) => (
                <li key={item.id} className={`list-item${item.completed ? ' done' : ''}`}>
                  <input type="checkbox" checked={!!item.completed} onChange={() => toggleItem(item.id)} />
                  <span style={{ flex: 1 }}>{item.title}</span>
                  <button type="button" className="btn btn-ghost" onClick={() => removeItem(item.id)}>Delete</button>
                </li>
              ))}"""
        visible_logic = """
  const visible = (items || []).filter((i) => {
    if (filter === 'active') return !i.completed;
    if (filter === 'completed') return i.completed;
    return true;
  });"""
        extra_state = "const [filter, setFilter] = useState('all');"
    else:
        list_render = """
              {items.map((item) => (
                <li key={item.id} className="list-item">
                  <span style={{ flex: 1 }}>{item.title}</span>
                  <button type="button" className="btn btn-ghost" onClick={() => removeItem(item.id)}>Delete</button>
                </li>
              ))}"""
        visible_logic = "  const visible = items || [];"
        extra_state = ""

    return f'''import {{ useEffect, useState }} from 'react';
import {{ apiFetch }} from './api/client.js';

const API = '{api_path}';
const STORAGE_KEY = '{table}_data';

export default function App() {{
  const [items, setItems] = useState([]);
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  {extra_state}

  useEffect(() => {{
    let cancelled = false;
    (async () => {{
      try {{
        const data = await apiFetch(API);
        if (!cancelled && Array.isArray(data)) setItems(data);
      }} catch {{
        try {{
          const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
          if (!cancelled) setItems(saved);
        }} catch {{ /* empty */ }}
      }} finally {{
        if (!cancelled) setLoading(false);
      }}
    }})();
    return () => {{ cancelled = true; }};
  }}, []);

  useEffect(() => {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  }}, [items]);

  const addItem = async (e) => {{
    e.preventDefault();
    const text = title.trim();
    if (!text) return;
    const optimistic = {{ id: Date.now(), title: text, completed: false }};
    setItems((prev) => [optimistic, ...prev]);
    setTitle('');
    try {{
      const created = await apiFetch(API, {{ method: 'POST', body: JSON.stringify({{ title: text, completed: false }}) }});
      if (created?.id) setItems((prev) => prev.map((i) => (i.id === optimistic.id ? created : i)));
    }} catch {{ /* keep optimistic */ }}
  }};

  const removeItem = async (id) => {{
    setItems((prev) => prev.filter((i) => i.id !== id));
    try {{ await apiFetch(`${{API}}/${{id}}`, {{ method: 'DELETE' }}); }} catch {{ /* removed locally */ }}
  }};
{toggle_block}
{visible_logic}

  return (
    <div className="app"><div className="app-container">
      <header className="app-header">
        <h1 className="app-title">{title}</h1>
        <p className="app-subtitle">Manage your {table}</p>
      </header>
      {{error && <div className="error-banner">{{error}}</div>}}
      <div className="card">
        <form className="form-row" onSubmit={{addItem}}>
          <input className="input" placeholder="Add new..." value={{title}} onChange={{(e) => setTitle(e.target.value)}} />
          <button type="submit" className="btn btn-primary" disabled={{!title.trim()}}>Add</button>
        </form>
        {filter_block}
        {{loading ? <p className="empty">Loading...</p> : visible.length === 0 ? <p className="empty">No items yet.</p> : (
          <ul className="list">{list_render}
          </ul>
        )}}
      </div>
    </div></div>
  );
}}
'''


# ---------------------------------------------------------------------------
# Content / form / dashboard
# ---------------------------------------------------------------------------

def generate_content_app_jsx(spec: Dict[str, Any], title: str = "Generator") -> str:
    return f'''import {{ useState }} from 'react';
import {{ apiFetch }} from './api/client.js';

export default function App() {{
  const [topic, setTopic] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const generate = async (e) => {{
    e.preventDefault();
    const text = topic.trim();
    if (!text) return;
    setLoading(true);
    try {{
      const data = await apiFetch(`${{API}}/generate`, {{ method: 'POST', body: JSON.stringify({{ topic: text }}) }});
      const items = data?.content || data?.results || data?.ideas || [];
      setResults(Array.isArray(items) ? items : [String(items)]);
    }} catch {{
      setResults([`Idea for "${{text}}": Explore creative approaches and unique angles on this topic.`]);
    }} finally {{ setLoading(false); }}
  }};

  return (
    <div className="app"><div className="app-container">
      <header className="app-header"><h1 className="app-title">{title}</h1></header>
      <div className="card">
        <form className="form-row" onSubmit={{generate}}>
          <input className="input" placeholder="Enter topic..." value={{topic}} onChange={{(e) => setTopic(e.target.value)}} />
          <button type="submit" className="btn btn-primary" disabled={{loading || !topic.trim()}}>{{loading ? 'Generating...' : 'Generate'}}</button>
        </form>
        {{results.length > 0 && (
          <ul className="list">{{results.map((r, i) => <li key={{i}} className="list-item">{{typeof r === 'string' ? r : JSON.stringify(r)}}</li>)}}</ul>
        )}}
      </div>
    </div></div>
  );
}}
'''


def generate_form_app_jsx(spec: Dict[str, Any], title: str = "Form App") -> str:
    prefix = spec.get("api_prefix", "/items")
    endpoint = "/translate" if "translat" in spec.get("requirement", "").lower() else "/process"
    return f'''import {{ useState }} from 'react';
import {{ apiFetch }} from './api/client.js';

const API = '{prefix}';

export default function App() {{
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {{
    e.preventDefault();
    if (!input.trim()) return;
    setLoading(true);
    try {{
      const data = await apiFetch(`${{API}}{endpoint}`, {{ method: 'POST', body: JSON.stringify({{ text: input }}) }});
      setOutput(data?.result || data?.output || data?.translation || JSON.stringify(data));
    }} catch {{
      setOutput(`Processed: ${{input}}`);
    }} finally {{ setLoading(false); }}
  }};

  return (
    <div className="app"><div className="app-container">
      <header className="app-header"><h1 className="app-title">{title}</h1></header>
      <div className="card">
        <form onSubmit={{submit}}>
          <textarea className="textarea" placeholder="Enter text..." value={{input}} onChange={{(e) => setInput(e.target.value)}} />
          <div style={{{{ marginTop: '1rem' }}}}>
            <button type="submit" className="btn btn-primary" disabled={{loading}}>{{loading ? 'Processing...' : 'Submit'}}</button>
          </div>
        </form>
        {{output && <div className="results-box">{{output}}</div>}}
      </div>
    </div></div>
  );
}}
'''


def generate_dashboard_app_jsx(spec: Dict[str, Any], title: str = "Dashboard") -> str:
    prefix = spec.get("api_prefix", "/items")
    return f'''import {{ useEffect, useState }} from 'react';
import {{ apiFetch }} from './api/client.js';

const API = '{prefix}';

export default function App() {{
  const [stats, setStats] = useState({{ total: 0, active: 0 }});
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {{
    (async () => {{
      try {{
        const [s, list] = await Promise.all([
          apiFetch(`${{API}}/stats`).catch(() => null),
          apiFetch(API).catch(() => []),
        ]);
        if (s) setStats(s);
        else if (Array.isArray(list)) setStats({{ total: list.length, active: list.filter(i => !i.completed).length }});
        if (Array.isArray(list)) setItems(list.slice(0, 5));
      }} finally {{ setLoading(false); }}
    }})();
  }}, []);

  return (
    <div className="app"><div className="app-container">
      <header className="app-header"><h1 className="app-title">{title}</h1></header>
      {{loading ? <p className="empty">Loading...</p> : (
        <>
          <div className="metrics">
            <div className="metric-card"><div className="metric-value">{{stats.total ?? 0}}</div><div>Total</div></div>
            <div className="metric-card"><div className="metric-value">{{stats.active ?? 0}}</div><div>Active</div></div>
          </div>
          <div className="card"><h3>Recent items</h3>
            <ul className="list">{{items.map((i) => <li key={{i.id}} className="list-item">{{i.title || i.name || JSON.stringify(i)}}</li>)}}</ul>
          </div>
        </>
      )}}
    </div></div>
  );
}}
'''


def generate_service_backend(spec: Dict[str, Any]) -> Dict[str, str]:
    """Minimal backend for content/form/dashboard apps."""
    kind = spec.get("app_kind", "custom")
    table = spec.get("primary_table", "items")
    entity = _entity_class(spec)
    prefix = spec.get("api_prefix", f"/{table}")

    base_crud = generate_crud_backend(spec)

    extra_routes = ""
    if kind == "content":
        extra_routes = '''

@router.post("/generate")
def generate_content(payload: dict):
    topic = (payload or {}).get("topic", "something")
    return {"content": [f"Idea 1 for {topic}", f"Idea 2 for {topic}", f"Idea 3 for {topic}"]}
'''
    elif kind == "form":
        extra_routes = '''

@router.post("/process")
def process_text(payload: dict):
    text = (payload or {}).get("text", "")
    return {"result": f"Processed: {text}"}


@router.post("/translate")
def translate_text(payload: dict):
    text = (payload or {}).get("text", "")
    return {"translation": f"[Translated] {text}"}
'''
    elif kind == "dashboard":
        extra_routes = f'''

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query({entity}).count()
    return {{"total": total, "active": total}}
'''

    routes = base_crud["backend/routes.py"]
    if extra_routes:
        routes = routes.rstrip() + "\n" + extra_routes

    return {**base_crud, "backend/routes.py": routes}


def generate_minimal_backend_stubs() -> Dict[str, str]:
    return {
        "backend/models.py": "from database import Base\n",
        "backend/schemas.py": "from pydantic import BaseModel\n",
        "backend/routes.py": '''from fastapi import APIRouter
router = APIRouter(tags=["app"])
@router.get("/info")
def app_info():
    return {"status": "ok"}
''',
    }


def generate_quiz_app_jsx(uiux: str = "", prd: str = "", title: str = "Quiz") -> str:
    """API-backed quiz fallback: embedded MCQ UI plus backend health ping for validation."""
    base = generate_static_quiz_app_jsx(uiux, prd, title=title)
    return base.replace(
        "import { useState } from 'react';",
        "import { useState, useEffect } from 'react';\nimport { apiFetch } from './api/client.js';",
        1,
    ).replace(
        "export default function App() {",
        "export default function App() {\n  useEffect(() => { apiFetch('/info').catch(() => {}); }, []);",
        1,
    )


def _stack_for_kind(spec: Dict[str, Any], uiux: str, prd: str) -> Dict[str, str]:
    kind = spec.get("app_kind", "custom")
    title = spec.get("title", "My App")
    css = generate_app_css()
    files: Dict[str, str] = {"frontend/src/styles/app.css": css}

    if kind == "static_quiz":
        files["frontend/src/App.jsx"] = generate_static_quiz_app_jsx(uiux, prd, title=title)
        files.update(generate_minimal_backend_stubs())
        return files

    if kind in ("static",) or (spec.get("frontend_only") and kind != "static_quiz"):
        files["frontend/src/App.jsx"] = generate_static_quiz_app_jsx(uiux, prd, title=title)
        files.update(generate_minimal_backend_stubs())
        return files

    if kind == "quiz":
        files["frontend/src/App.jsx"] = generate_quiz_app_jsx(uiux, prd, title=title)
        files.update(generate_crud_backend(spec))
        return files

    if kind == "content":
        files["frontend/src/App.jsx"] = generate_content_app_jsx(spec, title)
        files.update(generate_service_backend(spec))
        return files

    if kind == "form":
        files["frontend/src/App.jsx"] = generate_form_app_jsx(spec, title)
        files.update(generate_service_backend(spec))
        return files

    if kind == "dashboard":
        files["frontend/src/App.jsx"] = generate_dashboard_app_jsx(spec, title)
        files.update(generate_service_backend(spec))
        return files

    # crud, quiz (API), custom → CRUD stack
    files["frontend/src/App.jsx"] = generate_crud_app_jsx(spec, title)
    files.update(generate_crud_backend(spec))
    return files


def apply_deterministic_fallback(
    files: Dict[str, str],
    app_spec: Dict[str, Any],
    uiux: str = "",
    prd: str = "",
) -> Dict[str, str]:
    """Apply built-in generators for any supported app kind."""
    generated = _stack_for_kind(app_spec, uiux, prd)
    out = dict(files)
    out.update(generated)
    return out


# Backward-compat alias
generate_static_quiz_css = generate_app_css
