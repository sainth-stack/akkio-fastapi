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


def generate_questions_json(uiux: str = "", prd: str = "") -> str:
    questions = _parse_static_questions(uiux, prd) or _DEFAULT_QUESTIONS
    return json.dumps({"questions": questions}, indent=2)


def generate_static_quiz_app_jsx(uiux: str = "", prd: str = "", title: str = "Quiz") -> str:
    return f'''import {{ useState }} from 'react';
import questionsData from './data/questions.json';

const QUESTIONS = questionsData.questions || [];

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


def _soften_page_bg(bg: str) -> str:
    """Pure white page backgrounds look flat — use a soft slate canvas."""
    normalized = (bg or "").strip().upper().replace(" ", "")
    if normalized in ("#FFFFFF", "#FFF", "WHITE", "#FEFEFE"):
        return "#f1f5f9"
    return bg


def build_saas_app_css(tokens: Dict[str, Any] | None = None, primary: str = "#4f46e5") -> str:
    """
    Premium SaaS stylesheet — injected for every generated app.
    Works with standard classes AND bare form-row/header markup from LLM output.
    """
    colors = (tokens or {}).get("colors") if isinstance(tokens, dict) else {}
    if not isinstance(colors, dict):
        colors = tokens if isinstance(tokens, dict) else {}

    primary = str(colors.get("primary") or primary)
    bg = _soften_page_bg(str(colors.get("background") or colors.get("bg") or "#f1f5f9"))
    surface = str(colors.get("surface") or "#ffffff")
    text = str(colors.get("text") or "#0f172a")
    muted = str(colors.get("muted") or "#64748b")
    border = str(colors.get("border") or "#e2e8f0")
    danger = str(colors.get("danger") or "#ef4444")
    hover = _primary_hover(primary)
    font = "'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
    if isinstance(tokens, dict) and isinstance(tokens.get("typography"), dict):
        font = tokens["typography"].get("fontFamily") or font
    max_w = "56rem"
    if isinstance(tokens, dict) and isinstance(tokens.get("layout"), dict):
        max_w = tokens["layout"].get("maxContentWidth") or max_w

    return f"""/* Akkio Premium SaaS design system */
:root {{
  --color-primary: {primary};
  --color-primary-hover: {hover};
  --color-primary-soft: color-mix(in srgb, {primary} 12%, #ffffff);
  --color-bg: {bg};
  --color-bg-start: color-mix(in srgb, {primary} 6%, {bg});
  --color-background: {bg};
  --color-surface: {surface};
  --color-text: {text};
  --color-text-muted: {muted};
  --color-muted: {muted};
  --color-border: {border};
  --color-danger: {danger};
  --font-sans: {font};
  --radius-sm: 0.375rem;
  --radius-md: 0.625rem;
  --radius-lg: 0.875rem;
  --radius-xl: 1.125rem;
  --shadow-sm: 0 1px 2px rgb(15 23 42 / 0.05);
  --shadow-md: 0 4px 12px rgb(15 23 42 / 0.08), 0 2px 4px rgb(15 23 42 / 0.04);
  --shadow-lg: 0 12px 32px rgb(15 23 42 / 0.1), 0 4px 12px rgb(15 23 42 / 0.06);
  --shadow-xl: 0 20px 48px rgb(15 23 42 / 0.12);
  --max-content-width: {max_w};
}}

*, *::before, *::after {{ box-sizing: border-box; -webkit-font-smoothing: antialiased; }}

html, body {{
  margin: 0;
  padding: 0;
  min-height: 100%;
  font-family: var(--font-sans);
  color: var(--color-text);
  line-height: 1.55;
  background: var(--color-bg);
}}

#root {{ min-height: 100vh; }}

/* ─── Page shell ─────────────────────────────────────────────────────── */
.app {{
  min-height: 100vh;
  background:
    radial-gradient(ellipse 90% 60% at 50% -10%, color-mix(in srgb, var(--color-primary) 14%, transparent), transparent 55%),
    linear-gradient(180deg, var(--color-bg-start) 0%, var(--color-bg) 35%, var(--color-bg) 100%);
  color: var(--color-text);
}}

.app-container {{
  max-width: var(--max-content-width);
  margin: 0 auto;
  padding: 0 1.25rem 3rem;
}}

/* ─── Hero header ────────────────────────────────────────────────────── */
.app-header, .app > header, header.navbar {{
  text-align: center;
  padding: 2.75rem 1rem 1.75rem;
  margin-bottom: 0.25rem;
}}

.app-title, .app-header h1, .app > header h1, .navbar h1, .app > h1:first-child {{
  font-size: clamp(1.875rem, 4vw, 2.625rem);
  font-weight: 800;
  letter-spacing: -0.035em;
  line-height: 1.15;
  margin: 0 0 0.625rem;
  color: var(--color-text);
  background: linear-gradient(135deg, var(--color-text) 20%, color-mix(in srgb, var(--color-primary) 55%, var(--color-text)) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}

.app-subtitle, .app-header p, .app > header p {{
  font-size: 1.0625rem;
  color: var(--color-text-muted);
  margin: 0 auto;
  max-width: 36rem;
  line-height: 1.6;
}}

/* ─── Elevated panels (card + auto-wrap bare forms) ────────────────────── */
.card,
.app-container > .form-row,
.app-container > form,
.app > .form-row,
.app > form:not(.card form),
.app .task-form,
.app .todo-form,
.main-content > .form-row,
.main-content > .card {{
  background: var(--color-surface);
  border: 1px solid color-mix(in srgb, var(--color-border) 75%, var(--color-primary) 12%);
  border-radius: var(--radius-xl);
  padding: 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-lg);
  backdrop-filter: blur(8px);
}}

.card .form-row, .card form.form-row {{
  background: transparent;
  border: none;
  box-shadow: none;
  padding: 0;
  margin-bottom: 1.25rem;
  backdrop-filter: none;
}}

/* ─── Forms & inputs ─────────────────────────────────────────────────── */
.form-row, .task-form, .todo-form {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.875rem;
  align-items: stretch;
}}

.input, .textarea,
.form-row input, .form-row select, .form-row textarea,
.task-form input, .todo-form input,
.card input, .card select, .card textarea,
.app input[type="text"], .app input[type="email"],
.app input[type="password"], .app input[type="search"],
.app input[type="date"], .app input[type="datetime-local"],
.app input[type="number"], .app input:not([type]),
.app select, .app textarea {{
  flex: 1;
  min-width: 150px;
  min-height: 44px;
  padding: 0.6875rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font: inherit;
  font-size: 0.9375rem;
  background: color-mix(in srgb, var(--color-surface) 92%, var(--color-bg) 8%);
  color: var(--color-text);
  transition: border-color 0.18s, box-shadow 0.18s, background 0.18s;
}}

.textarea {{ min-height: 120px; resize: vertical; width: 100%; }}

.input:focus, .textarea:focus,
.form-row input:focus, .card input:focus,
.app input:focus, .app select:focus, .app textarea:focus {{
  outline: none;
  border-color: var(--color-primary);
  background: var(--color-surface);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--color-primary) 18%, transparent);
}}

.input::placeholder, .app input::placeholder {{ color: #94a3b8; }}

/* ─── Buttons ────────────────────────────────────────────────────────── */
.btn,
.app button,
.form-row button, .card button,
.task-form button, .task-card button {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.375rem;
  min-height: 44px;
  padding: 0.6875rem 1.375rem;
  border-radius: var(--radius-md);
  font: inherit;
  font-size: 0.9375rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  white-space: nowrap;
  transition: transform 0.15s, box-shadow 0.15s, filter 0.15s, background 0.15s;
}}

.btn-primary,
button[type="submit"],
.form-row button[type="submit"],
.card button.btn-primary,
.app button.btn-primary,
.app .form-row button:last-child:not(.btn-ghost),
.task-form button:first-of-type {{
  background: linear-gradient(135deg, var(--color-primary) 0%, color-mix(in srgb, var(--color-primary) 75%, #1e1b4b) 100%);
  color: #fff;
  box-shadow: 0 2px 8px color-mix(in srgb, var(--color-primary) 35%, transparent), var(--shadow-sm);
}}

.btn-primary:hover:not(:disabled),
button[type="submit"]:hover:not(:disabled),
.form-row button[type="submit"]:hover:not(:disabled),
.app button:hover:not(:disabled):not(.btn-ghost) {{
  filter: brightness(1.06);
  transform: translateY(-1px);
  box-shadow: 0 6px 20px color-mix(in srgb, var(--color-primary) 32%, transparent);
}}

.btn-primary:active:not(:disabled),
.app button:active:not(:disabled) {{
  transform: translateY(0);
}}

.btn-primary:disabled, .btn:disabled, .app button:disabled {{
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}}

.btn-ghost {{
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
}}

.btn-ghost:hover:not(:disabled) {{
  border-color: var(--color-primary);
  color: var(--color-primary);
  background: var(--color-primary-soft);
}}

.btn-danger {{
  background: linear-gradient(135deg, var(--color-danger), #b91c1c);
  color: #fff;
}}

/* ─── Lists & data rows ──────────────────────────────────────────────── */
.list, .task-list, .todo-list {{
  list-style: none;
  padding: 0;
  margin: 1rem 0 0;
  display: flex;
  flex-direction: column;
  gap: 0.625rem;
}}

.list-item, .task-card, .todo-card {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.875rem;
  padding: 1rem 1.25rem;
  background: color-mix(in srgb, var(--color-surface) 96%, var(--color-bg) 4%);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  transition: border-color 0.18s, box-shadow 0.18s, transform 0.18s;
}}

.list-item:hover, .task-card:hover, .todo-card:hover {{
  border-color: color-mix(in srgb, var(--color-primary) 35%, var(--color-border));
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}}

.task-card, .todo-card {{ flex-direction: column; align-items: stretch; }}

.task-card h2, .todo-card h2, .list-item strong {{
  margin: 0 0 0.35rem;
  font-size: 1.0625rem;
  font-weight: 600;
  color: var(--color-text);
}}

.task-card p, .todo-card p {{
  margin: 0 0 0.75rem;
  color: var(--color-text-muted);
  font-size: 0.875rem;
}}

.list-item.done span, .completed h2, .task-card.completed h2 {{
  text-decoration: line-through;
  color: var(--color-text-muted);
}}

.task-card.completed, .todo-card.completed, .completed {{
  opacity: 0.88;
  background: color-mix(in srgb, var(--color-bg) 50%, var(--color-surface));
}}

/* ─── Layout helpers ─────────────────────────────────────────────────── */
.navbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem 1.5rem;
  background: color-mix(in srgb, var(--color-surface) 88%, transparent);
  border-bottom: 1px solid var(--color-border);
  box-shadow: var(--shadow-sm);
  backdrop-filter: blur(12px);
  text-align: left;
}}

.sidebar {{
  width: 240px;
  min-height: 200px;
  padding: 1.25rem;
  background: color-mix(in srgb, var(--color-surface) 90%, var(--color-bg));
  border-right: 1px solid var(--color-border);
}}

.main-content {{
  flex: 1;
  padding: 1.5rem;
  max-width: var(--max-content-width);
  margin: 0 auto;
  width: 100%;
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
  font-size: 0.9375rem;
}}

.error-banner {{
  background: linear-gradient(135deg, #fef2f2, #fff1f2);
  color: #991b1b;
  border: 1px solid #fecaca;
  border-radius: var(--radius-md);
  padding: 0.875rem 1.125rem;
  margin-bottom: 1rem;
  font-size: 0.9375rem;
}}

.success-banner {{
  background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
  color: #166534;
  border: 1px solid #bbf7d0;
  border-radius: var(--radius-md);
  padding: 0.875rem 1.125rem;
  margin-bottom: 1rem;
}}

.options {{ display: flex; flex-direction: column; gap: 0.625rem; margin: 1rem 0; }}

.option-row {{
  text-align: left;
  padding: 1rem 1.125rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  cursor: pointer;
  font: inherit;
  width: 100%;
  transition: all 0.18s;
}}

.option-row:hover:not(:disabled) {{
  border-color: var(--color-primary);
  background: var(--color-primary-soft);
  box-shadow: var(--shadow-sm);
}}

.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}}

.metric-card {{
  padding: 1.25rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  text-align: center;
  background: var(--color-surface);
  box-shadow: var(--shadow-sm);
}}

.metric-value {{
  font-size: 1.625rem;
  font-weight: 800;
  color: var(--color-primary);
  letter-spacing: -0.02em;
}}

.question-text {{ font-size: 1.125rem; font-weight: 600; margin: 0 0 1rem; }}

/* ─── LLM results / markdown prose ───────────────────────────────────── */
.form-col {{
  display: flex;
  flex-direction: column;
  gap: 1rem;
}}

.form-actions {{
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  align-items: center;
}}

.error-text {{
  color: var(--color-danger);
  font-size: 0.875rem;
  margin-top: 0.75rem;
  padding: 0.75rem 1rem;
  background: color-mix(in srgb, var(--color-danger) 8%, var(--color-surface));
  border-radius: var(--radius-md);
  border: 1px solid color-mix(in srgb, var(--color-danger) 25%, transparent);
}}

.results-box, .results-prose {{
  margin-top: 1.5rem;
  padding: 1.5rem 1.75rem;
  background: linear-gradient(180deg, var(--color-surface) 0%, color-mix(in srgb, var(--color-surface) 96%, var(--color-primary) 4%) 100%);
  border: 1px solid color-mix(in srgb, var(--color-border) 80%, var(--color-primary) 15%);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
  line-height: 1.75;
  color: var(--color-text);
  max-height: min(70vh, 720px);
  overflow-y: auto;
  text-align: left;
}}

.results-prose h2 {{
  font-size: 1.375rem;
  font-weight: 700;
  margin: 1.5rem 0 0.75rem;
  color: var(--color-text);
  letter-spacing: -0.02em;
}}

.results-prose h3 {{
  font-size: 1.125rem;
  font-weight: 700;
  margin: 1.25rem 0 0.5rem;
  color: var(--color-primary);
}}

.results-prose h4 {{
  font-size: 1rem;
  font-weight: 600;
  margin: 1rem 0 0.5rem;
  color: var(--color-text);
}}

.results-prose p {{
  margin: 0 0 0.875rem;
  color: var(--color-text-muted);
  font-size: 0.9375rem;
}}

.results-prose ul {{
  margin: 0.5rem 0 1rem;
  padding-left: 1.25rem;
  color: var(--color-text-muted);
}}

.results-prose li {{
  margin-bottom: 0.375rem;
  font-size: 0.9375rem;
}}

.results-prose strong {{
  color: var(--color-text);
  font-weight: 600;
}}

.results-ideas {{
  margin-top: 1.5rem;
}}

/* Center content when LLM omits app-container wrapper */
.app:not(:has(.app-container)) > .app-header,
.app:not(:has(.app-container)) > header,
.app:not(:has(.app-container)) > .form-row,
.app:not(:has(.app-container)) > form,
.app:not(:has(.app-container)) > .card,
.app:not(:has(.app-container)) > .list,
.app:not(:has(.app-container)) > ul {{
  max-width: var(--max-content-width);
  margin-left: auto;
  margin-right: auto;
  width: calc(100% - 2.5rem);
}}

@media (max-width: 768px) {{
  .app-container {{ padding: 0 1rem 2rem; }}
  .app-header {{ padding: 2rem 0.5rem 1.25rem; }}
  .form-row, .task-form, .card {{ padding: 1.25rem; }}
  .form-row, .task-form {{ flex-direction: column; }}
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
# LLM / content / form / dashboard
# ---------------------------------------------------------------------------

def generate_llm_app_jsx(spec: Dict[str, Any], title: str = "AI App") -> str:
    """LLM app UI — calls platform proxy at POST /generate (OpenAI via Akkio backend)."""
    gen_type = spec.get("gen_type") or "general"
    input_mode = spec.get("llm_input_mode") or ("textarea" if gen_type in ("summarize", "translate") else "input")
    is_textarea = input_mode == "textarea"
    placeholder = {
        "summarize": "Paste text to summarize...",
        "translate": "Enter text to translate...",
        "travel": "Destination or travel interests (e.g. 5 days in Tokyo)...",
        "linkedin_post": "Topic for your LinkedIn post...",
        "ideas": "Enter a topic to brainstorm ideas...",
    }.get(gen_type, "Enter your prompt...")
    btn_label = {
        "summarize": "Summarize",
        "translate": "Translate",
        "travel": "Plan Trip",
        "linkedin_post": "Generate Post",
        "ideas": "Generate Ideas",
    }.get(gen_type, "Generate")
    subtitle = {
        "summarize": "AI-powered text summarization",
        "translate": "AI translation powered by OpenAI",
        "travel": "AI travel planner — get itinerary ideas",
        "linkedin_post": "Generate professional LinkedIn posts",
        "ideas": "Brainstorm creative ideas instantly",
    }.get(gen_type, "Powered by AI")

    input_block = (
        f'''<textarea className="textarea" rows={{6}} placeholder="{placeholder}" value={{input}} onChange={{(e) => setInput(e.target.value)}} aria-label="User input" />'''
        if is_textarea else
        f'''<input className="input" placeholder="{placeholder}" value={{input}} onChange={{(e) => setInput(e.target.value)}} aria-label="User input" />'''
    )
    form_class = "form-col" if is_textarea else "form-row"

    return f'''import {{ useState }} from 'react';
import {{ apiFetch }} from './api/client.js';

const GEN_TYPE = '{gen_type}';

function escapeHtml(s) {{
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

function inlineFormat(s) {{
  return escapeHtml(s).replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
}}

/** Lightweight markdown → HTML for travel plans, summaries, posts */
function formatMarkdown(text) {{
  if (!text) return '';
  const lines = String(text).split('\\n');
  let html = '';
  let inList = false;
  for (const line of lines) {{
    const t = line.trim();
    if (!t) {{
      if (inList) {{ html += '</ul>'; inList = false; }}
      continue;
    }}
    if (t.startsWith('#### ')) {{
      if (inList) {{ html += '</ul>'; inList = false; }}
      html += `<h4>${{inlineFormat(t.slice(5))}}</h4>`;
    }} else if (t.startsWith('### ')) {{
      if (inList) {{ html += '</ul>'; inList = false; }}
      html += `<h3>${{inlineFormat(t.slice(4))}}</h3>`;
    }} else if (t.startsWith('## ')) {{
      if (inList) {{ html += '</ul>'; inList = false; }}
      html += `<h2>${{inlineFormat(t.slice(3))}}</h2>`;
    }} else if (t.startsWith('- ')) {{
      if (!inList) {{ html += '<ul>'; inList = true; }}
      html += `<li>${{inlineFormat(t.slice(2))}}</li>`;
    }} else {{
      if (inList) {{ html += '</ul>'; inList = false; }}
      html += `<p>${{inlineFormat(t)}}</p>`;
    }}
  }}
  if (inList) html += '</ul>';
  return html;
}}

export default function App() {{
  const [input, setInput] = useState('');
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generate = async (e) => {{
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    setLoading(true);
    setError('');
    setResults([]);
    setSummary('');
    try {{
      const data = await apiFetch('/generate', {{
        method: 'POST',
        body: JSON.stringify({{ topic: text, text, gen_type: GEN_TYPE, count: 5 }}),
      }});
      const items = data?.content || [];
      const raw = data?.raw_text || '';
      const fullText = raw || (Array.isArray(items) ? items.join('\\n\\n') : String(items || ''));

      // Short numbered ideas → list; long-form (travel, summarize, posts) → prose
      const useList = GEN_TYPE === 'ideas'
        && Array.isArray(items)
        && items.length > 1
        && items.every((i) => typeof i === 'string' && i.length < 280);

      if (useList) {{
        setResults(items);
      }} else {{
        setSummary(fullText);
      }}
    }} catch (err) {{
      setError(err?.message || 'Generation failed. Try again.');
    }} finally {{
      setLoading(false);
    }}
  }};

  const clearAll = () => {{ setInput(''); setResults([]); setSummary(''); setError(''); }};

  return (
    <div className="app"><div className="app-container">
      <header className="app-header">
        <h1 className="app-title">{title}</h1>
        <p className="app-subtitle">{subtitle}</p>
      </header>
      <div className="card">
        <form className="{form_class}" onSubmit={{generate}}>
          {input_block}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={{loading || !input.trim()}}>
              {{loading ? 'Working...' : '{btn_label}'}}
            </button>
            {{(summary || results.length > 0) && (
              <button type="button" className="btn btn-ghost" onClick={{clearAll}}>Clear</button>
            )}}
          </div>
        </form>
        {{error && <p className="error-text">{{error}}</p>}}
        {{summary && (
          <div
            className="results-box results-prose"
            dangerouslySetInnerHTML={{{{ __html: formatMarkdown(summary) }}}}
          />
        )}}
        {{results.length > 0 && (
          <ul className="list results-ideas">
            {{results.map((r, i) => (
              <li key={{i}} className="list-item">{{typeof r === 'string' ? r : JSON.stringify(r)}}</li>
            ))}}
          </ul>
        )}}
      </div>
    </div></div>
  );
}}
'''


def generate_content_app_jsx(spec: Dict[str, Any], title: str = "Generator") -> str:
    return generate_llm_app_jsx(spec, title=title)


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
        files["frontend/src/data/questions.json"] = generate_questions_json(uiux, prd)
        files["frontend/src/App.jsx"] = generate_static_quiz_app_jsx(uiux, prd, title=title)
        files.update(generate_minimal_backend_stubs())
        return files

    if kind == "llm":
        files["frontend/src/App.jsx"] = generate_llm_app_jsx(spec, title)
        files.update(generate_minimal_backend_stubs())
        return files

    if kind in ("static",) or (spec.get("frontend_only") and kind not in ("static_quiz", "llm")):
        files["frontend/src/data/questions.json"] = generate_questions_json(uiux, prd)
        files["frontend/src/App.jsx"] = generate_static_quiz_app_jsx(uiux, prd, title=title)
        files.update(generate_minimal_backend_stubs())
        return files

    if kind == "quiz":
        files["frontend/src/App.jsx"] = generate_quiz_app_jsx(uiux, prd, title=title)
        files.update(generate_crud_backend(spec))
        return files

    if kind == "content":
        files["frontend/src/App.jsx"] = generate_content_app_jsx(spec, title)
        files.update(generate_minimal_backend_stubs())
        return files

    if kind == "form":
        llm_spec = {**spec, "gen_type": spec.get("gen_type") or "translate", "llm_input_mode": "textarea"}
        files["frontend/src/App.jsx"] = generate_llm_app_jsx(llm_spec, title)
        files.update(generate_minimal_backend_stubs())
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
