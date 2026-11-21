# Akkio — App Builder (monorepo)

Single GitHub repository for the **FastAPI backend** and **App Builder UI**.

| Path | Role |
|------|------|
| `/` | FastAPI (`main.py`, `api/`, `app_builder/`) |
| `akkio-frontend/` | React + Vite UI (port **3002**) |

Deploy and VPS notes: [docs/HOSTINGER.md](docs/HOSTINGER.md).

## Local development

### Backend

```bash
cd akkio-fastapi   # repo root
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # if present
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd akkio-frontend
cp .env.example .env
npm install
npm start            # Vite dev server on PORT from .env (default 3002)
```

Set `VITE_BASE_URL` in `akkio-frontend/.env` to your API origin (e.g. `http://localhost:8000`).

## Clone on VPS (one repo)

```bash
git clone https://github.com/sainth-stack/akkio-fastapi.git
cd akkio-fastapi
# backend + akkio-frontend/ are in this tree — no separate akkio-frontend repo
```