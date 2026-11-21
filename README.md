# Akkio

**One git repo** — FastAPI backend + Vite App Builder UI.

```
akkio/                  ← clone root (repo: sainth-stack/akkio-fastapi)
├── backend/            ← Python / FastAPI (port 8000)
├── frontend/           ← React + Vite (port 3002)
├── docs/               ← VPS / deploy
└── README.md
```

Deploy: [docs/HOSTINGER.md](docs/HOSTINGER.md).

## Local development

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_server.py
```

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm start
```

## Clone (VPS or laptop)

```bash
git clone https://github.com/sainth-stack/akkio-fastapi.git akkio
cd akkio
git checkout only-ai-models
```

pm2:

```bash
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name akkio-api --cwd /path/to/akkio/backend
pm2 start npm --name akkio-ui --cwd /path/to/akkio/frontend -- start
```

## What belongs at repo root

Only **`backend/`**, **`frontend/`**, **`docs/`**, and repo metadata. No extra nested `akkio-fastapi/` folder, no root `venv/` (use `backend/venv` only).