# Akkio

Monorepo for **App Builder**: FastAPI backend + Vite React UI.

```
akkio-fastapi/          ← GitHub repo (clone root)
├── backend/            ← Python / FastAPI (port 8000)
├── frontend/           ← React + Vite (port 3002)
├── docs/               ← VPS / deploy notes
└── README.md
```

Deploy: [docs/HOSTINGER.md](docs/HOSTINGER.md).

## Local development

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # edit MongoDB, keys, PUBLIC_BASE_URL
python run_server.py    # or: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
cp .env.example .env    # VITE_BASE_URL=http://localhost:8000
npm install
npm start
```

## VPS (one clone)

```bash
git clone https://github.com/sainth-stack/akkio-fastapi.git
cd akkio-fastapi
git pull origin only-ai-models

# API
cd backend && source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000

# UI
cd ../frontend && npm install && npm start
```

pm2 example:

```bash
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name akkio-api --cwd /path/to/akkio-fastapi/backend
pm2 start npm --name akkio-ui --cwd /path/to/akkio-fastapi/frontend -- start
```