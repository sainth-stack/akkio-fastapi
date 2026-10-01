# Generated Fullstack Application

Production-ready React + TypeScript + Vite + Material UI frontend with FastAPI + PostgreSQL backend.

## Stack

- Frontend: React, TypeScript, Vite, Material UI, React Router, TanStack Query, Recharts
- Backend: Python, FastAPI, Pydantic, SQLAlchemy, PostgreSQL, JWT
- Run locally: `docker compose up --build`

## Local development

```bash
cp .env.example .env
docker compose up --build
```

- UI: http://localhost:5173
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Frontend only (with mock fallback)

```bash
cd frontend
npm install
npm run dev
```

If the backend is down, every screen still works using `src/api/mock.ts`.

### Backend only

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
uvicorn main:app --reload --port 8000
```

Set `DATABASE_URL` to PostgreSQL for production. SQLite is used only when `DATABASE_URL` is unset (preview fallback).

## Seed

```bash
cd backend
python seed.py
```
