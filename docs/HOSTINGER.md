# Hostinger VPS Setup — App Builder POC

Direct **HTTP on IP + ports** only. No Nginx, no Docker, no EC2 SSH deploy.

## URLs

| Service | URL |
|---------|-----|
| App Builder UI | `http://YOUR_IP:3002` |
| FastAPI (API + WebSocket + preview) | `http://YOUR_IP:8000` |
| Generated app preview | `http://YOUR_IP:8000/app/{project_name}` |
| Generated app API | `http://YOUR_IP:8000/api/apps/{project_name}` |

WebSockets connect directly to `ws://YOUR_IP:8000/api/codegen/...` (no reverse proxy).

## Prerequisites

- **Node.js 20+** (`node -v`)
- **Python 3.9+** with project venv
- **PostgreSQL** (Neon or local)
- **pm2** or **systemd** for process restart

## Firewall

Allow inbound TCP:

- **3002** — React frontend
- **8000** — FastAPI

## Environment

### Backend (`akkio-fastapi/.env`)

```bash
ENV=production
HOST=0.0.0.0
PORT=8000
PUBLIC_BASE_URL=http://YOUR_IP:8000
APP_BUILDER_RUNTIME_DIR=/var/akkio/runtime/projects
CORS_ORIGINS=http://YOUR_IP:3002
OPENAI_API_KEY=...
PGHOST=...
JWT_SECRET=...
NPM_INSTALL_TIMEOUT=300
NPM_BUILD_TIMEOUT=600
DEPLOY_RUN_TESTS=false
```

Deploy runs **locally on this VPS** (no SSH). Flow: Run App (build) → Deploy tab → health check at `PUBLIC_BASE_URL/app/{project}`.

Do **not** set `DEPLOY_EC2_IP` / `DEPLOY_SSH_KEY_PATH`.

### Frontend (`akkio-frontend/.env`)

```bash
REACT_APP_BASE_URL=http://YOUR_IP:8000
PORT=3002
HOST=0.0.0.0
```

`HOST=0.0.0.0` is required so the UI is reachable via IP, not only localhost.

## Run services

### FastAPI

```bash
cd akkio-fastapi
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### React (POC dev server)

```bash
cd akkio-frontend
npm start
```

### pm2 example

```bash
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name akkio-api --cwd /path/to/akkio-fastapi
pm2 start npm --name akkio-ui --cwd /path/to/akkio-frontend -- start
pm2 save
```

## Runtime disk

Generated projects are stored under `APP_BUILDER_RUNTIME_DIR` (default: `app_builder/runtime/projects` in dev).

On Hostinger use a persistent path, e.g. `/var/akkio/runtime/projects`, not `/tmp`.

## Admin login

Run once after DB is configured:

```bash
cd akkio-fastapi
./venv/bin/python scripts/seed_admin.py
```

Default: `admin@gmail.com` / `Admin@123` (override via `SEED_ADMIN_EMAIL`, `SEED_ADMIN_PASSWORD` in `.env`).

## Smoke test

1. Open `http://YOUR_IP:3002` → login
2. Create app → PRD → UI/UX → Architecture → **Generate Code**
3. **Run App** → preview at `http://YOUR_IP:8000/app/{project_name}`
4. **Deploy to Hostinger** → live URL after health check

## End-to-end validation checklist

Run manually on Hostinger (document pass/fail):

| # | Test | Expected |
|---|------|----------|
| 1 | Prompt: **warehouse parts inventory** (not todo/ideas keywords) | Custom entities in PRD/architecture, not template app |
| 2 | PRD → UI/UX → Architecture | Named entities + color palette in saved sections |
| 3 | Generate Code | Entity names appear in `backend/` and `frontend/` files |
| 4 | Run App | Preview at `http://IP:8000/app/{project}`; CRUD works at `/api/apps/{project}` |
| 5 | 3 users in parallel | User A cannot access User B projects (403 on `/api/app-builder/*` and `/api/apps/*`) |
| 6 | Refresh mid-codegen | Pipeline status restores from DB; job poll shows progress |
| 7 | Deploy | `BUILD_SUCCESS` required; live URL after `RUNNING` health check |

No Docker, Redis, Nginx, or per-app ports required.
