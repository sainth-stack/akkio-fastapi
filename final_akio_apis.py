"""
Akkio FastAPI — minimal surface for Multi-Agent, App Builder, and health checks.
Legacy BI/SharePoint/automl endpoints were removed; use git history if needed.
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from database import PostgresDatabase

app = FastAPI(title="Akkio", version="2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
async def _on_startup():
    try:
        PostgresDatabase._ensure_pool()
        print("✅ Database connection pool initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize database pool: {e}")
    try:
        _db = PostgresDatabase()
        _db.ensure_connection()
        _db.create_llm_settings_table()
        _db.close()
        print("✅ LLM settings table initialized")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize LLM settings table: {e}")


@app.on_event("shutdown")
async def _on_shutdown():
    try:
        PostgresDatabase.close_all_connections()
        print("✅ Database connections closed cleanly")
    except Exception as e:
        print(f"⚠️ Warning: Error closing database connections: {e}")


db = PostgresDatabase()

from api.akkio.main import akkio_router

app.include_router(akkio_router, prefix="/api")

from api.app_creator.dynamic_app_router import router as dynamic_app_router
from api.app_creator.static_app_router import router as static_app_router
from api.app_creator.app_creator import router as app_builder_router
from api.app_creator.apps_api import router as app_builder_apps_router
from api.app_creator.prd_api import router as prd_router
from api.app_creator.agent_api import router as agent_router
from api.app_creator.codegen_api import router as codegen_router
from api.app_creator.planning_api import router as planning_router
from api.app_creator.deployment_api import router as deployment_router
from api.app_creator.github_api import router as github_router
from api.app_creator.test_api import router as test_router

app.include_router(dynamic_app_router)
app.include_router(static_app_router)
app.include_router(app_builder_router)
app.include_router(app_builder_apps_router, prefix="/api/app-builder")
app.include_router(prd_router)
app.include_router(agent_router)
app.include_router(codegen_router)
app.include_router(planning_router)
app.include_router(deployment_router)
app.include_router(github_router)
app.include_router(test_router)


@app.get("/health")
@app.get("/api/health")
async def health_check():
    health_status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": {},
    }
    try:
        pool_status = PostgresDatabase.get_pool_status()
        health_status["database"] = pool_status
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            health_status["database"]["connectivity"] = "connected"
        except Exception as db_err:
            health_status["database"]["connectivity"] = "error"
            health_status["database"]["error"] = str(db_err)
            health_status["api"] = "degraded"
    except Exception as e:
        health_status["database"] = {"status": "error", "error": str(e)}
        health_status["api"] = "degraded"
    status_code = 200 if health_status["api"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


if __name__ == "__main__":
    import uvicorn

    from app_builder.services.runtime_paths import get_runtime_root

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    _runtime_root = os.path.abspath(get_runtime_root())
    os.makedirs(_runtime_root, exist_ok=True)
    _candidates = [
        _runtime_root,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "app_builder", ".runtime")),
        os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime"),
    ]
    _reload_excludes = [d for d in _candidates if os.path.isdir(d)]
    uvicorn.run(
        "final_akio_apis:app",
        host=host,
        port=port,
        reload=True,
        reload_excludes=_reload_excludes,
    )
