"""
MongoDB database module for App Creator (app builder) only.
Used when users create apps (frontend + backend) - stores apps, codegen sessions, deployments.
Akkio main app continues to use Postgres only.
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Any, Dict

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

# Collections
COLL_APPS = "app_builder_apps"
COLL_CODEGEN = "app_builder_codegen_sessions"
COLL_DEPLOYMENTS = "app_builder_deployments"


def _get_mongo_url() -> str:
    """Get MongoDB connection URL from environment."""
    url = os.environ.get(
        "MONGODB_APP_BUILDER_URL",
        os.environ.get(
            "MONGODB_URL",
            "mongodb+srv://prashanth:BnHRQrqZHdnosfEe@cluster0.cpydc.mongodb.net/akkio?retryWrites=true&w=majority"
        )
    )
    return url


class AppBuilderMongoDB:
    """MongoDB backend for App Creator (apps, codegen sessions, deployments)."""

    _client: Optional[MongoClient] = None
    _db = None

    def __init__(self):
        self._ensure_client()

    def _ensure_client(self):
        if AppBuilderMongoDB._client is None:
            try:
                url = _get_mongo_url()
                AppBuilderMongoDB._client = MongoClient(url, serverSelectionTimeoutMS=5000)
                AppBuilderMongoDB._db = AppBuilderMongoDB._client.get_database()
                # Verify connection
                AppBuilderMongoDB._client.admin.command("ping")
                self._ensure_indexes()
            except Exception as e:
                raise RuntimeError(f"MongoDB connection failed: {e}")

    def _ensure_indexes(self):
        """Create indexes for efficient queries."""
        try:
            self._db[COLL_APPS].create_index("user_email")
            self._db[COLL_APPS].create_index([("user_email", 1), ("updated_at", -1)])
            self._db[COLL_CODEGEN].create_index("session_id", unique=True)
            self._db[COLL_CODEGEN].create_index("project_name")
            self._db[COLL_DEPLOYMENTS].create_index("app_id")
            self._db[COLL_DEPLOYMENTS].create_index("project_name")
            self._db[COLL_DEPLOYMENTS].create_index([("project_name", 1), ("deployed_at", -1)])
        except Exception as e:
            print(f"App Builder MongoDB: index creation warning: {e}")

    def _doc_to_app(self, doc: dict) -> dict:
        """Convert MongoDB doc to API-friendly app dict with 'id' as string."""
        if not doc:
            return None
        out = dict(doc)
        out["id"] = str(doc["_id"])
        del out["_id"]
        # Ensure datetime fields are serializable
        for k in ("created_at", "updated_at"):
            if k in out and hasattr(out[k], "isoformat"):
                out[k] = out[k].isoformat()
        return out

    def _doc_to_deployment(self, doc: dict) -> dict:
        """Convert MongoDB doc to API-friendly deployment dict."""
        if not doc:
            return None
        out = dict(doc)
        out["id"] = str(doc["_id"])
        del out["_id"]
        # app_id in MongoDB is stored as string
        if "app_id" in out and out["app_id"] is not None:
            out["app_id"] = str(out["app_id"])
        for k in ("deployed_at", "updated_at"):
            if k in out and hasattr(out.get(k), "isoformat"):
                out[k] = out[k].isoformat()
        return out

    # -------- App Builder Apps --------
    def create_app_builder_app(
        self,
        user_email: str,
        app_name: str,
        prompt: str,
        project_name: str,
        prd: str = None,
        generated_uiux: str = None,
        plan: list = None,
        architecture: dict = None,
        api_contract: str = None,
        db_schema: str = None,
        agents_state: dict = None,
        generated_code_json: dict = None,
        generated_files: dict = None,
    ) -> dict:
        """Create a new app builder app record."""
        coll = self._db[COLL_APPS]
        now = datetime.utcnow()
        doc = {
            "user_email": user_email,
            "app_name": app_name,
            "prompt": prompt,
            "project_name": project_name,
            "prd": prd,
            "generated_uiux": generated_uiux,
            "plan": plan,
            "architecture": architecture,
            "api_contract": api_contract,
            "db_schema": db_schema,
            "agents_state": agents_state,
            "generated_code_json": generated_code_json,
            "generated_files": generated_files,
            "created_at": now,
            "updated_at": now,
        }
        result = coll.insert_one(doc)
        doc["_id"] = result.inserted_id
        return self._doc_to_app(doc)

    def get_user_app_builder_apps(self, user_email: str) -> list:
        """Get all app builder apps for a user."""
        coll = self._db[COLL_APPS]
        cursor = coll.find({"user_email": user_email}).sort("updated_at", -1)
        return [self._doc_to_app(d) for d in cursor]

    def get_app_by_project_name(self, project_name: str) -> Optional[dict]:
        """Get the most recently updated app for a given project_name. Used when PRD/UIUX not in request."""
        if not project_name or not project_name.strip():
            return None
        coll = self._db[COLL_APPS]
        doc = coll.find_one(
            {"project_name": project_name.strip()},
            sort=[("updated_at", -1)]
        )
        return self._doc_to_app(doc)

    def get_app_builder_app(self, app_id, user_email: str = None) -> Optional[dict]:
        """Get a single app by id (supports both str ObjectId and legacy int). Scoped by user_email if provided."""
        coll = self._db[COLL_APPS]
        query = {}
        try:
            if isinstance(app_id, int):
                # Legacy: treat as numeric - we don't have int ids in Mongo, skip
                return None
            oid = ObjectId(app_id)
            query["_id"] = oid
        except (TypeError, ValueError):
            return None
        if user_email:
            query["user_email"] = user_email
        doc = coll.find_one(query)
        return self._doc_to_app(doc)

    def update_app_builder_app(
        self,
        app_id,
        user_email: str,
        app_name: str = None,
        prompt: str = None,
        project_name: str = None,
        prd: str = None,
        generated_uiux: str = None,
        plan: list = None,
        architecture: dict = None,
        api_contract: str = None,
        db_schema: str = None,
        agents_state: dict = None,
        generated_code_json: dict = None,
        generated_files: dict = None,
    ) -> int:
        """Update an app builder app. Returns number of docs modified."""
        coll = self._db[COLL_APPS]
        try:
            oid = ObjectId(app_id)
        except (TypeError, ValueError):
            return 0
        update = {"$set": {"updated_at": datetime.utcnow()}}
        if app_name is not None:
            update["$set"]["app_name"] = app_name
        if prompt is not None:
            update["$set"]["prompt"] = prompt
        if project_name is not None:
            update["$set"]["project_name"] = project_name
        if prd is not None:
            update["$set"]["prd"] = prd
        if generated_uiux is not None:
            update["$set"]["generated_uiux"] = generated_uiux
        if plan is not None:
            update["$set"]["plan"] = plan
        if architecture is not None:
            update["$set"]["architecture"] = architecture
        if api_contract is not None:
            update["$set"]["api_contract"] = api_contract
        if db_schema is not None:
            update["$set"]["db_schema"] = db_schema
        if agents_state is not None:
            update["$set"]["agents_state"] = agents_state
        if generated_code_json is not None:
            update["$set"]["generated_code_json"] = generated_code_json
        if generated_files is not None:
            update["$set"]["generated_files"] = generated_files
        r = coll.update_one({"_id": oid, "user_email": user_email}, update)
        return r.modified_count

    def delete_app_builder_app(self, app_id, user_email: str) -> int:
        """Delete an app builder app."""
        coll = self._db[COLL_APPS]
        try:
            oid = ObjectId(app_id)
        except (TypeError, ValueError):
            return 0
        r = coll.delete_one({"_id": oid, "user_email": user_email})
        return r.deleted_count

    # -------- Deployments --------
    def create_deployment(
        self,
        app_id=None,
        project_name: str = None,
        frontend_url: str = None,
        backend_url: str = None,
        frontend_port: int = None,
        backend_port: int = None,
        deployment_status: str = "pending",
        error_message: str = None,
    ) -> Optional[dict]:
        """Create a new deployment record."""
        coll = self._db[COLL_DEPLOYMENTS]
        now = datetime.utcnow()
        app_id_val = str(app_id) if app_id is not None else None
        doc = {
            "app_id": app_id_val,
            "project_name": project_name,
            "frontend_url": frontend_url,
            "backend_url": backend_url,
            "frontend_port": frontend_port,
            "backend_port": backend_port,
            "deployment_status": deployment_status,
            "error_message": error_message,
            "deployed_at": now,
            "updated_at": now,
        }
        result = coll.insert_one(doc)
        doc["_id"] = result.inserted_id
        return self._doc_to_deployment(doc)

    def update_deployment(
        self,
        deployment_id,
        frontend_url: str = None,
        backend_url: str = None,
        deployment_status: str = None,
        error_message: str = None,
    ) -> int:
        """Update an existing deployment record."""
        coll = self._db[COLL_DEPLOYMENTS]
        try:
            oid = ObjectId(deployment_id) if isinstance(deployment_id, str) else deployment_id
        except (TypeError, ValueError):
            return 0
        update = {"$set": {"updated_at": datetime.utcnow()}}
        if frontend_url is not None:
            update["$set"]["frontend_url"] = frontend_url
        if backend_url is not None:
            update["$set"]["backend_url"] = backend_url
        if deployment_status is not None:
            update["$set"]["deployment_status"] = deployment_status
        if error_message is not None:
            update["$set"]["error_message"] = error_message
        r = coll.update_one({"_id": oid}, update)
        return r.modified_count

    def get_deployment_by_app_id(self, app_id) -> Optional[dict]:
        """Get the latest deployment for an app."""
        coll = self._db[COLL_DEPLOYMENTS]
        app_id_str = str(app_id) if app_id is not None else None
        doc = coll.find_one(
            {"app_id": app_id_str},
            sort=[("deployed_at", -1)]
        )
        return self._doc_to_deployment(doc)

    def get_deployment_by_project_name(self, project_name: str) -> Optional[dict]:
        """Get the latest deployment for a project."""
        coll = self._db[COLL_DEPLOYMENTS]
        doc = coll.find_one(
            {"project_name": project_name},
            sort=[("deployed_at", -1)]
        )
        return self._doc_to_deployment(doc)

    def get_deployment_by_id(self, deployment_id) -> Optional[dict]:
        """Get deployment by its MongoDB _id."""
        coll = self._db[COLL_DEPLOYMENTS]
        try:
            oid = ObjectId(deployment_id) if isinstance(deployment_id, str) else deployment_id
        except (TypeError, ValueError):
            return None
        doc = coll.find_one({"_id": oid})
        return self._doc_to_deployment(doc)

    # -------- Codegen Sessions --------
    def create_or_update_codegen_session(
        self,
        session_id: str,
        project_name: str,
        requirement: str = None,
        prd: str = None,
        plan: list = None,
        architecture: Any = None,
        api_contract: str = None,
        db_schema: str = None,
        generated_code_json: dict = None,
        generated_files: dict = None,
        app_id: str = None,
    ) -> int:
        """Create or update a codegen session."""
        coll = self._db[COLL_CODEGEN]
        now = datetime.utcnow()
        existing = coll.find_one({"session_id": session_id})
        if existing:
            update = {"$set": {"updated_at": now}}
            if project_name:
                update["$set"]["project_name"] = project_name
            if requirement is not None:
                update["$set"]["requirement"] = requirement
            if prd is not None:
                update["$set"]["prd"] = prd
            if plan is not None:
                update["$set"]["plan"] = plan
            if architecture is not None:
                update["$set"]["architecture"] = architecture
            if api_contract is not None:
                update["$set"]["api_contract"] = api_contract
            if db_schema is not None:
                update["$set"]["db_schema"] = db_schema
            if generated_code_json is not None:
                update["$set"]["generated_code_json"] = generated_code_json
            if generated_files is not None:
                update["$set"]["generated_files"] = generated_files
            if app_id is not None:
                update["$set"]["app_id"] = app_id
            r = coll.update_one({"session_id": session_id}, update)
            return r.modified_count
        doc = {
            "session_id": session_id,
            "project_name": project_name,
            "requirement": requirement,
            "prd": prd,
            "plan": plan,
            "architecture": architecture,
            "api_contract": api_contract,
            "db_schema": db_schema,
            "generated_code_json": generated_code_json,
            "generated_files": generated_files,
            "app_id": app_id,
            "created_at": now,
            "updated_at": now,
        }
        coll.insert_one(doc)
        return 1

    def get_codegen_session(self, session_id: str) -> Optional[dict]:
        """Get a codegen session by session_id."""
        coll = self._db[COLL_CODEGEN]
        doc = coll.find_one({"session_id": session_id})
        if not doc:
            return None
        out = dict(doc)
        out["id"] = str(doc["_id"])
        del out["_id"]
        for k in ("created_at", "updated_at"):
            if k in out and hasattr(out.get(k), "isoformat"):
                out[k] = out[k].isoformat()
        return out


# Singleton for app creator usage
_app_builder_db: Optional[AppBuilderMongoDB] = None


def get_app_builder_db() -> AppBuilderMongoDB:
    """Get or create the App Builder MongoDB client."""
    global _app_builder_db
    if _app_builder_db is None:
        _app_builder_db = AppBuilderMongoDB()
    return _app_builder_db
