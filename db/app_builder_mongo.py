"""MongoDB-backed App Builder storage (apps, codegen sessions, deployments)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from pymongo import ASCENDING

from db.mongodb import MongoPool, get_collection, next_seq, utc_now

logger = logging.getLogger("app_builder")

_METADATA_FIELDS = (
    "prompt",
    "prd",
    "generated_uiux",
    "plan",
    "architecture",
    "api_contract",
    "db_schema",
    "agents_state",
    "generated_code_json",
    "generated_files",
    "user_email",
    "pipeline_status",
    "pipeline_error",
    "build_status",
    "build_log",
    "build_error",
    "preview_url",
    "live_url",
    "design_tokens",
    "design_system_md",
    "llm_model",
    "builder_kind",
    "track",
    "plan_json",
    "app_spec_json",
)


def _serialize_ts(value) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _app_row_to_dict(row: dict, user_email: Optional[str] = None) -> dict:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    email = user_email or meta.get("user_email") or ""
    out = {
        "id": str(row["id"]),
        "user_email": email,
        "app_name": row["name"],
        "name": row["name"],
        "project_name": row["project_name"],
        "created_at": _serialize_ts(row.get("created_at")),
        "updated_at": _serialize_ts(row.get("updated_at")),
    }
    for key in _METADATA_FIELDS:
        if key in meta and key != "user_email":
            out[key] = meta[key]
    out.setdefault("builder_kind", "app")
    return out


def _app_doc_to_row(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "user_id": doc["user_id"],
        "name": doc["name"],
        "project_name": doc["project_name"],
        "metadata": doc.get("metadata") or {},
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


class AppBuilderStore(MongoPool):
    _schema_ready = False
    _deployment_fk_migrated = False

    def init_schema(self) -> None:
        if self.__class__._schema_ready:
            return
        apps = get_collection("builder_apps")
        apps.create_index([("user_id", ASCENDING), ("project_name", ASCENDING)], unique=True)
        apps.create_index([("user_id", ASCENDING)])
        apps.create_index([("project_name", ASCENDING)])

        get_collection("app_builder_codegen_sessions").create_index(
            [("session_id", ASCENDING)], unique=True
        )
        get_collection("builder_app_jobs").create_index([("session_id", ASCENDING)], unique=True)
        get_collection("builder_app_jobs").create_index([("app_id", ASCENDING)])

        deployments = get_collection("app_builder_deployments")
        deployments.create_index([("app_id", ASCENDING)])
        deployments.create_index([("project_name", ASCENDING)])

        self.__class__._schema_ready = True

    def _ensure_deployment_fk(self) -> None:
        if self.__class__._deployment_fk_migrated:
            return
        self.init_schema()
        self.__class__._deployment_fk_migrated = True

    def _builder_app_exists(self, app_id: int) -> bool:
        return (
            get_collection("builder_apps").find_one({"id": app_id}, {"_id": 1}) is not None
        )

    def _resolve_deployment_app_id(self, app_id) -> Optional[int]:
        if app_id is None:
            return None
        try:
            app_id_int = int(app_id)
        except (TypeError, ValueError):
            return None
        if self._builder_app_exists(app_id_int):
            return app_id_int
        logger.warning(
            "[deployment] app_id=%s not found in builder_apps — deployment will use project_name only",
            app_id_int,
        )
        return None

    def _resolve_user_id(self, user_email: str) -> int:
        email = user_email.lower()
        users = get_collection("auth_users")
        try:
            doc = users.find_one({"email": email, "app": "akkio"}, {"id": 1})
            if doc and doc.get("id") is not None:
                return int(doc["id"])
            doc = users.find_one({"email": email}, {"id": 1})
            if doc and doc.get("id") is not None:
                return int(doc["id"])
        except Exception:
            pass
        return abs(int(hashlib.sha256(email.encode()).hexdigest()[:12], 16)) % (2**31 - 1)

    def _user_email_for_id(self, user_id: int) -> Optional[str]:
        doc = get_collection("auth_users").find_one({"id": user_id}, {"email": 1})
        if doc:
            return doc.get("email")
        return None

    def create_app_builder_app(
        self,
        user_email: str,
        app_name: str,
        prompt: str,
        project_name: str,
        prd: Optional[str] = None,
        generated_uiux: Optional[str] = None,
        plan: Optional[list] = None,
        architecture: Optional[dict] = None,
        api_contract: Optional[str] = None,
        db_schema: Optional[str] = None,
        agents_state: Optional[dict] = None,
        generated_code_json: Optional[dict] = None,
        generated_files: Optional[dict] = None,
        user_id: Optional[int] = None,
        builder_kind: Optional[str] = None,
    ) -> dict:
        self.init_schema()
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        metadata = {
            "prompt": prompt,
            "user_email": user_email.lower(),
            "prd": prd,
            "generated_uiux": generated_uiux,
            "plan": plan,
            "architecture": architecture,
            "api_contract": api_contract,
            "db_schema": db_schema,
            "agents_state": agents_state,
            "generated_code_json": generated_code_json,
            "generated_files": generated_files,
            "builder_kind": (builder_kind or "app").strip().lower() or "app",
        }
        coll = get_collection("builder_apps")
        now = utc_now()
        existing = coll.find_one({"user_id": uid, "project_name": project_name})
        if existing:
            merged = {**(existing.get("metadata") or {}), **metadata}
            coll.update_one(
                {"_id": existing["_id"]},
                {"$set": {"name": app_name, "metadata": merged, "updated_at": now}},
            )
            doc = coll.find_one({"_id": existing["_id"]})
        else:
            doc = {
                "id": next_seq("builder_apps"),
                "user_id": uid,
                "name": app_name,
                "project_name": project_name,
                "metadata": metadata,
                "created_at": now,
                "updated_at": now,
            }
            coll.insert_one(doc)
        return _app_row_to_dict(_app_doc_to_row(doc), user_email)

    def get_user_app_builder_apps(
        self,
        user_email: str,
        user_id: Optional[int] = None,
        builder_kind: Optional[str] = None,
    ) -> list:
        self.init_schema()
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        coll = get_collection("builder_apps")
        cursor = coll.find({"user_id": uid}).sort("updated_at", -1)
        apps = [_app_row_to_dict(_app_doc_to_row(dict(d)), user_email) for d in cursor]
        kind = (builder_kind or "").strip().lower()
        if not kind:
            return apps
        if kind == "app":
            return [a for a in apps if (a.get("builder_kind") or "app") in ("app", "", None)]
        return [a for a in apps if a.get("builder_kind") == kind]

    def get_app_by_project_name(
        self,
        project_name: str,
        user_id: Optional[int] = None,
        user_email: Optional[str] = None,
    ) -> Optional[dict]:
        self.init_schema()
        if not project_name or not project_name.strip():
            return None
        uid = user_id
        if uid is None and user_email:
            try:
                uid = self._resolve_user_id(user_email)
            except ValueError:
                return None
        coll = get_collection("builder_apps")
        pn = project_name.strip()
        if uid is not None:
            doc = coll.find_one({"project_name": pn, "user_id": uid})
        else:
            doc = coll.find_one({"project_name": pn}, sort=[("updated_at", -1)])
        if not doc:
            return None
        row = _app_doc_to_row(dict(doc))
        email = user_email or self._user_email_for_id(row["user_id"])
        return _app_row_to_dict(row, email)

    def get_app_builder_app(
        self, app_id, user_email: Optional[str] = None, user_id: Optional[int] = None
    ) -> Optional[dict]:
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return None
        query: dict[str, Any] = {"id": aid}
        if user_id is not None:
            query["user_id"] = user_id
        elif user_email:
            query["user_id"] = self._resolve_user_id(user_email)
        doc = get_collection("builder_apps").find_one(query)
        if not doc:
            return None
        row = _app_doc_to_row(dict(doc))
        email = user_email or self._user_email_for_id(row["user_id"])
        return _app_row_to_dict(row, email)

    def get_app_builder_app_poll_snapshot(
        self,
        app_id,
        user_email: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[dict]:
        """Status polling only — avoids loading generated_code_json / plan_json from metadata."""
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return None
        uid = user_id
        if uid is None and user_email:
            uid = self._resolve_user_id(user_email)
        if uid is None:
            return None
        doc = get_collection("builder_apps").find_one(
            {"id": aid, "user_id": uid},
            {
                "id": 1,
                "project_name": 1,
                "metadata.pipeline_status": 1,
                "metadata.pipeline_error": 1,
                "metadata.build_status": 1,
                "metadata.build_error": 1,
                "metadata.preview_url": 1,
                "metadata.live_url": 1,
                "metadata.track": 1,
                "metadata.build_log": 1,
            },
        )
        if not doc:
            return None
        meta = doc.get("metadata") or {}
        build_log = meta.get("build_log") or ""
        if build_log and len(build_log) > 8000:
            build_log = "…" + build_log[-8000:]
        return {
            "id": doc["id"],
            "project_name": doc.get("project_name"),
            "pipeline_status": meta.get("pipeline_status"),
            "pipeline_error": meta.get("pipeline_error"),
            "build_status": meta.get("build_status"),
            "build_error": meta.get("build_error"),
            "preview_url": meta.get("preview_url"),
            "live_url": meta.get("live_url"),
            "track": meta.get("track"),
            "build_log": build_log or None,
        }

    def update_app_builder_app(
        self,
        app_id,
        user_email: str,
        app_name: Optional[str] = None,
        prompt: Optional[str] = None,
        project_name: Optional[str] = None,
        prd: Optional[str] = None,
        generated_uiux: Optional[str] = None,
        plan: Optional[list] = None,
        architecture: Optional[dict] = None,
        api_contract: Optional[str] = None,
        db_schema: Optional[str] = None,
        agents_state: Optional[dict] = None,
        generated_code_json: Optional[dict] = None,
        generated_files: Optional[dict] = None,
        pipeline_status: Optional[str] = None,
        pipeline_error: Optional[str] = None,
        build_status: Optional[str] = None,
        build_log: Optional[str] = None,
        build_error: Optional[str] = None,
        preview_url: Optional[str] = None,
        live_url: Optional[str] = None,
        user_id: Optional[int] = None,
        design_tokens: Optional[dict] = None,
        design_system_md: Optional[str] = None,
        llm_model: Optional[str] = None,
        builder_kind: Optional[str] = None,
        track: Optional[str] = None,
        plan_json: Optional[dict] = None,
        app_spec_json: Optional[dict] = None,
    ) -> int:
        self.init_schema()
        existing = self.get_app_builder_app(app_id, user_email=user_email, user_id=user_id)
        if not existing:
            return 0
        meta = {k: existing.get(k) for k in _METADATA_FIELDS if k in existing}
        meta["user_email"] = user_email.lower()
        updates = {
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
            "pipeline_status": pipeline_status,
            "pipeline_error": pipeline_error,
            "build_status": build_status,
            "build_log": build_log,
            "build_error": build_error,
            "preview_url": preview_url,
            "live_url": live_url,
            "design_tokens": design_tokens,
            "design_system_md": design_system_md,
            "llm_model": llm_model,
            "builder_kind": builder_kind,
            "track": track,
            "plan_json": plan_json,
            "app_spec_json": app_spec_json,
        }
        key_map = {"app_name": "name"}
        name_val = app_name
        project_val = project_name
        for k, v in updates.items():
            if v is not None:
                field = key_map.get(k, k)
                if field == "name":
                    name_val = v
                elif field == "project_name":
                    project_val = v
                else:
                    meta[field] = v
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        set_fields: dict[str, Any] = {"metadata": meta, "updated_at": utc_now()}
        if name_val is not None:
            set_fields["name"] = name_val
        if project_val is not None:
            set_fields["project_name"] = project_val
        result = get_collection("builder_apps").update_one(
            {"id": int(app_id), "user_id": uid},
            {"$set": set_fields},
        )
        return result.modified_count

    def delete_app_builder_app(self, app_id, user_email: str, user_id: Optional[int] = None) -> int:
        self.init_schema()
        uid = user_id if user_id is not None else self._resolve_user_id(user_email)
        aid = int(app_id)
        get_collection("app_builder_deployments").delete_many({"app_id": aid})
        result = get_collection("builder_apps").delete_one({"id": aid, "user_id": uid})
        return result.deleted_count

    def create_deployment(
        self,
        app_id=None,
        project_name: Optional[str] = None,
        frontend_url: Optional[str] = None,
        backend_url: Optional[str] = None,
        frontend_port: Optional[int] = None,
        backend_port: Optional[int] = None,
        deployment_status: str = "pending",
        error_message: Optional[str] = None,
        deploy_log: Optional[str] = None,
    ) -> Optional[dict]:
        self.init_schema()
        self._ensure_deployment_fk()
        app_id_int = self._resolve_deployment_app_id(app_id)
        now = utc_now()
        doc = {
            "id": next_seq("app_builder_deployments"),
            "app_id": app_id_int,
            "project_name": project_name,
            "frontend_url": frontend_url,
            "backend_url": backend_url,
            "frontend_port": frontend_port,
            "backend_port": backend_port,
            "deployment_status": deployment_status,
            "error_message": error_message,
            "deploy_log": deploy_log,
            "deployed_at": now,
            "updated_at": now,
        }
        get_collection("app_builder_deployments").insert_one(doc)
        row = dict(doc)
        return self._deployment_row(row)

    def update_deployment(
        self,
        deployment_id,
        frontend_url: Optional[str] = None,
        backend_url: Optional[str] = None,
        deployment_status: Optional[str] = None,
        error_message: Optional[str] = None,
        deploy_log: Optional[str] = None,
    ) -> int:
        self.init_schema()
        try:
            did = int(deployment_id)
        except (TypeError, ValueError):
            return 0
        set_fields: dict[str, Any] = {"updated_at": utc_now()}
        for col, val in (
            ("frontend_url", frontend_url),
            ("backend_url", backend_url),
            ("deployment_status", deployment_status),
            ("error_message", error_message),
            ("deploy_log", deploy_log),
        ):
            if val is not None:
                set_fields[col] = val
        if len(set_fields) == 1:
            return 0
        result = get_collection("app_builder_deployments").update_one(
            {"id": did},
            {"$set": set_fields},
        )
        return result.modified_count

    def _deployment_row(self, row: dict) -> dict:
        row = dict(row)
        row.pop("_id", None)
        row["id"] = str(row["id"])
        if row.get("app_id") is not None:
            row["app_id"] = str(row["app_id"])
        row["deployed_at"] = _serialize_ts(row.get("deployed_at"))
        row["updated_at"] = _serialize_ts(row.get("updated_at"))
        return row

    def _find_deployment(self, query: dict) -> Optional[dict]:
        doc = get_collection("app_builder_deployments").find_one(
            query, sort=[("deployed_at", -1)]
        )
        return self._deployment_row(dict(doc)) if doc else None

    def get_deployment_by_app_id(self, app_id) -> Optional[dict]:
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return None
        return self._find_deployment({"app_id": aid})

    def get_deployment_by_project_name(self, project_name: str) -> Optional[dict]:
        self.init_schema()
        return self._find_deployment({"project_name": project_name})

    def upsert_local_deployment(
        self,
        *,
        app_id=None,
        project_name: str,
        frontend_url: str,
        backend_url: Optional[str] = None,
        deploy_log: Optional[str] = None,
    ) -> dict:
        """Create or update a RUNNING deployment for local preview after Run App succeeds."""
        existing = None
        if app_id is not None:
            existing = self.get_deployment_by_app_id(app_id)
        if not existing:
            existing = self.get_deployment_by_project_name(project_name)

        log_line = deploy_log or "Registered local preview after successful build"
        if existing:
            self.update_deployment(
                existing["id"],
                frontend_url=frontend_url,
                backend_url=backend_url,
                deployment_status="LOCAL_PREVIEW",
                error_message=None,
                deploy_log=log_line,
            )
            return self.get_deployment_by_id(existing["id"]) or existing

        return self.create_deployment(
            app_id=app_id,
            project_name=project_name,
            frontend_url=frontend_url,
            backend_url=backend_url,
            deployment_status="LOCAL_PREVIEW",
            deploy_log=log_line,
        )

    def get_deployment_by_id(self, deployment_id) -> Optional[dict]:
        self.init_schema()
        try:
            did = int(deployment_id)
        except (TypeError, ValueError):
            return None
        doc = get_collection("app_builder_deployments").find_one({"id": did})
        return self._deployment_row(dict(doc)) if doc else None

    def list_deployments_by_app_id(self, app_id, limit: int = 20) -> list:
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return []
        cursor = (
            get_collection("app_builder_deployments")
            .find({"app_id": aid})
            .sort("deployed_at", -1)
            .limit(limit)
        )
        return [self._deployment_row(dict(r)) for r in cursor]

    def create_or_update_codegen_session(
        self,
        session_id: str,
        project_name: str,
        requirement: Optional[str] = None,
        prd: Optional[str] = None,
        plan: Optional[list] = None,
        architecture: Any = None,
        api_contract: Optional[str] = None,
        db_schema: Optional[str] = None,
        generated_code_json: Optional[dict] = None,
        generated_files: Optional[dict] = None,
        app_id: Optional[str] = None,
    ) -> int:
        self.init_schema()
        app_id_int = None
        if app_id:
            try:
                app_id_int = int(app_id)
            except (TypeError, ValueError):
                pass
        coll = get_collection("app_builder_codegen_sessions")
        existing = coll.find_one({"session_id": session_id}, {"_id": 1})
        now = utc_now()
        if existing:
            set_fields: dict[str, Any] = {"updated_at": now}
            if project_name is not None:
                set_fields["project_name"] = project_name
            for field, val in (
                ("requirement", requirement),
                ("prd", prd),
                ("plan", plan),
                ("architecture", architecture),
                ("api_contract", api_contract),
                ("db_schema", db_schema),
                ("generated_code_json", generated_code_json),
                ("generated_files", generated_files),
                ("app_id", app_id_int),
            ):
                if val is not None:
                    set_fields[field] = val
            result = coll.update_one({"session_id": session_id}, {"$set": set_fields})
            return result.modified_count
        doc = {
            "id": next_seq("app_builder_codegen_sessions"),
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
            "app_id": app_id_int,
            "created_at": now,
            "updated_at": now,
        }
        coll.insert_one(doc)
        return 1

    def create_or_update_job(
        self,
        session_id: str,
        job_type: str = "codegen",
        app_id: int | Optional[str] = None,
        status: str = "queued",
        step: Optional[str] = None,
        logs: Optional[str] = None,
        error: Optional[str] = None,
        finished: bool = False,
    ) -> int:
        self.init_schema()
        app_id_int = None
        if app_id is not None:
            try:
                app_id_int = int(app_id)
            except (TypeError, ValueError):
                app_id_int = None
        coll = get_collection("builder_app_jobs")
        existing = coll.find_one({"session_id": session_id}, {"_id": 1})
        now = utc_now()
        if existing:
            set_fields: dict[str, Any] = {"updated_at": now}
            if app_id_int is not None:
                set_fields["app_id"] = app_id_int
            if job_type is not None:
                set_fields["job_type"] = job_type
            if status is not None:
                set_fields["status"] = status
            if step is not None:
                set_fields["step"] = step
            if logs is not None:
                set_fields["logs"] = logs
            if error is not None:
                set_fields["error"] = error
            if finished:
                set_fields["finished_at"] = now
            result = coll.update_one({"session_id": session_id}, {"$set": set_fields})
            return result.modified_count
        doc = {
            "id": next_seq("builder_app_jobs"),
            "app_id": app_id_int,
            "session_id": session_id,
            "job_type": job_type,
            "status": status,
            "step": step,
            "logs": logs,
            "error": error,
            "started_at": now,
            "finished_at": now if finished else None,
            "updated_at": now,
        }
        coll.insert_one(doc)
        return 1

    def get_job_by_session_id(self, session_id: str) -> Optional[dict]:
        self.init_schema()
        doc = get_collection("builder_app_jobs").find_one({"session_id": session_id})
        if not doc:
            return None
        return self._job_row_to_dict(dict(doc))

    def get_active_job_for_app(self, app_id) -> Optional[dict]:
        """Latest in-progress job for an app (finished_at IS NULL)."""
        self.init_schema()
        try:
            aid = int(app_id)
        except (TypeError, ValueError):
            return None
        doc = get_collection("builder_app_jobs").find_one(
            {"app_id": aid, "finished_at": None},
            sort=[("started_at", -1)],
        )
        if not doc:
            return None
        return self._job_row_to_dict(dict(doc))

    def _job_row_to_dict(self, row: dict) -> dict:
        out = dict(row)
        out.pop("_id", None)
        out["id"] = str(out["id"])
        if out.get("app_id") is not None:
            out["app_id"] = str(out["app_id"])
        for k in ("started_at", "finished_at", "updated_at"):
            out[k] = _serialize_ts(out.get(k))
        return out

    def get_codegen_session(self, session_id: str) -> Optional[dict]:
        self.init_schema()
        doc = get_collection("app_builder_codegen_sessions").find_one({"session_id": session_id})
        if not doc:
            return None
        out = dict(doc)
        out.pop("_id", None)
        out["id"] = str(out["id"])
        if out.get("app_id") is not None:
            out["app_id"] = str(out["app_id"])
        for k in ("created_at", "updated_at"):
            out[k] = _serialize_ts(out.get(k))
        return out


_store: Optional[AppBuilderStore] = None


def get_app_builder_db() -> AppBuilderStore:
    global _store
    if _store is None:
        _store = AppBuilderStore()
    return _store
