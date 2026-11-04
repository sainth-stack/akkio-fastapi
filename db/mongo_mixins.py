"""MongoDB domain mixins (LLM settings, multi-model, tabular uploads)."""
from __future__ import annotations

import pickle
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from db.mongodb import get_collection, utc_now


class LLMSettingsMixin:
    def create_llm_settings_table(self) -> None:
        get_collection("llm_settings").create_index("email", unique=True)

    def save_llm_settings(
        self,
        email: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.ensure_connection()
        coll = get_collection("llm_settings")
        existing = coll.find_one({"email": email})
        doc = existing or {"email": email, "created_at": utc_now()}
        if provider is not None:
            doc["provider"] = provider
        if api_key is not None:
            doc["api_key"] = api_key
        if model_name is not None:
            doc["model_name"] = model_name
        doc["updated_at"] = utc_now()
        coll.update_one({"email": email}, {"$set": doc}, upsert=True)

    def get_llm_settings(self, email: str) -> Optional[dict]:
        self.ensure_connection()
        row = get_collection("llm_settings").find_one({"email": email})
        return dict(row) if row else None

    def delete_llm_settings(self, email: str) -> str:
        self.ensure_connection()
        res = get_collection("llm_settings").delete_one({"email": email})
        return f"{res.deleted_count} LLM settings deleted"


class TabularDataMixin:
    def create_table(self) -> None:
        get_collection("akio_data_fastapi").create_index([("email", 1), ("name", 1)], unique=True)

    def _ensure_type_columns(self) -> None:
        return None

    def insert_or_update(
        self,
        email,
        data,
        tb_name,
        data_type: Optional[str] = None,
        data_subtype: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
    ):
        self.ensure_connection()
        tb_name_clean = self._clean_name(tb_name)
        coll = get_collection("akio_data_fastapi")
        blob = pickle.dumps(data)
        now = utc_now()
        existing = coll.find_one({"email": email, "name": tb_name_clean})
        if existing:
            coll.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "lastupdate": now,
                        "fileobj": blob,
                        "rawfile": raw_bytes if raw_bytes is not None else existing.get("rawfile"),
                        "type": data_type or existing.get("type"),
                        "subtype": data_subtype or existing.get("subtype"),
                    }
                },
            )
            return "updated"
        coll.insert_one(
            {
                "email": email,
                "name": tb_name_clean,
                "lastupdate": now,
                "datecreated": now,
                "fileobj": blob,
                "rawfile": raw_bytes,
                "type": data_type,
                "subtype": data_subtype,
            }
        )
        return "inserted"

    def get_table_data(self, table_name, limit: Optional[int] = None):
        try:
            clean_name = self._clean_name(table_name)
            row = get_collection("akio_data_fastapi").find_one({"name": clean_name})
            if not row or not row.get("fileobj"):
                return pd.DataFrame()
            table_data = pickle.loads(row["fileobj"])
            if isinstance(table_data, pd.DataFrame):
                df = table_data
            else:
                df = pd.DataFrame(table_data)
            if limit is not None and limit > 0:
                return df.head(limit)
            return df
        except Exception as e:
            print(f"Error retrieving table data: {e}")
            raise


class MultiModelMixin:
    _SESSION_KEYS = (
        "session_id",
        "model_name",
        "user_email",
        "system_prompt",
        "status",
        "progress",
        "stage",
        "temperature",
        "workflow",
        "output_format",
        "published",
        "public_id",
        "published_at",
        "error_message",
        "created_at",
        "updated_at",
        "completed_at",
    )

    def ensure_training_tables(self) -> None:
        if getattr(self, "_tables_initialized", False):
            return
        sessions = get_collection("multi_model_sessions")
        sessions.create_index([("session_id", 1)], unique=True)
        sessions.create_index([("user_email", 1), ("model_name", 1)], unique=True)
        sessions.create_index(
            [("public_id", 1)],
            unique=True,
            partialFilterExpression={"public_id": {"$type": "string"}},
        )
        get_collection("multi_model_files").create_index([("session_id", 1)])
        self._tables_initialized = True

    def _session_row_to_dict(self, doc: Optional[dict]) -> Optional[dict]:
        if not doc:
            return None
        out = {k: doc.get(k) for k in self._SESSION_KEYS}
        if out.get("published") is None:
            out["published"] = False
        return out

    def create_multi_model_session(
        self,
        session_id: str,
        model_name: str,
        user_email: str,
        system_prompt: str,
        temperature: float = 0.0,
        workflow: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> None:
        self.ensure_training_tables()
        coll = get_collection("multi_model_sessions")
        now = utc_now()
        coll.update_one(
            {"user_email": user_email, "model_name": model_name},
            {
                "$set": {
                    "session_id": session_id,
                    "model_name": model_name,
                    "user_email": user_email,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "workflow": workflow,
                    "output_format": output_format,
                    "status": "pending",
                    "progress": 0,
                    "stage": "Initializing",
                    "error_message": None,
                    "updated_at": now,
                    "completed_at": None,
                },
                "$setOnInsert": {"created_at": now, "published": False},
            },
            upsert=True,
        )

    def add_multi_model_file(
        self,
        session_id: str,
        file_name: str,
        file_type: str,
        storage_path: Optional[str] = None,
        vector_collection_id: Optional[str] = None,
        db_table_name: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.ensure_training_tables()
        get_collection("multi_model_files").insert_one(
            {
                "session_id": session_id,
                "file_name": file_name,
                "file_type": file_type,
                "storage_path": storage_path,
                "vector_collection_id": vector_collection_id,
                "db_table_name": db_table_name,
                "processed": False,
                "error_message": error_message,
                "created_at": utc_now(),
            }
        )

    def update_multi_model_progress(
        self, session_id: str, progress: int, stage: str, status: Optional[str] = None
    ) -> None:
        self.ensure_training_tables()
        upd: dict[str, Any] = {"progress": progress, "stage": stage, "updated_at": utc_now()}
        if status:
            upd["status"] = status
        get_collection("multi_model_sessions").update_one({"session_id": session_id}, {"$set": upd})

    def complete_multi_model_training(self, session_id: str) -> None:
        self.ensure_training_tables()
        now = utc_now()
        get_collection("multi_model_sessions").update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": "completed",
                    "progress": 100,
                    "stage": "Training completed",
                    "updated_at": now,
                    "completed_at": now,
                }
            },
        )

    def fail_multi_model_training(self, session_id: str, error_message: str) -> None:
        self.ensure_training_tables()
        get_collection("multi_model_sessions").update_one(
            {"session_id": session_id},
            {"$set": {"status": "failed", "error_message": error_message, "updated_at": utc_now()}},
        )

    def get_multi_model_session(
        self,
        session_id: Optional[str] = None,
        user_email: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Optional[dict]:
        self.ensure_training_tables()
        coll = get_collection("multi_model_sessions")
        if session_id:
            doc = coll.find_one({"session_id": session_id})
        elif user_email and model_name:
            doc = coll.find_one(
                {"user_email": user_email, "model_name": model_name},
                sort=[("updated_at", -1)],
            )
        else:
            return None
        return self._session_row_to_dict(doc)

    def get_multi_model_files(self, session_id: str) -> list:
        self.ensure_training_tables()
        keys = (
            "id",
            "session_id",
            "file_name",
            "file_type",
            "storage_path",
            "vector_collection_id",
            "db_table_name",
            "processed",
            "error_message",
            "created_at",
        )
        out = []
        for idx, doc in enumerate(
            get_collection("multi_model_files").find({"session_id": session_id}).sort("created_at", 1),
            start=1,
        ):
            row = {k: doc.get(k if k != "id" else "_id") for k in keys}
            row["id"] = idx
            out.append(row)
        return out

    def get_user_multi_models(self, user_email: str) -> list:
        self.ensure_training_tables()
        keys = (
            "session_id",
            "model_name",
            "system_prompt",
            "status",
            "progress",
            "stage",
            "temperature",
            "workflow",
            "output_format",
            "published",
            "public_id",
            "published_at",
            "error_message",
            "created_at",
            "updated_at",
            "completed_at",
        )
        return [
            {k: doc.get(k) for k in keys}
            for doc in get_collection("multi_model_sessions")
            .find({"user_email": user_email})
            .sort("updated_at", -1)
        ]

    def get_multi_model_session_by_public_id(self, public_id: str) -> Optional[dict]:
        self.ensure_training_tables()
        doc = get_collection("multi_model_sessions").find_one({"public_id": public_id})
        return self._session_row_to_dict(doc)

    def set_multi_model_published(
        self, session_id: str, user_email: str, published: bool, public_id: Optional[str] = None
    ) -> int:
        self.ensure_training_tables()
        coll = get_collection("multi_model_sessions")
        if published:
            upd: dict[str, Any] = {"published": True, "updated_at": utc_now()}
            if public_id:
                upd["public_id"] = public_id
            existing = coll.find_one({"session_id": session_id, "user_email": user_email})
            if existing and not existing.get("published_at"):
                upd["published_at"] = utc_now()
            res = coll.update_one({"session_id": session_id, "user_email": user_email}, {"$set": upd})
        else:
            res = coll.update_one(
                {"session_id": session_id, "user_email": user_email},
                {"$set": {"published": False, "published_at": None, "updated_at": utc_now()}},
            )
        return res.modified_count

    def update_multi_model_session_config(
        self,
        session_id: str,
        user_email: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        workflow: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> int:
        self.ensure_training_tables()
        upd: dict[str, Any] = {"updated_at": utc_now()}
        if model_name is not None:
            upd["model_name"] = model_name
        if system_prompt is not None:
            upd["system_prompt"] = system_prompt
        if temperature is not None:
            upd["temperature"] = temperature
        if workflow is not None:
            upd["workflow"] = workflow
        if output_format is not None:
            upd["output_format"] = output_format
        res = get_collection("multi_model_sessions").update_one(
            {"session_id": session_id, "user_email": user_email},
            {"$set": upd},
        )
        return res.modified_count

    def mark_file_processed(self, session_id: str, file_name: str) -> None:
        self.ensure_training_tables()
        get_collection("multi_model_files").update_one(
            {"session_id": session_id, "file_name": file_name},
            {"$set": {"processed": True}},
        )

    def delete_multi_model_session(self, session_id: str) -> None:
        self.ensure_training_tables()
        get_collection("multi_model_files").delete_many({"session_id": session_id})
        get_collection("multi_model_sessions").delete_one({"session_id": session_id})

    def delete_table(self, table_name: str) -> None:
        clean = self._clean_name(table_name)
        get_collection("akio_data_fastapi").delete_many({"name": clean})
