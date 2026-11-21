import copy
import json
import os
import tempfile
import threading
from typing import Any, Dict, Optional

from app_builder.services.runtime_paths import resolve_project_root


_locks: Dict[str, threading.RLock] = {}
_guard = threading.Lock()


def project_lock(project_id: str) -> threading.RLock:
    with _guard:
        return _locks.setdefault(project_id, threading.RLock())


def load_definition(project_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(resolve_project_root(project_id), "runtime_definition.json")
    with project_lock(project_id):
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)


def save_definition(project_id: str, definition: Dict[str, Any]) -> None:
    root = resolve_project_root(project_id)
    if not os.path.isdir(root):
        raise ValueError("Project directory does not exist")
    with project_lock(project_id):
        fd, temporary = tempfile.mkstemp(prefix=".definition-", suffix=".json", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(definition, handle, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, os.path.join(root, "runtime_definition.json"))
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)


def mutate_record(definition: Dict[str, Any], data_key: str, record: Dict[str, Any], index: Optional[int] = None) -> Dict[str, Any]:
    out = copy.deepcopy(definition)
    table = next((b for p in out.get("pages", []) for b in p.get("blocks", []) if b.get("type") == "dataTable" and b.get("dataKey") == data_key), None)
    if table is None:
        raise ValueError("Unknown table dataset")
    fields = {c["field"]: c for c in table["columns"]}
    if set(record) != set(fields):
        raise ValueError("Record fields must match table columns")
    for field, column in fields.items():
        value = record[field]
        if column.get("required") and (value is None or value == ""):
            raise ValueError(f"{field} is required")
        if column.get("inputType") == "number" and value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{field} must be numeric")
    bag = out.setdefault("data", out.pop("mock", {}))
    rows = bag.get(data_key)
    if not isinstance(rows, list):
        raise ValueError("Table dataset is missing")
    if index is None:
        rows.append(record)
    elif index < 0 or index >= len(rows):
        raise ValueError("Record no longer exists")
    else:
        rows[index] = record
    out["meta"]["dataSource"] = "user_entered"
    out["meta"]["containsSyntheticData"] = definition.get("meta", {}).get("containsSyntheticData", definition.get("meta", {}).get("dataSource") == "llm_generated")
    from app_builder.services.definition_metrics import compute_definition

    return compute_definition(out)