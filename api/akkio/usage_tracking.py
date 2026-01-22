from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional, Dict

from database import PostgresDatabase

from .usage_store import deduct_tokens


_current_email: ContextVar[Optional[str]] = ContextVar("current_email", default=None)
# Context var to aggregate tokens across multiple calls (e.g. multi-agent)
_token_aggregator: ContextVar[Optional[Dict[str, int]]] = ContextVar("token_aggregator", default=None)


def set_current_email(email: Optional[str]):
    return _current_email.set((email or "").strip().lower() or None)


def reset_current_email(token) -> None:
    try:
        _current_email.reset(token)
    except Exception:
        pass


def get_current_email() -> Optional[str]:
    try:
        return _current_email.get()
    except Exception:
        return None

def start_token_aggregation():
    """Start aggregating tokens in the current context."""
    return _token_aggregator.set({"total_tokens": 0})

def end_token_aggregation(token, email: str = None, description: str = "Multi-Model Agent Task"):
    """
    Stop aggregation and deduct the total accumulated tokens in one transaction.
    Returns the result of the deduction or None if no tokens used.
    """
    try:
        agg = _token_aggregator.get()
        _token_aggregator.reset(token)
        
        total = agg.get("total_tokens", 0)
        if total > 0:
            if not email:
                email = get_current_email()
            
            db = PostgresDatabase()
            db.ensure_connection()
            # We use deduct_tokens directly, which will log a single transaction
            # Note: deduct_tokens inside usage_store.py needs to support a custom description if we want "Multi-Model Agent Task"
            # For now, it logs "Used X tokens". We can enhance deduct_tokens or just let it be.
            # To pass description, we'd need to update deduct_tokens signature.
            # But wait, deduct_tokens calls log_transaction with details=f"Used {total_tokens} tokens".
            # We can override that behavior if we want, but "Used X tokens" is fine for now, or we update usage_store.py.
            
            # Let's verify usage_store.py first. It hardcodes details.
            row = deduct_tokens(db, email, total)
            try:
                db.close()
            except Exception:
                pass
            return {"total_tokens": total, **row}
    except Exception:
        pass
    return None

def extract_total_tokens(obj: Any) -> int:
    """
    Best-effort token extraction for:
    - OpenAI python responses: resp.usage.total_tokens
    - LangChain AIMessage: msg.usage_metadata or msg.response_metadata
    """
    if obj is None:
        return 0

    # OpenAI python SDK style
    usage = getattr(obj, "usage", None)
    if usage is not None:
        total = getattr(usage, "total_tokens", None)
        if isinstance(total, int):
            return total
        # sometimes dict-like
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            return int(usage["total_tokens"])

    # LangChain AIMessage usage metadata (newer)
    usage_md = getattr(obj, "usage_metadata", None)
    if isinstance(usage_md, dict):
        # common keys: input_tokens/output_tokens/total_tokens
        if isinstance(usage_md.get("total_tokens"), int):
            return int(usage_md["total_tokens"])
        it = usage_md.get("input_tokens")
        ot = usage_md.get("output_tokens")
        if isinstance(it, int) or isinstance(ot, int):
            return int(it or 0) + int(ot or 0)

    # LangChain response metadata (older)
    resp_md = getattr(obj, "response_metadata", None)
    if isinstance(resp_md, dict):
        token_usage = resp_md.get("token_usage") or resp_md.get("usage")
        if isinstance(token_usage, dict) and isinstance(token_usage.get("total_tokens"), int):
            return int(token_usage["total_tokens"])

    return 0


def record_llm_usage(email: Optional[str], llm_response_obj: Any) -> Optional[dict]:
    """
    Deduct credits based on token usage. Returns updated usage row (or None on failure).
    If token aggregation is active, just adds to the aggregator and returns None.
    """
    try:
        if not email:
            email = get_current_email()
            
        total_tokens = extract_total_tokens(llm_response_obj)
        if total_tokens <= 0:
            return None
            
        # Check if aggregation is active
        agg = _token_aggregator.get()
        if agg is not None:
            agg["total_tokens"] = agg.get("total_tokens", 0) + total_tokens
            return None # Deferred
            
        # Immediate deduction
        db = PostgresDatabase()
        db.ensure_connection()
        row = deduct_tokens(db, email, total_tokens)
        try:
            db.close()
        except Exception:
            pass
        return {"total_tokens": total_tokens, **row}
    except LookupError:
        # ContextVar not found (shouldn't happen with default=None but good safety)
        pass
    except Exception:
        return None
