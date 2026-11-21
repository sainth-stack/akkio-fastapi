"""
Per-app generation metrics: stage durations, retry counts, fix success rates.

Stored inside app metadata JSONB under the key "generation_metrics".
Lightweight — pure Python, no external deps.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class GenerationMetrics:
    """Collects metrics for one generation run."""

    def __init__(self, app_id: Optional[str] = None, project_name: str = ""):
        self.app_id = app_id
        self.project_name = project_name
        self._start = time.monotonic()
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.fix_attempts: List[Dict[str, Any]] = []
        self.gate_results: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    # ── Stage timing ─────────────────────────────────────────────────────────

    def stage_start(self, stage: str) -> None:
        self.stages[stage] = {"start": time.monotonic(), "end": None, "duration_s": None}

    def stage_end(self, stage: str) -> float:
        if stage not in self.stages:
            self.stage_start(stage)
        s = self.stages[stage]
        s["end"] = time.monotonic()
        s["duration_s"] = round(s["end"] - s["start"], 2)
        return s["duration_s"]

    @contextmanager
    def time_stage(self, stage: str):
        self.stage_start(stage)
        try:
            yield
        finally:
            self.stage_end(stage)

    # ── Fix tracking ──────────────────────────────────────────────────────────

    def record_fix_attempt(
        self,
        stage: str,
        attempt: int,
        offending_file: str,
        succeeded: bool,
    ) -> None:
        self.fix_attempts.append({
            "stage": stage,
            "attempt": attempt,
            "file": offending_file,
            "succeeded": succeeded,
            "ts": time.monotonic() - self._start,
        })

    @property
    def fix_success_rate(self) -> float:
        if not self.fix_attempts:
            return 1.0
        succeeded = sum(1 for a in self.fix_attempts if a["succeeded"])
        return round(succeeded / len(self.fix_attempts), 2)

    @property
    def total_fix_attempts(self) -> int:
        return len(self.fix_attempts)

    # ── Gate results ──────────────────────────────────────────────────────────

    def record_gate(
        self,
        route: str,
        passed: bool,
        console_errors: int = 0,
        screenshot_path: str = "",
        error: str = "",
    ) -> None:
        self.gate_results.append({
            "route": route,
            "passed": passed,
            "console_errors": console_errors,
            "screenshot": screenshot_path,
            "error": error,
            "ts": round(time.monotonic() - self._start, 1),
        })

    @property
    def gate_passed(self) -> bool:
        if not self.gate_results:
            return True  # no gate = not blocked
        return all(r["passed"] for r in self.gate_results)

    # ── Summary ───────────────────────────────────────────────────────────────

    def record_error(self, error: str) -> None:
        self.errors.append(error[:500])

    @property
    def total_duration_s(self) -> float:
        return round(time.monotonic() - self._start, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict for storage in app metadata JSONB."""
        return {
            "total_duration_s": self.total_duration_s,
            "stages": {
                k: {"duration_s": v.get("duration_s")}
                for k, v in self.stages.items()
            },
            "fix_attempts": self.fix_attempts,
            "fix_success_rate": self.fix_success_rate,
            "total_fixes": self.total_fix_attempts,
            "gate_results": self.gate_results,
            "gate_passed": self.gate_passed,
            "errors": self.errors,
        }

    def save_to_db(self, db: Any, user_email: str, uid: Optional[int] = None) -> None:
        """Persist metrics into app metadata. Best-effort — never raises."""
        if not self.app_id:
            return
        try:
            db.update_app_builder_app(
                app_id=self.app_id,
                user_email=user_email,
                user_id=uid,
                plan_json={"_metrics": self.to_dict()},  # merges into JSONB
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level convenience: attach metrics to pipeline run_pipeline()
# ---------------------------------------------------------------------------

_NOOP = GenerationMetrics()

def get_noop() -> GenerationMetrics:
    return GenerationMetrics()
