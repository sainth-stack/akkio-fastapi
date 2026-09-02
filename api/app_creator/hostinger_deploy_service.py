"""
Local Hostinger deploy — same VPS as the API, no SSH.

Flow: verify build (or rebuild) → optional pytest → health check → RUNNING.
Serves at {PUBLIC_BASE_URL}/app/{project_name}
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable, Optional

from api.app_creator.pipeline_helpers import public_base_url
from api.app_creator.static_app_router import _resolve_static_dir
from app_builder.schemas.files import GeneratedFiles
from app_builder.services.file_writer import file_writer
from app_builder.services.runtime_paths import resolve_project_root
from db.app_builder import get_app_builder_db

logger = logging.getLogger("app_builder")

IN_PROGRESS_STATUSES = frozenset({"QUEUED", "BUILDING", "TESTING", "DEPLOYING"})
TERMINAL_SUCCESS = "RUNNING"
TERMINAL_FAILURE = "FAILED"

StatusCallback = Callable[[str, str, Optional[str]], None]


class HostingerDeployService:
    def __init__(self, db=None):
        self.db = db or get_app_builder_db()

    def _should_run_tests(self, override: Optional[bool]) -> bool:
        if override is not None:
            return override
        return os.environ.get("DEPLOY_RUN_TESTS", "").lower() in ("1", "true", "yes")

    def _append_log(self, logs: list[str], line: str) -> str:
        logs.append(line)
        return "\n".join(logs)

    def _sync_disk_from_db(
        self,
        project_name: str,
        user_email: str,
        user_id: int | None,
    ) -> int:
        app = self.db.get_app_by_project_name(
            project_name, user_id=user_id, user_email=user_email
        )
        if not app:
            return 0
        files = app.get("generated_code_json") or {}
        if not isinstance(files, dict) or not files:
            return 0
        file_writer(project_name, GeneratedFiles(files=files))
        return len(files)

    def _run_pytest(self, project_name: str) -> tuple[bool, str]:
        project_root = resolve_project_root(project_name)
        backend_path = os.path.join(project_root, "backend")
        tests_dir = os.path.join(backend_path, "tests")

        if not os.path.isdir(backend_path):
            return True, "No backend directory — tests skipped"
        if not os.path.isdir(tests_dir):
            return True, "No tests/ directory — tests skipped"

        has_tests = any(
            f.startswith("test_") and f.endswith(".py")
            for f in os.listdir(tests_dir)
        )
        if not has_tests:
            return True, "No pytest files found — tests skipped"

        try:
            result = subprocess.run(
                ["pytest", "tests/"],
                cwd=backend_path,
                capture_output=True,
                timeout=int(os.environ.get("DEPLOY_PYTEST_TIMEOUT", "120")),
            )
            output = (result.stdout or b"").decode("utf-8", errors="replace")
            output += (result.stderr or b"").decode("utf-8", errors="replace")
            if result.returncode != 0:
                return False, output.strip() or f"pytest failed with exit code {result.returncode}"
            return True, output.strip() or "All tests passed"
        except subprocess.TimeoutExpired:
            return False, "pytest timed out"
        except FileNotFoundError:
            return True, "pytest not installed — tests skipped"
        except Exception as e:
            return False, str(e)

    def _health_check(self, project_name: str, public_base: str | None = None) -> tuple[bool, str]:
        base = public_base or public_base_url()
        url = f"{base.rstrip('/')}/app/{project_name}"
        try:
            import urllib.request

            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                code = resp.getcode()
                if code < 500:
                    return True, f"GET {url} → HTTP {code}"
                return False, f"GET {url} → HTTP {code}"
        except Exception as e:
            return False, f"Health check failed for {url}: {e}"

    def deploy(
        self,
        *,
        deployment_id: str,
        app_id: str | int | None,
        project_name: str,
        user_email: str,
        user_id: int | None,
        rebuild: bool = False,
        run_tests: Optional[bool] = None,
        on_status: Optional[StatusCallback] = None,
        public_base: str | None = None,
    ) -> dict:
        logs: list[str] = []
        base = (public_base or public_base_url()).rstrip("/")

        def emit(status: str, line: str, error: Optional[str] = None) -> None:
            self._append_log(logs, line)
            if on_status:
                on_status(status, line, error)

        project_root = resolve_project_root(project_name)
        if not os.path.isdir(project_root):
            msg = f"Project not found: {project_root}"
            emit(TERMINAL_FAILURE, msg, msg)
            return {"status": "error", "message": msg, "logs": "\n".join(logs)}

        app = None
        if app_id:
            app = self.db.get_app_builder_app(app_id, user_email=user_email, user_id=user_id)
        if not app:
            app = self.db.get_app_by_project_name(
                project_name, user_id=user_id, user_email=user_email
            )

        build_status = (app or {}).get("build_status")
        if not rebuild and build_status != "BUILD_SUCCESS":
            msg = (
                "Build must succeed before deploy. Run the app in the Build tab first, "
                "or redeploy with rebuild=true."
            )
            emit(TERMINAL_FAILURE, msg, msg)
            if app_id and user_email:
                self.db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=user_id,
                    pipeline_status="DEPLOY_FAILED",
                    pipeline_error=msg,
                )
            return {"status": "error", "message": msg, "logs": "\n".join(logs)}

        emit("QUEUED", f"Deploy queued for {project_name}")

        synced = self._sync_disk_from_db(project_name, user_email, user_id)
        if synced:
            emit("QUEUED", f"Synced {synced} files from database to disk")

        static_dir = _resolve_static_dir(project_name)
        needs_build = rebuild or not static_dir

        if needs_build:
            emit("BUILDING", "Running npm install + build...")
            from api.app_creator.app_creator import _build_frontend

            static_dir, err = _build_frontend(project_name, install=True, use_sandbox=False)
            if err:
                emit(TERMINAL_FAILURE, err, err)
                if app_id and user_email:
                    self.db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=user_id,
                        build_status="BUILD_FAILED",
                        build_error=err,
                        pipeline_status="DEPLOY_FAILED",
                        pipeline_error=err,
                    )
                return {"status": "error", "message": err, "logs": "\n".join(logs)}
            emit("BUILDING", "Build completed successfully")
            if app_id and user_email:
                self.db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=user_id,
                    build_status="BUILD_SUCCESS",
                    build_error=None,
                )
        else:
            emit("BUILDING", "Using existing build output (BUILD_SUCCESS)")

        if self._should_run_tests(run_tests):
            emit("TESTING", "Running pytest...")
            passed, test_output = self._run_pytest(project_name)
            if test_output:
                emit("TESTING", test_output[:4000])
            if not passed:
                msg = test_output or "Tests failed"
                emit(TERMINAL_FAILURE, msg, msg)
                if app_id and user_email:
                    self.db.update_app_builder_app(
                        app_id=app_id,
                        user_email=user_email,
                        user_id=user_id,
                        pipeline_status="TEST_FAILED",
                        pipeline_error=msg,
                    )
                return {"status": "error", "message": msg, "logs": "\n".join(logs)}
            emit("TESTING", "Tests passed (or skipped)")
        else:
            emit("TESTING", "Tests skipped (DEPLOY_RUN_TESTS not enabled)")

        live_url = f"{base}/app/{project_name}"
        backend_url = f"{base}/api/apps/{project_name}"
        emit("DEPLOYING", f"Registering live URL: {live_url}")

        if not static_dir:
            static_dir = _resolve_static_dir(project_name)
        if not static_dir:
            msg = "Build output not found after deploy build step"
            emit(TERMINAL_FAILURE, msg, msg)
            return {"status": "error", "message": msg, "logs": "\n".join(logs)}

        ok, health_msg = self._health_check(project_name, public_base=base)
        emit("DEPLOYING", health_msg)
        if not ok:
            emit(TERMINAL_FAILURE, health_msg, health_msg)
            if app_id and user_email:
                self.db.update_app_builder_app(
                    app_id=app_id,
                    user_email=user_email,
                    user_id=user_id,
                    pipeline_status="DEPLOY_FAILED",
                    pipeline_error=health_msg,
                )
            return {"status": "error", "message": health_msg, "logs": "\n".join(logs)}

        if app_id and user_email:
            self.db.update_app_builder_app(
                app_id=app_id,
                user_email=user_email,
                user_id=user_id,
                preview_url=live_url,
                live_url=live_url,
                pipeline_status=None,
                pipeline_error=None,
            )

        emit("RUNNING", f"Deploy complete — app live at {live_url}")
        return {
            "status": "success",
            "message": "Deployment successful",
            "frontend_url": live_url,
            "backend_url": backend_url,
            "live_url": live_url,
            "logs": "\n".join(logs),
        }
