"""
Build generated frontends inside an E2B cloud sandbox (isolated Linux + Node),
then copy dist/ or build/ back into the project on this machine.

Architecture (unchanged from single-backend mode):
- Runtime API for the generated app stays on this FastAPI process: ``/api/apps/{projectId}``.
- E2B is used only for ``npm install`` + ``npm run build`` when enabled — not for hosting the backend.

Requires: pip package ``e2b``, env ``E2B_API_KEY``. Enable with ``APP_BUILDER_USE_E2B=1`` or ``use_sandbox`` on /run.
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import tarfile
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.app_creator.cra_npm_patch import prepare_frontend_dir_on_disk

logger = logging.getLogger("app_builder")


def _is_transient_network_error(exc: BaseException) -> bool:
    """Dropped connections, read timeouts on large downloads — safe to reconnect and retry."""
    try:
        import httpcore
    except ImportError:
        httpcore = None  # type: ignore
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore
    if httpcore is not None and isinstance(
        exc,
        (
            httpcore.ReadError,
            httpcore.ConnectError,
            httpcore.WriteError,
            httpcore.ReadTimeout,
            httpcore.ConnectTimeout,
        ),
    ):
        return True
    if httpx is not None and isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (54, 104, 32):  # reset, pipe, broken pipe
        return True
    msg = str(exc).lower()
    if "connection reset" in msg or "broken pipe" in msg or "connection aborted" in msg:
        return True
    if "read operation timed out" in msg or "timed out" in msg:
        return True
    return False


def _output_tar_read_timeout() -> float:
    """
    Seconds for HTTP read when downloading /tmp/frontend-out.tar.gz (can be large / slow).
    Set E2B_OUTPUT_TAR_READ_TIMEOUT_SEC=0 for no cap (SDK passes through to httpx).
    Default 3600.
    """
    raw = os.environ.get("E2B_OUTPUT_TAR_READ_TIMEOUT_SEC", "3600").strip().lower()
    if raw in ("0", "none", "unlimited"):
        return 0.0
    return float(raw)


def _reconnect_sandbox(holder: List[Any], sandbox_timeout_sec: int, request_timeout_val: int) -> None:
    """Replace holder[0] with a fresh client for the same sandbox_id (new HTTP connections)."""
    from e2b import Sandbox

    sid = holder[0].sandbox_id
    holder[0] = Sandbox.connect(
        sandbox_id=sid,
        timeout=sandbox_timeout_sec,
        request_timeout=request_timeout_val,
    )
    logger.info("[e2b_build] Reconnected to sandbox %s", sid)


def _with_reconnect(
    holder: List[Any],
    sandbox_timeout_sec: int,
    request_timeout_val: int,
    operation: str,
    fn: Callable[[Any], Any],
    max_attempts: int = 8,
) -> Any:
    """Run fn(sandbox); on transient network errors (incl. read timeout on large downloads) reconnect and retry."""
    last: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn(holder[0])
        except Exception as e:
            last = e
            if not _is_transient_network_error(e) or attempt == max_attempts - 1:
                raise
            logger.warning(
                "[e2b_build] %s transient error (attempt %s/%s): %s",
                operation,
                attempt + 1,
                max_attempts,
                e,
            )
            _reconnect_sandbox(holder, sandbox_timeout_sec, request_timeout_val)
    raise last  # pragma: no cover

# Default E2B templates do not provide a writable /workspace; use /tmp (override with E2B_SANDBOX_WORKDIR).
def _sandbox_frontend_root() -> str:
    return os.environ.get("E2B_SANDBOX_WORKDIR", "/tmp/akkio-frontend-build").rstrip("/")


# Match app_creator exclusions for tarball size
_TAR_SKIP_NAMES = frozenset(
    {
        "node_modules",
        "dist",
        "build",
        ".git",
        ".next",
        "__pycache__",
        ".venv",
        "venv",
    }
)


def _tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    parts = tarinfo.name.strip("./").split("/")
    if any(p in _TAR_SKIP_NAMES for p in parts):
        return None
    return tarinfo


def _make_frontend_tarball(frontend_dir: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    with tarfile.open(path, "w:gz") as tar:
        for name in sorted(os.listdir(frontend_dir)):
            if name in _TAR_SKIP_NAMES:
                continue
            full = os.path.join(frontend_dir, name)
            tar.add(full, arcname=name, filter=_tar_filter)
    return path


def _cmd_error(prefix: str, result) -> str:
    out = (result.stdout or "") + (result.stderr or "")
    out = out.strip()[:2000]
    if not out:
        out = f"exit_code={result.exit_code}"
    return f"{prefix} (exit {result.exit_code}): {out}"


def _export_env_sh(env: Dict[str, str]) -> str:
    return " && ".join(f"export {k}={shlex.quote(str(v))}" for k, v in env.items())


def _run_npm_step_bg(
    holder: List[Any],
    sandbox_timeout_sec: int,
    request_timeout_val: int,
    work: str,
    build_env: Dict[str, str],
    npm_invocation: str,
    exit_file: str,
    log_file: str,
    phase: str,
) -> Optional[str]:
    """
    Run npm in a background subshell, then poll via filesystem API (HTTP GET), not process RPC.
    Process streams often hit 'connection reset' after the bg job; file reads use a separate path.
    """
    from e2b import FileNotFoundException

    env_sh = _export_env_sh(build_env)
    work_q = shlex.quote(work)
    exit_q = shlex.quote(exit_file)
    log_q = shlex.quote(log_file)
    inner = f"{env_sh} && cd {work_q} && {npm_invocation}; echo $? > {exit_q}"
    start_cmd = f"rm -f {exit_q} {log_q} && ({inner}) > {log_q} 2>&1 &"

    r = _with_reconnect(
        holder,
        sandbox_timeout_sec,
        request_timeout_val,
        "start_" + phase.replace(" ", "_"),
        lambda s: s.commands.run(start_cmd, cwd="/", timeout=120),
    )
    if r.exit_code != 0:
        return _cmd_error(f"Failed to start {phase}", r)

    max_wait = int(os.environ.get("E2B_NPM_STEP_MAX_SEC", "2400"))
    poll_interval = float(os.environ.get("E2B_NPM_POLL_INTERVAL_SEC", "4"))
    rt_files = float(os.environ.get("E2B_FILES_REQUEST_TIMEOUT_SEC", "120"))
    t0 = time.time()

    while time.time() - t0 < max_wait:
        try:
            raw = holder[0].files.read(exit_file, format="text", request_timeout=rt_files)
        except FileNotFoundException:
            time.sleep(poll_interval)
            continue
        except Exception as e:
            if _is_transient_network_error(e):
                logger.warning("[e2b_build] poll %s: reconnect after %s", phase, e)
                _reconnect_sandbox(holder, sandbox_timeout_sec, request_timeout_val)
                time.sleep(0.5)
                continue
            raise

        try:
            code = int((raw or "").strip())
        except ValueError:
            return f"{phase}: could not read exit code from sandbox"

        if code != 0:
            try:
                log_body = holder[0].files.read(log_file, format="text", request_timeout=rt_files)
            except Exception:
                log_body = ""
            tail = (log_body or "")[-8000:]
            return f"{phase} failed (exit {code}). Log tail:\n{tail}"
        return None

    return f"{phase} timed out after {max_wait}s (see E2B_NPM_STEP_MAX_SEC)"


def build_frontend_in_e2b(
    project_name: str,
    frontend_dir: str,
    install: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run npm install + npm run build in E2B, copy dist or build back to frontend_dir.

    Returns (static_dir_path, error_message).
    """
    try:
        from e2b import Sandbox
    except ImportError:
        return None, "E2B sandbox build requires the `e2b` package. Add it to requirements and reinstall."

    if not os.environ.get("E2B_API_KEY"):
        return None, "E2B sandbox build requires E2B_API_KEY in the environment."

    if not os.path.isdir(frontend_dir) or not os.path.isfile(os.path.join(frontend_dir, "package.json")):
        return None, "No frontend found"

    tarball = None
    sandbox_holder: List[Any] = [None]
    try:
        if prepare_frontend_dir_on_disk(frontend_dir):
            logger.info("[e2b_build] Patched frontend for CRA (ajv overrides + .env)")
        tarball = _make_frontend_tarball(frontend_dir)
        sandbox_timeout_sec = int(os.environ.get("E2B_SANDBOX_TIMEOUT_SEC", "3600"))
        request_timeout_val = int(os.environ.get("E2B_REQUEST_TIMEOUT_SEC", "0"))
        template = os.environ.get("E2B_SANDBOX_TEMPLATE") or None
        create_kw = {
            "timeout": sandbox_timeout_sec,
            "metadata": {"project": project_name, "purpose": "akkio-frontend-build"},
            "request_timeout": request_timeout_val,
        }
        if template:
            create_kw["template"] = template
        sandbox_holder[0] = Sandbox.create(**create_kw)

        def log(msg: str) -> None:
            logger.info("[e2b_build] %s", msg)
            if on_log:
                on_log(msg)

        work = _sandbox_frontend_root()
        log("Uploading frontend sources to sandbox...")
        with open(tarball, "rb") as f:
            blob = f.read()

        def _upload_src(s):
            s.files.write("/tmp/frontend-src.tar.gz", blob)

        _with_reconnect(
            sandbox_holder,
            sandbox_timeout_sec,
            request_timeout_val,
            "upload_sources",
            _upload_src,
        )

        r = _with_reconnect(
            sandbox_holder,
            sandbox_timeout_sec,
            request_timeout_val,
            "extract_sources",
            lambda s: s.commands.run(
                f"rm -rf {work} && mkdir -p {work} && tar xzf /tmp/frontend-src.tar.gz -C {work}",
                cwd="/",
                timeout=120,
            ),
        )
        if r.exit_code != 0:
            return None, _cmd_error("Failed to extract sources in sandbox", r)

        public_url = f"/app/{project_name}"
        build_env = {"PUBLIC_URL": public_url, "CI": "true"}

        if install:
            log("Running npm install in sandbox (background + poll; avoids stream timeouts)...")
            err = _run_npm_step_bg(
                sandbox_holder,
                sandbox_timeout_sec,
                request_timeout_val,
                work,
                build_env,
                "npm install --legacy-peer-deps",
                "/tmp/akkio-npm-install.exit",
                "/tmp/akkio-npm-install.log",
                "npm install",
            )
            if err:
                return None, err

        log("Running npm run build in sandbox (background + poll)...")
        err = _run_npm_step_bg(
            sandbox_holder,
            sandbox_timeout_sec,
            request_timeout_val,
            work,
            build_env,
            "npm run build",
            "/tmp/akkio-npm-build.exit",
            "/tmp/akkio-npm-build.log",
            "npm run build",
        )
        if err:
            return None, err

        r = _with_reconnect(
            sandbox_holder,
            sandbox_timeout_sec,
            request_timeout_val,
            "archive_dist",
            lambda s: s.commands.run(
                f"if [ -d {work}/dist ]; then cd {work}/dist && tar czf /tmp/frontend-out.tar.gz .; "
                f"elif [ -d {work}/build ]; then cd {work}/build && tar czf /tmp/frontend-out.tar.gz .; "
                "else echo NO_OUTPUT_DIR; exit 1; fi",
                cwd="/",
                timeout=600,
            ),
        )
        if r.exit_code != 0:
            return None, _cmd_error("Build produced no dist/ or build/ in sandbox", r)

        rt_files = float(os.environ.get("E2B_FILES_REQUEST_TIMEOUT_SEC", "120"))
        tar_read_timeout = _output_tar_read_timeout()
        logger.info(
            "[e2b_build] Downloading build output tarball (timeout=%ss; set E2B_OUTPUT_TAR_READ_TIMEOUT_SEC if needed)",
            tar_read_timeout if tar_read_timeout else "unlimited",
        )

        def _read_out(s):
            return s.files.read(
                "/tmp/frontend-out.tar.gz",
                format="bytes",
                request_timeout=tar_read_timeout,
            )

        data = _with_reconnect(
            sandbox_holder,
            sandbox_timeout_sec,
            request_timeout_val,
            "read_output_tar",
            _read_out,
            max_attempts=12,
        )
        if not data:
            return None, "Sandbox build output archive was empty"

        def _detect_out_sub(s):
            if s.files.exists(f"{work}/dist", request_timeout=rt_files):
                return "dist"
            if s.files.exists(f"{work}/build", request_timeout=rt_files):
                return "build"
            return "dist"

        out_sub = _with_reconnect(
            sandbox_holder,
            sandbox_timeout_sec,
            request_timeout_val,
            "detect_out_dir",
            _detect_out_sub,
        )

        out_dir = os.path.join(frontend_dir, out_sub)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(bytes(data))
            tmp.flush()
            with tarfile.open(tmp.name, "r:gz") as tar:
                # Python 3.12+ safe extraction
                kw = {}
                if hasattr(tarfile, "data_filter"):
                    kw["filter"] = tarfile.data_filter
                tar.extractall(out_dir, **kw)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        log(f"Synced sandbox build to ./{out_sub}/")
        return out_dir, None

    except Exception as e:
        logger.exception("[e2b_build] failed: %s", e)
        return None, f"E2B sandbox build error: {e}"
    finally:
        if sandbox_holder[0] is not None:
            try:
                sandbox_holder[0].kill()
            except Exception as e:
                logger.warning("[e2b_build] sandbox.kill failed: %s", e)
        if tarball and os.path.isfile(tarball):
            try:
                os.unlink(tarball)
            except OSError:
                pass


def e2b_available() -> bool:
    try:
        __import__("e2b")
    except ImportError:
        return False
    return bool(os.environ.get("E2B_API_KEY"))
