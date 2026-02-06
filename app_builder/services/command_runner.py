from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class ManagedProcess:
    id: str
    name: str
    command: List[str]
    cwd: str
    env: Dict[str, str]
    started_at: float = field(default_factory=time.time)
    pid: Optional[int] = None
    _proc: Optional[subprocess.Popen] = None
    _log_lines: List[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _max_lines: int = 5000

    def append_log(self, line: str) -> None:
        with self._lock:
            self._log_lines.append(line)
            if len(self._log_lines) > self._max_lines:
                # Keep a rolling window
                self._log_lines = self._log_lines[-self._max_lines :]

    def get_logs(self, since: int = 0) -> Tuple[int, List[str]]:
        with self._lock:
            lines = self._log_lines[since:]
            return len(self._log_lines), lines

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def return_code(self) -> Optional[int]:
        if self._proc is None:
            return None
        return self._proc.poll()

    def start(self) -> None:
        if self._proc is not None:
            return

        self.append_log(f"$ (cwd={self.cwd}) {' '.join(self.command)}\n")
        self._proc = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.pid = self._proc.pid

        def _reader():
            assert self._proc is not None
            try:
                if self._proc.stdout is None:
                    return
                for line in self._proc.stdout:
                    self.append_log(line)
            finally:
                rc = self.return_code()
                self.append_log(f"\n[process exited: {rc}]\n")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def terminate(self, timeout_s: float = 5.0) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout_s)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


class ProcessRegistry:
    """In-memory process registry for generated projects."""

    def __init__(self):
        self._lock = threading.Lock()
        self._procs: Dict[str, ManagedProcess] = {}

    def create(
        self, *, name: str, command: List[str], cwd: str, env: Optional[Dict[str, str]] = None
    ) -> ManagedProcess:
        proc_id = str(uuid.uuid4())
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)

        mp = ManagedProcess(
            id=proc_id,
            name=name,
            command=command,
            cwd=cwd,
            env=merged_env,
        )
        with self._lock:
            self._procs[proc_id] = mp
        return mp

    def get(self, proc_id: str) -> Optional[ManagedProcess]:
        with self._lock:
            return self._procs.get(proc_id)

    def terminate(self, proc_id: str) -> None:
        mp = self.get(proc_id)
        if not mp:
            return
        mp.terminate()

    def list(self) -> List[ManagedProcess]:
        with self._lock:
            return list(self._procs.values())


process_registry = ProcessRegistry()


def find_free_port() -> int:
    """Public helper used by API layer."""
    return _find_free_port()
