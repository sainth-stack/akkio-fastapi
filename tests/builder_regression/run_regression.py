#!/usr/bin/env python3
"""
Builder regression CI script.

Usage:
    python run_regression.py [--only new|legacy] [--ids fo_crm_saas,fo_hr_platform]

Exit codes:
    0  all checks passed
    1  one or more prompts failed

Requires:
    - AKKIO_API_URL  (default: http://localhost:8000)
    - AKKIO_TEST_TOKEN  (service account JWT)

Output:
    - Console table: id | track | pass/fail | build_time | tsc_errors | console_errors
    - tests/builder_regression/results/<run_id>.json  (full results)
    - Fails release if any legacy prompt regresses OR new-track pass rate < 95 %
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("AKKIO_API_URL", "http://localhost:8000")
TOKEN = os.environ.get("AKKIO_TEST_TOKEN", "")
RESULTS_DIR = Path(__file__).parent / "results"
TIMEOUT_GENERATE_S = 300  # 5 min max per prompt
NEW_TRACK_PASS_THRESHOLD = 0.95  # 95 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


async def _wait_for_generation(
    client: httpx.AsyncClient, project_name: str, deadline: float
) -> Dict[str, Any]:
    """Poll /api/apps/status until terminal state or deadline."""
    while time.time() < deadline:
        try:
            resp = await client.get(
                f"{BASE_URL}/api/apps/status",
                params={"project_name": project_name},
                headers=_headers(),
                timeout=15,
            )
            data = resp.json()
            status = data.get("generation_status") or data.get("status") or ""
            if status in {"complete", "failed", "error", "BUILD_COMPLETE", "BUILD_FAILED"}:
                return data
        except Exception:
            pass
        await asyncio.sleep(5)
    return {"status": "timeout", "error": "timed out waiting for generation"}


async def run_prompt(prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single prompt through the builder API and collect results.
    Returns a result dict with keys: id, track, passed, build_time_s,
    tsc_errors, console_errors, gate_passed, error.
    """
    pid = prompt_cfg["id"]
    t0 = time.time()
    result: Dict[str, Any] = {
        "id": pid,
        "track": prompt_cfg["track"],
        "prompt": prompt_cfg["prompt"][:80] + "…",
        "passed": False,
        "build_time_s": 0,
        "tsc_errors": [],
        "console_errors": [],
        "gate_passed": None,
        "error": None,
        "files_count": 0,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Create app
        try:
            cr = await client.post(
                f"{BASE_URL}/api/apps/create",
                json={"name": f"regtest_{pid}_{uuid.uuid4().hex[:6]}",
                      "requirement": prompt_cfg["prompt"]},
                headers=_headers(),
            )
            cr.raise_for_status()
            app_data = cr.json()
            project_name = app_data.get("project_name") or app_data.get("name")
            app_id = app_data.get("id") or app_data.get("app_id")
        except Exception as e:
            result["error"] = f"create failed: {e}"
            result["build_time_s"] = round(time.time() - t0, 1)
            return result

        # 2. Trigger codegen via planning API
        try:
            pr = await client.post(
                f"{BASE_URL}/api/planning/start",
                json={"project_name": project_name, "app_id": app_id,
                      "requirement": prompt_cfg["prompt"]},
                headers=_headers(),
                timeout=60,
            )
            pr.raise_for_status()
        except Exception as e:
            result["error"] = f"planning start failed: {e}"
            result["build_time_s"] = round(time.time() - t0, 1)
            return result

        # 3. Wait for generation
        deadline = t0 + TIMEOUT_GENERATE_S
        status_data = await _wait_for_generation(client, project_name, deadline)

        if status_data.get("status") == "timeout":
            result["error"] = "generation timed out"
            result["build_time_s"] = round(time.time() - t0, 1)
            return result

        # 4. tsc verify via run-checks
        try:
            checks_resp = await client.post(
                f"{BASE_URL}/api/test-suite/run-checks",
                json={"project_name": project_name},
                headers=_headers(),
                timeout=120,
            )
            checks = checks_resp.json()
            tsc_errs = checks.get("errors") or []
            gate_passed = checks.get("passed", False)
            gate_data = checks.get("gate_results") or {}
            console_errs: List[str] = []
            for route_r in (gate_data.get("failing_routes") or []):
                console_errs.extend(route_r.get("console_errors") or [])

            result["tsc_errors"] = tsc_errs
            result["console_errors"] = console_errs
            result["gate_passed"] = gate_passed
            result["passed"] = gate_passed and len(tsc_errs) == 0

            # File count from status
            files = status_data.get("files") or {}
            result["files_count"] = len(files)

        except Exception as e:
            result["error"] = f"checks failed: {e}"

    result["build_time_s"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_table(results: List[Dict[str, Any]]) -> None:
    header = f"{'ID':<30} {'TRACK':<15} {'PASS':<6} {'TIME':>6}s {'TSC':>4} {'CONSOLE':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(
            f"{r['id']:<30} {r['track']:<15} {icon:<6} {r['build_time_s']:>6.1f}  "
            f"{len(r['tsc_errors']):>4}  {len(r['console_errors']):>7}"
        )
        if r.get("error"):
            print(f"  ⚠  {r['error']}")
    print("=" * len(header))


def _save_results(results: List[Dict[str, Any]], run_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_id}.json"
    out.write_text(json.dumps({
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }, indent=2))
    return out


def _evaluate(results: List[Dict[str, Any]]) -> bool:
    legacy = [r for r in results if r["track"] == "legacy"]
    new_track = [r for r in results if r["track"] == "frontend_only"]

    legacy_ok = all(r["passed"] for r in legacy)
    if not legacy_ok:
        failed_ids = [r["id"] for r in legacy if not r["passed"]]
        print(f"\n🔴 LEGACY REGRESSION: {failed_ids}")

    if new_track:
        pass_rate = sum(1 for r in new_track if r["passed"]) / len(new_track)
        rate_ok = pass_rate >= NEW_TRACK_PASS_THRESHOLD
        print(f"\nNew-track pass rate: {pass_rate:.0%}  (threshold {NEW_TRACK_PASS_THRESHOLD:.0%})")
        if not rate_ok:
            print(f"🔴 NEW-TRACK BELOW THRESHOLD ({pass_rate:.0%} < {NEW_TRACK_PASS_THRESHOLD:.0%})")
    else:
        rate_ok = True

    medians = [r["build_time_s"] for r in new_track if r["passed"]]
    if medians:
        medians.sort()
        median = medians[len(medians) // 2]
        print(f"Median e2e time (passing new-track): {median:.1f}s")

    return legacy_ok and rate_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    from tests.builder_regression.prompts import ALL_PROMPTS

    prompts = ALL_PROMPTS
    if args.only == "new":
        prompts = [p for p in prompts if p["track"] == "frontend_only"]
    elif args.only == "legacy":
        prompts = [p for p in prompts if p["track"] == "legacy"]

    if args.ids:
        wanted = set(args.ids.split(","))
        prompts = [p for p in prompts if p["id"] in wanted]

    if not prompts:
        print("No prompts selected.")
        return 0

    print(f"\nRunning {len(prompts)} regression prompt(s) against {BASE_URL}")

    tasks = [run_prompt(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    results = list(results)

    _print_table(results)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = _save_results(results, run_id)
    print(f"\nResults saved → {out_path}")

    passed = _evaluate(results)
    print("\n" + ("✅ All checks passed." if passed else "❌ One or more checks FAILED."))
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Akkio builder regression suite")
    parser.add_argument("--only", choices=["new", "legacy"], default=None,
                        help="Run only new-track or legacy prompts")
    parser.add_argument("--ids", default=None,
                        help="Comma-separated list of prompt IDs to run")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
