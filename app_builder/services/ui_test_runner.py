"""Run Playwright UI tests for generated app frontends."""
from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional, Tuple

from api.app_creator.pipeline_helpers import npm_build_timeout, npm_install_timeout, public_base_url
from api.app_creator.static_app_router import _resolve_static_dir
from api.app_creator.vite_build import is_vite_frontend, run_vite_build
from app_builder.services.runtime_paths import get_projects_dir

PLAYWRIGHT_VERSION = "^1.49.1"
E2E_DIR_NAME = "e2e"


def project_frontend_dir(project_name: str) -> str:
    return os.path.join(get_projects_dir(), project_name, "frontend")


def project_e2e_dir(project_name: str) -> str:
    return os.path.join(project_frontend_dir(project_name), E2E_DIR_NAME)


def list_ui_test_files(project_name: str) -> List[str]:
    e2e_dir = project_e2e_dir(project_name)
    if not os.path.isdir(e2e_dir):
        return []
    files = []
    for name in sorted(os.listdir(e2e_dir)):
        if name.endswith(".spec.ts") or name.endswith(".spec.js"):
            files.append(name)
    return files


def merge_playwright_package_json(package_path: str) -> None:
    if not os.path.isfile(package_path):
        return
    with open(package_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scripts = data.setdefault("scripts", {})
    scripts["test:e2e"] = "playwright test"
    dev = data.setdefault("devDependencies", {})
    dev["@playwright/test"] = PLAYWRIGHT_VERSION
    with open(package_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def write_playwright_config(project_name: str) -> str:
    rel_path = f"frontend/playwright.config.ts"
    content = """import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:8000';

export default defineConfig({
  testDir: './e2e',
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
"""
    from app_builder.services.file_writer import write_project_file

    write_project_file(project_name, rel_path, content)
    return rel_path


def default_smoke_spec() -> str:
    return """import { test, expect } from '@playwright/test';

async function openApp(page: import('@playwright/test').Page, baseURL?: string) {
  const url = (baseURL || process.env.PLAYWRIGHT_BASE_URL || '').trim();
  if (!url) {
    throw new Error('PLAYWRIGHT_BASE_URL is not set');
  }
  await page.goto(url, { waitUntil: 'domcontentloaded' });
}

test.describe('App smoke', () => {
  test('loads without console errors', async ({ page, baseURL }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    await openApp(page, baseURL);
    await expect(page.locator('body')).toBeVisible();
    await page.waitForTimeout(500);
    expect(errors, errors.join('\\n')).toHaveLength(0);
  });

  test('shows primary heading or landmark main content', async ({ page, baseURL }) => {
    await openApp(page, baseURL);
    const heading = page.getByRole('heading').first();
    const main = page.getByRole('main').first();
    const hasHeading = await heading.isVisible().catch(() => false);
    const hasMain = await main.isVisible().catch(() => false);
    expect(hasHeading || hasMain).toBeTruthy();
  });
});
"""


def ensure_smoke_spec(project_name: str) -> str:
    from app_builder.services.file_writer import write_project_file

    rel = f"frontend/{E2E_DIR_NAME}/smoke.spec.ts"
    write_project_file(project_name, rel, default_smoke_spec())
    return rel


def ensure_frontend_built(project_name: str) -> Tuple[bool, str]:
    if _resolve_static_dir(project_name):
        return True, ""
    frontend_dir = project_frontend_dir(project_name)
    if not os.path.isdir(frontend_dir):
        return False, "Frontend directory not found. Generate and build the app first."
    if is_vite_frontend(frontend_dir):
        dist, err, log = run_vite_build(frontend_dir, project_name, install=True)
        if dist:
            return True, log
        return False, err or "Vite build failed"
    return False, "App is not built. Open App View and run the app (build) before running UI tests."


def build_preview_url(project_name: str, access_token: Optional[str], request_base: Optional[str]) -> str:
    base = public_base_url(request_base=request_base).rstrip("/")
    url = f"{base}/app/{project_name}"
    token = (access_token or "").strip()
    if token:
        url = f"{url}?access_token={token}"
    return url


async def _run_cmd(cmd: List[str], cwd: str, env: dict, timeout: int) -> Tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        return 1, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    out = (stdout or b"").decode("utf-8", errors="replace")
    out += (stderr or b"").decode("utf-8", errors="replace")
    return process.returncode or 0, out


async def run_ui_tests(
    project_name: str,
    test_files: Optional[List[str]] = None,
    access_token: Optional[str] = None,
    request_base: Optional[str] = None,
) -> Tuple[bool, str, int]:
    frontend_dir = project_frontend_dir(project_name)
    e2e_dir = project_e2e_dir(project_name)
    if not os.path.isdir(e2e_dir):
        return False, "No UI tests found. Click Generate Test Scripts first.", 1

    ok, build_log = ensure_frontend_built(project_name)
    if not ok:
        return False, build_log, 1

    package_json = os.path.join(frontend_dir, "package.json")
    if not os.path.isfile(package_json):
        return False, "frontend/package.json not found", 1

    merge_playwright_package_json(package_json)
    write_playwright_config(project_name)

    env = os.environ.copy()
    env["PLAYWRIGHT_BASE_URL"] = build_preview_url(project_name, access_token, request_base)

    logs: List[str] = []
    if build_log.strip():
        logs.append("=== Build ===\n" + build_log.strip())

    logs.append(f"=== Preview URL ===\n{env['PLAYWRIGHT_BASE_URL']}\n")

    install_code, install_out = await _run_cmd(
        ["npm", "install", "--legacy-peer-deps"],
        frontend_dir,
        env,
        npm_install_timeout(),
    )
    logs.append("=== npm install ===\n" + install_out[-6000:])
    if install_code != 0:
        return False, "\n".join(logs), install_code

    pw_install_code, pw_install_out = await _run_cmd(
        ["npx", "playwright", "install", "chromium"],
        frontend_dir,
        env,
        max(npm_install_timeout(), 180),
    )
    logs.append("=== playwright install ===\n" + pw_install_out[-4000:])
    if pw_install_code != 0:
        return False, "\n".join(logs), pw_install_code

    cmd = ["npx", "playwright", "test"]
    if test_files:
        for tf in test_files:
            cmd.append(os.path.join(E2E_DIR_NAME, tf))

    test_code, test_out = await _run_cmd(cmd, frontend_dir, env, npm_build_timeout() + 120)
    logs.append("=== playwright test ===\n" + test_out)

    return test_code == 0, "\n".join(logs), test_code
