import json
import os
import re
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from app_builder.services.file_writer import write_project_file
from app_builder.services.ui_test_runner import (
    E2E_DIR_NAME,
    default_smoke_spec,
    ensure_smoke_spec,
    merge_playwright_package_json,
    project_frontend_dir,
    write_playwright_config,
)


def _strip_markdown_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```typescript"):
        content = content[13:]
    elif content.startswith("```ts"):
        content = content[5:]
    elif content.startswith("```javascript"):
        content = content[13:]
    elif content.startswith("```js"):
        content = content[5:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _collect_frontend_sources(files: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path, content in files.items():
        if not path.startswith("frontend/"):
            continue
        if "node_modules" in path or path.endswith(".css"):
            continue
        if path.endswith((".tsx", ".ts", ".jsx", ".js")):
            if len(content) > 12000:
                content = content[:12000] + "\n// ... truncated ...\n"
            out[path] = content
    return out


def _parse_llm_test_files(raw: str) -> Dict[str, str]:
    cleaned = _strip_markdown_fence(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return {
                k: str(v)
                for k, v in data["files"].items()
                if k.startswith(f"frontend/{E2E_DIR_NAME}/")
            }
    except json.JSONDecodeError:
        pass

    # Fallback: single TypeScript file body
    if "import" in cleaned and "@playwright/test" in cleaned:
        return {f"frontend/{E2E_DIR_NAME}/app-flows.spec.ts": cleaned}

    return {}


def _sanitize_spec_filename(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    if not base.endswith(".spec.ts") and not base.endswith(".spec.js"):
        if base.endswith(".ts") or base.endswith(".js"):
            base = re.sub(r"\.(ts|js)$", ".spec.ts", base)
        else:
            base = f"{base}.spec.ts"
    return base


async def generate_tests(
    project_name: str,
    project_files: Dict[str, str],
    llm,
) -> Dict[str, str]:
    """
    Generates Playwright UI e2e tests for the generated frontend.
    """
    frontend_sources = _collect_frontend_sources(project_files)
    if not frontend_sources:
        return {}

    # Prioritize routing shell and pages
    priority_keys = sorted(
        frontend_sources.keys(),
        key=lambda p: (
            0 if p.endswith("App.tsx") or p.endswith("App.jsx") else 1,
            0 if "/pages/" in p else 1,
            p,
        ),
    )
    context_blocks: List[str] = []
    total = 0
    for key in priority_keys:
        chunk = frontend_sources[key]
        if total + len(chunk) > 48000:
            break
        context_blocks.append(f"**{key}**:\n{chunk}")
        total += len(chunk)

    system_prompt = """You are an expert QA engineer writing Playwright UI end-to-end tests for a React app.

**SCOPE — UI ONLY**
- Test user-visible behavior in the browser (navigation, forms, buttons, tables, dialogs).
- Do NOT call REST APIs directly or use pytest/FastAPI TestClient.
- Do NOT mock fetch unless the UI is explicitly a static mock-only demo.

**PLAYWRIGHT RULES**
1. Use `@playwright/test` only (`test`, `expect` from '@playwright/test').
2. Always open the app with:
   `async function openApp(page, baseURL) { await page.goto(baseURL!, { waitUntil: 'domcontentloaded' }); }`
   and call `openApp` in each test (or in `test.beforeEach`).
3. Prefer `getByRole`, `getByLabel`, `getByText` over CSS selectors.
4. Wait for UI: `await expect(locator).toBeVisible()` — avoid arbitrary long sleeps except brief 200–500ms after navigation if needed.
5. Cover: main page load, primary navigation between routes, one critical user flow per major page (create/edit/delete or submit) when the UI supports it.
6. Tests must be deterministic and pass against the built app served at PLAYWRIGHT_BASE_URL (already includes auth query param when needed).

**OUTPUT**
Return ONLY valid JSON (no markdown fences):
{
  "files": {
    "frontend/e2e/navigation.spec.ts": "<full file>",
    "frontend/e2e/flows.spec.ts": "<full file>"
  }
}
Use 2–4 spec files. Each file must be complete and runnable.
"""

    user_prompt = f"""Generate Playwright UI e2e tests for this project.

Frontend source files:
{chr(10).join(context_blocks)}

Return JSON with spec files under frontend/e2e/.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    written: Dict[str, str] = {}

    llm_files = _parse_llm_test_files(content)
    for rel_path, code in llm_files.items():
        fname = _sanitize_spec_filename(rel_path)
        normalized = f"frontend/{E2E_DIR_NAME}/{fname}"
        write_project_file(project_name, normalized, code.strip())
        written[normalized] = code.strip()

    smoke_path = ensure_smoke_spec(project_name)
    written[smoke_path] = default_smoke_spec()

    write_playwright_config(project_name)
    pkg = os.path.join(project_frontend_dir(project_name), "package.json")
    if os.path.isfile(pkg):
        merge_playwright_package_json(pkg)

    return written
