import os
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from llm_helper import get_llm_for_user
from app_builder.services.surgical_patch import apply_surgical_patch


async def tailor_template_to_requirement(
    template_files: Dict[str, str],
    user_requirement: str,
    template_name: str,
) -> Dict[str, str]:
    """
    Tailor template files to user's requirement: update styles, colors, field names, functionality.
    Uses same surgical update logic as update_code_from_chat but on in-memory files.
    """
    if not template_files:
        return template_files
    req = (user_requirement or "").strip()
    req_lower = req.lower()
    files_to_tailor = [
        p for p in ["frontend/src/styles.css", "frontend/styles.css", "frontend/src/App.js", "frontend/App.js"]
        if p in template_files
    ]
    if not files_to_tailor:
        files_to_tailor = [p for p in template_files if p.endswith((".css", ".js", ".jsx")) and "node_modules" not in p][:3]
    if not files_to_tailor:
        return template_files
    llm = get_llm_for_user(None, temperature=0.15)
    result = dict(template_files)
    for rel_path in files_to_tailor:
        existing = result.get(rel_path, "")
        if not existing or len(existing) > 15000:
            continue
        prompt = f"""You are updating a file for a "{template_name}" template. The user requested: "{req}"

CURRENT FILE ({rel_path}):
```
{existing[:8000]}{"..." if len(existing) > 8000 else ""}
```

Update this file to match the user's requirement:
- STYLING: Adjust colors, fonts, spacing, borders, shadows if they mentioned appearance, theme, or layout
- ALIGNMENT: Fix alignment, padding, margins, centering, grid/flex layout per their preference
- LAYOUT: Improve spacing, whitespace, component positioning for a clean, polished look
- FIELDS/FEATURES: Change field names, labels, or add features if they mentioned functionality
- FOR ideas-generator (LLM template) App.js: If user asks for LinkedIn post/social media → change GEN_TYPE to "linkedin_post". For travel planner/suggestions → "travel". For translator/translate → "translate". For generic GenAI → "general". Keep "ideas" only for ideas/brainstorm.
Keep the structure and don't break the app. Return the COMPLETE updated file.

Respond ONLY with a JSON object: {{"content": "FULL_UPDATED_FILE_CONTENT"}}"""
        try:
            resp = await llm.ainvoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r"\{[\s\S]*\"content\"[\s\S]*\}", text)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    try:
                        from json_repair import repair_json
                        data = json.loads(repair_json(m.group(0)))
                    except Exception:
                        continue
                content = data.get("content", "").strip()
                if content and len(content) > 50:
                    result[rel_path] = content
                    print(f"[tailor] updated {rel_path} for requirement")
        except Exception as e:
            print(f"[tailor] skip {rel_path}: {e}")
            continue
    return result


_DESIGN_INTENT_PROMPT = """You are a senior product designer with deep knowledge of global brands, design systems, and color psychology.

The user wants to update an app's visual design. Understand their intent and extract the exact color scheme they want.

User request: "{request}"
Current primary color: {current_primary}
Current background: {current_background}

Your job:
- Understand ANY reference: brand names (Myntra, Zomato, Spotify, Netflix, Airbnb...), moods ("luxury", "playful", "dark"), industries ("healthcare", "fintech"), or explicit colors ("#FF3F6C", "deep orange")
- Use your knowledge of real brand color systems
- If they say "like Myntra" → use Myntra's actual brand colors (hot pink #FF3F6C)
- If they say "dark mode" → dark background #121212, light text
- If no visual change is requested → return the current colors unchanged

Return ONLY a valid JSON object, nothing else:
{{
  "primary": "#HEX",
  "primary_dark": "#HEX",
  "primary_light": "#HEX",
  "secondary": "#HEX",
  "accent": "#HEX",
  "background": "#HEX",
  "surface": "#HEX",
  "text": "#HEX",
  "muted": "#HEX",
  "border": "#HEX",
  "style": "minimal|bold|dark|corporate|playful|luxury",
  "reasoning": "one sentence explaining the color choices"
}}"""


async def _extract_design_intent_with_llm(
    user_request: str,
    existing_tokens: Optional[Dict[str, Any]],
    llm,
) -> Dict[str, Any]:
    """
    LLM-powered design intent extraction.
    Understands brands, moods, color names, hex codes — no hardcoding needed.
    Falls back to existing tokens if LLM fails or no color change is requested.
    """
    existing_colors = (existing_tokens or {}).get("colors") or {}
    current_primary = existing_colors.get("primary", "#1565C0")
    current_background = existing_colors.get("background", "#FFFFFF")

    prompt = _DESIGN_INTENT_PROMPT.format(
        request=user_request,
        current_primary=current_primary,
        current_background=current_background,
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)

        # Extract JSON
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return dict(existing_tokens or {})

        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(m.group(0)))
            except Exception:
                return dict(existing_tokens or {})

        # Validate — must have real hex codes
        valid_hex = re.compile(r'^#[0-9a-fA-F]{6}$')
        extracted = {
            k: v for k, v in data.items()
            if isinstance(v, str) and (valid_hex.match(v) or k in ("style", "reasoning"))
        }
        if not extracted.get("primary"):
            return dict(existing_tokens or {})

        # Merge into existing tokens
        merged = dict(existing_tokens or {})
        merged["colors"] = {**existing_colors, **{k: v for k, v in extracted.items() if k != "reasoning"}}
        print(f"[design-intent] LLM extracted: primary={extracted.get('primary')} style={extracted.get('style')} | reasoning: {extracted.get('reasoning', '')}")
        return merged

    except Exception as e:
        print(f"[design-intent] LLM extraction failed ({e}) — using existing tokens")
        return dict(existing_tokens or {})


def _color_uiux_from_tokens(tokens: Dict[str, Any]) -> str:
    """Build a UIUX hint string from extracted design tokens."""
    colors = (tokens or {}).get("colors") or {}
    hints = []
    if colors.get("primary"):
        hints.append(f"Primary: {colors['primary']}")
    if colors.get("background"):
        hints.append(f"Background: {colors['background']}")
    if colors.get("style"):
        hints.append(f"Style: {colors['style']}")
    return "\n".join(hints)


# _color_uiux_from_request removed — replaced by LLM-driven _extract_design_intent_with_llm


_UPDATE_CLASSIFIER_PROMPT = """You are a software architect. Classify what kind of update the user is requesting.

User request: "{request}"

## EXISTING PAGES IN THIS APP
{pages_list}

## PROJECT FILES
{file_tree}

Respond with ONLY a JSON object:
{{
  "type": "theme_only | add_page | modify_page | full_regen | synthetic_data",
  "scope": "brief description of what changes",
  "pages_affected": ["PageName1", "PageName2"],
  "reason": "one sentence explanation"
}}

Classification rules:
- "theme_only": ONLY if user asks to change colors, theme, style, fonts, dark mode, or brand look with NO data/feature changes
- "add_page": user wants a new page/screen/section added to an existing app
- "modify_page": user wants specific page(s) functionality or content changed (but same domain)
- "synthetic_data": user asks to generate/create/add/populate dummy data, fake data, synthetic data, sample data, mock data, test data, seed data, or says the app is showing zeros/no data/empty
- "full_regen": ANY of these → domain change (electronics → pharma, fashion → healthcare), 
  "make it pharma/food/medical/...", changing the core product/data type, 
  "data should be X", "products should be X", major restructuring

Important:
- Use the EXISTING PAGES list above to correctly identify which page(s) the user is referring to
- If user says "fix the dashboard" → look for a page with "Dashboard" in its name in the list
- If user says "update the products page" → look for "ProductsPage" or "ProductPage" in the list
- Always populate pages_affected using the exact page names from the EXISTING PAGES list
- If the user is changing the DOMAIN or PRODUCT TYPE of the app → always "full_regen"
Examples of synthetic_data: "create dummy data", "add sample data", "populate with fake data",
"generate synthetic data", "add test data", "showing zeros fix it", "no data showing"
Examples of full_regen: "make it pharma", "products should be medicines",
"change to food delivery", "create hospital management system"

Return ONLY valid JSON."""


async def _classify_update(
    user_request: str,
    llm,
    existing_pages: List[str] = None,
    project_tree: List[str] = None,
) -> Dict[str, Any]:
    """Use LLM to classify update scope — avoids unnecessary full regeneration."""
    pages_list = "\n".join(f"- {p}" for p in (existing_pages or [])) or "(no pages found)"
    file_tree = "\n".join(f"- {f}" for f in (project_tree or [])) or "(no files found)"
    prompt = _UPDATE_CLASSIFIER_PROMPT.format(
        request=user_request,
        pages_list=pages_list,
        file_tree=file_tree,
    )
    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                data = json.loads(m.group(0))
                update_type = data.get("type", "full_regen")
                pages_affected = data.get("pages_affected") or []

                # ── If modify_page but no pages identified, infer them ───────
                if update_type == "modify_page" and not pages_affected and existing_pages:
                    req_lower = user_request.lower()

                    # Step 1: fuzzy string matching against page base names
                    for page in existing_pages:
                        page_base = page.replace("Page", "").lower()
                        if page_base and (page_base in req_lower or page.lower() in req_lower):
                            pages_affected.append(page)

                    # Step 2: if still empty, ask LLM to infer
                    if not pages_affected:
                        infer_prompt = (
                            f'Given this user request: "{user_request}"\n'
                            f"And these existing page names: {existing_pages}\n"
                            "Which page(s) is the user most likely referring to?\n"
                            'Return ONLY a JSON array of page names: ["PageName1"]\n'
                            "If unclear, return the single most likely page."
                        )
                        try:
                            infer_resp = await llm.ainvoke(infer_prompt)
                            infer_text = infer_resp.content if hasattr(infer_resp, "content") else str(infer_resp)
                            arr_match = re.search(r'\[[\s\S]*?\]', infer_text)
                            if arr_match:
                                pages_affected = json.loads(arr_match.group(0))
                                print(f"[update-classifier] inferred pages via LLM: {pages_affected}")
                        except Exception as ie:
                            print(f"[update-classifier] page inference failed: {ie}")

                    data["pages_affected"] = pages_affected

                print(f"[update-classifier] type={update_type} | pages={pages_affected} | {data.get('reason','')}")
                return data
            except Exception:
                pass
    except Exception as e:
        print(f"[update-classifier] failed: {e}")
    # Default: full_regen to be safe
    return {"type": "full_regen", "scope": "unknown", "pages_affected": [], "reason": "classification failed"}


async def _maybe_update_backend_routes(
    project_root: str,
    page_name: str,
    page_code: str,
    llm,
) -> Optional[Tuple[str, str]]:
    """
    Check if a new page calls API endpoints that don't yet exist in the backend.
    Returns (backend_rel_path, updated_content) if changes are needed, else None.
    """
    # Find the backend file to update
    backend_candidates = [
        ("backend/routes.py", os.path.join(project_root, "backend", "routes.py")),
        ("backend/main.py", os.path.join(project_root, "backend", "main.py")),
    ]
    backend_rel_path: Optional[str] = None
    backend_content: Optional[str] = None
    for rel, abs_path in backend_candidates:
        if os.path.exists(abs_path):
            backend_rel_path = rel
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as _fh:
                    backend_content = _fh.read()
            except Exception as _e:
                print(f"[add_page] Could not read {abs_path}: {_e}")
            break

    if not backend_rel_path or not backend_content:
        print(f"[add_page] Backend file not found — skipping route update")
        return None

    prompt = (
        f'You are a FastAPI backend developer.\n'
        f'A new frontend page "{page_name}" was just added with this code:\n\n'
        f'```tsx\n{page_code[:3000]}{"..." if len(page_code) > 3000 else ""}\n```\n\n'
        f"Current backend file ({backend_rel_path}):\n"
        f'```python\n{backend_content[:4000]}{"..." if len(backend_content) > 4000 else ""}\n```\n\n'
        "Does this new page call any API endpoints (via apiFetch, fetch, axios, etc.) "
        "that do NOT already exist in the current backend?\n"
        "If YES: Add ONLY the missing endpoints to the backend. "
        "Return the COMPLETE updated Python file (no markdown, no explanation).\n"
        'If NO: Return exactly the string "NO_CHANGE".'
    )
    try:
        resp = await llm.ainvoke(prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if text.startswith("NO_CHANGE"):
            print(f"[add_page] Backend routes: no new endpoints needed")
            return None
        # Strip markdown fences if present
        code_match = re.search(r"```(?:python)?\n([\s\S]*?)\n```", text)
        if code_match:
            text = code_match.group(1).strip()
        if text and len(text) > 100 and "def " in text:
            print(f"[add_page] Backend updated with new routes for {page_name} ({len(text)} chars)")
            return backend_rel_path, text
    except Exception as _e:
        print(f"[add_page] Backend route update failed: {_e}")
    return None


async def update_code_from_chat(
    project_name: str,
    project_root: str,
    user_request: str,
    prd: str = "",
    original_requirement: str = "",
    architecture: Dict[str, Any] = None,
    uiux: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
    websocket: Optional[Any] = None,
    builder_kind: str = "",
) -> Dict[str, Any]:
    """
    Code-only update: classify the request, then surgically update only what's needed.
    PRD and architecture are used as read-only context — they are NOT regenerated.
    Step 1: Classify update type (theme_only / add_page / modify_page / full_regen).
    Step 2: Extract design intent (colors/theme).
    Step 3: Surgical code generation — only regenerate what's needed.
    """
    print(f"\n{'='*60}")
    print(f"Code update for: {project_name}")
    print(f"Request: {user_request}")
    print(f"{'='*60}")

    llm = get_llm_for_user(None, temperature=0.15)
    # Use stored PRD and architecture as context only — do NOT update them
    current_prd = prd
    current_architecture = architecture or {}

    if websocket:
        try:
            await websocket.send_text(json.dumps({
                "event": "agent_start",
                "agent": "update_code_agent",
                "message": "Analyzing your request..."
            }))
        except Exception:
            pass

    combined_spec = "\n".join([
        original_requirement or "",
        user_request or "",
        current_prd or "",
        prd or "",
    ])
    try:
        from app_builder.services.fullstack_stack import is_fullstack
        from app_builder.services.fullstack_app_generator import generate_fullstack_application
        from app_builder.services.fullstack_codegen import (
            post_process_fullstack_files,
            is_fullstack_generated_files,
        )
        from app_builder.services.file_writer import file_writer
        from app_builder.schemas.files import GeneratedFiles

        # Check if this is a fullstack app (by builder_kind OR by existing files on disk)
        disk_files: Dict[str, str] = {}
        try:
            for root, dirs, fnames in os.walk(project_root):
                dirs[:] = [d for d in dirs if d not in ("node_modules", "__pycache__", "dist", "build", ".git")]
                for fn in fnames:
                    if fn.endswith(('.ts', '.tsx', '.py', '.json')):
                        rel = os.path.relpath(os.path.join(root, fn), project_root)
                        try:
                            with open(os.path.join(root, fn), 'r', encoding='utf-8', errors='replace') as fh:
                                _raw = fh.read()
                                disk_files[rel.replace('\\', '/')] = _raw if len(_raw) < 50000 else _raw[:15000]
                        except Exception:
                            pass
        except Exception:
            pass

        is_fs = is_fullstack(builder_kind) or is_fullstack_generated_files(disk_files)

        if is_fs:
            from app_builder.services.fullstack_ai_generator import (
                generate_fullstack_app_with_ai,
                generate_layout_with_ai,
                generate_login_with_ai,
                generate_page_with_ai,
                _build_app_tsx_for_pages,
                _build_app_layout_tsx,
                _build_login_page_tsx,
                _page_name_to_path,
            )
            from app_builder.services.fullstack_frontend_generator import build_theme_ts

            # ── Step 1: Classify the update scope with LLM ────────────────────
            if websocket:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "update_code_agent",
                    "message": "🧠 Analysing what needs to change...",
                }))

            # Extract existing page names and file tree from disk for the classifier
            _existing_pages: List[str] = [
                os.path.splitext(os.path.basename(p))[0]
                for p in sorted(disk_files)
                if p.startswith("frontend/src/pages/") and p.endswith(".tsx")
            ]
            _project_tree: List[str] = sorted(disk_files.keys())

            update_classification = await _classify_update(
                user_request, llm,
                existing_pages=_existing_pages,
                project_tree=_project_tree,
            )
            update_type = update_classification.get("type", "full_regen")

            # ── Pre-check: synthetic data keywords override classifier ────────
            _SYNTHETIC_DATA_KEYWORDS = {
                "synthetic data", "dummy data", "fake data", "sample data",
                "mock data", "test data", "seed data", "populate data",
                "generate data", "create data", "add data", "showing zeros",
                "no data", "empty data", "showing 0", "all zeros", "all zero",
                "populate with", "fill with data", "add some data",
            }
            req_lower = user_request.lower()
            if any(kw in req_lower for kw in _SYNTHETIC_DATA_KEYWORDS):
                update_type = "synthetic_data"
                update_classification["type"] = "synthetic_data"

            # ── Pre-check: only do full_regen if user EXPLICITLY asked for it ─
            _EXPLICIT_REGEN_WORDS = {
                "rebuild", "regenerate", "start over", "redo", "start fresh",
                "from scratch", "make new", "recreate",
            }
            if update_type == "full_regen" and not any(w in req_lower for w in _EXPLICIT_REGEN_WORDS):
                print(f"[update-classifier] Downgrading full_regen → modify_page (no explicit rebuild keyword in request)")
                update_type = "modify_page"
                update_classification["type"] = "modify_page"

            # ── Step 2: Extract design intent with LLM ────────────────────────
            if websocket:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "update_code_agent",
                    "message": "🎨 AI extracting colors, theme, style from your request...",
                }))

            merged_tokens = await _extract_design_intent_with_llm(user_request, design_tokens, llm)
            merged_uiux = "\n".join(filter(None, [uiux, _color_uiux_from_tokens(merged_tokens)]))
            update_spec = "\n".join(filter(None, [user_request, current_prd]))
            primary = (merged_tokens.get("colors") or {}).get("primary", "")

            if websocket:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "update_code_agent",
                    "message": f"🎨 Theme extracted{f' — {primary}' if primary else ''}. Update type: {update_type}",
                }))

            new_files: Dict[str, str] = {}
            colors = (merged_tokens.get("colors") or merged_tokens)
            from app_builder.services.fullstack_app_generator import extract_app_title
            title = extract_app_title(original_requirement or update_spec, current_prd)

            # ── Step 3: Smart update — only regenerate what's needed ──────────
            if update_type == "theme_only":
                # Fastest path: only update theme.ts + AppLayout + LoginPage
                # Pages keep their existing content — only colors change
                if websocket:
                    await websocket.send_text(json.dumps({
                        "event": "agent_progress",
                        "agent": "update_code_agent",
                        "message": "⚡ Theme update — regenerating theme, layout and login page...",
                    }))

                # Extract pages from disk for layout/routing
                pages_from_disk = [
                    (os.path.splitext(os.path.basename(p))[0], "")
                    for p in disk_files
                    if p.startswith("frontend/src/pages/") and p.endswith(".tsx")
                    and "LoginPage" not in p
                ]

                # Generate theme, layout, login in parallel (not all pages)
                theme_ts = build_theme_ts(colors)
                layout_code, login_code = await asyncio.gather(
                    generate_layout_with_ai(title, pages_from_disk, colors, original_requirement or update_spec, llm),
                    generate_login_with_ai(title, colors, original_requirement or update_spec, llm),
                )

                new_files["frontend/src/theme.ts"] = theme_ts
                new_files["frontend/src/layout/AppLayout.tsx"] = (
                    layout_code if (layout_code and len(layout_code) > 300)
                    else _build_app_layout_tsx(title, pages_from_disk, colors)
                )
                new_files["frontend/src/pages/LoginPage.tsx"] = (
                    login_code if (login_code and len(login_code) > 300)
                    else _build_login_page_tsx(title, colors)
                )

            elif update_type == "add_page":
                # Surgical: generate ONLY the new page + update App.tsx + AppLayout
                if websocket:
                    await websocket.send_text(json.dumps({
                        "event": "agent_progress",
                        "agent": "update_code_agent",
                        "message": "➕ Adding new page — generating page component + updating router...",
                    }))

                pages_to_add = update_classification.get("pages_affected") or []

                # Collect existing pages from disk (for router + sidebar)
                all_pages: List[Tuple[str, str]] = [
                    (os.path.splitext(os.path.basename(p))[0], "")
                    for p in sorted(disk_files)
                    if p.startswith("frontend/src/pages/") and p.endswith(".tsx")
                    and "LoginPage" not in p
                ]

                for raw_name in pages_to_add:
                    # Normalize to PascalCase + "Page" suffix
                    page_comp = raw_name if raw_name.endswith("Page") else raw_name + "Page"
                    if not page_comp[0].isupper():
                        page_comp = page_comp[0].upper() + page_comp[1:]

                    if websocket:
                        try:
                            await websocket.send_text(json.dumps({
                                "event": "agent_progress",
                                "agent": "update_code_agent",
                                "message": f"⚙️ Generating {page_comp}...",
                            }))
                        except Exception:
                            pass

                    page_code = await generate_page_with_ai(
                        page_comp, user_request, title, colors, current_architecture, llm,
                        requirement=original_requirement or update_spec,
                        prd=current_prd,
                        all_pages=all_pages,
                    )
                    if page_code:
                        new_files[f"frontend/src/pages/{page_comp}.tsx"] = page_code
                        if not any(name == page_comp for name, _ in all_pages):
                            all_pages.append((page_comp, ""))

                        # ── Update backend routes if the new page needs new endpoints ──
                        if websocket:
                            try:
                                await websocket.send_text(json.dumps({
                                    "event": "agent_progress",
                                    "agent": "update_code_agent",
                                    "message": f"🔗 Checking if {page_comp} needs new backend routes...",
                                }))
                            except Exception:
                                pass
                        backend_result = await _maybe_update_backend_routes(
                            project_root, page_comp, page_code, llm
                        )
                        if backend_result:
                            backend_rel, backend_updated = backend_result
                            new_files[backend_rel] = backend_updated
                            # Write backend file to disk immediately
                            _bp = os.path.join(project_root, backend_rel)
                            try:
                                with open(_bp, "w", encoding="utf-8") as _bfh:
                                    _bfh.write(backend_updated)
                                print(f"[add_page] Wrote updated backend: {backend_rel}")
                            except Exception as _bwe:
                                print(f"[add_page] Could not write backend {backend_rel}: {_bwe}")

                # Regenerate App.tsx (router) and AppLayout.tsx (sidebar) to include the new page
                new_files["frontend/src/App.tsx"] = _build_app_tsx_for_pages(all_pages)
                layout_code = await generate_layout_with_ai(
                    title, all_pages, colors, original_requirement or update_spec, llm
                )
                new_files["frontend/src/layout/AppLayout.tsx"] = (
                    layout_code if (layout_code and len(layout_code) > 300)
                    else _build_app_layout_tsx(title, all_pages, colors)
                )

                # Write files directly — don't call post_process which would fill/overwrite other pages
                if new_files:
                    file_writer(project_name, GeneratedFiles(files=new_files))

                file_list = sorted(new_files.keys())
                n = len(file_list)
                return {
                    "status": "success",
                    "analysis": f"New page added. {n} files updated (page + router + layout).",
                    "updated_files": file_list,
                    "updated_code_dict": new_files,
                    "updated_prd": current_prd,
                    "updated_architecture": current_architecture,
                    "merged_design_tokens": merged_tokens,
                    "update_type": update_type,
                    "changes": [{"file": k, "action": "updated"} for k in file_list[:60]],
                }

            elif update_type == "synthetic_data":
                # Generate realistic domain-specific data and update mock.ts + seed_data.json
                if websocket:
                    await websocket.send_text(json.dumps({
                        "event": "agent_progress",
                        "agent": "update_code_agent",
                        "message": "🗄️ Generating realistic synthetic data for your app...",
                    }))

                # Build context: existing pages + backend routes for the LLM
                existing_pages_list = "\n".join(f"- {p}" for p in _existing_pages) or "(none)"
                existing_backend = ""
                for backend_key in ("backend/main.py", "backend/routes.py"):
                    bp = os.path.join(project_root, backend_key)
                    if os.path.exists(bp):
                        try:
                            with open(bp, "r", encoding="utf-8") as _bh:
                                existing_backend = _bh.read()[:6000]
                            break
                        except Exception:
                            pass

                _SYNTH_DATA_PROMPT = f"""You are a data engineer. Generate realistic, domain-specific synthetic data for a web application.

APP TITLE: {title}
APP REQUIREMENT: {(original_requirement or current_prd or user_request)[:1000]}

PAGES IN APP:
{existing_pages_list}

EXISTING BACKEND ROUTES (extract the API paths from this):
{existing_backend[:3000] if existing_backend else '(not available)'}

USER REQUEST: {user_request}

Generate TWO things:

1. A TypeScript mock.ts file with realistic data for ALL API endpoints used by the pages above.
   - At least 15-25 records per entity (products, orders, users, etc.)
   - Use domain-appropriate field names and realistic values
   - Include a `_ENDPOINTS` map for exact paths like `/api/dashboard/kpis`, `/api/stats`, etc. with non-zero realistic numbers
   - Format: export const _ENDPOINTS: Record<string, unknown> = {{ ... }}; export const _store: Record<string, unknown[]> = {{ ... }};
   - The mockFetch function should check _ENDPOINTS first (exact path match), then _store (collection match)

2. A seed_data.json with the same data in JSON format for server-side use.
   Format: {{ "kpis": {{...}}, "entities": {{ "products": [...], "orders": [...], ... }} }}

Output format (use these exact markers):
===MOCK_TS_START===
(complete mock.ts TypeScript file)
===MOCK_TS_END===

===SEED_JSON_START===
(complete seed_data.json)
===SEED_JSON_END===

Make the data realistic and substantial. No placeholder values. No zeros for meaningful metrics."""

                try:
                    _synth_response = await llm.ainvoke(_SYNTH_DATA_PROMPT)
                    _synth_raw = _synth_response.content if hasattr(_synth_response, "content") else str(_synth_response)

                    # Parse mock.ts
                    _mock_ts = None
                    if "===MOCK_TS_START===" in _synth_raw and "===MOCK_TS_END===" in _synth_raw:
                        _mock_ts = _synth_raw.split("===MOCK_TS_START===")[1].split("===MOCK_TS_END===")[0].strip()

                    # Parse seed_data.json
                    _seed_json = None
                    if "===SEED_JSON_START===" in _synth_raw and "===SEED_JSON_END===" in _synth_raw:
                        _seed_json = _synth_raw.split("===SEED_JSON_START===")[1].split("===SEED_JSON_END===")[0].strip()

                    if _mock_ts and len(_mock_ts) > 100:
                        new_files["frontend/src/api/mock.ts"] = _mock_ts
                        if websocket:
                            await websocket.send_text(json.dumps({
                                "event": "agent_progress",
                                "agent": "update_code_agent",
                                "message": "✅ mock.ts updated with realistic data",
                            }))

                    if _seed_json and len(_seed_json) > 10:
                        # Validate JSON
                        try:
                            json.loads(_seed_json)
                            new_files["backend/seed_data.json"] = _seed_json
                            if websocket:
                                await websocket.send_text(json.dumps({
                                    "event": "agent_progress",
                                    "agent": "update_code_agent",
                                    "message": "✅ seed_data.json written — backend will serve realistic data",
                                }))
                        except json.JSONDecodeError:
                            pass

                    if not new_files:
                        raise ValueError("LLM did not produce usable data output")

                except Exception as _synth_err:
                    print(f"[synthetic_data] LLM generation failed: {_synth_err}. Using domain defaults.")
                    # Fallback: write a generic but non-zero seed_data.json
                    from app_builder.services.fullstack_ai_generator import _default_seed_data
                    _detected_domain = title.lower()
                    _fallback_seed = _default_seed_data(_detected_domain)
                    new_files["backend/seed_data.json"] = json.dumps(_fallback_seed, indent=2)

                # Write files directly (skip post_process which may overwrite pages)
                if new_files:
                    file_writer(project_name, GeneratedFiles(files=new_files))

                file_list = sorted(new_files.keys())
                if websocket:
                    await websocket.send_text(json.dumps({
                        "event": "agent_complete",
                        "agent": "update_code_agent",
                        "message": f"🎉 Synthetic data generated! {len(file_list)} file(s) updated. Refresh the app to see realistic data.",
                    }))

                return {
                    "status": "success",
                    "analysis": f"Synthetic data generated for '{title}'. Updated {len(file_list)} file(s) with domain-specific realistic data. Refresh the app preview to see the changes.",
                    "updated_files": file_list,
                    "updated_code_dict": new_files,
                    "updated_prd": current_prd,
                    "updated_architecture": current_architecture,
                    "merged_design_tokens": merged_tokens,
                    "update_type": "synthetic_data",
                    "changes": [{"file": k, "action": "updated"} for k in file_list],
                }

            elif update_type == "modify_page":
                # Surgical: regenerate only the page(s) the user's message refers to
                if websocket:
                    await websocket.send_text(json.dumps({
                        "event": "agent_progress",
                        "agent": "update_code_agent",
                        "message": "✏️ Updating code — regenerating affected page(s) only...",
                    }))

                pages_to_update = update_classification.get("pages_affected") or []

                all_pages_on_disk: List[Tuple[str, str]] = [
                    (os.path.splitext(os.path.basename(p))[0], "")
                    for p in sorted(disk_files)
                    if p.startswith("frontend/src/pages/") and p.endswith(".tsx")
                    and "LoginPage" not in p
                ]

                if not pages_to_update:
                    # Fallback: never blindly update ALL pages.
                    # If there's only one page, update it; otherwise bail out gracefully.
                    non_login = [name for name, _ in all_pages_on_disk]
                    if len(non_login) == 1:
                        pages_to_update = non_login
                        print(f"[modify_page] Single page found — updating {non_login[0]}")
                    else:
                        print(f"[modify_page] Could not identify page to update — skipping to avoid touching all pages")
                        return {
                            "status": "error",
                            "analysis": "Could not determine which page to update from your request. Please mention the page name explicitly (e.g. 'fix the Dashboard page').",
                            "updated_files": [],
                            "updated_code_dict": {},
                            "updated_prd": current_prd,
                            "updated_architecture": current_architecture,
                            "merged_design_tokens": merged_tokens,
                            "update_type": update_type,
                            "changes": [],
                        }

                for raw_name in pages_to_update:
                    page_comp = raw_name if raw_name.endswith("Page") else raw_name + "Page"
                    if not page_comp[0].isupper():
                        page_comp = page_comp[0].upper() + page_comp[1:]

                    # Verify it exists on disk (fuzzy match if needed)
                    disk_path = f"frontend/src/pages/{page_comp}.tsx"
                    if disk_path not in disk_files:
                        candidates = [
                            p for p in disk_files
                            if p.startswith("frontend/src/pages/") and raw_name.lower() in p.lower()
                        ]
                        if candidates:
                            disk_path = candidates[0]
                            page_comp = os.path.splitext(os.path.basename(disk_path))[0]
                        else:
                            print(f"[modify_page] Skipping {page_comp} — not found on disk")
                            continue

                    if websocket:
                        try:
                            await websocket.send_text(json.dumps({
                                "event": "agent_progress",
                                "agent": "update_code_agent",
                                "message": f"⚙️ Updating {page_comp} (preserving existing code)...",
                            }))
                        except Exception:
                            pass

                    # ── Read existing page content so we preserve prior customizations ──
                    existing_page_content: Optional[str] = None
                    full_disk_path = os.path.join(project_root, disk_path)
                    try:
                        with open(full_disk_path, "r", encoding="utf-8", errors="replace") as _fh:
                            existing_page_content = _fh.read()
                        print(f"[modify_page] Read existing {disk_path} ({len(existing_page_content)} chars)")
                    except Exception as _re:
                        print(f"[modify_page] Could not read existing {disk_path}: {_re}")

                    page_code = await generate_page_with_ai(
                        page_comp, user_request, title, colors, current_architecture, llm,
                        requirement=original_requirement or update_spec,
                        prd=current_prd,
                        all_pages=all_pages_on_disk,
                        existing_content=existing_page_content,
                    )
                    if page_code:
                        new_files[f"frontend/src/pages/{page_comp}.tsx"] = page_code

                # Write files directly — don't call post_process which would fill/overwrite other pages
                if new_files:
                    file_writer(project_name, GeneratedFiles(files=new_files))

                file_list = sorted(new_files.keys())
                n = len(file_list)
                return {
                    "status": "success",
                    "analysis": f"Page logic updated. {n} page(s) regenerated.",
                    "updated_files": file_list,
                    "updated_code_dict": new_files,
                    "updated_prd": current_prd,
                    "updated_architecture": current_architecture,
                    "merged_design_tokens": merged_tokens,
                    "update_type": update_type,
                    "changes": [{"file": k, "action": "updated"} for k in file_list[:60]],
                }

            else:
                # full_regen: user explicitly requested a full rebuild
                if websocket:
                    await websocket.send_text(json.dumps({
                        "event": "agent_progress",
                        "agent": "update_code_agent",
                        "message": f"🤖 AI regenerating app — {update_classification.get('scope', 'applying changes')}...",
                    }))
                try:
                    new_files = await generate_fullstack_app_with_ai(
                        update_spec, current_prd, merged_uiux, current_architecture, merged_tokens, llm,
                    )
                except Exception as ai_exc:
                    print(f"[update_code] AI regen failed ({ai_exc}) — deterministic fallback")
                    from app_builder.services.fullstack_app_generator import generate_fullstack_application
                    new_files = generate_fullstack_application(
                        update_spec, current_prd, merged_uiux, current_architecture, merged_tokens,
                    )

            if new_files:
                # For theme_only updates, don't pass requirement/prd to post_process:
                # fill_missing_fullstack_files would see 0 AI pages and overwrite them
                # with generic deterministic pages (killing the user's actual app content).
                _pp_req = "" if update_type == "theme_only" else update_spec
                _pp_prd = "" if update_type == "theme_only" else current_prd
                new_files = post_process_fullstack_files(
                    new_files,
                    design_tokens=merged_tokens,
                    uiux=merged_uiux,
                    requirement=_pp_req,
                    prd=_pp_prd,
                )
                file_writer(project_name, GeneratedFiles(files=new_files))

            file_list = sorted(new_files.keys())
            n = len(file_list)
            summary = {
                "theme_only": f"Theme updated with new colors ({primary}). {n} files refreshed — no page logic changed.",
                "full_regen": f"App fully regenerated with your request. {n} files updated.",
            }.get(update_type, f"{n} files updated.")

            return {
                "status": "success",
                "analysis": summary,
                "updated_files": file_list,
                "updated_code_dict": new_files,
                "updated_prd": current_prd,
                "updated_architecture": current_architecture,
                "merged_design_tokens": merged_tokens,
                "update_type": update_type,
                "changes": [{"file": k, "action": "updated"} for k in file_list[:60]],
            }
    except Exception as regen_exc:
        print(f"[update_code] fullstack regenerate failed: {regen_exc}")

    if websocket:
        await websocket.send_text(json.dumps({
            "event": "agent_progress",
            "agent": "update_code_agent",
            "message": "Planning file modifications..."
        }))

    # ── 1. Walk project and collect ALL source files ──────────────────────────
    files_context: Dict[str, str] = {}
    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in (
            "node_modules", "__pycache__", ".git", ".next",
            "venv", "dist", "build", ".cache", "coverage"
        )]
        for f in filenames:
            if f.endswith(('.js', '.jsx', '.ts', '.tsx', '.py',
                           '.css', '.html', '.json', '.md', '.env.example')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, project_root)
                try:
                    with open(full_path, "r", encoding="utf-8") as fh:
                        files_context[rel_path] = fh.read()
                except Exception:
                    pass

    if not files_context:
        return {"status": "error", "message": "No source files found in project directory."}

    def _fmt_file_entry(path: str, content: str) -> str:
        first_line = (content.split('\n', 1)[0] or '').strip()[:80]
        size = len(content.encode('utf-8'))
        suffix = f" — {first_line}" if first_line else ""
        return f"  - {path} ({size} bytes){suffix}"

    files_list = "\n".join(_fmt_file_entry(p, files_context[p]) for p in sorted(files_context.keys()))

    context_block = f"""
### Current Specification
**PRD:**
{current_prd[:2000]}{'...' if len(current_prd) > 2000 else ''}

**Architecture:**
{json.dumps(current_architecture, indent=2)[:2000]}{'...' if len(json.dumps(current_architecture)) > 2000 else ''}

**UI/UX Design (preserve unless user asks to change layout/theme):**
{uiux[:2000] if uiux else 'Use existing app styling (app-container, card, btn, input classes).'}

**Design System Tokens:**
{json.dumps(design_tokens, indent=2)[:1200] if design_tokens else 'Derived from UI/UX palette.'}
"""

    if original_requirement:
        context_block = f"\nOriginal App Requirement:\n{original_requirement}\n" + context_block

    # ── STEP 2: Planning — identify which files need to change ────────────────
    planning_prompt = f"""You are an expert full-stack developer reviewing a project update request.

Project: "{project_name}"
{context_block}

User's Request:
"{user_request}"

All project files (path, size in bytes, first line / export summary):
{files_list}

Your job: Decide which files need to be CREATED or MODIFIED to fulfill the request.

Rules:
1. If the request is about UI (inputs, buttons, forms, tables, styling, layout) → include frontend JS/JSX/CSS files.
2. If the request mentions colors, theme, appearance, or styles → ALWAYS include ALL .css files (e.g. frontend/src/styles.css, frontend/src/App.css).
3. If the request is about data, API, or backend logic → include backend Python files.
4. If both are needed, include both.
5. Be precise — only list files that genuinely need changes.
6. For new files that don't exist yet, include them with path relative to project root.

Respond ONLY with a valid JSON object (no markdown):
{{
  "reasoning": "Brief explanation of what needs to change",
  "files_to_change": [
    {{
      "path": "relative/path/to/file.js",
      "action": "modify",
      "reason": "Why this file needs to change"
    }}
  ]
}}"""

    try:
        plan_response = await llm.ainvoke(planning_prompt)
        plan_text = plan_response.content if hasattr(plan_response, 'content') else str(plan_response)
        print(f"[Step 2] Planning response: {plan_text[:300]}")

        # Parse planning JSON
        plan_json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', plan_text, re.DOTALL)
        if plan_json_match:
            plan_data = json.loads(plan_json_match.group(1))
        else:
            obj_match = re.search(r'\{[\s\S]*\}', plan_text)
            plan_data = json.loads(obj_match.group(0)) if obj_match else json.loads(plan_text.strip())

        files_to_change = plan_data.get("files_to_change", [])
        reasoning = plan_data.get("reasoning", "")
        style_keywords = ("color", "colour", "style", "theme", "appearance", "css", "styling")
        if any(k in user_request.lower() for k in style_keywords):
            css_in_project = [p for p in files_context if p.endswith(".css")]
            paths_in_plan = {f.get("path", "") for f in files_to_change}
            for css_path in css_in_project:
                if css_path not in paths_in_plan:
                    files_to_change.append({"path": css_path, "action": "modify", "reason": "User requested style/color changes"})
        print(f"[Step 2] Files to change: {[f['path'] for f in files_to_change]}")

        if not files_to_change:
            return {
                "status": "error",
                "message": "Could not determine which files to update. Please be more specific.",
                "analysis": reasoning,
                "updated_prd": current_prd,
                "updated_architecture": current_architecture
            }

    except Exception as e:
        print(f"[Step 2] Planning failed: {e}. Falling back to single-step update.")
        candidates = [(p, 1 if p.endswith(".css") and any(k in user_request.lower() for k in ("color", "colour", "style", "theme", "css")) else 0)
                     for p in files_context if p.endswith(('.js', '.jsx', '.tsx', '.css'))]
        candidates.sort(key=lambda x: -x[1])
        files_to_change = [{"path": p, "action": "modify", "reason": "fallback"} for p, _ in candidates[:8]]
        reasoning = ""

    # ── STEP 3: Surgical Re-generation — update files ────────────────────────
    if websocket:
        await websocket.send_text(json.dumps({
            "event": "agent_progress",
            "agent": "update_code_agent",
            "message": "Updating code..."
        }))

    applied_changes = []
    updated_code_dict = {}
    all_analyses = []

    for file_info in files_to_change:
        rel_path = file_info.get("path", "").strip().lstrip("/")
        action = file_info.get("action", "modify")
        reason = file_info.get("reason", "")

        if not rel_path:
            continue

        existing_content = files_context.get(rel_path, "")
        is_new_file = (action == "create") or (not existing_content)

        if websocket:
            try:
                await websocket.send_text(json.dumps({
                    "event": "agent_progress",
                    "agent": "update_code_agent",
                    "message": f"{'Creating' if is_new_file else 'Updating'} {rel_path}..."
                }))
            except Exception:
                pass

        if is_new_file:
            file_prompt = f"""You are an expert developer. Create a new file for the project "{project_name}".

User's Request: "{user_request}"
Reason this file is needed: {reason}
{context_block}
File to create: {rel_path}

Other relevant files for context:
"""
            # Add a few related files as context (up to 15 KB each)
            related = [(p, c if len(c) < 50000 else c[:15000]) for p, c in files_context.items()
                       if p != rel_path][:4]
            for rp, rc in related:
                file_prompt += f"\n--- {rp} ---\n{rc}\n"

            file_prompt += f"""
Write the COMPLETE content for {rel_path}.
Respond ONLY with a JSON object:
{{
  "analysis": "What this file does",
  "content": "COMPLETE file content"
}}"""
        else:
            # ── Try surgical SEARCH/REPLACE patch first ───────────────────────
            surgical_content, surgical_ok = await apply_surgical_patch(
                existing_content=existing_content,
                user_request=f"{user_request}\nWhat to change: {reason}",
                file_path=rel_path,
                llm_client=llm,
                context=context_block,
            )
            if surgical_ok and surgical_content:
                _fp_on_disk = os.path.join(project_root, rel_path)
                os.makedirs(os.path.dirname(_fp_on_disk), exist_ok=True)
                with open(_fp_on_disk, "w", encoding="utf-8") as _f:
                    _f.write(surgical_content)
                applied_changes.append(rel_path)
                updated_code_dict[rel_path] = surgical_content
                all_analyses.append(f"• {rel_path}: surgical patch applied")
                if websocket:
                    try:
                        await websocket.send_text(json.dumps({
                            "event": "agent_progress",
                            "agent": "update_code_agent",
                            "message": f"✂️ Surgically patched {rel_path}"
                        }))
                    except Exception:
                        pass
                continue  # skip full-rewrite for this file

            # Surgical patch failed — fall back to full-rewrite prompt
            file_prompt = f"""You are an expert developer updating a file in project "{project_name}".

User's Request: "{user_request}"
What needs to change in this file: {reason}
{context_block}
File to update: {rel_path}

CURRENT COMPLETE CONTENT OF {rel_path}:
```
{existing_content}
```

TASK: Produce the COMPLETE updated content of {rel_path} that:
1. Preserves ALL existing code, imports, functions, and logic that are NOT related to the change.
2. Adds/modifies ONLY what the user requested, following the PRD and Architecture context.
3. Is a complete, working file — not a snippet or partial update.

DO NOT omit any existing code. DO NOT use "..." or placeholders. Write the full file.

Respond ONLY with a JSON object:
{{
  "analysis": "What you changed and why",
  "content": "COMPLETE updated file content"
}}"""

        try:
            file_response = await llm.ainvoke(file_prompt)
            file_text = file_response.content if hasattr(file_response, 'content') else str(file_response)

            fj_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', file_text, re.DOTALL)
            json_str = fj_match.group(1).strip() if fj_match else (
                obj_match.group(0) if (obj_match := re.search(r'\{[\s\S]*\}', file_text)) else file_text.strip()
            )
            try:
                file_data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    file_data = json.loads(repair_json(json_str))
                except ImportError:
                    raise

            new_content = file_data.get("content", "").strip()
            analysis = file_data.get("analysis", "")

            if not new_content:
                continue

            # Sanity check: new content should be at least 50% the size of existing
            if existing_content and len(new_content) < len(existing_content) * 0.4:
                retry_prompt = file_prompt + f"""

IMPORTANT: Your previous response was too short ({len(new_content)} vs {len(existing_content)} original).
You likely truncated the file. Write the FULL content."""
                retry_response = await llm.ainvoke(retry_prompt)
                retry_text = retry_response.content if hasattr(retry_response, 'content') else str(retry_response)
                
                fj2 = re.search(r'```(?:json)?\s*\n(.*?)\n```', retry_text, re.DOTALL)
                json_str2 = fj2.group(1).strip() if fj2 else (
                    obj2.group(0) if (obj2 := re.search(r'\{[\s\S]*\}', retry_text)) else "{}"
                )
                try:
                    file_data = json.loads(json_str2)
                except json.JSONDecodeError:
                    try:
                        from json_repair import repair_json
                        file_data = json.loads(repair_json(json_str2))
                    except ImportError:
                        file_data = {}
                
                new_content = file_data.get("content", new_content).strip()
                analysis = file_data.get("analysis", analysis)

            # Write to disk
            file_path = os.path.join(project_root, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            applied_changes.append(rel_path)
            updated_code_dict[rel_path] = new_content
            all_analyses.append(f"• {rel_path}: {analysis}")

        except Exception as e:
            print(f"  ❌ Error processing {rel_path}: {e}")
            continue

    if not applied_changes:
        return {
            "status": "error",
            "message": "No files were successfully updated.",
            "analysis": reasoning,
            "updated_prd": current_prd,
            "updated_architecture": current_architecture
        }

    combined_analysis = reasoning + "\n\n" + "\n".join(all_analyses) if all_analyses else reasoning

    return {
        "status": "success",
        "analysis": combined_analysis.strip(),
        "updated_files": applied_changes,
        "updated_code_dict": updated_code_dict,
        "updated_prd": current_prd,
        "updated_architecture": current_architecture
    }
