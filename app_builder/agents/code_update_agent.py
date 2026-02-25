import os
import re
import json
from typing import Dict, Any, Optional
from llm_helper import get_llm_for_user


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


async def update_code_from_chat(
    project_name: str,
    project_root: str,
    user_request: str,
    prd: str = "",
    original_requirement: str = "",
    architecture: Dict[str, Any] = None,
    websocket: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Spec-driven code update:
    Step 0: Update PRD based on user request.
    Step 1: Update Architecture based on updated PRD.
    Step 2: Planning — identify which files need to change.
    Step 3: Surgical Re-generation — update files.
    """
    print(f"\n{'='*60}")
    print(f"Spec-driven code update for: {project_name}")
    print(f"Request: {user_request}")
    print(f"{'='*60}")

    llm = get_llm_for_user(None, temperature=0.15)
    updated_prd = prd
    updated_architecture = architecture or {}

    if websocket:
        try:
            await websocket.send_text(json.dumps({
                "event": "agent_start",
                "agent": "update_code_agent",
                "message": "Step 1/4: Analyzing and updating specification (PRD)..."
            }))
        except Exception:
            pass

    # ── STEP 0: Update PRD ────────────────────────────────────────────────────
    prd_update_prompt = f"""You are a product owner. Update the existing PRD to incorporate the user's new request.
    
Existing PRD:
{prd}

User Request:
{user_request}

Respond with the COMPLETE UPDATED PRD in Markdown format. Keep the same structure:
1. Product Overview
2. Business Requirements (FR/NFR)
3. Core Features
4. Process Flows

Output ONLY the Markdown content."""

    try:
        prd_resp = await llm.ainvoke(prd_update_prompt)
        updated_prd = prd_resp.content if hasattr(prd_resp, 'content') else str(prd_resp)
        if websocket:
            await websocket.send_text(json.dumps({
                "event": "prd_updated",
                "data": updated_prd
            }))
            await websocket.send_text(json.dumps({
                "event": "agent_progress",
                "agent": "update_code_agent",
                "message": "PRD updated successfully."
            }))
    except Exception as e:
        print(f"PRD update failed: {e}")

    # ── STEP 1: Update Architecture ───────────────────────────────────────────
    if websocket:
        await websocket.send_text(json.dumps({
            "event": "agent_progress",
            "agent": "update_code_agent",
            "message": "Step 2/4: Updating system architecture design..."
        }))

    arch_update_prompt = f"""You are a system architect. Update the existing architecture JSON to reflect the changes in the updated PRD.

Existing Architecture:
{json.dumps(updated_architecture, indent=2)}

Updated PRD:
{updated_prd}

User Request:
{user_request}

Rules:
1. Preserve existing tables, columns, and components unless they need modification.
2. Add new tables, columns, or components required by the update.
3. Stay consistent with SQLite/SQLAlchemy and the current tech stack.
4. Python 3.13 compatibility. Use Pydantic v2 (model_config, .model_dump()).
5. Use dynamic backend URL: env vars first, then derive from window.location (same host + :5004) for deployment.

Respond ONLY with a valid JSON object matching the architecture schema."""

    try:
        arch_resp = await llm.ainvoke(arch_update_prompt)
        arch_text = arch_resp.content if hasattr(arch_resp, 'content') else str(arch_resp)
        
        # Parse JSON
        obj_match = re.search(r'\{[\s\S]*\}', arch_text)
        if obj_match:
            updated_architecture = json.loads(obj_match.group(0))
        
        if websocket:
            await websocket.send_text(json.dumps({
                "event": "architecture_updated",
                "data": updated_architecture
            }))
            await websocket.send_text(json.dumps({
                "event": "agent_progress",
                "agent": "update_code_agent",
                "message": "Architecture updated successfully."
            }))
    except Exception as e:
        print(f"Architecture update failed: {e}")

    if websocket:
        await websocket.send_text(json.dumps({
            "event": "agent_progress",
            "agent": "update_code_agent",
            "message": "Step 3/4: Planning file modifications..."
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

    files_list = "\n".join(f"  - {p}" for p in sorted(files_context.keys()))

    context_block = f"""
### Current Specification
**Updated PRD:**
{updated_prd[:2000]}{'...' if len(updated_prd) > 2000 else ''}

**Updated Architecture:**
{json.dumps(updated_architecture, indent=2)[:2000]}{'...' if len(json.dumps(updated_architecture)) > 2000 else ''}
"""

    if original_requirement:
        context_block = f"\nOriginal App Requirement:\n{original_requirement}\n" + context_block

    # ── STEP 2: Planning — identify which files need to change ────────────────
    planning_prompt = f"""You are an expert full-stack developer reviewing a project update request.

Project: "{project_name}"
{context_block}

User's Update Request:
"{user_request}"

All project files:
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
                "updated_prd": updated_prd,
                "updated_architecture": updated_architecture
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
            "message": "Step 4/4: Executing code modifications..."
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
            # Add a few related files as context
            related = [(p, c) for p, c in files_context.items()
                       if p != rel_path and len(c) < 3000][:4]
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
2. Adds/modifies ONLY what the user requested, strictly following the Updated PRD and Architecture.
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
            "updated_prd": updated_prd,
            "updated_architecture": updated_architecture
        }

    combined_analysis = reasoning + "\n\n" + "\n".join(all_analyses) if all_analyses else reasoning

    return {
        "status": "success",
        "analysis": combined_analysis.strip(),
        "updated_files": applied_changes,
        "updated_code_dict": updated_code_dict,
        "updated_prd": updated_prd,
        "updated_architecture": updated_architecture
    }
