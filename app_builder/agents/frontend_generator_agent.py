import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

async def frontend_generator_agent(
    structured_requirement: Dict[str, Any],
    blueprint: Dict[str, Any],
    api_contract: Dict[str, Any],
    llm=None
) -> Dict[str, str]:
    """
    Frontend Code Generator Agent - Generates React app files.
    Strictly follows the API contract and blueprint.
    """
    if llm is None:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.2)

    system_prompt = """You are a Senior Frontend Developer. Your task is to generate high-quality, professional React code.
You MUST strictly follow the provided API contract and blueprint components.

CRITICAL FRONTEND RULES:
1. Use Tailwind CSS for ALL styling. Do NOT use vanilla CSS files.
2. Style the application with a PREMIUM, modern look. Use vibrant yet professional colors.
3. Hardcode the Backend Base URL to `http://localhost:5002`.
4. Include the Tailwind Play CDN in `frontend/public/index.html`:
   `<script src="https://cdn.tailwindcss.com"></script>`
5. **FRONTEND MUST WORK END-TO-END EVEN IF BACKEND IS DOWN — USE LOCALSTORAGE**:
   - Every component MUST load data from localStorage FIRST, then try API fetch.
   - Every state change (add/edit/delete) MUST sync to localStorage immediately.
   - CRUD operations MUST work 100% with localStorage alone. API calls are optional enhancements.
   - Example pattern:
     Initialize state from localStorage: const [items, setItems] = useState(() => JSON.parse(localStorage.getItem(STORAGE_KEY)) || []);
     Sync to localStorage on change: useEffect(() => localStorage.setItem(STORAGE_KEY, JSON.stringify(items)), [items]);
     Optional API fetch on mount: useEffect(() => fetch(url).then(r => r.json()).then(d => Array.isArray(d) && setItems(d)).catch(() => {{}}), []);
6. **ZERO-CRASH FETCHING**:
   - Wrap ALL `fetch()` and `await response.json()` calls in `try/catch` blocks.
   - NEVER let an API error crash the component or the React tree.
7. **NO EXTERNAL STATE MANAGERS**:
   - Use ONLY plain React `useState` and `useEffect`.
   - NEVER use Zustand, Redux, MobX, or any other state management library.
   - KEEP IT SIMPLE. Pass state via props if needed.

Output ONLY a JSON mapping of filenames to file content:
{
  "frontend/src/App.js": "content",
  "frontend/src/components/ComponentName.js": "content",
  "frontend/package.json": "content",
  "frontend/public/index.html": "content",
  "frontend/src/index.js": "content"
}

CRITICAL RULES:
1. Use standard React with fetch.
2. All components in the blueprint must be implemented.
3. Every file in the JSON mapping above (especially frontend/package.json) MUST be generated and complete.
4. Output ONLY the raw JSON string. Do NOT include markdown blocks.
5. If the code is long, do NOT truncate. The output must be valid, parseable JSON.
6. NEVER use `import create from 'zustand'`. Use plain React state only.
7. package.json MUST NOT include version numbers for dependencies.
"""

    context = {
        "structured_requirement": structured_requirement,
        "blueprint": blueprint,
        "api_contract": api_contract
    }
    
    user_prompt = f"Context: {json.dumps(context, indent=2)}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        return {}
