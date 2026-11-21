"""
Surgical SEARCH/REPLACE patch system — like Cursor's edit format.
LLM outputs SEARCH/REPLACE blocks; we apply them programmatically.
"""
from typing import Optional
import re

SEARCH_REPLACE_SYSTEM_PROMPT = '''
You are a surgical code editor. You receive an existing file and a change request.
Output ONLY the minimal changes needed using SEARCH/REPLACE blocks.

Format (use exactly this):
<<<<<<< SEARCH
[exact existing code to find — must match character-for-character]
=======
[replacement code]
>>>>>>> REPLACE

Rules:
- One block per logical change
- SEARCH must be an exact substring of the existing file
- Include enough context lines (2-3 before/after) for uniqueness
- Never include unchanged code outside the blocks
- For adding new code at the end, SEARCH the last line and REPLACE with last line + new code
- If you need to delete code, REPLACE with empty string
- Multiple blocks are fine — apply them in order

After the blocks, output:
FILES_CHANGED: comma-separated list of file paths that were changed
'''


def parse_search_replace_blocks(llm_output: str) -> list[dict]:
    """Parse SEARCH/REPLACE blocks from LLM output."""
    blocks = []
    pattern = re.compile(
        r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
        re.DOTALL
    )
    for match in pattern.finditer(llm_output):
        search = match.group(1)
        replace = match.group(2)
        blocks.append({'search': search, 'replace': replace})
    return blocks


def apply_patches(original: str, blocks: list[dict]) -> tuple[str, bool]:
    """Apply search/replace blocks to original content. Returns (result, success)."""
    result = original
    for block in blocks:
        search = block['search']
        replace = block['replace']
        if search in result:
            result = result.replace(search, replace, 1)
        else:
            # Try stripping trailing whitespace per line
            search_stripped = '\n'.join(line.rstrip() for line in search.split('\n'))
            result_stripped_lines = '\n'.join(line.rstrip() for line in result.split('\n'))
            if search_stripped in result_stripped_lines:
                result = result_stripped_lines.replace(search_stripped, replace, 1)
            else:
                return result, False  # patch failed, caller should fall back
    return result, True


async def apply_surgical_patch(
    existing_content: str,
    user_request: str,
    file_path: str,
    llm_client,  # the openai/langchain client already used in the codebase
    context: str = ''
) -> tuple[Optional[str], bool]:
    """
    Ask LLM for SEARCH/REPLACE blocks, apply them.
    Returns (patched_content, success). On failure returns (None, False).
    """
    prompt = f"""{SEARCH_REPLACE_SYSTEM_PROMPT}

## FILE: {file_path}
```
{existing_content}
```

## ADDITIONAL CONTEXT
{context}

## CHANGE REQUEST
{user_request}

Output SEARCH/REPLACE blocks now:"""

    try:
        # Use the same LLM pattern as the rest of the codebase
        response = await llm_client.ainvoke(prompt)
        llm_output = response.content if hasattr(response, 'content') else str(response)

        blocks = parse_search_replace_blocks(llm_output)
        if not blocks:
            return None, False

        patched, success = apply_patches(existing_content, blocks)
        if not success:
            return None, False

        # Sanity check: patched should not be empty or dramatically shorter
        if len(patched) < len(existing_content) * 0.3:
            return None, False

        return patched, True
    except Exception:
        return None, False
