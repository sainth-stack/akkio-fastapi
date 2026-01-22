"""
Multi-Model Agent System (Refactored)
Pipeline: Context/Memory -> Plan -> Answer -> Web Search (if needed)
"""

import json
import asyncio
import base64
import os
import io
import sys
from typing import List, Dict, Any, Optional, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
from .usage_tracking import record_llm_usage, start_token_aggregation, end_token_aggregation
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from llm_helper import get_llm_for_user

# Web search imports - Using Google for better results
try:
    from googlesearch import search as google_search
    import requests
    from bs4 import BeautifulSoup
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("Warning: googlesearch-python not installed. Web search will be disabled.")
    print("Install with: pip install googlesearch-python beautifulsoup4 requests")


class ContextBuilder:
    """
    Stage 1: Context & Memory Builder
    Collects relevant context from files and conversation history.
    """
    def __init__(self, db_connection, chroma_client):
        self.db = db_connection
        self.chroma_client = chroma_client

    async def build_context(self, query: str, files: List[Dict], messages: List[Dict]) -> Dict[str, Any]:
        """
        Aggregates context from:
        1. Conversation History (Memory)
        2. Tabular Files (Schema + Sample)
        3. Document Files (Vector Search Excerpts)
        4. Image/Audio Files (Pass paths/content for multimodal LLM)
        """
        context = {
            'text_context': [],
            'tabular_context': [],
            'image_context': [],
            'sources': [],
            'memory_summary': []
        }

        # 1. Memory (Last N messages)
        # We assume messages are passed in a standard list of dicts format
        # Filter strictly for user/bot conversation to avoid blowing up context with metadata
        if messages:
             # Take last 6 messages for immediate context
            recent_msgs = messages[-6:]
            for msg in recent_msgs:
                role = msg.get('type', 'user')
                content = msg.get('content', '')
                # Attempt to parse if content is JSON string (common in this app)
                try: 
                    if isinstance(content, str) and (content.startswith('{') or content.startswith('[')):
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and 'answer' in parsed:
                            content = parsed['answer']
                except: pass
                
                # Truncate very long messages
                if len(str(content)) > 500:
                    content = str(content)[:500] + "...(truncated)"
                context['memory_summary'].append(f"{role.upper()}: {content}")

        # 2. File Processing
        tabular_files = [f for f in files if f.get('file_type') == 'tabular']
        doc_files = [f for f in files if f.get('file_type') == 'document']
        image_files = [f for f in files if f.get('file_type') == 'image']

        # Process Tabular
        for f in tabular_files:
            tbl_context = self._get_tabular_context(f)
            if tbl_context:
                context['tabular_context'].append(tbl_context)
                context['sources'].append(f['file_name'])

        # Process Documents (Vector Search)
        # Only search if query implies looking for info, or always?
        # "Always" is safer for "Context Builder" pattern, but we can limit results.
        if doc_files:
            doc_context = await self._get_document_context(query, doc_files)
            if doc_context:
                context['text_context'].extend(doc_context)
                context['sources'].extend([f['file_name'] for f in doc_files]) # Rough attribution

        # Process Images
        # We just pass the base64 data to the final context for the VLM
        for f in image_files:
            img_data = self._get_image_data(f)
            if img_data:
                context['image_context'].append(img_data)
                context['sources'].append(f['file_name'])

        return context

    def _get_tabular_context(self, file_info: Dict) -> Optional[str]:
        try:
            if file_info.get('db_table_name'):
                # Get schema and head
                # Note: db.get_table_schema is hypothetical, using get_table_data for now but limiting
                # In prod, you'd want a lightweight schema query.
                df = self.db.get_table_data(file_info['db_table_name'], limit=3)
                if df is not None:
                    columns = list(df.columns)
                    sample = df.to_string(index=False)
                    return f"Table '{file_info['file_name']}':\nColumns: {columns}\nSample Data:\n{sample}"
        except Exception:
            pass
        return None

    async def _get_document_context(self, query: str, files: List[Dict]) -> List[str]:
        results_list = []
        for f in files:
            if not f.get('vector_collection_id'): continue
            try:
                collection = self.chroma_client.get_collection(f['vector_collection_id'])
                # Semantic search
                results = collection.query(query_texts=[query], n_results=3)
                if results and results['documents']:
                    for doc in results['documents'][0]:
                        results_list.append(f"Excerpt from {f['file_name']}:\n{doc}")
            except Exception:
                pass
        return results_list

    def _get_image_data(self, file_info: Dict) -> Optional[str]:
        # Return base64 string
        path = file_info.get('storage_path')
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            except: pass
        return None


class SimplePlanner:
    """
    Stage 2: Planner
    Generates a fast, short bulleted plan.
    """
    def __init__(self, system_prompt: str, user_email: str = None):
        # Use llm_helper to get user's configured LLM
        self.llm = get_llm_for_user(user_email=user_email, temperature=0.1)
        self.system_prompt = system_prompt
        self.user_email = user_email

    async def stream_plan(self, query: str, context: Dict, callback: Callable):
        """
        Generates and streams the plan.
        """
        sources_list = ", ".join(context['sources']) if context['sources'] else "Memory only"
        
        # simplified context summary for planning
        context_summary = f"Available Sources: {sources_list}\n"
        if context['tabular_context']:
            context_summary += f"Tabular Data Available: {len(context['tabular_context'])} tables\n"
        if context['text_context']:
             context_summary += f"Document Excerpts: {len(context['text_context'])} found\n"
        
        prompt = f"""
Input:
User Query: {query}
Context Summary:
{context_summary}

Task:
Create a concise, 3-6 bullet point plan to answer the user query based on the available context.
Format: Plain text bullets, each starting with "- ".
No intro, no outro, no markup. Just the bullets.
"""
        messages = [
            SystemMessage(content="You are a precise planning assistant. Output strictly bullets."),
            HumanMessage(content=prompt)
        ]

        await callback("plan_start", {})
        
        collected_plan = ""
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content:
                collected_plan += content
                await callback("plan_chunk", {"chunk": content})

        await callback("plan_complete", {"full_plan": collected_plan})
        return collected_plan
    
    async def stream_plan_with_sources(self, query: str, context: Dict, web_results: List[Dict], callback: Callable):
        """
        Generates plan and appends web sources if available.
        """
        # First generate the plan
        plan = await self.stream_plan(query, context, callback)
        
        # If we have web results, append them to the plan
        if web_results:
            sources_section = "\n\n📚 Web Sources Found:\n"
            for idx, result in enumerate(web_results, 1):
                sources_section += f"{idx}. {result['title']} - {result['link']}\n"
            
            # Send the sources as additional plan chunks
            await callback("plan_chunk", {"chunk": sources_section})
            plan += sources_section
            await callback("plan_complete", {"full_plan": plan})
        
        return plan


class WebSearchAgent:
    """
    Web Search Agent
    Performs web search when local context is insufficient.
    """
    def __init__(self):
        self.enabled = WEB_SEARCH_AVAILABLE
        print(f"[WebSearchAgent] Initialized. Enabled: {self.enabled}")
        if self.enabled:
            print("[WebSearchAgent] Using Google Search for better accuracy")
        else:
            print("[WebSearchAgent] WARNING: Google search libraries not available!")
            print("[WebSearchAgent] Install with: pip install googlesearch-python beautifulsoup4 requests")
    
    def _optimize_search_query(self, query: str) -> str:
        """
        Optimizes search query for better relevance.
        Uses quotes for exact matching of full names to avoid partial matches.
        """
        # Clean basic whitespace issues
        cleaned = query.strip().replace('\n', ' ').replace('\r', '')
        cleaned = ' '.join(cleaned.split())
        
        # Detect "Name, Company" or "Person, Organization" patterns
        if ',' in cleaned:
            parts = [part.strip() for part in cleaned.split(',')]
            if len(parts) == 2:
                name_part = parts[0]
                company_part = parts[1]
                
                # Check if this is a person name (has at least 2 words - first and last name)
                name_words = name_part.split()
                if len(name_words) >= 2 and name_part and company_part:
                    # Use quotes around FULL NAME to force exact match
                    # This prevents matching partial names like "Sainath" -> "Palagummi Sainath"
                    optimized = f'"{name_part}" {company_part} LinkedIn'
                    print(f"[WebSearchAgent] Person/Company lookup - using exact name match")
                    print(f"[WebSearchAgent] Query: '{optimized}'")
                    return optimized
        
        # Check if query is a person's full name (2-4 words, capitalized)
        words = cleaned.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w and len(w) > 1):
            # Use quotes for exact full name match + LinkedIn for professional results
            optimized = f'"{cleaned}" LinkedIn'
            print(f"[WebSearchAgent] Full name detected - using exact match + LinkedIn")
            print(f"[WebSearchAgent] Query: '{optimized}'")
            return optimized
        
        # For other queries, just clean them
        return cleaned
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Performs web search and returns results with title, link, and snippet.
        """
        if not self.enabled:
            print("[WebSearchAgent] Search called but agent is disabled")
            return []
        
        print(f"[WebSearchAgent] Searching for: '{query}' (max {max_results} results)")
        
        try:
            # Optimize query before searching
            optimized_query = self._optimize_search_query(query)
            
            # Run synchronous search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._sync_search, optimized_query, max_results)
            print(f"[WebSearchAgent] Search completed. Found {len(results)} results")
            return results
        except Exception as e:
            print(f"[WebSearchAgent] ERROR in async search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _sync_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Synchronous Google search implementation - query should already be optimized"""
        print(f"[WebSearchAgent] Starting Google search with query: '{query}'")
        
        try:
            results = []
            # Use Google search - more accurate than Bing/DuckDuckGo
            search_results = google_search(query, num_results=max_results, sleep_interval=1, advanced=True)
            
            for r in search_results:
                # Extract snippet from the page if possible
                snippet = r.description if hasattr(r, 'description') and r.description else ""
                
                # If no description, try to fetch from the page
                if not snippet:
                    try:
                        response = requests.get(r.url, timeout=3, headers={'User-Agent': 'Mozilla/5.0'})
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            # Get meta description
                            meta_desc = soup.find('meta', attrs={'name': 'description'})
                            if meta_desc and meta_desc.get('content'):
                                snippet = meta_desc.get('content')
                            else:
                                # Get first paragraph as fallback
                                p = soup.find('p')
                                if p:
                                    snippet = p.get_text()[:200]
                    except:
                        snippet = "No description available"
                
                result = {
                    'title': r.title if hasattr(r, 'title') else 'No title',
                    'link': r.url if hasattr(r, 'url') else '',
                    'snippet': snippet[:300] if snippet else 'No description'
                }
                results.append(result)
                print(f"[WebSearchAgent] Found: {result['title']}")
                
                if len(results) >= max_results:
                    break
            
            print(f"[WebSearchAgent] Google search complete: {len(results)} results")
            return results
        except Exception as e:
            print(f"[WebSearchAgent] ERROR in Google search: {e}")
            import traceback
            traceback.print_exc()
            return []


class AnswerAgent:
    """
    Stage 3: Final Answer Agent
    Generates the grounded answer using all context.
    """
    def __init__(self, system_prompt: str, temperature: float = 0.0, output_format: str = None, user_email: str = None):
        # Use llm_helper to get user's configured LLM
        self.llm = get_llm_for_user(user_email=user_email, temperature=temperature)
        self.system_prompt = system_prompt
        self.output_format = output_format
        self.user_email = user_email
        self.web_search = WebSearchAgent()

    async def stream_answer(self, query: str, context: Dict, plan: str, callback: Callable):
        """
        Generates and streams the final answer.
        If context is insufficient, performs web search and includes results.
        """
        # Construct full context string
        full_context_str = ""
        
        if context['tabular_context']:
            full_context_str += "## Tabular Data\n" + "\n\n".join(context['tabular_context']) + "\n\n"
        
        if context['text_context']:
            full_context_str += "## Document Excerpts\n" + "\n\n".join(context['text_context']) + "\n\n"
            
        if context['memory_summary']:
             full_context_str += "## Conversation History\n" + "\n".join(context['memory_summary']) + "\n\n"

        # Build data-only context (excluding conversation history) for web search decision
        data_context_str = ""
        if context['tabular_context']:
            data_context_str += "\n\n".join(context['tabular_context']) + "\n\n"
        if context['text_context']:
            data_context_str += "\n\n".join(context['text_context']) + "\n\n"

        # Check if we need web search (based on data context, not conversation)
        web_search_results = []
        needs_web_search = self._needs_web_search(data_context_str, query)
        
        print(f"[WEB SEARCH] Decision: needs_web_search={needs_web_search}, enabled={self.web_search.enabled}")
        print(f"[WEB SEARCH] Data context length: {len(data_context_str)} chars")
        print(f"[WEB SEARCH] Query: {query}")
        
        if needs_web_search and not self.web_search.enabled:
            warning_msg = "⚠️ Web search is needed but DISABLED. Please install: pip install duckduckgo-search"
            print(f"[WEB SEARCH] {warning_msg}")
            await callback("web_search_unavailable", {
                "message": warning_msg,
                "install_command": "pip install duckduckgo-search"
            })
        elif needs_web_search and self.web_search.enabled:
            try:
                # Notify that web search is starting
                print(f"[WEB SEARCH] Starting web search for query: {query}")
                await callback("web_search_start", {"message": "Searching the web for information..."})
                
                web_search_results = await self.web_search.search(query, max_results=5)
                print(f"[WEB SEARCH] Got {len(web_search_results)} results")
                
                if web_search_results:
                    await callback("web_search_complete", {"results_count": len(web_search_results)})
                    # Add web results to context
                    web_context = "## Web Search Results\n"
                    for idx, result in enumerate(web_search_results, 1):
                        web_context += f"{idx}. {result['title']}\n   {result['snippet']}\n   URL: {result['link']}\n\n"
                    full_context_str += web_context
                    print(f"[WEB SEARCH] Added web context: {len(web_context)} chars")
                else:
                    await callback("web_search_complete", {"results_count": 0, "message": "No web results found"})
                    print("[WEB SEARCH] No results found")
            except Exception as e:
                print(f"[WEB SEARCH] ERROR: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail the whole request if web search fails
                await callback("web_search_error", {"message": f"Web search failed: {str(e)}"})
        else:
            print("[WEB SEARCH] Skipping web search - sufficient context")

        prompt = f"""
{self.system_prompt}

OUTPUT FORMAT GUIDELINES:
{self.output_format or "STRICTLY use valid HTML formatting (<h3>, <p>, <ul>, <li>, <strong>, <code>, <a> for links)."}
DO NOT use Markdown syntax (no `**bold**`, no `* list`, no `# header`).
Everything must be proper HTML tags.
For links, use: <a href="URL" target="_blank">Link Text</a>
For code blocks, use: <pre><code class="language-python">...</code></pre>
If the user asks for a specific format, prioritize that but keep it as HTML.
DO NOT echo the context items back to the user unless explicitly asked.
DO NOT start your response with blockquotes (>) or excessive headers.
Start the answer directly.

CONTEXT:
{full_context_str}

PLAN FOLLOWED:
{plan}

USER QUERY:
{query}

INSTRUCTIONS:
Answer the user query comprehensively using the provided context and history.
If web search results are provided, incorporate them naturally into your answer and include the source links.
When referencing web sources, format them as clickable links using HTML <a> tags.
If the context is still insufficient, state clearly what is missing.
Do not mention "In the provided context" or "Based on the files" repeatedly; just answer naturally.
"""
        messages = [HumanMessage(content=prompt)]
        
        # Inject images if present
        if context['image_context']:
            content_blocks = [{"type": "text", "text": prompt}]
            for img_b64 in context['image_context']:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            messages = [HumanMessage(content=content_blocks)]

        await callback("answer_start", {})
        
        collected_answer = ""
        async for chunk in self.llm.astream(messages):
             content = chunk.content
             if content:
                 collected_answer += content
                 await callback("answer_chunk", {"chunk": content})
        
        # If we used web search, append the sources in an accordion format
        if web_search_results:
            sources_html = self._format_web_sources_accordion(web_search_results)
            collected_answer += sources_html
            await callback("answer_chunk", {"chunk": sources_html})
                 
        await callback("answer_complete", {"full_answer": collected_answer})
        return collected_answer
    
    async def stream_answer_with_web_results(self, query: str, context: Dict, plan: str, web_results: List[Dict], callback: Callable):
        """
        Generates answer using pre-fetched web search results.
        """
        # Construct full context string
        full_context_str = ""
        
        if context['tabular_context']:
            full_context_str += "## Tabular Data\n" + "\n\n".join(context['tabular_context']) + "\n\n"
        
        if context['text_context']:
            full_context_str += "## Document Excerpts\n" + "\n\n".join(context['text_context']) + "\n\n"
            
        if context['memory_summary']:
             full_context_str += "## Conversation History\n" + "\n".join(context['memory_summary']) + "\n\n"

        # Add web results to context if provided
        if web_results:
            print(f"[ANSWER] Using {len(web_results)} pre-fetched web results")
            web_context = "## Web Search Results\n"
            for idx, result in enumerate(web_results, 1):
                web_context += f"{idx}. {result['title']}\n   {result['snippet']}\n   URL: {result['link']}\n\n"
            full_context_str += web_context

        prompt = f"""
{self.system_prompt}

OUTPUT FORMAT GUIDELINES:
{self.output_format or "STRICTLY use valid HTML formatting (<h3>, <p>, <ul>, <li>, <strong>, <code>, <a> for links)."}
DO NOT use Markdown syntax (no `**bold**`, no `* list`, no `# header`).
Everything must be proper HTML tags.
For links, use: <a href="URL" target="_blank" rel="noopener noreferrer">Link Text</a>
For code blocks, use: <pre><code class="language-python">...</code></pre>
If the user asks for a specific format, prioritize that but keep it as HTML.
DO NOT echo the context items back to the user unless explicitly asked.
DO NOT start your response with blockquotes (>) or excessive headers.
Start the answer directly.
When referencing web sources, use the source numbers (1, 2, 3, etc.) from the Web Search Results section.

CONTEXT:
{full_context_str}

PLAN FOLLOWED:
{plan}

USER QUERY:
{query}

INSTRUCTIONS:
Answer the user query comprehensively using the provided context and history.
The web search results are already provided in the context above.
When referencing information from web sources, mention the source number.
Provide a comprehensive answer based on all available information.
"""
        messages = [HumanMessage(content=prompt)]
        
        # Inject images if present
        if context['image_context']:
            content_blocks = [{"type": "text", "text": prompt}]
            for img_b64 in context['image_context']:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            messages = [HumanMessage(content=content_blocks)]

        await callback("answer_start", {})
        
        collected_answer = ""
        async for chunk in self.llm.astream(messages):
             content = chunk.content
             if content:
                 collected_answer += content
                 await callback("answer_chunk", {"chunk": content})
        
        # NOTE: We don't append accordion here since links are in planning
        await callback("answer_complete", {"full_answer": collected_answer})
        return collected_answer
    
    def _needs_web_search(self, context_str: str, query: str) -> bool:
        """
        Determines if web search is needed based on available context.
        VERY AGGRESSIVE: Always search when no meaningful data context exists.
        """
        # If we have NO or very little actual data context, ALWAYS search
        # This is the most important check
        if len(context_str.strip()) < 500:  # Increased threshold for better coverage
            print(f"[WEB SEARCH] Triggering due to low context: {len(context_str.strip())} chars")
            return True
        
        # Check if query is about real-world entities, people, companies, current events
        web_indicators = [
            'who is', 'what is', 'tell me about', 'information about',
            'latest', 'current', 'recent', 'news', 'today',
            'company', 'person', 'celebrity', 'organization',
            'about', 'details', 'background', 'profile',
            'find', 'search', 'look up'
        ]
        query_lower = query.lower()
        for indicator in web_indicators:
            if indicator in query_lower:
                print(f"[WEB SEARCH] Triggering due to keyword: '{indicator}'")
                return True
        
        # Check if query looks like a person's name (2-3 capitalized words)
        words = query.strip().split()
        if len(words) in [2, 3]:
            if all(word[0].isupper() for word in words if word and len(word) > 1):
                # Likely a person or company name
                print(f"[WEB SEARCH] Triggering due to name detection: {query}")
                return True
        
        # Check if query is a single entity (company, product, etc.)
        if len(words) <= 3 and any(word[0].isupper() for word in words if word and len(word) > 1):
            print(f"[WEB SEARCH] Triggering due to entity detection: {query}")
            return True
        
        # If query is short (< 10 words) and has no context, probably needs web search
        if len(words) < 10:
            print(f"[WEB SEARCH] Triggering due to short query with low context")
            return True
        
        return False
    
    def _format_web_sources_accordion(self, results: List[Dict[str, str]]) -> str:
        """
        Formats web search results as an HTML accordion/details element.
        """
        if not results:
            return ""
        
        html = '<details style="margin-top: 1.5rem; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; background-color: #f9fafb;">'
        html += '<summary style="font-weight: 600; cursor: pointer; color: #374151; font-size: 0.95rem;">🔗 Web Sources ({} links found)</summary>'.format(len(results))
        html += '<div style="margin-top: 1rem;">'
        html += '<ul style="list-style: none; padding: 0; margin: 0;">'
        
        for idx, result in enumerate(results, 1):
            html += '<li style="margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #e5e7eb;">'
            html += f'<div style="font-weight: 600; margin-bottom: 0.25rem;"><a href="{result["link"]}" target="_blank" style="color: #2563eb; text-decoration: none;">{idx}. {result["title"]}</a></div>'
            html += f'<div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem;">{result["snippet"]}</div>'
            html += f'<div style="font-size: 0.75rem; color: #9ca3af; word-break: break-all;">{result["link"]}</div>'
            html += '</li>'
        
        html += '</ul>'
        html += '</div>'
        html += '</details>'
        
        return html


class MultiModelAgentSystem:
    """
    Main Orchestrator
    Orchestrates: Context Building -> Planning -> Answer Generation -> Web Search (if needed)
    """
    def __init__(self, session_id: str, system_prompt: str, db_connection, chroma_client, temperature: float = 0.0, workflow: str = None, output_format: str = None):
        self.session_id = session_id
        self.db = db_connection
        
        self.context_builder = ContextBuilder(db_connection, chroma_client)
        self.planner = SimplePlanner(system_prompt)
        self.answer_agent = AnswerAgent(system_prompt, temperature, output_format)
        
        self.email = None # For usage tracking

    async def query_async(self, user_query: str, messages: List[Dict] = None, stream_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Executes the pipeline with web search BEFORE planning for better context.
        """
        token_ctx = start_token_aggregation()
        
        try:
            # 0. Setup
            if not stream_callback:
                # Dummy callback if none provided (for non-streaming compat)
                async def noop(ev, dt): pass
                stream_callback = noop

            # 1. Get Files
            files = self._get_files()

            # 2. Build Context
            context = await self.context_builder.build_context(user_query, files, messages)
            
            # 2.5. Check if we need web search EARLY (before planning)
            data_context_str = ""
            if context['tabular_context']:
                data_context_str += "\n\n".join(context['tabular_context']) + "\n\n"
            if context['text_context']:
                data_context_str += "\n\n".join(context['text_context']) + "\n\n"
            
            web_search_results = []
            needs_web_search = self.answer_agent._needs_web_search(data_context_str, user_query)
            
            print(f"[WEB SEARCH] Early check: needs_web_search={needs_web_search}, enabled={self.answer_agent.web_search.enabled}")
            
            if needs_web_search and not self.answer_agent.web_search.enabled:
                # Notify user that web search is needed but unavailable
                warning_msg = "⚠️ Web search needed but disabled. Install: pip install duckduckgo-search"
                print(f"[WEB SEARCH] {warning_msg}")
                await stream_callback("web_search_unavailable", {
                    "message": warning_msg,
                    "install_command": "pip install duckduckgo-search"
                })
            
            if needs_web_search and self.answer_agent.web_search.enabled:
                try:
                    print(f"[WEB SEARCH] Performing early web search for planning")
                    await stream_callback("web_search_start", {"message": "Searching the web..."})
                    
                    web_search_results = await self.answer_agent.web_search.search(user_query, max_results=5)
                    print(f"[WEB SEARCH] Early search got {len(web_search_results)} results")
                    
                    if web_search_results:
                        await stream_callback("web_search_complete", {"results_count": len(web_search_results)})
                except Exception as e:
                    print(f"[WEB SEARCH] Early search error: {e}")
                    await stream_callback("web_search_error", {"message": str(e)})
            
            # 3. Stream Plan (with web sources if available)
            if web_search_results:
                plan = await self.planner.stream_plan_with_sources(user_query, context, web_search_results, stream_callback)
            else:
                plan = await self.planner.stream_plan(user_query, context, stream_callback)
            
            # 4. Stream Answer (use existing web results)
            answer = await self.answer_agent.stream_answer_with_web_results(
                user_query, context, plan, web_search_results, stream_callback
            )
            
            agents_used = ['ContextBuilder', 'SimplePlanner', 'AnswerAgent']
            if self.answer_agent.web_search.enabled and web_search_results:
                agents_used.append('WebSearchAgent')
            
            # Final Return
            return {
                'answer': answer,
                'sources': context['sources'],
                'plan': plan,
                'agents_used': agents_used,
                'web_sources': web_search_results
            }

        except Exception as e:
            await stream_callback("error", {"message": str(e)})
            raise e
        finally:
             end_token_aggregation(token_ctx, self.email, description="Multi-Model Pipeline")

    def _get_files(self):
        # Sync wrapper for DB
        try:
            raw_files = self.db.get_multi_model_files(self.session_id)
            return [
                {
                    'file_name': f['file_name'],
                    'file_type': f['file_type'],
                    'storage_path': f['storage_path'],
                    'vector_collection_id': f.get('vector_collection_id'),
                    'db_table_name': f.get('db_table_name')
                }
                for f in raw_files
            ]
        except:
             return []
