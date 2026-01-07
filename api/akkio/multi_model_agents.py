"""
Multi-Model Agentic System
Implements: Intent Interpreter → Planner → Dynamic Agent Generator → Specialist Agents → Validator
"""

import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from chromadb.config import Settings
import pandas as pd
import os


class IntentInterpreterAgent:
    """Analyzes user query to understand intent and required capabilities"""
    
    def __init__(self, system_prompt: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.system_prompt = system_prompt
        self.id = None  # Add id attribute to prevent serialization errors
        self.email = None  # Add email attribute to prevent serialization errors
    
    def interpret(self, query: str, available_sources: List[Dict[str, Any]], messages: List[Dict] = None) -> Dict[str, Any]:
        """
        Interpret user query and determine what kind of analysis is needed
        Returns: {
            'intent': str,  # 'data_analysis', 'document_search', 'image_analysis', 'multi_domain'
            'required_sources': List[str],  # Which files/sources are needed
            'analysis_type': str,  # 'statistical', 'semantic', 'visual', 'hybrid'
            'complexity': str  # 'simple', 'moderate', 'complex'
        }
        """
        # Fast path for greetings/chat
        import re
        simple_chat_patterns = [
            r'^(hi|hello|hey|greetings)(?:\s+(?:there|all|everyone))?[\.!]*$', 
            r'^how are you\??$',
            r'^thanks(?:\s+you)?[\.!]*$',
            r'^good\s+(?:morning|afternoon|evening)[\.!]*$'
        ]
        
        normalized_query = query.lower().strip()
        for pattern in simple_chat_patterns:
            if re.match(pattern, normalized_query):
                return {
                    'intent': 'chat',
                    'required_sources': [],
                    'analysis_type': 'none',
                    'complexity': 'simple',
                    'key_entities': [],
                    'expected_output': 'text',
                    'generation_format': None
                }
                
        sources_info = "\n".join([
            f"- {s['file_name']} ({s['file_type']}): {s.get('description', 'No description')}"
            for s in available_sources
        ])
        
        prompt = f"""
{self.system_prompt}

Available Data Sources:
{sources_info}

Conversation History:
{json.dumps(messages, indent=2) if messages else "No history"}

User Query: {query}

Analyze this query and provide a JSON response with:
1. intent: What is the user trying to do? (chat/data_analysis/document_search/image_analysis/multi_domain). Use "chat" for greetings, general questions, or small talk not requiring specific files.
2. required_sources: Which files are needed? (list of file names, empty if "chat")
3. analysis_type: What kind of analysis? (statistical/semantic/visual/hybrid/none)
4. complexity: How complex is this query? (simple/moderate/complex)
5. key_entities: Important entities or concepts mentioned
6. expected_output: What format should the answer be in?
7. generation_format: If user asks to "generate", "create", "download", "build", "develop", "implement", or "give" a file/report/pdf/code, specify "pdf", "csv", or "excel". Default to "pdf" for "report" or code generation requests. Otherwise null.

Respond ONLY with valid JSON.
"""
        
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
            return {
                'intent': 'multi_domain',
                'required_sources': [s['file_name'] for s in available_sources],
                'analysis_type': 'hybrid',
                'complexity': 'moderate',
                'key_entities': [],
                'expected_output': 'text'
            }


class PlannerAgent:
    """Creates execution plan based on intent"""
    
    def __init__(self, system_prompt: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.system_prompt = system_prompt
        self.id = None  # Add id attribute to prevent serialization errors
        self.email = None  # Add email attribute to prevent serialization errors
    
    def create_plan(self, intent_analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Create step-by-step execution plan
        Returns: {
            'steps': List[Dict],  # Each step with action, agent_type, inputs
            'dependencies': Dict,  # Step dependencies
            'estimated_time': str
        }
        """
        prompt = f"""
{self.system_prompt}

Intent Analysis:
{json.dumps(intent_analysis, indent=2)}

User Query: {query}

Create a detailed execution plan as JSON with:
1. steps: List of steps, each with:
   - step_id: unique identifier
   - action: what to do
   - agent_type: which specialist agent (data_analyst/document_expert/image_analyzer/synthesizer)
   - inputs: what data/sources needed
   - output: what this step produces
2. dependencies: which steps depend on others (dict of step_id: [prerequisite_step_ids])
3. estimated_time: rough estimate

Respond ONLY with valid JSON.
"""
        
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            # Fallback plan
            return {
                'steps': [
                    {
                        'step_id': '1',
                        'action': 'analyze_query',
                        'agent_type': 'synthesizer',
                        'inputs': intent_analysis['required_sources'],
                        'output': 'comprehensive_answer'
                    }
                ],
                'dependencies': {},
                'estimated_time': '10-30 seconds'
            }


class DynamicAgentGenerator:
    """Generates specialist agents on-demand based on plan"""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.id = None  # Add id attribute to prevent serialization errors
        self.email = None  # Add email attribute to prevent serialization errors
    
    def generate_agent(self, agent_type: str, task_description: str) -> 'SpecialistAgent':
        """Generate a specialist agent for a specific task"""
        agent_prompts = {
            'data_analyst': f"""
{self.system_prompt}

You are a DATA ANALYST specialist. Your role:
- Analyze tabular data (CSV, Excel)
- Perform statistical analysis
- Identify patterns and trends
- Provide data-driven insights
- Always cite specific data points

Task: {task_description}
""",
            'document_expert': f"""
{self.system_prompt}

You are a DOCUMENT ANALYSIS specialist. Your role:
- Extract information from documents (PDF, Word, Text)
- Understand context and semantics
- Find relevant passages
- Summarize complex information
- Always cite source documents and page numbers

Task: {task_description}
""",
            'image_analyzer': f"""
{self.system_prompt}

You are an IMAGE ANALYSIS specialist. Your role:
- Analyze images and visual data
- Identify patterns in images
- Extract visual information
- Describe image content
- Always reference specific images

Task: {task_description}
""",
            'synthesizer': f"""
{self.system_prompt}

You are a SYNTHESIS specialist. Your role:
- Combine insights from multiple sources
- Provide comprehensive answers and elaborate as much as you can for production more like chatgpt and if user asks to create code then you must shoudl provide proper code based on previous history and based on his requirements step by step.
- Ensure consistency across sources
- Highlight agreements and conflicts
- Provide confidence scores

Task: {task_description}
"""
        }
        
        prompt = agent_prompts.get(agent_type, agent_prompts['synthesizer'])
        return SpecialistAgent(prompt, agent_type, self.llm)


class SpecialistAgent:
    """Individual specialist agent for specific tasks"""
    
    def __init__(self, prompt: str, agent_type: str, llm):
        self.prompt = prompt
        self.agent_type = agent_type
        self.llm = llm
        self.id = None  # Add id attribute to prevent serialization errors
        self.email = None  # Add email attribute to prevent serialization errors
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the specialist task
        Returns: {
            'result': Any,
            'confidence': float,
            'sources_used': List[str],
            'reasoning': str
        }
        """
        # Build context from inputs
        context = self._build_context(inputs)
        
        query = inputs.get('query', '')
        full_prompt = f"""
{self.prompt}

Context/Data:
{context}

Query: {query}

Provide your analysis as JSON with:
1. result: Your answer/analysis
2. confidence: Your confidence level (0-1)
3. sources_used: Which sources you used
4. reasoning: Your reasoning process

Respond ONLY with valid JSON.
"""
        
        response = self.llm.invoke(full_prompt)
        try:
            return json.loads(response.content)
        except:
            return {
                'result': response.content,
                'confidence': 0.7,
                'sources_used': list(inputs.get('sources', {}).keys()),
                'reasoning': 'Analysis completed'
            }
    
    def _build_context(self, inputs: Dict[str, Any]) -> str:
        """Build context string from various input types"""
        context_parts = []
        
        # Add data sources
        if 'sources' in inputs:
            for source_name, source_data in inputs['sources'].items():
                if isinstance(source_data, pd.DataFrame):
                    context_parts.append(f"\n=== Data from {source_name} ===")
                    context_parts.append(f"Shape: {source_data.shape}")
                    context_parts.append(f"Columns: {', '.join(source_data.columns.tolist())}")
                    context_parts.append(f"Sample data:\n{source_data.head(10).to_string()}")
                elif isinstance(source_data, str):
                    context_parts.append(f"\n=== Content from {source_name} ===")
                    context_parts.append(source_data[:2000])  # Limit to 2000 chars
                elif isinstance(source_data, list):
                    context_parts.append(f"\n=== Items from {source_name} ===")
                    context_parts.append("\n".join(str(item)[:500] for item in source_data[:5]))
        
        # Add vector search results
        if 'vector_results' in inputs:
            context_parts.append(f"\n=== Relevant Document Excerpts ===")
            for i, result in enumerate(inputs['vector_results'][:5], 1):
                context_parts.append(f"\n{i}. From {result.get('source', 'Unknown')}:")
                context_parts.append(f"   {result.get('content', '')[:500]}")
        
        return "\n".join(context_parts)


class ValidatorAgent:
    """Validates and refines outputs from specialist agents"""
    
    def __init__(self, system_prompt: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.system_prompt = system_prompt
        self.id = None  # Add id attribute to prevent serialization errors
        self.email = None  # Add email attribute to prevent serialization errors
    
    def validate_and_refine(self, specialist_outputs: List[Dict[str, Any]], original_query: str, intent_analysis: Optional[Dict[str, Any]] = None, messages: List[Dict] = None) -> Dict[str, Any]:
        """
        Validate outputs and create final refined answer
        Returns: {
            'answer': str,
            'confidence': float,
            'sources': List[Dict],
            'reasoning': str,
            'agents_used': List[str]
        }
        """
        outputs_summary = json.dumps(specialist_outputs, indent=2, default=str)
        intent_info = json.dumps(intent_analysis, indent=2) if intent_analysis else "Not available"
        
        # Check if generation_format indicates document creation
        generation_format = intent_analysis.get('generation_format') if intent_analysis else None
        
        # Detect code generation requests
        code_generation_keywords = ['build', 'create', 'generate code', 'develop', 'implement', 'write code', 'code for', 'build website', 'build application', 'create website', 'create application']
        is_code_generation = any(keyword in original_query.lower() for keyword in code_generation_keywords)
        
        should_create_report = (
            generation_format and generation_format != 'null' and 
            generation_format.lower() in ['pdf', 'doc', 'document']
        ) or any(keyword in original_query.lower() for keyword in ['create pdf', 'create doc', 'create document', 'generate pdf', 'generate doc', 'generate document', 'download pdf', 'download doc']) or is_code_generation
        
        # Extract previous documents and context from conversation history
        previous_docs_context = ""
        previous_queries = []
        if messages:
            # Extract last 20 messages for better context
            recent_messages = messages[-20:] if len(messages) > 20 else messages
            for msg in recent_messages:
                if isinstance(msg, dict):
                    # Extract user queries
                    if msg.get('type') == 'user' or 'user' in str(msg.get('role', '')).lower():
                        query_text = msg.get('content', '') or msg.get('message', '') or str(msg)
                        if query_text and len(query_text) > 10:
                            previous_queries.append(query_text[:200])
                    
                    # Extract document content from bot responses
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                # Extract from answer field
                                if parsed.get('answer'):
                                    prev_answer = parsed.get('answer', '')
                                    # Check for document-related content
                                    doc_keywords = ['document', 'BRD', 'wireframe', 'architecture', 'requirement', 'specification', 'design']
                                    if any(keyword.lower() in prev_answer.lower() for keyword in doc_keywords):
                                        previous_docs_context += f"\n--- Previous Document Context ---\n{prev_answer[:1000]}\n"
                                
                                # Extract from multi_model_metadata
                                if parsed.get('multi_model_metadata', {}).get('answer'):
                                    prev_answer = parsed.get('multi_model_metadata', {}).get('answer', '')
                                    doc_keywords = ['document', 'BRD', 'wireframe', 'architecture', 'requirement', 'specification', 'design']
                                    if any(keyword.lower() in prev_answer.lower() for keyword in doc_keywords):
                                        previous_docs_context += f"\n--- Previous Document Context (Metadata) ---\n{prev_answer[:1000]}\n"
                        except:
                            # If not JSON, check if it contains document keywords
                            doc_keywords = ['document', 'BRD', 'wireframe', 'architecture', 'requirement', 'specification', 'design']
                            if any(keyword.lower() in content.lower() for keyword in doc_keywords):
                                previous_docs_context += f"\n--- Previous Context ---\n{content[:1000]}\n"
        
        # Format previous queries for context
        queries_context = "\n".join([f"- {q}" for q in previous_queries[-5:]]) if previous_queries else "No previous queries"
        
        prompt = f"""
{self.system_prompt}

Original Query: {original_query}

Conversation History (Last 20 messages for full context):
{json.dumps(messages[-20:] if messages else [], indent=2) if messages else "No history"}

Previous User Queries (Last 5):
{queries_context}

Previous Documents/Context Referenced:
{previous_docs_context if previous_docs_context else "No previous documents referenced - User may be starting a new task"}

Intent Analysis:
{intent_info}

Generation Format: {generation_format or 'null'}
Should Create Report: {should_create_report}
Is Code Generation Request: {is_code_generation}

{"⚠️ CODE GENERATION MODE ⚠️" if is_code_generation else ""}
If this is a CODE GENERATION request, you MUST generate ACTUAL, COMPLETE CODE FILES, not descriptions or explanations!

Specialist Agent Outputs:
{outputs_summary}

Your task:
1. **CRITICAL: Analyze the conversation history carefully**. If the user is asking you to build, create, or generate something based on previous documents (like BRD, wireframes, architecture docs), you MUST reference those documents from the conversation history.

2. **CODE GENERATION DETECTION**: If the user asks to "build", "create", "generate code", "develop", or "implement" a website/application/system, you MUST:
   - Generate ACTUAL, COMPLETE CODE FILES with FULL IMPLEMENTATIONS (not descriptions, not folder structures, not pseudo-code)
   - For EACH file, provide the COMPLETE code with:
     * All imports and dependencies
     * All classes, functions, and methods fully implemented
     * All business logic, error handling, and validation
     * No placeholders, no "TODO" comments, no incomplete code
   - Create file-by-file code structure with proper file paths as headers
   - Include ALL necessary files: backend, frontend, database schemas, config files, README, etc.
   - Format code in proper code blocks with file paths as <h3> headers followed by <pre><code> blocks
   - Ensure code is production-ready, error-free, and runnable
   - Include database migrations, seed files, environment templates
   - Add comprehensive README with setup instructions
   - Follow the technology stack from previous documents
   - **CRITICAL**: Show actual code for each file, not just file names or descriptions

3. For complex multi-step tasks (like "build an e-commerce website based on these documents"):
   - Review ALL previous messages in the conversation history
   - Extract relevant information from previous documents mentioned (BRD, wireframes, architecture)
   - Generate COMPLETE CODE FILES for each component:
     * Backend API files (routes, controllers, models, middleware)
     * Frontend components (React components, pages, utilities)
     * Database schema and migration files
     * Configuration files (.env templates, package.json, etc.)
     * Test files (unit tests, integration tests)
     * Documentation (README with complete setup instructions)
   - Each code file should be clearly marked with its file path
   - Code should be complete, runnable, and follow best practices
   - Reference specific requirements, technologies, and specifications from previous documents

4. Validate the outputs for consistency and accuracy
5. Identify any conflicts or inconsistencies
6. Synthesize a comprehensive final answer in HTML format with CODE BLOCKS
7. ALWAYS construct a "report" object IF:
   - The user EXPLICITLY asks for a "report", "pdf", "file", "document", "doc", or "download", OR
   - The generation_format is "pdf", "doc", or "document", OR
   - The user asks to "build", "create", or "generate" code/website/application
   Otherwise, set "report" to null.
8. List all sources with relevance scores (if any used)
9. Provide a detailed, reasoning-based answer similar to ChatGPT (structure, depth, clarity).

CRITICAL FOR CODE GENERATION:
- **YOU MUST GENERATE ACTUAL CODE FOR EACH FILE, NOT JUST FOLDER STRUCTURES OR DESCRIPTIONS**
- When generating code, use this EXACT format in your answer:
  <h3>File: backend/app/models/user.py</h3>
  <pre><code class="language-python">
  from sqlalchemy import Column, Integer, String, DateTime
  from sqlalchemy.ext.declarative import declarative_base
  from datetime import datetime
  
  Base = declarative_base()
  
  class User(Base):
      __tablename__ = 'users'
      
      id = Column(Integer, primary_key=True, index=True)
      email = Column(String, unique=True, index=True, nullable=False)
      password_hash = Column(String, nullable=False)
      full_name = Column(String)
      created_at = Column(DateTime, default=datetime.utcnow)
      
      def __repr__(self):
          return f"<User(id={self.id}, email={self.email})>"
  </code></pre>
  
  <h3>File: backend/app/routes/user_routes.py</h3>
  <pre><code class="language-python">
  from fastapi import APIRouter, Depends, HTTPException
  from sqlalchemy.orm import Session
  from app.models.user import User
  from app.database import get_db
  
  router = APIRouter()
  
  @router.post("/users/register")
  async def register_user(email: str, password: str, db: Session = Depends(get_db)):
      # Complete implementation here
      pass
  </code></pre>
  
- **DO NOT** just list folder structures like:
  ❌ WRONG: "backend/models/user.py - Contains user model"
  ❌ WRONG: "Create a file at backend/models/user.py with user class"
  
- **DO** provide complete, actual code:
  ✅ CORRECT: Show the full file path as <h3> header, then complete code in <pre><code> block
  
- Generate ALL files needed for a complete, working application with ACTUAL CODE:
  * backend/app/models/user.py - COMPLETE CODE with imports, class definition, all methods
  * backend/app/routes/user_routes.py - COMPLETE CODE with all route handlers
  * backend/app/controllers/user_controller.py - COMPLETE CODE with business logic
  * frontend/src/components/ProductList.jsx - COMPLETE CODE with React component
  * database/schema.sql - COMPLETE SQL with all CREATE TABLE statements
  * database/seed.py - COMPLETE Python code with all seed data
  * README.md - COMPLETE markdown with setup instructions
  * .env.example - COMPLETE environment variables template
  * requirements.txt - COMPLETE list of dependencies (without versions)
  * package.json - COMPLETE JSON with all dependencies and scripts

- Code must be COMPLETE, RUNNABLE, and PRODUCTION-READY
- Each file must have ALL necessary imports, classes, functions, and logic
- Include proper error handling, validation, and security measures
- Follow the exact technology stack mentioned in previous documents
- Ensure database setup instructions are clear and error-free
- NO placeholders, NO "TODO" comments, NO incomplete implementations

IMPORTANT FOR MULTI-STEP TASKS:
- When the user asks to "build", "create", or "generate" something based on previous documents, you MUST:
  * Reference the specific documents from conversation history
  * Extract key requirements, technologies, and specifications
  * Generate ACTUAL CODE FILES (not just descriptions)
  * Provide complete file structure with all necessary files
  * Ensure code is production-ready and follows all requirements
  * Break down complex tasks into clear, implementable code files
  * Address ALL requirements mentioned in the query with actual code

CRITICAL INSTRUCTIONS:
1. DO NOT wrap your answer in markdown code blocks (no ```json or ```).
2. Return ONLY raw JSON, nothing else.
3. Your 'answer' MUST be detailed and explain the "Why" and "How", not just the result. mimic a thoughtful AI assistant.
4. Format the answer field as clean, well-structured HTML with:
   - Use <h3>, <h4> for headings
   - Use <p> tags for paragraphs
   - Use <ul>, <li> for lists
   - Use <strong> for emphasis
   - Use <table border='1'> for tabular data if needed
   - Use <br> for line breaks where needed
   - Make it visually appealing and easy to read

Provide response as RAW JSON (no markdown, no code blocks) with these exact fields:
{{
  "answer": "Your HTML formatted answer here",
  "sources": [{{"file_name": "...", "file_type": "...", "relevance": 0.0-1.0, "excerpt": "..."}}],
  "reasoning": "Your reasoning in plain text",
  "agents_used": ["agent1", "agent2"],
  "validation_notes": "Any notes in plain text",
  "generation_format": "pdf/csv/excel/null",
  "report": {{ 
      "heading": "Report Title",
      "paragraphs": ["<p>...</p>"], 
      "table": {{...}}, 
      "charts": [] 
  }}
}}

CRITICAL FOR REPORT OBJECT:
- If "Should Create Report" is True, you MUST create a report object from your answer.
- The 'report.paragraphs' MUST contain the FULL, DETAILED content of your answer, split into logical blocks. Do not just summarize. 
- **FOR CODE GENERATION**: If the user asked to build/create/generate code, the report MUST include ALL code files with proper formatting:
  * Each file should be in a separate paragraph with <h4>File: path/to/file</h4> header
  * Code should be wrapped in <pre><code> blocks for proper formatting
  * Include complete file structure and all necessary files
  * Preserve code formatting, indentation, and syntax
- The 'report.paragraphs' must contain ALL that content so the generated PDF is complete. You can use HTML tags inside the paragraphs strings.
- Extract the heading from the first <h3> tag in your answer, or use a meaningful title based on the query.
- Split your answer into paragraphs preserving the HTML structure (p, h4, ul, ol, pre, code tags).
- For code generation tasks, ensure each code file is clearly separated and properly formatted for PDF export.

Example answer field:
"answer": "<h3>Image Analysis Overview</h3><p>The image file <strong>'23-1.png'</strong> is referenced, but without access to its actual content, a specific explanation cannot be provided.</p><ul><li><strong>Data Visualizations:</strong> Graphs or charts that present data in a visual format.</li><li><strong>Illustrations:</strong> Artistic representations that convey information.</li></ul>"

REMEMBER: Return ONLY the JSON object, no markdown formatting around it.
"""
        
        response = self.llm.invoke(prompt)
        
        # Try to parse JSON from response
        try:
            # First try direct JSON parsing
            result = json.loads(response.content)
            return result
        except:
            # Try to extract JSON from markdown code blocks
            import re
            content = response.content
            
            # Remove markdown code blocks (```json ... ``` or ``` ... ```)
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except:
                    pass
            
            # If still can't parse, return a simple HTML answer
            return {
                'answer': f'<p>{content}</p>',
                'sources': [],
                'reasoning': 'Synthesized from specialist agents',
                'agents_used': [output.get('agent_type', 'unknown') for output in specialist_outputs],
                'validation_notes': 'Response generated successfully',
                'report': None,
                'generation_format': None
            }


class MultiModelAgentSystem:
    """Main orchestrator for the multi-model agentic system"""
    
    def __init__(self, session_id: str, system_prompt: str, db_connection, chroma_client):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.db = db_connection
        self.chroma_client = chroma_client
        self.id = None  # Add id attribute to prevent serialization errors
        self.email = None  # Add email attribute to prevent serialization errors
        
        # Initialize agents
        self.intent_interpreter = IntentInterpreterAgent(system_prompt)
        self.planner = PlannerAgent(system_prompt)
        self.agent_generator = DynamicAgentGenerator(system_prompt)
        self.validator = ValidatorAgent(system_prompt)
    
    def query(self, user_query: str, messages: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a user query through the full agentic pipeline
        """
        # Get available sources
        available_sources = self._get_available_sources()
        
        # Step 1: Intent Interpretation
        intent_analysis = self.intent_interpreter.interpret(user_query, available_sources, messages)
        
        # Step 2: Optimization for Chat/Simple queries
        # If intent is 'chat' or no specific sources required and intent is simple, skip planning/agents
        if intent_analysis.get('intent') == 'chat' or (not intent_analysis.get('required_sources') and intent_analysis.get('complexity') == 'simple'):
             specialist_outputs = [{
                 'agent_type': 'chat_response',
                 'result': 'Direct chat response',
                 'confidence': 1.0,
                 'sources_used': [],
                 'reasoning': 'Direct chat interaction'
             }]
        else:
            # Step 2: Planning (Standard Path)
            execution_plan = self.planner.create_plan(intent_analysis, user_query)
            
            # Step 3: Execute plan with specialist agents
            specialist_outputs = []
            for step in execution_plan['steps']:
                agent = self.agent_generator.generate_agent(
                    step['agent_type'],
                    step['action']
                )
                
                # Prepare inputs for this agent
                inputs = self._prepare_agent_inputs(step, user_query)
                
                # Execute
                output = agent.execute(inputs)
                output['agent_type'] = step['agent_type']
                output['step_id'] = step['step_id']
                specialist_outputs.append(output)
        
        # Step 4: Validation and Refinement
        final_result = self.validator.validate_and_refine(specialist_outputs, user_query, intent_analysis, messages)
        
        # If generation_format indicates document creation but report is null, create report from answer
        generation_format = final_result.get('generation_format')
        if generation_format and generation_format != 'null' and generation_format.lower() in ['pdf', 'doc', 'document']:
            if not final_result.get('report') and final_result.get('answer'):
                # Convert answer to report format
                answer_html = final_result['answer']
                
                # Extract heading from first h3 tag or use default
                import re
                heading_match = re.search(r'<h3[^>]*>(.*?)</h3>', answer_html, re.IGNORECASE)
                heading = heading_match.group(1).strip() if heading_match else 'Generated Document'
                # Clean HTML tags from heading
                heading = re.sub(r'<[^>]+>', '', heading)
                
                # Split answer into paragraphs
                # Replace h3 with h4 for better formatting
                processed_html = re.sub(r'<h3([^>]*)>', r'<h4\1>', answer_html, flags=re.IGNORECASE)
                processed_html = re.sub(r'</h3>', r'</h4>', processed_html, flags=re.IGNORECASE)
                
                paragraphs = []
                # Extract all block-level elements (p, h4, ul, ol, div)
                # Use a pattern that captures the full element
                block_pattern = r'<(p|h4|ul|ol|div)[^>]*>.*?</\1>'
                full_blocks = re.finditer(block_pattern, processed_html, re.DOTALL | re.IGNORECASE)
                
                blocks_found = False
                for match in full_blocks:
                    block = match.group(0)
                    if block.strip():
                        paragraphs.append(block.strip())
                        blocks_found = True
                
                if not blocks_found:
                    # Fallback: split by line breaks or use whole content
                    lines = processed_html.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            if not line.startswith('<'):
                                paragraphs.append(f'<p>{line}</p>')
                            else:
                                paragraphs.append(line)
                
                # If still no paragraphs, use the whole HTML
                if not paragraphs:
                    paragraphs = [processed_html]
                
                # Create report object
                final_result['report'] = {
                    'heading': heading or 'Generated Document',
                    'paragraphs': paragraphs,
                    'table': None,
                    'charts': []
                }
        
        return final_result
    
    def _get_available_sources(self) -> List[Dict[str, Any]]:
        """Get all available data sources for this session"""
        files = self.db.get_multi_model_files(self.session_id)
        return [
            {
                'file_name': f['file_name'],
                'file_type': f['file_type'],
                'storage_path': f['storage_path'],
                'vector_collection_id': f['vector_collection_id'],
                'db_table_name': f['db_table_name']
            }
            for f in files
        ]
    
    def _prepare_agent_inputs(self, step: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Prepare inputs for a specialist agent"""
        inputs = {'query': query, 'sources': {}}
        
        # Get required sources
        required_files = step.get('inputs', [])
        files = self.db.get_multi_model_files(self.session_id)
        
        for file_info in files:
            if file_info['file_name'] in required_files or not required_files:
                # Load data based on type
                if file_info['file_type'] == 'tabular' and file_info['db_table_name']:
                    try:
                        data = self.db.get_table_data(file_info['db_table_name'])
                        inputs['sources'][file_info['file_name']] = data
                    except:
                        pass
                
                elif file_info['file_type'] in ['document', 'image'] and file_info['vector_collection_id']:
                    try:
                        # Query vector database
                        collection = self.chroma_client.get_collection(file_info['vector_collection_id'])
                        results = collection.query(
                            query_texts=[query],
                            n_results=5
                        )
                        if 'vector_results' not in inputs:
                            inputs['vector_results'] = []
                        
                        for i, doc in enumerate(results['documents'][0]):
                            inputs['vector_results'].append({
                                'source': file_info['file_name'],
                                'content': doc,
                                'distance': results['distances'][0][i] if 'distances' in results else 0
                            })
                    except:
                        pass
        
        return inputs

