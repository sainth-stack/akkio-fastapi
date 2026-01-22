
import asyncio
import sys
import os
import io
import pandas as pd
from unittest.mock import MagicMock, AsyncMock

# Add path to find modules
sys.path.append('/Users/sainathreddy/tech/styli/styli-tk/akkio/akkio-fastapi')

from api.akkio.multi_model_agents import MultiModelAgentSystem

# Mock DB
class MockDB:
    def get_multi_model_files(self, session_id):
        return [
            {
                'file_name': 'test_data.csv',
                'file_type': 'tabular',
                'storage_path': '/tmp/test_data.csv',
                'vector_collection_id': None,
                'db_table_name': 'test_table'
            },
            {
                'file_name': 'test_image.png',
                'file_type': 'image',
                'storage_path': '/tmp/test_image.png',
                'vector_collection_id': 'img_col',
                'db_table_name': None
            }
        ]
    
    def get_table_data(self, table_name):
        return pd.DataFrame({
            'doctor': ['Dr. Smith', 'Dr. Jones', 'Dr. Strange'],
            'hospital': ['KIMS', 'Apollo', 'Rainbow'],
            'availability': ['Mon-Fri', 'Sat-Sun', 'Always']
        })

# Mock Chroma
class MockChroma:
    def get_collection(self, name):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            'documents': [['Some document content relevant to query']],
            'distances': [[0.1]]
        }
        return mock_col

# Mock LLM slightly to avoid hefty API calls if possible, or just let it run if we want real integration test.
# For this verify, we want to see if the ORCHESTRATION works.
# But we also want to see if code gen works. Real LLM is better for code gen.

# Setup dummy files
with open('/tmp/test_data.csv', 'w') as f:
    f.write("doctor,hospital,availability\nDr. Smith,KIMS,Mon-Fri\nDr. Jones,Apollo,Sat-Sun")

# Mock LLM and Env
os.environ['OPENAI_API_KEY'] = 'sk-dummy-key-for-testing'

# Create a small valid white image
from PIL import Image
img = Image.new('RGB', (60, 30), color = 'white')
img.save('/tmp/test_image.png')

async def main():
    print("Initializing System...")
    sys = MultiModelAgentSystem(
        session_id="test_session",
        system_prompt="You are a helpful assistant.",
        db_connection=MockDB(),
        chroma_client=MockChroma()
    )
    
    # MOCK THE LLMS to return valid JSON
    mock_llm_response = MagicMock()
    
    # 1. Intent Interpreter Mock
    sys.intent_interpreter.llm = MagicMock()
    sys.intent_interpreter.llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "intent": "multi_domain",
        "required_sources": ["test_data.csv"],
        "analysis_type": "hybrid",
        "complexity": "moderate"
    }
    '''))
    
    # 2. Planner Mock
    sys.planner.llm = MagicMock()
    sys.planner.llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "steps": [
            {"step_id": "1", "action": "analyze_csv", "agent_type": "data_analyst", "inputs": ["test_data.csv"], "output": "csv_analysis"},
            {"step_id": "2", "action": "check_image", "agent_type": "image_analyzer", "inputs": ["test_image.png"], "output": "img_analysis"}
        ],
        "dependencies": {}, 
        "estimated_time": "10s"
    }
    '''))
    
    # 3. Dynamic Agent / Specialist Mock
    # We need to mock the LLM that is assigned to the specialist agent.
    # The generator creates a new SpecialistAgent with a NEW LLM instance (from self.llm).
    # so we need to mock the generator's llm.
    sys.agent_generator.llm = MagicMock()
    
    # Helper to return different responses based on prompt or context (simplified for test)
    # We'll just return a success message for any agent
    async def specialist_side_effect(*args, **kwargs):
        # We need to return valid python code for data analyst
        # And valid text/json for others.
        # Check args/prompt to decide
        prompt_str = str(args)
        
        if "DATA ANALYST" in prompt_str or "Pandas" in prompt_str:
            return MagicMock(content='''
```python
result = "Dr. Smith is at KIMS on Mon-Fri."
```
''')
        elif "IMAGE ANALYSIS" in prompt_str:
             return MagicMock(content='The image is white.')
        else:
             return MagicMock(content='{"result": "Generic result", "confidence": 1.0, "sources_used": [], "reasoning": "mock"}')

    sys.agent_generator.llm.ainvoke = AsyncMock(side_effect=specialist_side_effect)

    # 4. Validator Mock
    sys.validator.llm = MagicMock()
    sys.validator.llm.ainvoke = AsyncMock(return_value=MagicMock(content='''
    {
        "answer": "<h3>Analysis Result</h3><p>Dr. Smith is at KIMS and the image is white.</p>",
        "sources": [],
        "reasoning": "Combined mock results",
        "agents_used": ["data_analyst", "image_analyzer"],
        "validation_notes": "Verified",
        "report": null,
        "generation_format": null
    }
    '''))
    
    # Test 3: Parallel Execution
    print("\n--- TEST 3: Parallel Execution (Docs + Data) ---")
    query3 = "Compare the doctor in KIMS with the color of the image."
    
    start_time = asyncio.get_event_loop().time()
    result3 = await sys.query_async(query3)
    end_time = asyncio.get_event_loop().time()
    
    print("Result Answer:", result3['answer'])
    print(f"Execution Time: {end_time - start_time:.4f}s")
    print("Agents used:", result3.get('multi_model_metadata', {}).get('agents_used'))
    
    # Test 4: Report Suppression Test
    print("\n--- TEST 4: Report Suppression (Should be Null) ---")
    
    # We need to mock validator again to respect the new instruction?
    # Actually, since we are mocking the LLM response, we can't test if the PROMPT actually works on a real LLM.
    # But we can test if the 'should_create_report' logic is working by inspecting the prompt passed to the mock.
    
    query4 = "Just tell me who is available."
    # Validtor mock needs to be updated or we check the call args
    
    # Reset mock to capture new call
    sys.validator.llm.ainvoke.reset_mock()
    # Return a response that MIGHT have a report if the prompt didn't forbid it
    # But here we just want to see if the boolean in prompt was False.
    sys.validator.llm.ainvoke.return_value = MagicMock(content='{"answer": "ok", "report": null}')
    
    await sys.query_async(query4)
    
    # Inspect the prompt sent to Validator
    call_args = sys.validator.llm.ainvoke.call_args
    if call_args:
        prompt_sent = str(call_args[0][0]) # The prompt string
        if "Should Create Report: False" in prompt_sent:
            print("SUCCESS: Validated that 'Should Create Report' was False for simple query.")
        else:
            print("FAILURE: 'Should Create Report' was NOT False.")
            print(prompt_sent[:500]) # Print start of prompt
            
    # Test 5: Report Trigger Test
    print("\n--- TEST 5: Report Trigger (Should be True) ---")
    query5 = "Create a PDF report of availability."
    sys.validator.llm.ainvoke.reset_mock()
    sys.validator.llm.ainvoke.return_value = MagicMock(content='{"answer": "ok", "report": {}}')
    
    await sys.query_async(query5)
    
    call_args = sys.validator.llm.ainvoke.call_args
    if call_args:
        prompt_sent = str(call_args[0][0])
        if "Should Create Report: True" in prompt_sent:
            print("SUCCESS: Validated that 'Should Create Report' was True for PDF request.")
        else:
            print("FAILURE: 'Should Create Report' was NOT True.")

if __name__ == "__main__":
    asyncio.run(main())
