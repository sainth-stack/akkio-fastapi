import asyncio
from typing import TypedDict, Annotated, Dict, Any, List
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # If langgraph is not installed, we assume it will be available in the environment.
    pass

from ..schemas.requirements import UserRequirement
from ..agents.requirement_structuring_agent import requirement_structuring_agent
from ..agents.architecture_agent import stream_architecture_generation
from ..agents.api_contract_agent import api_contract_agent
from ..agents.database_schema_agent import database_schema_agent
from ..agents.backend_generator_agent import backend_generator_agent
from ..agents.frontend_generator_agent import frontend_generator_agent
from ..agents.validation_agent import validate_and_fix_code
from ..services.file_writer import file_writer
from ..schemas.files import GeneratedFiles

class BuilderState(TypedDict):
    user_requirement: UserRequirement
    project_name: str
    structured_requirement: Dict[str, Any]
    architecture: Dict[str, Any]
    api_contract: Dict[str, Any]
    db_schema: Dict[str, Any]
    generated_files: Dict[str, str]
    validation_results: Dict[str, Any]
    error: str

async def run_structuring_step(state: BuilderState):
    print(f"[builder_graph] Starting structuring_step...")
    try:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)
        req = state.get("user_requirement")
        structured = await requirement_structuring_agent(req, llm=llm)
        print(f"[builder_graph] Completed structuring_step.")
        return {"structured_requirement": structured, "project_name": structured.get("project_name", "app")}
    except Exception as e:
        print(f"[builder_graph] Structuring failed: {e}")
        return {"error": f"Structuring failed: {str(e)}"}

async def run_architecture_step(state: BuilderState):
    print(f"[builder_graph] Starting architecture_step...")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.7)
        structured = state.get("structured_requirement")
        requirement_text = str(structured)
        
        # Use stream_architecture_generation and pick the final result
        arch_dict = {}
        async for event in stream_architecture_generation(requirement_text, requirement_text, [], llm):
            if event["event"] == "architecture_complete":
                arch_dict = event["data"]
        
        print(f"[builder_graph] Completed architecture_step.")
        return {"architecture": arch_dict}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Architecture failed: {str(e)}"}

async def run_contract_step(state: BuilderState):
    print(f"[builder_graph] Starting contract_step...")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)
        structured = state.get("structured_requirement")
        arch = state.get("architecture")
        contract = await api_contract_agent(structured, arch, llm=llm)
        print(f"[builder_graph] Completed contract_step.")
        return {"api_contract": contract}
    except Exception as e:
        print(f"[builder_graph] Contract failed: {e}")
        return {"error": f"Contract generation failed: {str(e)}"}

async def run_schema_step(state: BuilderState):
    print(f"[builder_graph] Starting schema_step...")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)
        structured = state.get("structured_requirement")
        arch = state.get("architecture")
        schema = await database_schema_agent(structured, llm=llm)
        print(f"[builder_graph] Completed schema_step.")
        return {"db_schema": {"schema": schema}}
    except Exception as e:
        print(f"[builder_graph] Schema failed: {e}")
        return {"error": f"Schema generation failed: {str(e)}"}

async def run_coding_step(state: BuilderState):
    print(f"[builder_graph] Starting coding_step (this may take a while)...")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.2)
        structured = state.get("structured_requirement")
        arch = state.get("architecture")
        contract = state.get("api_contract")
        schema_data = state.get("db_schema", {})
        schema_text = schema_data.get("schema", "")
        
        backend_files, frontend_files = await asyncio.gather(
            backend_generator_agent(structured, arch, contract, schema_text, llm=llm),
            frontend_generator_agent(structured, arch, contract, llm=llm)
        )
        
        all_files = {**backend_files, **frontend_files}
        project_name = state.get("project_name", "app")
        
        # PERSIST TO DISK IMMEDIATELY
        print(f"[builder_graph] Persisting {len(all_files)} files to disk for project: {project_name}")
        file_writer(project_name, GeneratedFiles(files=all_files))
        
        print(f"[builder_graph] Completed coding_step. Generated {len(all_files)} files.")
        return {"generated_files": all_files}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Coding failed: {str(e)}"}

async def run_validation_step(state: BuilderState):
    print(f"[builder_graph] Starting validation_step...")
    try:
        if state.get("error"): return {}
        files = state.get("generated_files")
        arch = state.get("architecture")
        # validate_and_fix_code is synchronous but we wrapped it in run_validation_step async
        fixed_files = validate_and_fix_code(files, arch)
        project_name = state.get("project_name", "app")
        
        # PERSIST FIXED FILES IMMEDIATELY
        print(f"[builder_graph] Persisting validated/fixed files to disk for project: {project_name}")
        file_writer(project_name, GeneratedFiles(files=fixed_files))
        
        print(f"[builder_graph] Completed validation_step.")
        return {"generated_files": fixed_files, "validation_results": {"status": "success"}}
    except Exception as e:
        print(f"[builder_graph] Validation failed: {e}")
        return {"error": f"Validation failed: {str(e)}"}

# Initialize Graph
workflow = StateGraph(BuilderState)

workflow.add_node("structuring_step", run_structuring_step)
workflow.add_node("architecture_step", run_architecture_step)
workflow.add_node("contract_step", run_contract_step)
workflow.add_node("schema_step", run_schema_step)
workflow.add_node("coding_step", run_coding_step)
workflow.add_node("validation_step", run_validation_step)

workflow.set_entry_point("structuring_step")
workflow.add_edge("structuring_step", "architecture_step")
workflow.add_edge("architecture_step", "contract_step")
workflow.add_edge("contract_step", "schema_step")
workflow.add_edge("schema_step", "coding_step")
workflow.add_edge("coding_step", "validation_step")
workflow.add_edge("validation_step", END)

app_builder_graph = workflow.compile()
