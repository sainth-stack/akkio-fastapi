import asyncio
import logging
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

logger = logging.getLogger("app_builder")

class BuilderState(TypedDict):
    user_requirement: UserRequirement
    project_name: str
    template_name: str
    structured_requirement: Dict[str, Any]
    architecture: Dict[str, Any]
    api_contract: Dict[str, Any]
    db_schema: Dict[str, Any]
    generated_files: Dict[str, str]
    validation_results: Dict[str, Any]
    error: str

async def run_structuring_step(state: BuilderState):
    logger.info("[structuring] START")
    try:
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)
        req = state.get("user_requirement")
        desc = (req.description if hasattr(req, "description") else str(req)) if req else ""
        logger.info("[structuring] requirement=%r", desc[:100] + "..." if len(desc) > 100 else desc)
        structured = await requirement_structuring_agent(req, llm=llm)
        proj = structured.get("project_name", "app")
        tmpl = structured.get("template_name")
        logger.info("[structuring] DONE | project_name=%s | template_name=%s", proj, tmpl)
        return {
            "structured_requirement": structured,
            "project_name": proj,
            "template_name": tmpl,
        }
    except Exception as e:
        logger.exception("[structuring] FAILED: %s", e)
        return {"error": f"Structuring failed: {str(e)}"}

async def run_architecture_step(state: BuilderState):
    logger.info("[architecture] START")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0.7)
        structured = state.get("structured_requirement")
        requirement_text = str(structured)
        arch_dict = {}
        async for event in stream_architecture_generation(requirement_text, requirement_text, [], llm):
            if event["event"] == "architecture_complete":
                arch_dict = event["data"]
        logger.info("[architecture] DONE")
        return {"architecture": arch_dict}
    except Exception as e:
        logger.exception("[architecture] FAILED: %s", e)
        return {"error": f"Architecture failed: {str(e)}"}

async def run_contract_step(state: BuilderState):
    logger.info("[contract] START")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)
        structured = state.get("structured_requirement")
        arch = state.get("architecture")
        contract = await api_contract_agent(structured, arch, llm=llm)
        logger.info("[contract] DONE")
        return {"api_contract": contract}
    except Exception as e:
        logger.exception("[contract] FAILED: %s", e)
        return {"error": f"Contract generation failed: {str(e)}"}

async def run_schema_step(state: BuilderState):
    logger.info("[schema] START")
    try:
        if state.get("error"): return {}
        from llm_helper import get_llm_for_user
        llm = get_llm_for_user(user_email=None, temperature=0)
        structured = state.get("structured_requirement")
        arch = state.get("architecture")
        schema = await database_schema_agent(structured, llm=llm)
        logger.info("[schema] DONE")
        return {"db_schema": {"schema": schema}}
    except Exception as e:
        logger.exception("[schema] FAILED: %s", e)
        return {"error": f"Schema generation failed: {str(e)}"}

async def run_coding_step(state: BuilderState):
    logger.info("[coding] START | project_name=%s", state.get("project_name", "app"))
    try:
        if state.get("error"): return {}
        project_name = state.get("project_name", "app")
        template_name = state.get("template_name")
        if not template_name:
            req = state.get("user_requirement")
            requirement_text = (req.description if hasattr(req, "description") else str(req)) if req else ""
            try:
                from ..services.template_service import detect_template
                template_name = detect_template(requirement_text)
                if template_name:
                    logger.info("[coding] template detected from fallback: %s", template_name)
            except Exception as e:
                logger.debug("[coding] template detect fallback failed: %s", e)
                template_name = None
        try:
            if template_name:
                from ..services.template_service import get_template_code_files
                from ..agents.validation_agent import polish_template_output
                from ..agents.code_update_agent import tailor_template_to_requirement
                template_files = get_template_code_files(template_name)
                if template_files:
                    logger.info("[coding] USING TEMPLATE=%s | files=%d", template_name, len(template_files))
                    req = state.get("user_requirement")
                    requirement_text = (req.description if hasattr(req, "description") else str(req)) if req else ""
                    tailored = await tailor_template_to_requirement(template_files, requirement_text, template_name)
                    polished = polish_template_output(tailored, template_name)
                    file_writer(project_name, GeneratedFiles(files=polished))
                    logger.info("[coding] DONE (template) | project=%s | files_written=%d", project_name, len(polished))
                    return {"generated_files": polished}
        except Exception as t_err:
            logger.warning("[coding] template failed, falling back to LLM: %s", t_err)

        logger.info("[coding] generating via LLM (no template match)")
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
        logger.info("[coding] LLM generated %d files", len(all_files))
        file_writer(project_name, GeneratedFiles(files=all_files))
        logger.info("[coding] DONE (LLM) | project=%s", project_name)
        return {"generated_files": all_files}
    except Exception as e:
        logger.exception("[coding] FAILED: %s", e)
        return {"error": f"Coding failed: {str(e)}"}

async def run_validation_step(state: BuilderState):
    logger.info("[validation] START")
    try:
        if state.get("error"): return {}
        files = state.get("generated_files")
        arch = state.get("architecture") or {}
        template_name = state.get("template_name")
        project_name = state.get("project_name", "app")
        if template_name and files:
            from ..agents.validation_agent import polish_template_output
            files = polish_template_output(files, template_name)
            logger.info("[validation] polished template=%s", template_name)
        fixed_files = validate_and_fix_code(files, arch)
        logger.info("[validation] persisting %d fixed files to project=%s", len(fixed_files), project_name)
        file_writer(project_name, GeneratedFiles(files=fixed_files))
        logger.info("[validation] DONE")
        return {"generated_files": fixed_files, "validation_results": {"status": "success"}}
    except Exception as e:
        logger.exception("[validation] FAILED: %s", e)
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
