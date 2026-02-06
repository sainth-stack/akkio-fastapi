from typing import TypedDict, Annotated, Dict
try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # If langgraph is not installed, we might fail here. 
    # But user asked to add it to requirements.txt, so we assume it will be available.
    pass

from ..schemas.requirements import UserRequirement
from ..schemas.plan import ProjectPlan
from ..schemas.architecture import ArchitectureDecision
from ..schemas.files import GeneratedFiles
from ..agents.requirement_agent import requirement_agent
from ..agents.planner_agent import planner_agent
from ..agents.architecture_agent import architecture_agent
from ..agents.code_generator_agent import code_generator_agent

class BuilderState(TypedDict):
    user_requirement: UserRequirement
    clarified_requirement: str
    plan: ProjectPlan
    architecture: ArchitectureDecision
    generated_files: GeneratedFiles
    error: str

def run_requirement_agent(state: BuilderState):
    try:
        req = state.get("user_requirement")
        clarified = requirement_agent(req)
        return {"clarified_requirement": clarified}
    except Exception as e:
        return {"error": str(e)}

def run_planner_agent(state: BuilderState):
    try:
        if state.get("error"):
            return {}
        clarified = state.get("clarified_requirement")
        plan = planner_agent(clarified)
        return {"plan": plan}
    except Exception as e:
        return {"error": str(e)}

def run_architecture_agent(state: BuilderState):
    try:
        if state.get("error"):
            return {}
        plan = state.get("plan")
        arch = architecture_agent(plan)
        return {"architecture": arch}
    except Exception as e:
        return {"error": str(e)}

def run_code_generator_agent(state: BuilderState):
    try:
        if state.get("error"):
            return {}
        arch = state.get("architecture")
        files_dict = code_generator_agent(arch)
        return {"generated_files": GeneratedFiles(files=files_dict)}
    except Exception as e:
        return {"error": str(e)}

# Initialize Graph
workflow = StateGraph(BuilderState)

workflow.add_node("planning_agent_step1_req", run_requirement_agent)
workflow.add_node("planning_agent_step2_plan", run_planner_agent)
workflow.add_node("planning_agent_step3_arch", run_architecture_agent)
workflow.add_node("coding_agent", run_code_generator_agent)

workflow.set_entry_point("planning_agent_step1_req")
workflow.add_edge("planning_agent_step1_req", "planning_agent_step2_plan")
workflow.add_edge("planning_agent_step2_plan", "planning_agent_step3_arch")
workflow.add_edge("planning_agent_step3_arch", "coding_agent")
workflow.add_edge("coding_agent", END)

app_builder_graph = workflow.compile()
