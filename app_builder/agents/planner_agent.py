from ..schemas.plan import ProjectPlan

def planner_agent(requirement: str) -> ProjectPlan:
    """
    Returns a dynamic project plan based on the requirement.
    """
    from .architecture_agent import infer_entities_from_requirement
    entities = infer_entities_from_requirement(requirement, "")
    
    steps = [
        "Initialize project structure and basic dependencies",
        "Setup database connection and basic models"
    ]
    
    for entity in entities:
        steps.append(f"Implement backend services and API for {entity}")
    
    steps.extend([
        "Develop core frontend user interface components",
        "Integrate frontend with backend APIs and perform testing"
    ])
    
    return ProjectPlan(steps=steps)
