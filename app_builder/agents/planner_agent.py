from ..schemas.plan import ProjectPlan

def planner_agent(requirement: str) -> ProjectPlan:
    """
    Generates a fixed 5-8 step plan for a CRUD app.
    """
    return ProjectPlan(steps=[
        "Initialize the project structure with FastAPI and React.",
        "Design the database schema for the entities.",
        "Implement the backend API endpoints for CRUD operations.",
        "Set up the frontend React application.",
        "Create the frontend components for listing and viewing items.",
        "Implement forms for creating and updating items.",
        "Connect the frontend to the backend API.",
        "Verify the application functionality."
    ])
