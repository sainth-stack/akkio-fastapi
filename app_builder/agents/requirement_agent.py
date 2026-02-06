from ..schemas.requirements import UserRequirement

def requirement_agent(requirement: UserRequirement) -> str:
    """
    Trims whitespace and normalizes the requirement to a single sentence.
    """
    if not requirement.description:
        return ""
    cleaned = requirement.description.strip()
    return " ".join(cleaned.split())
