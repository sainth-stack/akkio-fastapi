import os


def get_runtime_root() -> str:
    env_path = os.getenv("APP_BUILDER_RUNTIME_DIR")
    if env_path:
        return env_path
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(_root, "runtime")


def get_projects_dir() -> str:
    return os.path.join(get_runtime_root(), "projects")


def resolve_project_root(project_name: str) -> str:
    bases = [
        get_projects_dir(),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".runtime", "projects"),
        os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime", "projects"),
    ]
    for base in bases:
        path = os.path.join(base, project_name)
        if os.path.exists(path):
            return path
    return os.path.join(get_projects_dir(), project_name)
