import os


def get_runtime_root() -> str:
    env_path = os.getenv("APP_BUILDER_RUNTIME_DIR")
    if env_path:
        return os.path.expanduser(env_path)
    # Outside the FastAPI package tree so `uvicorn main:app --reload` does not
    # restart when generated apps are written to disk.
    return os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime")


def get_projects_dir() -> str:
    return os.path.join(get_runtime_root(), "projects")


def resolve_project_root(project_name: str) -> str:
    _pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bases = [
        os.path.join(_pkg, "runtime", "projects"),  # legacy in-repo (pre-~/.akkio default)
        os.path.join(_pkg, ".runtime", "projects"),
        get_projects_dir(),
    ]
    for base in bases:
        path = os.path.join(base, project_name)
        if os.path.exists(path):
            return path
    return os.path.join(get_projects_dir(), project_name)
