import os


def get_runtime_root() -> str:
    """
    Returns the root folder for app-builder runtime artifacts.
    Defaults outside the repo to avoid reload interruptions.
    """
    env_path = os.getenv("APP_BUILDER_RUNTIME_DIR")
    if env_path:
        return env_path
    return os.path.join(os.path.expanduser("~"), ".akkio", "app_builder", "runtime")


def get_projects_dir() -> str:
    return os.path.join(get_runtime_root(), "projects")
