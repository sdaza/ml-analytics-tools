"""
ML Analytics Tools Package
"""

from dotenv import load_dotenv

from .aws_auth import ensure_aws_authenticated, ensure_aws_sso_login
from .data_connector import DataConnector
from .gsheet_connector import GSheet
from .s3_connector import S3Connector
from .sf_connector import SFConnector, get_spark
from .slack_connector import SlackConnector
from .utils import (
    execute_sql_scripts,
    find_project_root,
    get_credential_value,
    get_logger,
    get_sql_files,
    load_sql_query,
    log_and_raise_error,
    resolve_sql_query_paths,
)

# Automatically load .env file when the package is imported
logger = get_logger("ml_analytics")
try:
    project_root = find_project_root(required=False)
    env_file = project_root / ".env" if project_root else None
    if env_file is not None and env_file.exists():
        if load_dotenv(env_file, override=True):
            logger.info(".env file loaded successfully.")
        else:
            logger.warning("Failed to load .env file.")
    else:
        # Expected when installed as a dependency (e.g. on Databricks); stay quiet.
        logger.debug("No .env file found.")
except Exception:
    logger.debug("No .env file loaded.")


# Lazily-loaded public symbols that depend on the optional `[modeling]` extra.
# Importing them eagerly would pull mlflow (and its numpy pins) into every
# `import ml_analytics`, which is exactly what we want to avoid. They are
# resolved on first access via the module-level __getattr__ below (PEP 562).
_OPTIONAL_ATTRS = {
    "ModelManager": ".model_manager",
}


def __getattr__(name):
    target = _OPTIONAL_ATTRS.get(name)
    if target is not None:
        import importlib

        try:
            module = importlib.import_module(target, __name__)
        except ImportError as exc:
            raise ImportError(
                f"{name} requires the optional 'modeling' dependencies "
                f"(mlflow, scikit-learn, catboost, shap, lifelines, seaborn). "
                f"Install them with `pip install ml-analytics-tools[modeling]`."
            ) from exc
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataConnector",
    "ensure_aws_authenticated",
    "ensure_aws_sso_login",
    "execute_sql_scripts",
    "find_project_root",
    "get_credential_value",
    "get_logger",
    "get_spark",
    "get_sql_files",
    "GSheet",
    "load_sql_query",
    "log_and_raise_error",
    "ModelManager",
    "resolve_sql_query_paths",
    "S3Connector",
    "SFConnector",
    "SlackConnector",
]
