"""
ML Analytics Tools Package
"""

from dotenv import load_dotenv

from .aws_auth import ensure_aws_authenticated, ensure_aws_sso_login
from .data_connector import DataConnector
from .gsheet_connector import GSheet
from .model_manager import ModelManager
from .s3_connector import S3Connector
from .sf_connector import SFConnector
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

__all__ = [
    "DataConnector",
    "ensure_aws_authenticated",
    "ensure_aws_sso_login",
    "execute_sql_scripts",
    "find_project_root",
    "get_credential_value",
    "get_logger",
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
