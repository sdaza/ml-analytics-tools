"""
ML Analytics Tools Package
"""

from dotenv import load_dotenv

from .aws_auth import ensure_aws_authenticated, ensure_aws_sso_login
from .data_connector import DataConnector
from .gsheet_connector import GSheet
from .model_manager import ModelManager
from .s3_connector import S3Connector
from .slack_connector import SlackConnector
from .utils import (
    execute_sql_scripts,
    find_project_root,
    get_credential_value,
    get_logger,
    get_sql_files,
    load_sql_query,
    log_and_raise_error,
)

# Automatically load .env file when the package is imported
logger = get_logger("ml_analytics")
try:
    project_root = find_project_root()
    env_file = project_root / ".env"
    if env_file.exists():
        if load_dotenv(env_file, override=True):
            logger.info(".env file loaded successfully.")
        else:
            logger.info("Failed to load .env file.")
    else:
        logger.info("No .env file present in project root.")
except Exception:
    logger.info("No .env file loaded.")

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
    "S3Connector",
    "SlackConnector",
]
