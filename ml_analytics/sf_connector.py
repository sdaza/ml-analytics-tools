"""
Spark-based Snowflake connector.

This is meant for Spark environments (e.g. Databricks) where you want to read
from and write to Snowflake through ``spark.read.format("snowflake")`` rather
than the pure-Python ``snowflake-connector-python`` used by
:class:`ml_analytics.data_connector.DataConnector`.

PySpark is intentionally NOT a dependency of this package and is imported
lazily, only when a method that actually needs a Spark session is called. This
keeps the rest of the package usable in environments without Spark installed.
"""

from .data_connector import (
    SNOWFLAKE_SPARK_SOURCE_NAME,
    _clean_env_value,
    _get_snowflake_config_value,
    _load_private_key_pem_for_spark,
    _snowflake_secret_scope,
)
from .utils import get_logger, log_and_raise_error

# Cached Spark session shared across SFConnector instances. Populated lazily by
# get_spark(); never created at import time so the package stays importable
# without PySpark.
_spark_ctx = None


def get_spark():
    """
    Get or create a cached Spark session, importing PySpark lazily.

    PySpark is not a dependency of this package; it must be provided by the
    runtime (e.g. Databricks) or installed separately. The active session is
    reused when one exists (the normal case on Databricks), so we attach to the
    cluster session rather than spinning up a local one.
    """
    global _spark_ctx
    if _spark_ctx is not None:
        return _spark_ctx

    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ImportError(
            "SFConnector requires PySpark, which is not a dependency of "
            "ml-analytics-tools. Run it on a Spark runtime (e.g. Databricks) "
            "or install it with `pip install pyspark`."
        ) from exc

    _spark_ctx = SparkSession.getActiveSession() or SparkSession.builder.appName("ml_analytics").getOrCreate()
    return _spark_ctx


def _snowflake_account_url(account: str) -> str:
    """Normalize a Snowflake account identifier into a full host URL."""
    account_url = account.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if account_url.endswith(".snowflakecomputing.com"):
        return account_url
    return f"{account_url}.snowflakecomputing.com"


class SFConnector:
    """
    Connect to Snowflake through Spark and read/write Spark or pandas DataFrames.

    Credentials and connection settings are resolved, for each field, in this
    order: explicit argument -> environment variable -> Databricks secret (when a
    secret scope is configured or can be inferred from the user email).

    Authentication uses key-pair (``SNOWFLAKE_JWT``) when a private key is
    available, otherwise an OAuth token, otherwise a password / authenticator.

    Parameters
    ----------
    account : str, optional
        Snowflake account identifier or full URL (e.g. ``"dr06406.eu-west-1"``
        or ``"dr06406.eu-west-1.snowflakecomputing.com"``). Falls back to
        ``SNOWFLAKE_ACCOUNT`` / ``SNOWFLAKE_URL``.
    user, database, schema, warehouse, role : str, optional
        Standard Snowflake connection settings. Fall back to the matching
        ``SNOWFLAKE_*`` environment variables / secrets.
    password : str, optional
        Password for password authentication (ignored when a private key or
        token is provided).
    authenticator : str, optional
        Snowflake authenticator (e.g. ``"oauth"``, ``"externalbrowser"``).
    private_key, private_key_path, private_key_passphrase : str, optional
        Key-pair authentication material. ``private_key`` may be the PEM string
        itself; ``private_key_path`` a path to a ``.pem`` file.
    secret_scope : str, optional
        Databricks secret scope to read credentials from. When omitted it is
        inferred from ``SNOWFLAKE_SECRET_SCOPE`` / the user email.
    source_format : str, optional
        Spark data source name. Defaults to ``"net.snowflake.spark.snowflake"``.
        Pass ``"snowflake"`` to use the short alias registered on Databricks.
    extra_options : dict, optional
        Additional Snowflake Spark options merged into every read/write
        (e.g. ``{"sfTimezone": "UTC"}``). These override resolved defaults.
    spark : SparkSession, optional
        Existing Spark session to reuse. If omitted, the active session is used,
        or one is created on first use.
    """

    def __init__(
        self,
        *,
        account=None,
        user=None,
        database=None,
        schema=None,
        warehouse=None,
        role=None,
        password=None,
        authenticator=None,
        token=None,
        private_key=None,
        private_key_path=None,
        private_key_passphrase=None,
        secret_scope=None,
        source_format=SNOWFLAKE_SPARK_SOURCE_NAME,
        extra_options=None,
        spark=None,
    ):
        self._logger = get_logger("SF Connector")
        self.source_format = source_format
        self.extra_options = dict(extra_options or {})
        self._spark = spark

        self._secret_scope = _snowflake_secret_scope(secret_scope, user=user)

        self.account = _get_snowflake_config_value(
            "SNOWFLAKE_ACCOUNT",
            explicit=account,
            secret_scope=self._secret_scope,
            aliases=("SNOWFLAKE_URL", "snowflake_account"),
        )
        self.user = _get_snowflake_config_value(
            "SNOWFLAKE_USER",
            explicit=user,
            secret_scope=self._secret_scope,
            aliases=("snowflake_user",),
        )
        self.database = _get_snowflake_config_value(
            "SNOWFLAKE_DATABASE", explicit=database, secret_scope=self._secret_scope
        )
        self.schema = _get_snowflake_config_value("SNOWFLAKE_SCHEMA", explicit=schema, secret_scope=self._secret_scope)
        self.warehouse = _get_snowflake_config_value(
            "SNOWFLAKE_WAREHOUSE", explicit=warehouse, secret_scope=self._secret_scope
        )
        self.role = _get_snowflake_config_value("SNOWFLAKE_ROLE", explicit=role, secret_scope=self._secret_scope)
        self.password = _get_snowflake_config_value(
            "SNOWFLAKE_PASSWORD", explicit=password, secret_scope=self._secret_scope
        )
        self.authenticator = _get_snowflake_config_value(
            "SNOWFLAKE_AUTHENTICATOR", explicit=authenticator, secret_scope=self._secret_scope
        )
        self.token = _get_snowflake_config_value(
            "SNOWFLAKE_TOKEN",
            explicit=token,
            secret_scope=self._secret_scope,
            aliases=("SNOWFLAKE_OAUTH_TOKEN", "SNOWFLAKE_ACCESS_TOKEN"),
        )

        self._private_key = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY",
            explicit=private_key,
            secret_scope=self._secret_scope,
            aliases=("snowflake_key",),
        )
        self._private_key_path = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY_PATH",
            explicit=private_key_path,
            secret_scope=self._secret_scope,
            aliases=("SNOWFLAKE_PRIVATE_KEY_FILE",),
        )
        self._private_key_passphrase = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
            explicit=private_key_passphrase,
            secret_scope=self._secret_scope,
            aliases=("PRIVATE_KEY_PASSPHRASE", "snowflake_key_pass"),
        )

    def _get_spark(self):
        """Return the Spark session: the one passed in, or the cached shared one."""
        if self._spark is not None:
            return self._spark
        self._spark = get_spark()
        return self._spark

    def spark_options(self, include_private_key: bool = True) -> dict[str, str]:
        """Build the option dict for ``spark.read.format(...).options(**opts)``."""
        if not self.account:
            log_and_raise_error(
                self._logger, "A Snowflake account/URL is required (set account=... or SNOWFLAKE_ACCOUNT)."
            )
        if not self.user:
            log_and_raise_error(self._logger, "A Snowflake user is required (set user=... or SNOWFLAKE_USER).")

        options = {
            "sfUrl": _snowflake_account_url(self.account),
            "sfUser": self.user,
            "sfDatabase": self.database,
            "sfSchema": self.schema,
            "sfWarehouse": self.warehouse,
            "sfRole": self.role,
        }
        options = {key: value for key, value in options.items() if _clean_env_value(value) is not None}

        if include_private_key and (self._private_key or self._private_key_path):
            options["pem_private_key"] = _load_private_key_pem_for_spark(
                private_key=self._private_key,
                private_key_path=self._private_key_path,
                passphrase=self._private_key_passphrase,
            )
        elif self.token:
            options["sfAuthenticator"] = self.authenticator or "oauth"
            options["sfToken"] = self.token
        elif self.password:
            options["sfPassword"] = self.password
            if self.authenticator:
                options["sfAuthenticator"] = self.authenticator
        elif self.authenticator:
            options["sfAuthenticator"] = self.authenticator
            if self.authenticator.lower() == "externalbrowser":
                self._logger.warning(
                    "Snowflake externalbrowser authentication is interactive and is not suitable for "
                    "Databricks/Spark jobs. Use key-pair or OAuth for Spark workloads."
                )

        # Caller-provided options win over resolved defaults.
        options.update({k: v for k, v in self.extra_options.items() if _clean_env_value(v) is not None})
        return options

    def sql(self, query: str, return_pandas: bool = False):
        """
        Execute a SQL query against Snowflake and return the result.

        Parameters
        ----------
        query : str
            SQL query to execute.
        return_pandas : bool, optional
            If True, return a pandas DataFrame; otherwise return a Spark
            DataFrame. Defaults to False.
        """
        spark = self._get_spark()
        try:
            df = spark.read.format(self.source_format).options(**self.spark_options()).option("query", query).load()
        except Exception as e:
            log_and_raise_error(self._logger, f"Error reading from Snowflake: {e}")

        if return_pandas:
            return df.toPandas()
        return df

    def save_table(self, df, table: str, mode: str = "overwrite", column_mapping: str = "name"):
        """
        Write a Spark DataFrame to a Snowflake table.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame to write.
        table : str
            Destination table name (``sfDatabase`` / ``sfSchema`` from the
            connector are used unless the name is fully qualified).
        mode : str, optional
            Spark write mode: 'overwrite', 'append', 'ignore', or 'error'.
            Defaults to 'overwrite'.
        column_mapping : str, optional
            Snowflake ``column_mapping`` option ('name' or 'order').
            Defaults to 'name' so columns are matched by name.
        """
        if not table:
            log_and_raise_error(self._logger, "A destination table name is required.")

        options = self.spark_options()
        options["dbtable"] = table
        options["column_mapping"] = column_mapping
        try:
            df.write.format(self.source_format).options(**options).mode(mode).save()
        except Exception as e:
            log_and_raise_error(self._logger, f"Error writing to Snowflake table '{table}': {e}")
        self._logger.info(f"Table '{table}' written successfully (mode={mode}).")
