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
from .utils import get_logger, load_sql_query, log_and_raise_error

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
            if self.authenticator.lower() == "externalbrowser":
                log_and_raise_error(
                    self._logger,
                    "Snowflake externalbrowser authentication is interactive and cannot be used by "
                    "SFConnector (Spark jobs block on the browser SSO handshake). Use key-pair "
                    "(SNOWFLAKE_PRIVATE_KEY/_PATH) or OAuth (SNOWFLAKE_TOKEN) for Spark workloads, "
                    "or use DataConnector for interactive local queries.",
                )
            options["sfAuthenticator"] = self.authenticator

        # Caller-provided options win over resolved defaults.
        options.update({k: v for k, v in self.extra_options.items() if _clean_env_value(v) is not None})
        return options

    def _resolve_query(self, query: str, **kwargs) -> str:
        """Resolve a query string: if it looks like a SQL file path, load it; otherwise return as-is."""
        if query and query.strip().endswith(".sql"):
            loaded = load_sql_query(query.strip(), **kwargs)
            if loaded is None:
                log_and_raise_error(self._logger, f"Could not load SQL file: {query}")
            self._logger.info(f"Loaded SQL from file: {query}")
            return loaded
        return query

    def sql(
        self,
        query: str,
        return_pandas: bool = False,
        save_table: bool = False,
        table: str = None,
        schema: str = None,
        catalog: str = None,
        mode: str = "overwrite",
        **kwargs,
    ):
        """
        Execute a SQL query against Snowflake and return the result.

        Optionally persist the result straight into a Databricks Unity Catalog
        table while pulling the data, by passing ``save_table=True`` along with a
        destination ``table`` (and optionally ``schema`` / ``catalog``).

        Parameters
        ----------
        query : str
            SQL query to execute, or a path to a ``.sql`` file (relative to the
            project root). When a ``.sql`` path is given, its contents are loaded
            automatically.
        return_pandas : bool, optional
            If True, return a pandas DataFrame; otherwise return a Spark
            DataFrame. Defaults to False.
        save_table : bool, optional
            If True, write the result to a Unity Catalog table via
            :meth:`save_to_uc` before returning. Defaults to False.
        table : str, optional
            Destination table name when ``save_table`` is True. May be fully
            qualified (``catalog.schema.table``), in which case ``schema`` /
            ``catalog`` are ignored.
        schema, catalog : str, optional
            Unity Catalog schema and catalog to qualify ``table`` with.
        mode : str, optional
            Spark write mode for the saved table ('overwrite', 'append',
            'ignore', 'error'). Defaults to 'overwrite'.
        **kwargs
            Template variables substituted into the SQL file using ``str.format()``.
        """
        query = self._resolve_query(query, **kwargs)
        spark = self._get_spark()
        try:
            df = spark.read.format(self.source_format).options(**self.spark_options()).option("query", query).load()
        except Exception as e:
            log_and_raise_error(self._logger, f"Error reading from Snowflake: {e}")

        if save_table:
            self.save_to_uc(df, table=table, schema=schema, catalog=catalog, mode=mode)

        if return_pandas:
            return df.toPandas()
        return df

    @staticmethod
    def _qualified_uc_name(table: str, schema: str = None, catalog: str = None) -> str:
        """Build a Unity Catalog table identifier from its parts.

        A ``table`` that already contains dots is treated as fully qualified and
        returned as-is; otherwise ``catalog`` / ``schema`` are prepended when given.
        """
        if "." in table:
            return table
        parts = [part for part in (catalog, schema, table) if part]
        return ".".join(parts)

    def save_to_uc(self, df, table: str, schema: str = None, catalog: str = None, mode: str = "overwrite"):
        """
        Write a Spark DataFrame to a Databricks Unity Catalog table.

        Uses Spark's native ``df.write.saveAsTable(...)`` (a managed UC table),
        not the Snowflake connector.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame to write.
        table : str
            Destination table name. May be fully qualified
            (``catalog.schema.table``), in which case ``schema`` / ``catalog``
            are ignored.
        schema, catalog : str, optional
            Unity Catalog schema and catalog to qualify ``table`` with.
        mode : str, optional
            Spark write mode: 'overwrite', 'append', 'ignore', or 'error'.
            Defaults to 'overwrite'.
        """
        if not table:
            log_and_raise_error(self._logger, "A destination table name is required.")

        full_name = self._qualified_uc_name(table, schema=schema, catalog=catalog)
        try:
            df.write.mode(mode).saveAsTable(full_name)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error writing to Unity Catalog table '{full_name}': {e}")
        self._logger.info(f"Table '{full_name}' written to Unity Catalog (mode={mode}).")
