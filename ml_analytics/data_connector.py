"""
Generic utility functions for data processing and database connection.
"""

import logging
import os
import re
import threading
import time
from pathlib import Path

import boto3
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import redshift_connector

from .s3_connector import S3Connector
from .utils import (
    _split_sql_statements,
    get_credential_value,
    get_logger,
    load_sql_query,
    log_and_raise_error,
)

SNOWFLAKE_SPARK_SOURCE_NAME = "net.snowflake.spark.snowflake"


def _clean_env_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _snowflake_secret_scope(scope: str = None, user: str = None) -> str | None:
    configured_scope = _clean_env_value(
        scope
        or os.getenv("SNOWFLAKE_SECRET_SCOPE")
        or os.getenv("ML_ANALYTICS_SNOWFLAKE_SECRET_SCOPE")
        or os.getenv("DATABRICKS_SECRET_SCOPE")
    )
    if configured_scope is not None:
        return configured_scope

    user = _clean_env_value(user or os.getenv("SNOWFLAKE_USER"))
    if user and "@" in user:
        return f"user-{user}"

    return None


def _get_databricks_dbutils():
    try:
        import builtins

        dbutils = getattr(builtins, "dbutils", None)
        if dbutils is not None:
            return dbutils
    except Exception:
        pass

    try:
        import __main__

        dbutils = getattr(__main__, "dbutils", None)
        if dbutils is not None:
            return dbutils
    except Exception:
        pass

    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            return shell.user_ns.get("dbutils")
    except Exception:
        pass

    return None


def _get_databricks_secret(key: str, scope: str = None):
    scope = _snowflake_secret_scope(scope)
    if not scope or not key:
        return None

    dbutils = _get_databricks_dbutils()
    if dbutils is None:
        return None

    try:
        return _clean_env_value(dbutils.secrets.get(scope=scope, key=key))
    except Exception:
        return None


def _get_snowflake_config_value(name: str, explicit=None, secret_scope: str = None, aliases: tuple[str, ...] = ()):
    explicit = _clean_env_value(explicit)
    if explicit is not None:
        return explicit

    candidate_names = (name, *aliases)
    for candidate in candidate_names:
        value = _clean_env_value(os.getenv(candidate))
        if value is not None:
            return value

    secret_scope = _snowflake_secret_scope(secret_scope)
    if not secret_scope:
        return None

    configured_secret_key = _clean_env_value(os.getenv(f"{name}_SECRET_KEY"))
    secret_keys = []
    for key in (configured_secret_key, *candidate_names):
        if key and key not in secret_keys:
            secret_keys.append(key)

    for secret_key in secret_keys:
        value = _get_databricks_secret(secret_key, scope=secret_scope)
        if value is not None:
            return value

    return None


def _read_private_key_pem(private_key: str | bytes = None, private_key_path: str = None) -> bytes | None:
    """Read Snowflake private key material from an env value or a local file path."""
    if private_key:
        pem_bytes = private_key if isinstance(private_key, bytes) else private_key.encode()
        if b"\\n" in pem_bytes and b"\n" not in pem_bytes:
            pem_bytes = pem_bytes.replace(b"\\n", b"\n")
        return pem_bytes

    if private_key_path:
        return Path(private_key_path).expanduser().read_bytes()

    return None


def _load_private_key_der(
    private_key: str | bytes = None,
    private_key_path: str = None,
    passphrase: str = None,
) -> bytes | None:
    """Load a Snowflake key-pair private key as DER bytes for the Python connector."""
    pem_bytes = _read_private_key_pem(private_key=private_key, private_key_path=private_key_path)
    if pem_bytes is None:
        return None

    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(
        pem_bytes,
        password=passphrase.encode() if passphrase else None,
        backend=default_backend(),
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _load_private_key_pem_for_spark(
    private_key: str | bytes = None,
    private_key_path: str = None,
    passphrase: str = None,
) -> str | None:
    """Load a Snowflake key-pair private key in the format expected by the Spark connector."""
    pem_bytes = _read_private_key_pem(private_key=private_key, private_key_path=private_key_path)
    if pem_bytes is None:
        return None

    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(
        pem_bytes,
        password=passphrase.encode() if passphrase else None,
        backend=default_backend(),
    )
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return re.sub(r"-*(BEGIN|END) PRIVATE KEY-*\n", "", pem).replace("\n", "")


class DataConnector:
    def __init__(
        self,
        *,
        database=None,
        user=None,
        password=None,
        host=None,
        port=None,
        s3_bucket=None,
        timeout=240,
        engine=None,
        account=None,
        warehouse=None,
        schema=None,
        role=None,
        authenticator=None,
        private_key=None,
        private_key_path=None,
        private_key_passphrase=None,
        secret_scope=None,
    ):
        """
        Initialize a DataConnector instance.
        Connection is established lazily and will time out after a period of inactivity.

        By default, this connector uses Redshift to preserve existing behavior.
        Pass ``engine="snowflake"`` or set ``ML_ANALYTICS_DB_ENGINE=snowflake``
        to use Snowflake connection settings from ``SNOWFLAKE_*`` environment variables.
        """
        self._logger = get_logger("Data Connector")
        self.engine = self._resolve_engine(engine)
        self._snowflake_private_key = private_key
        self._snowflake_private_key_path = private_key_path
        self._snowflake_private_key_passphrase = private_key_passphrase
        self._snowflake_secret_scope = _snowflake_secret_scope(secret_scope, user=user)

        if self.engine == "snowflake":
            self._db_params = self._build_snowflake_params(
                database=database,
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                schema=schema,
                role=role,
                authenticator=authenticator,
                private_key=private_key,
                private_key_path=private_key_path,
                private_key_passphrase=private_key_passphrase,
                secret_scope=self._snowflake_secret_scope,
            )
        else:
            self._db_params = {
                "database": database or get_credential_value("BI_REDSHIFT_DB"),
                "user": user or get_credential_value("BI_REDSHIFT_USER"),
                "password": password or get_credential_value("BI_REDSHIFT_PASSWORD"),
                "host": host or get_credential_value("BI_REDSHIFT_HOST"),
                "port": port or get_credential_value("BI_REDSHIFT_PORT"),
            }
        self._s3_bucket = s3_bucket or os.getenv("ML_ANALYTICS_S3_BUCKET")
        self.connection = None
        self.cursor = None
        self.s3 = None

        # Cache for S3 connectors by bucket name
        self._s3_connectors: dict[str, S3Connector] = {}

        # Timeout attributes
        self._timeout = timeout
        self._last_activity = None
        self._idle_timer = None
        # Use a re-entrant lock because some methods call others that also
        # acquire the lock (e.g. _close_if_idle -> close_redshift_connection ->
        # _cancel_idle_timer). An RLock avoids deadlocks in that scenario.
        self._lock = threading.RLock()

    @staticmethod
    def _resolve_engine(engine: str = None) -> str:
        selected = (
            engine
            or os.getenv("ML_ANALYTICS_DB_ENGINE")
            or os.getenv("ML_ANALYTICS_DATABASE_ENGINE")
            or "redshift"
        )
        normalized = selected.strip().lower().replace("-", "_")
        aliases = {
            "redshift": "redshift",
            "rs": "redshift",
            "snowflake": "snowflake",
            "snow_flake": "snowflake",
            "sf": "snowflake",
        }
        if normalized not in aliases:
            raise ValueError("Unsupported DataConnector engine. Use 'redshift' or 'snowflake'.")
        return aliases[normalized]

    @staticmethod
    def _import_snowflake_connector():
        try:
            import snowflake.connector as snowflake_connector
        except ImportError as exc:
            raise ImportError(
                "Snowflake support requires the optional package "
                "'snowflake-connector-python[pandas]'. Install it before using "
                "DataConnector(engine='snowflake')."
            ) from exc
        return snowflake_connector

    @staticmethod
    def _build_snowflake_params(
        *,
        database=None,
        user=None,
        password=None,
        account=None,
        warehouse=None,
        schema=None,
        role=None,
        authenticator=None,
        private_key=None,
        private_key_path=None,
        private_key_passphrase=None,
        secret_scope=None,
    ) -> dict:
        secret_scope = _snowflake_secret_scope(secret_scope)
        private_key = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY",
            explicit=private_key,
            secret_scope=secret_scope,
            aliases=("snowflake_key",),
        )
        private_key_path = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY_PATH",
            explicit=private_key_path,
            secret_scope=secret_scope,
            aliases=("SNOWFLAKE_PRIVATE_KEY_FILE",),
        )
        private_key_passphrase = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
            explicit=private_key_passphrase,
            secret_scope=secret_scope,
            aliases=("PRIVATE_KEY_PASSPHRASE", "snowflake_key_pass"),
        )
        token = _get_snowflake_config_value(
            "SNOWFLAKE_TOKEN",
            secret_scope=secret_scope,
            aliases=("SNOWFLAKE_OAUTH_TOKEN", "SNOWFLAKE_ACCESS_TOKEN"),
        )
        resolved_authenticator = _get_snowflake_config_value(
            "SNOWFLAKE_AUTHENTICATOR",
            explicit=authenticator,
            secret_scope=secret_scope,
        )

        params = {
            "user": _get_snowflake_config_value(
                "SNOWFLAKE_USER",
                explicit=user,
                secret_scope=secret_scope,
                aliases=("snowflake_user",),
            ),
            "password": _get_snowflake_config_value(
                "SNOWFLAKE_PASSWORD", explicit=password, secret_scope=secret_scope
            ),
            "account": _get_snowflake_config_value(
                "SNOWFLAKE_ACCOUNT", explicit=account, secret_scope=secret_scope
            ),
            "warehouse": _get_snowflake_config_value(
                "SNOWFLAKE_WAREHOUSE", explicit=warehouse, secret_scope=secret_scope
            ),
            "database": _get_snowflake_config_value(
                "SNOWFLAKE_DATABASE", explicit=database, secret_scope=secret_scope
            ),
            "schema": _get_snowflake_config_value("SNOWFLAKE_SCHEMA", explicit=schema, secret_scope=secret_scope),
            "role": _get_snowflake_config_value("SNOWFLAKE_ROLE", explicit=role, secret_scope=secret_scope),
            "autocommit": True,
        }

        if private_key or private_key_path:
            params["authenticator"] = "SNOWFLAKE_JWT"
            params.pop("password", None)
            params["private_key"] = _load_private_key_der(
                private_key=private_key,
                private_key_path=private_key_path,
                passphrase=private_key_passphrase,
            )
        elif token:
            params["authenticator"] = resolved_authenticator or "oauth"
            params["token"] = token
            params.pop("password", None)
        else:
            params["authenticator"] = resolved_authenticator

        return {key: value for key, value in params.items() if _clean_env_value(value) is not None}

    @staticmethod
    def _snowflake_account_url(account: str) -> str:
        account_url = account.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
        if account_url.endswith(".snowflakecomputing.com"):
            return account_url
        return f"{account_url}.snowflakecomputing.com"

    def snowflake_spark_options(self, include_private_key: bool = True) -> dict[str, str]:
        """Return options for ``spark.read.format(SNOWFLAKE_SPARK_SOURCE_NAME).options(...)``."""
        if self.engine != "snowflake":
            log_and_raise_error(
                self._logger,
                "snowflake_spark_options() is only available when DataConnector uses engine='snowflake'.",
                NotImplementedError,
            )

        account = self._db_params.get("account")
        if not account:
            log_and_raise_error(self._logger, "SNOWFLAKE_ACCOUNT is required to build Spark options.")

        options = {
            "sfURL": self._snowflake_account_url(account),
            "sfUser": self._db_params.get("user"),
            "sfDatabase": self._db_params.get("database"),
            "sfSchema": self._db_params.get("schema"),
            "sfWarehouse": self._db_params.get("warehouse"),
            "sfRole": self._db_params.get("role"),
        }
        options = {key: value for key, value in options.items() if _clean_env_value(value) is not None}

        private_key = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY",
            explicit=self._snowflake_private_key,
            secret_scope=self._snowflake_secret_scope,
            aliases=("snowflake_key",),
        )
        private_key_path = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY_PATH",
            explicit=self._snowflake_private_key_path,
            secret_scope=self._snowflake_secret_scope,
            aliases=("SNOWFLAKE_PRIVATE_KEY_FILE",),
        )
        private_key_passphrase = _get_snowflake_config_value(
            "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
            explicit=self._snowflake_private_key_passphrase,
            secret_scope=self._snowflake_secret_scope,
            aliases=("PRIVATE_KEY_PASSPHRASE", "snowflake_key_pass"),
        )
        token = _get_snowflake_config_value(
            "SNOWFLAKE_TOKEN",
            secret_scope=self._snowflake_secret_scope,
            aliases=("SNOWFLAKE_OAUTH_TOKEN", "SNOWFLAKE_ACCESS_TOKEN"),
        )

        if include_private_key and (private_key or private_key_path):
            options["pem_private_key"] = _load_private_key_pem_for_spark(
                private_key=private_key,
                private_key_path=private_key_path,
                passphrase=private_key_passphrase,
            )
        elif token:
            options["sfAuthenticator"] = self._db_params.get("authenticator", "oauth")
            options["sfToken"] = token
        else:
            authenticator = self._db_params.get("authenticator")
            if authenticator:
                options["sfAuthenticator"] = authenticator
                if authenticator.lower() == "externalbrowser":
                    self._logger.warning(
                        "Snowflake externalbrowser authentication is interactive and is not suitable for "
                        "Databricks/Spark jobs. Use key-pair or OAuth for Spark workloads."
                    )

        return options

    get_snowflake_spark_options = snowflake_spark_options

    def _is_connection_open(self) -> bool:
        """Return True when the underlying connection is open/usable.

        Prefer checking redshift_connector-specific APIs when available for reliable
        results. Fallback to a set of widely used attributes and lastly to a
        lightweight cursor probe.
        """
        # Quick negative checks
        if not getattr(self, "connection", None):
            return False

        conn = self.connection
        try:
            # If this is a redshift_connector connection prefer its public API
            try:
                module_name = conn.__class__.__module__
            except Exception:
                module_name = ""

            if module_name.startswith("redshift_connector"):
                # If the connection object became None while we inspected it
                if conn is None:
                    return False

                # redshift_connector exposes is_closed (callable or property)
                if hasattr(conn, "is_closed"):
                    val = conn.is_closed
                    if callable(val):
                        val = val()
                    return not bool(val)

                # Some versions might still have a 'closed' like psycopg2 (int)
                if hasattr(conn, "closed"):
                    val = conn.closed
                    if callable(val):
                        val = val()
                    if isinstance(val, int):
                        return val == 0
                    return not bool(val)

                # Last resort for redshift_connector: probe via cursor
                cur = getattr(self, "cursor", None)
                if cur is None:
                    return False
                try:
                    # A small probe that doesn't change transaction state
                    cur.execute("SELECT 1")
                    # Some cursor implementations keep a reference to the
                    # connection even after close; check connection object
                    # again for early returns.
                    if getattr(self, "connection", None) is None:
                        return False
                    return True
                except Exception:
                    return False

            # Generic checks for other DB drivers
            # psycopg2: .closed is an int (0=open, non-zero=closed)
            if hasattr(conn, "closed"):
                val = conn.closed
                if callable(val):
                    val = val()
                if isinstance(val, int):
                    return val == 0
                return not bool(val)

            # common boolean attributes
            if hasattr(conn, "is_closed"):
                val = conn.is_closed
                if callable(val):
                    val = val()
                return not bool(val)

            if hasattr(conn, "open"):
                val = conn.open
                if callable(val):
                    val = val()
                return bool(val)

            if hasattr(conn, "is_valid"):
                val = conn.is_valid
                if callable(val):
                    val = val()
                return bool(val)

            # Last-resort probe: try a very cheap operation with the cursor.
            cur = getattr(self, "cursor", None)
            if cur is None:
                return False
            try:
                cur.execute("SELECT 1")
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _start_idle_timer(self):
        """Starts or resets the idle connection timer."""
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(self._timeout, self._close_if_idle)
            self._idle_timer.daemon = True
            self._idle_timer.start()
            self._last_activity = time.time()

    def _cancel_idle_timer(self):
        """Cancels the idle timer if it's active."""
        with self._lock:
            if self._idle_timer:
                try:
                    self._idle_timer.cancel()
                except Exception:
                    self._logger.debug("Failed to cancel idle timer", exc_info=True)
                self._idle_timer = None

    def _close_if_idle(self):
        """Callback for the timer to close the connection if it's idle."""
        with self._lock:
            last_activity_snapshot = self._last_activity
            connection_open = self._is_connection_open()
            if connection_open:
                idle_duration = time.time() - last_activity_snapshot
                if idle_duration >= self._timeout:
                    self._logger.info(f"Connection has been idle for {idle_duration:.2f} seconds. Closing.")
                    self.close_redshift_connection()

    def _mark_activity(self):
        with self._lock:
            self._last_activity = time.time()
        self._start_idle_timer()

    def connect(self):
        """Establish a connection to the database if not already connected."""
        with self._lock:
            if self._is_connection_open():
                self._start_idle_timer()  # Reset timer on use
                return

            try:
                if self.engine == "snowflake":
                    snowflake_connector = self._import_snowflake_connector()
                    self.connection = snowflake_connector.connect(**self._db_params)
                else:
                    self.connection = redshift_connector.connect(**self._db_params)
                    self.connection.autocommit = True
                self.cursor = self.connection.cursor()
                self._start_idle_timer()  # Start idle timer on new connection

                # Initialize the default S3 connector only when a default bucket is configured.
                if self._s3_bucket:
                    self.s3 = self._get_s3_for_bucket(self._s3_bucket)

            except Exception as e:
                log_and_raise_error(self._logger, f"Failed to connect to {self.engine.title()}: {e}")

    def close_redshift_connection(self):
        """Close the database connection if it is open."""
        # Acquire lock to synchronize with the idle timer closure
        with self._lock:
            self._cancel_idle_timer()
            try:
                if self._is_connection_open():
                    try:
                        # Some redshift_connector connection objects set internal flags
                        # when closed; call close and then clear references.
                        self.connection.close()
                    except Exception:
                        # some drivers raise on close if already closed; ignore
                        pass
                    finally:
                        # Always clear connection and cursor references to avoid
                        # future probes erroneously believing a connection exists.
                        try:
                            self.cursor = None
                        except Exception:
                            pass
                        try:
                            self.connection = None
                        except Exception:
                            pass
            except Exception as e:
                self._logger.warning(f"Error closing connection: {e}")

    def close_connection(self):
        """Close the database connection if it is open."""
        self.close_redshift_connection()

    def __del__(self):
        """Ensure the database connection is closed when the instance is garbage-collected."""
        try:
            self.close_redshift_connection()
        except Exception:
            pass

    def __enter__(self):
        """Enter the context manager, establishing a connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, closing the connection."""
        self.close_redshift_connection()

    def _ensure_connected(self):
        """Ensure there is an active connection, creating one if necessary."""
        with self._lock:
            if not self._is_connection_open():
                self.connect()
            else:
                # If we are using an existing connection, reset its idle timer
                self._start_idle_timer()

    def _resolve_query(self, query: str, **kwargs) -> str:
        """Resolve a query string: if it looks like a SQL file path, load it; otherwise return as-is."""
        if query and query.strip().endswith(".sql"):
            loaded = load_sql_query(query.strip(), **kwargs)
            if loaded is None:
                log_and_raise_error(self._logger, f"Could not load SQL file: {query}")
            self._logger.info(f"Loaded SQL from file: {query}")
            return loaded
        return query

    def execute_sql(
        self, query: str, fetch_one: bool = False, fetch_all: bool = False, fetch_result: bool = False, **kwargs
    ):
        """
        Execute a SQL query with automatic connection management and activity tracking.

        This method is useful for executing DDL, DML, or queries that don't return
        data (or when you need to fetch results). It automatically:
        - Ensures the connection is active before executing
        - Resets the idle timer to prevent connection timeout
        - Optionally fetches and returns results

        For queries that return datasets (SELECT), prefer using the sql() method
        which returns pandas or polars DataFrames.

        Parameters
        ----------
        query : str
            The SQL query to execute, or a path to a .sql file (relative to project root).
            If a .sql file path is provided, its contents are loaded automatically.
        fetch_one : bool, optional
            If True, fetches and returns a single row result using fetchone().
            Defaults to False.
        fetch_all : bool, optional
            If True, fetches and returns all rows using fetchall().
            Takes precedence over fetch_one if both are True.
            Defaults to False.
        fetch_result : bool, optional
            Deprecated alias for fetch_one. Defaults to False.
        **kwargs
            Template variables to substitute in the SQL file using str.format().

        Returns
        -------
        tuple, list of tuples, or None
            - If fetch_all=True: list of row tuples
            - If fetch_one=True: single row tuple
            - Otherwise: None

        Examples
        --------
        # Execute a DDL command
        dc.execute_sql("CREATE TABLE test (id INT, name VARCHAR(100))")

        # Execute from a SQL file
        dc.execute_sql("queries/create_table.sql")

        # Execute query and fetch single result
        result = dc.execute_sql("SELECT COUNT(*) FROM test", fetch_one=True)
        count = result[0]  # Get the count value

        # Execute query and fetch all results
        results = dc.execute_sql("SELECT * FROM test", fetch_all=True)
        for row in results:
            print(row)
        """
        query = self._resolve_query(query, **kwargs)
        self._ensure_connected()
        self._mark_activity()
        try:
            self.cursor.execute(query)
            if fetch_all:
                return self.cursor.fetchall()
            elif fetch_one or fetch_result:
                return self.cursor.fetchone()
            return None
        finally:
            self._start_idle_timer()

    def _fetch_pandas_dataframe(self) -> pd.DataFrame:
        if self.engine == "snowflake" and hasattr(self.cursor, "fetch_pandas_all"):
            return self.cursor.fetch_pandas_all()
        if hasattr(self.cursor, "fetch_dataframe"):
            return self.cursor.fetch_dataframe()
        if hasattr(self.cursor, "fetch_pandas_all"):
            return self.cursor.fetch_pandas_all()

        rows = self.cursor.fetchall()
        description = getattr(self.cursor, "description", None) or []
        columns = [col[0] for col in description]
        return pd.DataFrame(rows, columns=columns or None)

    def sql(self, query: str = None, format: str = "pandas", **kwargs) -> pl.DataFrame | pd.DataFrame:
        """
        Execute a SQL query against the configured database and return the result.

        Parameters
        ----------
        query : str
            The SQL query to execute, or a path to a .sql file (relative to project root).
        format : str
            Output format: 'pandas' or 'polars'. Defaults to 'pandas'.
        **kwargs
            Template variables to substitute in the SQL file using str.format().
        """
        query = self._resolve_query(query, **kwargs)
        self._ensure_connected()
        self._mark_activity()
        try:
            if format not in ["pandas", "polars"]:
                log_and_raise_error(self._logger, "Invalid format. Use 'pandas' or 'polars'.")

            if format == "pandas":
                self.execute_sql(query)
                tmp = self._fetch_pandas_dataframe()
                self._logger.info("Data fetched successfully")
                return tmp
            elif format == "polars":
                if self.engine == "snowflake":
                    self.execute_sql(query)
                    tmp = pl.from_pandas(self._fetch_pandas_dataframe())
                else:
                    tmp = pl.read_database(query, connection=self.cursor)
                self._logger.info("Data fetched successfully")
                return tmp
        except Exception as e:
            log_and_raise_error(self._logger, f"Error fetching data: {e}")
        finally:
            self._start_idle_timer()  # Reset timer after operation

    def _get_s3_for_bucket(self, bucket: str = None) -> S3Connector:
        """Get an S3Connector for the specified bucket.

        If bucket is None, uses the default bucket. Returns a cached connector
        if available, otherwise creates and caches a new one.

        This avoids mutating a shared S3Connector instance and ensures each
        operation uses the correct bucket.
        """
        target_bucket = self._resolve_s3_bucket(bucket)

        # Check cache first
        if target_bucket in self._s3_connectors:
            return self._s3_connectors[target_bucket]

        # Create new connector and cache it
        s3_logger_instance = logging.getLogger(f"S3 Connector ({target_bucket})")
        s3_logger_instance.setLevel(logging.WARNING)
        connector = S3Connector(bucket=target_bucket, log_level="WARNING")
        self._s3_connectors[target_bucket] = connector

        # Also set self.s3 to the default bucket's connector for backward compatibility
        if target_bucket == self._s3_bucket:
            self.s3 = connector

        return connector

    def _resolve_s3_bucket(self, bucket: str = None) -> str:
        target_bucket = bucket or self._s3_bucket
        if not target_bucket:
            log_and_raise_error(
                self._logger,
                "No S3 bucket configured. Pass s3_bucket=... or set ML_ANALYTICS_S3_BUCKET.",
            )
        return target_bucket.rstrip("/").lstrip("/")

    def create_spectrum_table(
        self,
        table: str,
        schema: str,
        relative_path: str,
        partitions: list[tuple[str, str]] = None,
        force_table_creation: bool = False,
        sync_partitions_on_creation: bool = True,
        return_query: bool = False,
        s3_bucket: str = None,
    ) -> str | None:
        self._ensure_connected()
        self._mark_activity()

        # Get the appropriate S3 connector for this operation
        working_s3 = self._get_s3_for_bucket(s3_bucket)
        working_bucket = self._resolve_s3_bucket(s3_bucket)

        if relative_path:
            relative_path = relative_path.strip().lstrip("/")

        s3_path = working_s3.get_path(relative_path=relative_path)

        # Check if files exist in S3 before proceeding
        try:
            files_in_path = working_s3.list_files(prefix=relative_path, bucket=working_bucket)
            if not files_in_path:
                log_and_raise_error(
                    self._logger,
                    f"No files found at S3 path: {s3_path}. Please verify the path exists and contains data files.",
                    FileNotFoundError,
                )
            self._logger.info(f"Found {len(files_in_path)} file(s) in {s3_path}")
        except FileNotFoundError:
            raise
        except Exception as e:
            self._logger.warning(f"Could not verify files in S3 path {s3_path}: {e}")

        try:
            full_table_name = f"{schema}.{table}"
            table_exists = self._spectrum_table_exists(table_name=table, schema_name=schema)
            create_table_query = None
            partition_search_depth = len(partitions) if partitions else 2

            if table_exists:
                if force_table_creation:
                    self._logger.info(
                        f"Table {full_table_name} exists and force_table_creation is True. Dropping table."
                    )
                    drop_query = f"DROP TABLE IF EXISTS {full_table_name};"
                    try:
                        self.execute_sql(drop_query)
                    except Exception as e:
                        log_and_raise_error(self._logger, f"Error dropping table {full_table_name}: {e}")
                else:
                    self._logger.info(
                        f"Table {full_table_name} already exists and force_table_creation is False. Skipping creation."
                    )

            if self._spectrum_table_exists(table_name=table, schema_name=schema) is False:
                try:
                    if not relative_path.endswith("/"):
                        relative_path_for_listing = relative_path + "/"
                    else:
                        relative_path_for_listing = relative_path

                    original_relative_path = relative_path

                    parquet_file_key = None
                    candidates = [
                        relative_path_for_listing,
                        relative_path_for_listing.lstrip("/"),
                        f"{working_bucket}/{relative_path_for_listing}".lstrip("/"),
                        original_relative_path,
                        original_relative_path.rstrip("/"),
                    ]

                    for prefix_candidate in candidates:
                        try:
                            files_prefix = working_s3.list_files_in_prefix(prefix=prefix_candidate) or []
                        except Exception:
                            files_prefix = []
                        for file_key in files_prefix:
                            if file_key.lower().endswith(".parquet"):
                                parquet_file_key = file_key
                                self._logger.debug(
                                    "Parquet discovered (prefix-list): %s (tried %s)",
                                    parquet_file_key,
                                    prefix_candidate,
                                )
                                break
                        if parquet_file_key:
                            break

                    if parquet_file_key is None:
                        for prefix_candidate in candidates:
                            try:
                                potential_partition_dirs = (
                                    working_s3.list_partition_paths(
                                        prefix=prefix_candidate, depth=partition_search_depth
                                    )
                                    or []
                                )
                            except Exception:
                                potential_partition_dirs = []
                            for partition_dir in potential_partition_dirs:
                                try:
                                    files_in_partition = working_s3.list_files_in_prefix(prefix=partition_dir) or []
                                except Exception:
                                    files_in_partition = []
                                for file_key_in_part in files_in_partition:
                                    if file_key_in_part.lower().endswith(".parquet"):
                                        parquet_file_key = file_key_in_part
                                        self._logger.debug(
                                            "Parquet discovered in partition: %s (tried %s)",
                                            parquet_file_key,
                                            partition_dir,
                                        )
                                        break
                                if parquet_file_key:
                                    break
                            if parquet_file_key:
                                break

                    if parquet_file_key is None:
                        for prefix_candidate in candidates:
                            try:
                                files_candidate = working_s3.list_files(prefix=prefix_candidate) or []
                            except Exception:
                                files_candidate = []
                            for file_key in files_candidate:
                                if file_key.lower().endswith(".parquet"):
                                    parquet_file_key = file_key
                                    self._logger.debug(
                                        "Parquet discovered recursively: %s (tried %s)",
                                        parquet_file_key,
                                        prefix_candidate,
                                    )
                                    break
                            if parquet_file_key:
                                break

                    if not parquet_file_key:
                        log_and_raise_error(
                            self._logger,
                            f"No .parquet file found in '{relative_path}' or its immediate subdirectories "
                            "for schema inference.",
                            FileNotFoundError,
                        )

                    parquet_key_for_path = parquet_file_key
                    if parquet_key_for_path.startswith(f"{working_bucket}/"):
                        parquet_key_for_path = parquet_key_for_path[len(working_bucket) + 1 :]
                    parquet_s3_path = working_s3.get_path(parquet_key_for_path)
                    df_schema = pq.read_schema(parquet_s3_path)
                    columns_with_types = self._convert_pyarrow_schema_to_sql(df_schema, partitions)
                    column_names = [col for col, _ in columns_with_types]
                    self._check_redshift_reserved_words(column_names)

                except Exception as e:
                    log_and_raise_error(self._logger, f"Failed to infer schema or find Parquet file: {e}")

                create_table_query = f"CREATE EXTERNAL TABLE {full_table_name} (\n"
                create_table_query += ",\n".join([f"  {col} {dtype}" for col, dtype in columns_with_types])
                create_table_query += "\n)\n"

                if partitions:
                    partition_cols_str = ", ".join([f"{p_name} {p_type}" for p_name, p_type in partitions])
                    create_table_query += f"PARTITIONED BY ({partition_cols_str})\n"

                create_table_query += "STORED AS PARQUET\n"
                create_table_query += f"LOCATION '{s3_path}';"

                try:
                    self.execute_sql(create_table_query)
                    self._logger.info(f"Successfully created Spectrum table: {full_table_name}")
                except Exception as e:
                    log_and_raise_error(self._logger, f"Error creating Spectrum table {full_table_name}: {e}")

            if self._spectrum_table_exists(table_name=table, schema_name=schema):
                if partitions and sync_partitions_on_creation:
                    partition_column_names = [p_name for p_name, p_type in partitions]
                    self.sync_spectrum_partitions(
                        table=table,
                        schema=schema,
                        relative_path=relative_path,
                        partitions_columns=partition_column_names,
                        s3_bucket=working_bucket,
                    )

            if create_table_query and return_query:
                return create_table_query
        finally:
            self._start_idle_timer()

    def _spectrum_table_exists(self, table_name: str, schema_name: str) -> bool:
        self._ensure_connected()
        self._mark_activity()
        try:
            check_sql = f"""
                SELECT EXISTS (
                SELECT 1
                FROM svv_external_tables
                WHERE schemaname = '{schema_name}'
                AND tablename = '{table_name}'
                )
            """
            result = self.execute_sql(check_sql, fetch_one=True)
            exists = result[0]
            return exists
        finally:
            self._start_idle_timer()

    def sync_spectrum_data(
        self,
        table: str,
        schema: str,
        relative_path: str,
        partition_values: dict[str, str],
        s3_bucket: str = None,
    ):
        self._ensure_connected()
        self._mark_activity()
        try:
            # Get the appropriate S3 connector for this operation
            working_s3 = self._get_s3_for_bucket(s3_bucket)
            working_bucket = self._resolve_s3_bucket(s3_bucket)

            fully_qualified_table = f"{schema}.{table}"
            # Normalize the base relative path and build the partition suffix
            if relative_path is None:
                relative_path = ""
            if not relative_path.endswith("/"):
                relative_path_for_listing = relative_path + "/"
            else:
                relative_path_for_listing = relative_path

            partition_suffix = "".join(f"{col}={val}/" for col, val in partition_values.items())

            candidates = [
                relative_path_for_listing + partition_suffix,
                relative_path_for_listing.lstrip("/") + partition_suffix,
                f"{working_bucket}/{relative_path_for_listing}{partition_suffix}".lstrip("/"),
            ]

            found_files = []
            for candidate in candidates:
                try:
                    files = working_s3.list_files(prefix=candidate) or []
                except Exception:
                    files = []
                if files:
                    found_files = files
                    self._logger.debug("Found %d files for partition candidate '%s'", len(files), candidate)
                    break

            if not found_files:
                # No files found for the requested partition; raise a clear error
                log_and_raise_error(
                    self._logger,
                    f"No files found in S3 for partition values {partition_values} under '{relative_path}'.",
                    FileNotFoundError,
                )

            full_s3_location_for_partition = working_s3.get_path(relative_path)
            if not full_s3_location_for_partition.endswith("/"):
                full_s3_location_for_partition += "/"
            for col, val in partition_values.items():
                full_s3_location_for_partition += f"{col}={val}/"
            if not full_s3_location_for_partition.endswith("/"):
                full_s3_location_for_partition += "/"

            partition_spec = ", ".join(f"{col}='{val}'" for col, val in partition_values.items())
            query = f"""
                ALTER TABLE {fully_qualified_table}
                ADD IF NOT EXISTS PARTITION({partition_spec})
                LOCATION '{full_s3_location_for_partition}'
            """
            self.execute_sql(query)
            self._logger.info(f"Successful sync for table {fully_qualified_table}.")
        except Exception as e:
            log_and_raise_error(
                self._logger, f"Error syncing partition {partition_values} for table {fully_qualified_table}: {e}"
            )
        finally:
            self._start_idle_timer()

    def sync_spectrum_partitions(
        self, table: str, schema: str, relative_path: str, partitions_columns: list[str], s3_bucket: str = None
    ):
        self._ensure_connected()
        self._mark_activity()
        try:
            # Get the appropriate S3 connector for this operation
            working_s3 = self._get_s3_for_bucket(s3_bucket)

            fully_qualified_table = f"{schema}.{table}"
            current_relative_path = relative_path
            if not current_relative_path.endswith("/"):
                current_relative_path += "/"

            base_s3_prefix = current_relative_path
            discovered_partitions_s3_paths = working_s3.list_partition_paths(base_s3_prefix, len(partitions_columns))

            if not discovered_partitions_s3_paths:
                self._logger.info(
                    "No partition paths found under s3://%s/%s matching depth %d.",
                    working_s3.bucket,
                    base_s3_prefix,
                    len(partitions_columns),
                )
                return

            alter_queries = []
            for s3_partition_path in discovered_partitions_s3_paths:
                if not s3_partition_path.startswith(base_s3_prefix):
                    self._logger.warning(
                        "Skipping path %s, does not start with base prefix %s.",
                        s3_partition_path,
                        base_s3_prefix,
                    )
                    continue

                partition_key_value_str = s3_partition_path[len(base_s3_prefix) :].strip("/")
                parts = partition_key_value_str.split("/")

                if len(parts) != len(partitions_columns):
                    self._logger.warning(
                        "Skipping path %s, parts count (%d) != partition columns count (%d).",
                        s3_partition_path,
                        len(parts),
                        len(partitions_columns),
                    )
                    continue
                partition_spec_parts = []
                valid_partition = True
                for i, part_col_name in enumerate(partitions_columns):
                    key_value = parts[i].split("=", 1)
                    if len(key_value) == 2 and key_value[0] == part_col_name:
                        partition_value_str = key_value[1].replace("'", "''")
                        partition_spec_parts.append(f"{part_col_name}='{partition_value_str}'")
                    else:
                        valid_partition = False
                        break
                if not valid_partition:
                    continue
                partition_spec = ", ".join(partition_spec_parts)
                full_s3_location_for_partition = working_s3.get_path(s3_partition_path)
                if not full_s3_location_for_partition.endswith("/"):
                    full_s3_location_for_partition += "/"
                query = f"""
                ALTER TABLE {fully_qualified_table}
                ADD IF NOT EXISTS PARTITION({partition_spec})
                LOCATION '{full_s3_location_for_partition}';
                """
                alter_queries.append((query, partition_spec))

            if not alter_queries:
                self._logger.info(f"No new partitions to add for {fully_qualified_table}.")
                return

            for query, spec_for_log in alter_queries:
                try:
                    self.execute_sql(query)
                except Exception as e:
                    if "already exists" in str(e).lower():
                        self._logger.info(
                            f"Partition ({spec_for_log}) likely already exists for {fully_qualified_table}."
                        )
                    else:
                        self._logger.error(
                            f"Error adding partition for {fully_qualified_table} with spec ({spec_for_log}): {e}"
                        )
                    break
            else:
                self._logger.info(f"Finished syncing partitions for {fully_qualified_table}.")
        finally:
            self._start_idle_timer()

    def copy_table(self, source_table: str, destination_table: str, drop_destination_table: bool = True):
        self._ensure_connected()
        self._mark_activity()
        try:
            if not source_table or not destination_table:
                log_and_raise_error(self._logger, "Source and target table names must be provided.")
            if "." not in source_table or "." not in destination_table:
                log_and_raise_error(
                    self._logger, "Source and target table names must include schema (e.g., 'schema.table')."
                )
            if source_table == destination_table:
                log_and_raise_error(self._logger, "Source and target table names cannot be the same.")

            source_table = source_table.strip()
            destination_table = destination_table.strip()

            if drop_destination_table:
                self.execute_sql(f"DROP TABLE IF EXISTS {destination_table};")
                create_table_query = f"""CREATE TABLE {destination_table} AS SELECT * FROM {source_table}"""
                self.execute_sql(create_table_query)
            else:
                self.execute_sql(f"SELECT * FROM {source_table} LIMIT 0;")
                insert_query = f"INSERT INTO {destination_table} SELECT * FROM {source_table};"
                self.execute_sql(insert_query)
            self._logger.info(f"Data copied successfully from {source_table} to {destination_table}.")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error during table copy: {e}")
        finally:
            self._start_idle_timer()

    def unload_to_s3(
        self,
        query: str,
        relative_path: str,
        file_prefix: str = "data",
        s3_bucket: str = None,
        parallel: bool = True,
        overwrite: bool = True,
        drop_existing_files: bool = False,
        format: str = "PARQUET",
        max_file_size: str = None,
        partition_by: list[str] = None,
    ):
        """
        Execute a Redshift UNLOAD command to export query results directly to S3.

        Parameters
        ----------
        query : str
            The SELECT query to unload. Accepts:
            - A SELECT/WITH query string.
            - A table name (will be wrapped in ``SELECT * FROM``).
            - A path to a ``.sql`` file (relative to project root).
            - A multi-statement SQL string separated by semicolons.

            When multiple statements are provided (via file or inline string),
            all preceding statements are executed first (e.g., CREATE TEMP TABLE,
            INSERT, etc.) and the **last** statement is used as the UNLOAD query.
        relative_path : str
            The relative path within the S3 bucket where files will be saved (e.g., 'my-data/output/').
        file_prefix : str, optional
            Prefix for the output files. Defaults to 'data'.
        s3_bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.
        parallel : bool, optional
            If True, uses PARALLEL ON for faster unload with multiple files (one per slice).
            If False, uses PARALLEL OFF for a single output file. Defaults to True.
            Note: When using PARTITION BY, Redshift may create multiple files per partition
            even with PARALLEL OFF, as it parallelizes within each partition.
        overwrite : bool, optional
            If True, adds ALLOWOVERWRITE to replace existing files. Defaults to True.
        drop_existing_files : bool, optional
            If True, deletes all existing files matching the prefix before UNLOAD.
            This ensures a clean output directory. Defaults to False.
            Note: This happens before UNLOAD, regardless of the overwrite setting.
        format : str, optional
            Output format: 'PARQUET', 'CSV', or 'JSON'. Defaults to 'PARQUET'.
        max_file_size : str, optional
            Maximum size per file (e.g., '100 MB', '1 GB'). Only valid with PARALLEL ON.
            Causes Redshift to split files larger than this size. Use this to control file sizes
            when you have large datasets.
        partition_by : list[str], optional
            List of column names to partition by (Parquet only). Example: ['year', 'month'].

        Returns
        -------
        None

        Examples
        --------
        # Simple unload (multiple files, one per cluster slice)
        dc.unload_to_s3(
            query="SELECT * FROM my_schema.my_table WHERE date >= '2024-01-01'",
            relative_path="exports/my_table/",
            file_prefix="my_table_2024"
        )

        # Single file output
        dc.unload_to_s3(
            query="my_schema.summary_table",
            relative_path="exports/summary/",
            file_prefix="summary",
            parallel=False
        )

        # Control file size (will create more files if data exceeds max_file_size)
        dc.unload_to_s3(
            query="SELECT * FROM my_schema.large_table",
            relative_path="exports/large_table/",
            file_prefix="large",
            max_file_size="500 MB"
        )

        # Unload with partitioning
        dc.unload_to_s3(
            query="SELECT * FROM my_schema.events",
            relative_path="exports/events/",
            file_prefix="events",
            partition_by=["year", "month"]
        )

        # Unload from a .sql file (last SELECT is used for UNLOAD)
        dc.unload_to_s3(
            query="sql/my_export_query.sql",
            relative_path="exports/my_table/",
            file_prefix="my_table"
        )

        # Multi-statement SQL (preceding statements run first, last used for UNLOAD)
        dc.unload_to_s3(
            query=\"\"\"
            CREATE TEMP TABLE tmp AS SELECT id, name FROM users WHERE active;
            SELECT * FROM tmp
            \"\"\",
            relative_path="exports/active_users/",
            file_prefix="active_users"
        )

        Notes
        -----
        - With PARALLEL ON: Creates one file per cluster slice (typically 2-32 files).
        - With PARALLEL OFF: Creates a single file, UNLESS using PARTITION BY.
        - With PARTITION BY: Redshift parallelizes within each partition, creating multiple
          files per partition regardless of the PARALLEL setting. This is a Redshift limitation.
        - To control file sizes with large datasets, use the max_file_size parameter.
        """

        self._ensure_connected()
        self._mark_activity()

        try:
            # Get the appropriate S3 connector for this operation
            working_bucket = self._resolve_s3_bucket(s3_bucket)
            working_s3 = self._get_s3_for_bucket(working_bucket)

            if relative_path:
                relative_path = relative_path.strip().lstrip("/")
                # Remove double slashes
                relative_path = re.sub(r"/+", "/", relative_path)
                if not relative_path.endswith("/"):
                    relative_path += "/"
            else:
                relative_path = ""

            s3_path = f"s3://{working_bucket}/{relative_path}{file_prefix}"

            format = format.upper()
            if format not in ["PARQUET", "CSV", "JSON"]:
                log_and_raise_error(self._logger, f"Invalid format '{format}'. Must be PARQUET, CSV, or JSON.")

            if partition_by and format != "PARQUET":
                log_and_raise_error(self._logger, "partition_by parameter is only supported for PARQUET format.")

            try:
                existing_files = working_s3.list_files(prefix=f"{relative_path}{file_prefix}", bucket=working_bucket)
                if existing_files:
                    if drop_existing_files:
                        self._logger.info(
                            f"Found {len(existing_files)} existing file(s) at {s3_path}. "
                            "Deleting them (drop_existing_files=True)..."
                        )
                        for file_key in existing_files:
                            try:
                                working_s3.delete_file(file_key, bucket=working_bucket)
                            except Exception as delete_error:
                                self._logger.warning(f"Failed to delete {file_key}: {delete_error}")
                        self._logger.info(f"Deleted {len(existing_files)} file(s).")
                    elif overwrite:
                        self._logger.warning(
                            f"Found {len(existing_files)} existing file(s) at {s3_path}. "
                            "They will be overwritten (overwrite=True)."
                        )
                    else:
                        self._logger.warning(
                            f"Found {len(existing_files)} existing file(s) at {s3_path}. "
                            "UNLOAD will fail unless overwrite=True is set."
                        )
            except Exception as e:
                self._logger.debug(f"Could not check for existing files: {e}")

            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                log_and_raise_error(
                    self._logger,
                    "Unable to retrieve AWS credentials. Ensure boto3 is configured correctly.",
                )

            aws_access_key = credentials.access_key
            aws_secret_key = credentials.secret_key
            aws_session_token = credentials.token

            # Clean and validate query
            if query is None:
                log_and_raise_error(
                    self._logger,
                    "Query cannot be None. Please provide a valid SQL query string or ensure the SQL file exists.",
                )
            query = self._resolve_query(query)

            # Handle multi-statement SQL: split, execute preceding statements,
            # keep the last one as the UNLOAD query.
            statements = _split_sql_statements(query)
            if len(statements) > 1:
                preceding = statements[:-1]
                query = statements[-1]
                self._logger.info(
                    f"Multi-statement SQL detected: executing {len(preceding)} preceding statement(s) before UNLOAD"
                )
                for i, stmt in enumerate(preceding):
                    self._logger.debug(f"Executing preceding statement {i + 1}: {stmt[:100]}...")
                    self.execute_sql(stmt)
            elif len(statements) == 1:
                query = statements[0]

            # Check if it's a query statement (SELECT, WITH, etc.) or a table name
            # Skip leading comments to check actual SQL statement
            lines = query.split("\n")
            first_sql_line = None
            for line in lines:
                stripped = line.strip()
                # Skip empty lines and comments
                if stripped and not stripped.startswith("--"):
                    first_sql_line = stripped.upper()
                    break

            # If no SQL found or doesn't look like a query, treat as table name
            if first_sql_line is None or not (
                first_sql_line.startswith("SELECT")
                or first_sql_line.startswith("WITH")
                or first_sql_line.startswith("(")
            ):
                query = f"SELECT * FROM {query}"

            # Build UNLOAD query
            unload_query = f"""
            UNLOAD ($$
            {query}
            $$)
            TO '{s3_path}'
            ACCESS_KEY_ID '{aws_access_key}'
            SECRET_ACCESS_KEY '{aws_secret_key}'
            """

            if aws_session_token:
                unload_query += f"SESSION_TOKEN '{aws_session_token}'\n"

            # Add format
            if format == "PARQUET":
                unload_query += "FORMAT AS PARQUET\n"
            elif format == "CSV":
                unload_query += "FORMAT AS CSV\n"
            elif format == "JSON":
                unload_query += "FORMAT AS JSON\n"

            if partition_by:
                partition_cols = ", ".join(partition_by)
                unload_query += f"PARTITION BY ({partition_cols})\n"
                if not parallel:
                    self._logger.warning(
                        "Using PARTITION BY with PARALLEL OFF may still create multiple files per partition. "
                        "Redshift parallelizes within each partition even when PARALLEL OFF is specified."
                    )

            if parallel:
                unload_query += "PARALLEL ON\n"
            else:
                unload_query += "PARALLEL OFF\n"

            if max_file_size and parallel:
                unload_query += f"MAXFILESIZE {max_file_size}\n"
            elif max_file_size and not parallel:
                self._logger.warning("MAXFILESIZE is ignored when PARALLEL is OFF")

            if overwrite:
                unload_query += "ALLOWOVERWRITE\n"

            self._logger.info(f"Starting UNLOAD to {s3_path}")
            self._logger.debug(f"UNLOAD query: {unload_query}")

            with self._lock:
                self._cancel_idle_timer()
                self._last_activity = time.time()
                self._ensure_connected()
                self.cursor.execute(unload_query)
            self._mark_activity()

            self._logger.info(f"Successfully unloaded data to {s3_path}")

        except Exception as e:
            log_and_raise_error(self._logger, f"Error during UNLOAD to S3: {e}")
        finally:
            self._start_idle_timer()

    def load_from_s3(
        self,
        table: str,
        schema: str,
        relative_path: str,
        s3_bucket: str = None,
        format: str = "PARQUET",
        truncate_before_load: bool = False,
        drop_existing_table: bool = False,
        column_list: list[str] = None,
        column_types: dict[str, str] = None,
        ignore_header: int = None,
        delimiter: str = None,
        date_format: str = "auto",
        time_format: str = "auto",
        blank_as_null: bool = True,
        empty_as_null: bool = True,
        null_as: str = None,
        accept_invalid_chars: bool = False,
        max_error: int = 0,
        stat_update: bool = True,
        compupdate: bool = True,
        identity_column: str | bool = None,
    ):
        """
        Load data from S3 files into a Redshift table using the COPY command.

        Parameters
        ----------
        table : str
            The name of the target table (without schema).
        schema : str
            The schema name where the table exists or will be created.
        relative_path : str
            The relative path within the S3 bucket where files are located (e.g., 'my-data/input/').
            Can include wildcards or manifest files.
        s3_bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.
        format : str, optional
            Input format: 'PARQUET', 'CSV', 'JSON', 'AVRO', or 'ORC'. Defaults to 'PARQUET'.
        truncate_before_load : bool, optional
            If True, truncates the table before loading. Defaults to False.
        drop_existing_table : bool, optional
            If True, drops the existing table and recreates it from the source schema.
            Only works with PARQUET format (auto-infers schema from Parquet files).
            For other formats, you must create the table manually first.
            Defaults to False. Note: This is more destructive than truncate_before_load.
        column_list : list[str], optional
            List of column names in the target table to load data into.
            If None, assumes columns match the file structure.
        column_types : dict[str, str], optional
            Override specific column types when auto-creating tables (with drop_existing_table=True).
            Useful for null-only columns or forcing specific types.
            Example: {'created_date': 'DATE', 'status': 'VARCHAR(50)', 'amount': 'DECIMAL(10,2)'}
            Note: Only applies when the table is auto-created from PARQUET files.
        identity_column : str | bool, optional
            Add an auto-incrementing IDENTITY column when creating the table (requires drop_existing_table=True).
            - If True: Creates a column named 'id' as BIGINT IDENTITY(1,1)
            - If str: Creates a column with the specified name as BIGINT IDENTITY(1,1)
            - If None/False: No identity column (default)
            The identity column is added as the first column and excluded from the COPY column list.
            Example: identity_column=True or identity_column='row_id'
        ignore_header : int, optional
            Number of header lines to ignore (CSV only). Example: 1 for single header row.
        delimiter : str, optional
            Field delimiter for CSV files. Defaults to comma if not specified.
        date_format : str, optional
            Date format string. Defaults to 'auto'. Examples: 'YYYY-MM-DD', 'auto'.
            Note: Only supported for CSV and JSON formats.
        time_format : str, optional
            Timestamp format string. Defaults to 'auto'. Examples: 'YYYY-MM-DD HH:MI:SS', 'auto'.
            Note: Only supported for CSV and JSON formats.
        blank_as_null : bool, optional
            If True, treats blank values as NULL. Defaults to True.
            Note: Only supported for CSV format.
        empty_as_null : bool, optional
            If True, treats empty strings as NULL. Defaults to True.
            Note: Only supported for CSV format.
        null_as : str, optional
            String to interpret as NULL (e.g., 'NULL', '\\N'). Defaults to None.
            Note: Only supported for CSV format.
        accept_invalid_chars : bool, optional
            If True, replaces invalid UTF-8 characters with '?'. Defaults to False.
            Note: Only supported for CSV and JSON formats.
        max_error : int, optional
            Maximum number of errors allowed before failing. Defaults to 0 (no errors allowed).
        stat_update : bool, optional
            If True, updates table statistics after load. Defaults to True.
        compupdate : bool, optional
            If True, updates compression encodings. Defaults to True.
            Note: Only supported for CSV, JSON, AVRO, and ORC formats (not PARQUET).

        Returns
        -------
        str
            The fully qualified table name that was loaded.

        Examples
        --------
        # Simple Parquet load
        dc.load_from_s3(
            table="my_table",
            schema="my_schema",
            relative_path="imports/my_table/"
        )

        # CSV load with options
        dc.load_from_s3(
            table="my_table",
            schema="my_schema",
            relative_path="imports/my_table.csv",
            format="CSV",
            ignore_header=1,
            delimiter="|",
            truncate_before_load=True
        )

        # Auto-create table from Parquet if missing
        dc.load_from_s3(
            table="new_table",
            schema="my_schema",
            relative_path="imports/data/"
        )

        # Load specific columns
        dc.load_from_s3(
            table="my_table",
            schema="my_schema",
            relative_path="imports/data/",
            column_list=["id", "name", "created_at"]
        )
        """
        import boto3

        self._ensure_connected()
        self._mark_activity()

        # Get the appropriate S3 connector for this operation
        working_bucket = self._resolve_s3_bucket(s3_bucket)
        working_s3 = self._get_s3_for_bucket(working_bucket)

        if relative_path:
            relative_path = relative_path.strip().lstrip("/")
            # Remove double slashes
            relative_path = re.sub(r"/+", "/", relative_path)
        else:
            log_and_raise_error(self._logger, "relative_path cannot be empty")

        s3_path = f"s3://{working_bucket}/{relative_path}"

        format = format.upper()
        if format not in ["PARQUET", "CSV", "JSON", "AVRO", "ORC"]:
            log_and_raise_error(self._logger, f"Invalid format '{format}'. Must be PARQUET, CSV, JSON, AVRO, or ORC.")

        # Check if files exist in S3 before proceeding
        files_in_path = None
        try:
            files_in_path = working_s3.list_files(prefix=relative_path, bucket=working_bucket)
            if not files_in_path:
                log_and_raise_error(
                    self._logger,
                    f"No files found at S3 path: {s3_path}. Please verify the path exists and contains data files.",
                )
            self._logger.debug(f"Found {len(files_in_path)} file(s) in {s3_path}")
        except ValueError:
            # Re-raise errors from log_and_raise_error
            raise
        except Exception as e:
            self._logger.warning(f"Could not verify files in S3 path {s3_path}: {e}")

        try:
            fully_qualified_table = f"{schema}.{table}"

            table_exists = self._table_exists(table_name=table, schema_name=schema)

            # Handle table creation/recreation logic
            if not table_exists:
                if format == "PARQUET":
                    # If column_types provided, create table directly from those types
                    # Otherwise infer from Parquet schema
                    if column_types:
                        self._logger.debug(
                            f"Table {fully_qualified_table} does not exist. Creating from column_types..."
                        )
                        # Use all columns from column_types (not just those in Parquet)
                        # Some columns may be all-null and not present in Parquet file
                        column_order = list(column_types.keys())
                        self._create_table_from_column_types(
                            schema=schema,
                            table=table,
                            column_types=column_types,
                            column_order=column_order,
                            identity_column=identity_column,
                        )
                    else:
                        self._logger.debug(
                            f"Table {fully_qualified_table} does not exist. Creating from Parquet schema..."
                        )
                        self._create_table_from_parquet(
                            table=table,
                            schema=schema,
                            s3_bucket=working_bucket,
                            relative_path=relative_path,
                            known_files=files_in_path,
                            column_types=column_types,
                            identity_column=identity_column,
                        )
                else:
                    log_and_raise_error(
                        self._logger,
                        f"Table {fully_qualified_table} does not exist. "
                        f"Auto-creation only supported for PARQUET format. "
                        f"Please create the table manually for {format} format.",
                    )
            elif drop_existing_table:
                self._logger.debug(f"Dropping table {fully_qualified_table}")
                with self._lock:
                    self._cancel_idle_timer()
                    self._last_activity = time.time()
                    self._ensure_connected()
                    self.cursor.execute(f"DROP TABLE IF EXISTS {fully_qualified_table}")
                self._start_idle_timer()

                if format == "PARQUET":
                    # If column_types provided, create table directly from those types
                    # Otherwise infer from Parquet schema
                    if column_types:
                        self._logger.debug(f"Recreating table {fully_qualified_table} from column_types")
                        # Use all columns from column_types (not just those in Parquet)
                        # Some columns may be all-null and not present in Parquet file
                        column_order = list(column_types.keys())
                        self._create_table_from_column_types(
                            schema=schema,
                            table=table,
                            column_types=column_types,
                            column_order=column_order,
                            identity_column=identity_column,
                        )
                    else:
                        self._logger.debug(f"Recreating table {fully_qualified_table} from Parquet schema")
                        self._create_table_from_parquet(
                            schema=schema,
                            table=table,
                            s3_bucket=working_bucket,
                            relative_path=relative_path,
                            known_files=files_in_path,
                            column_types=column_types,
                            identity_column=identity_column,
                        )
                else:
                    log_and_raise_error(
                        self._logger,
                        f"Cannot auto-create table from {format} format. "
                        "Auto-creation only supported for PARQUET format. "
                        "Create the table manually before using drop_existing_table with non-Parquet formats.",
                    )

            # Truncate if requested (only if not already dropped)
            if truncate_before_load and not drop_existing_table:
                self._logger.debug(f"Truncating table {fully_qualified_table}")
                with self._lock:
                    self._cancel_idle_timer()
                    self._last_activity = time.time()
                    self._ensure_connected()
                    self.cursor.execute(f"TRUNCATE TABLE {fully_qualified_table}")
                self._start_idle_timer()

            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                log_and_raise_error(
                    self._logger, "Unable to retrieve AWS credentials. Ensure boto3 is configured correctly."
                )

            aws_access_key = credentials.access_key
            aws_secret_key = credentials.secret_key
            aws_session_token = credentials.token

            # Check for identity columns that should be excluded from COPY
            identity_columns = []
            if table_exists and format == "PARQUET":
                try:
                    # Use pg_catalog tables to detect identity columns via adsrc
                    identity_query = f"""
                        SELECT a.attname 
                        FROM pg_class c, pg_attribute a, pg_attrdef d, pg_namespace n
                        WHERE c.oid = a.attrelid 
                            AND c.relkind = 'r' 
                            AND a.attrelid = d.adrelid 
                            AND a.attnum = d.adnum 
                            AND d.adsrc LIKE '%identity%'
                            AND c.relnamespace = n.oid
                            AND n.nspname = '{schema}'
                            AND c.relname = '{table}'
                        ORDER BY a.attnum
                    """
                    identity_results = self.execute_sql(identity_query, fetch_all=True)
                    identity_columns = [row[0] for row in identity_results] if identity_results else []
                    if identity_columns:
                        self._logger.debug(f"Detected identity columns: {identity_columns}")
                except Exception as e:
                    self._logger.warning(f"Could not check for identity columns: {e}")

            # Build COPY query
            copy_query = f"COPY {fully_qualified_table}"

            # Add column list if provided or if we need to exclude identity columns
            # NOTE: column_list is NOT typically used for PARQUET format in Redshift
            # PARQUET columns must match table columns in order
            # EXCEPTION: When identity columns exist, we must specify column list excluding them
            if format == "PARQUET" and identity_columns:
                # Get all table columns and exclude identity columns
                try:
                    columns_query = f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = '{schema}'
                        AND table_name = '{table}'
                        ORDER BY ordinal_position
                    """
                    all_columns_results = self.execute_sql(columns_query, fetch_all=True)
                    all_columns = [row[0] for row in all_columns_results] if all_columns_results else []
                    # Exclude identity columns from COPY
                    copy_columns = [col for col in all_columns if col not in identity_columns]
                    columns_str = ", ".join(copy_columns)
                    copy_query += f" ({columns_str})"
                    self._logger.debug(f"COPY will use column list (excluding identity): {copy_columns}")
                except Exception as e:
                    self._logger.warning(f"Could not build column list for COPY: {e}")
            elif column_list and format != "PARQUET":
                columns_str = ", ".join(column_list)
                copy_query += f" ({columns_str})"

            copy_query += f"\nFROM '{s3_path}'\n"
            copy_query += f"ACCESS_KEY_ID '{aws_access_key}'\n"
            copy_query += f"SECRET_ACCESS_KEY '{aws_secret_key}'\n"

            # Add session token if present
            if aws_session_token:
                copy_query += f"SESSION_TOKEN '{aws_session_token}'\n"

            # Add format-specific options
            if format == "PARQUET":
                copy_query += "FORMAT AS PARQUET\n"
            elif format == "CSV":
                copy_query += "FORMAT AS CSV\n"
                if ignore_header:
                    copy_query += f"IGNOREHEADER {ignore_header}\n"
                if delimiter:
                    copy_query += f"DELIMITER '{delimiter}'\n"
            elif format == "JSON":
                copy_query += "FORMAT AS JSON 'auto'\n"
            elif format == "AVRO":
                copy_query += "FORMAT AS AVRO 'auto'\n"
            elif format == "ORC":
                copy_query += "FORMAT AS ORC\n"

            if format in ("CSV", "JSON") and date_format:
                copy_query += f"DATEFORMAT '{date_format}'\n"
            if format in ("CSV", "JSON") and time_format:
                copy_query += f"TIMEFORMAT '{time_format}'\n"

            if format == "CSV":
                if blank_as_null:
                    copy_query += "BLANKSASNULL\n"
                if empty_as_null:
                    copy_query += "EMPTYASNULL\n"
                if null_as:
                    copy_query += f"NULL AS '{null_as}'\n"

            if format in ("CSV", "JSON") and accept_invalid_chars:
                copy_query += "ACCEPTINVCHARS\n"

            if max_error > 0:
                copy_query += f"MAXERROR {max_error}\n"

            if stat_update:
                copy_query += "STATUPDATE ON\n"
            else:
                copy_query += "STATUPDATE OFF\n"

            if format != "PARQUET":
                if compupdate:
                    copy_query += "COMPUPDATE ON\n"
                else:
                    copy_query += "COMPUPDATE OFF\n"

            self._logger.debug(f"Starting COPY from {s3_path} to {fully_qualified_table}")
            self._logger.debug(f"COPY query: {copy_query}")

            with self._lock:
                self._cancel_idle_timer()
                self._last_activity = time.time()
                self._ensure_connected()
                self.cursor.execute(copy_query)
            self._mark_activity()

            self._logger.debug(f"Successfully copied data from {s3_path} to {fully_qualified_table}")

        except Exception as e:
            log_and_raise_error(self._logger, f"Error during COPY from S3: {e}")
        finally:
            self._start_idle_timer()

    def _table_exists(self, table_name: str, schema_name: str) -> bool:
        """Check if a regular (non-external) table exists in Redshift."""
        self._ensure_connected()
        self._mark_activity()
        try:
            check_sql = f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_tables
                    WHERE schemaname = '{schema_name}'
                    AND tablename = '{table_name}'
                )
            """
            result = self.execute_sql(check_sql, fetch_one=True)
            exists = result[0]
            return exists
        finally:
            self._start_idle_timer()

    def _create_table_from_parquet(
        self,
        table: str,
        schema: str,
        s3_bucket: str,
        relative_path: str,
        known_files: list[str] = None,
        column_types: dict[str, str] = None,
        identity_column: str | bool = None,
    ):
        """
        Helper method to create a table by inferring schema from a Parquet file.

        Parameters
        ----------
        column_types : dict[str, str], optional
            Override types for specific columns. Example: {'date_col': 'DATE', 'status': 'VARCHAR(50)'}
        identity_column : str | bool, optional
            Add an auto-incrementing IDENTITY column as the first column.
        """
        # Get the appropriate S3 connector for this operation
        working_s3 = self._get_s3_for_bucket(s3_bucket)

        try:
            if relative_path:
                relative_path = relative_path.strip().lstrip("/")
            if not relative_path.endswith("/"):
                relative_path_for_listing = relative_path + "/"
            else:
                relative_path_for_listing = relative_path

            # Find a Parquet file for schema inference
            parquet_file_key = None

            # Use known files if provided (avoids redundant S3 listing)
            if known_files:
                for file_key in known_files:
                    if file_key.lower().endswith(".parquet"):
                        parquet_file_key = file_key
                        self._logger.debug(f"Found Parquet file for schema inference: {parquet_file_key}")
                        break

            # Fall back to S3 listing only if no parquet found in known files
            if not parquet_file_key:
                candidates = [
                    relative_path_for_listing,
                    relative_path_for_listing.lstrip("/"),
                    relative_path,
                ]

                for prefix_candidate in candidates:
                    try:
                        files_prefix = working_s3.list_files_in_prefix(prefix=prefix_candidate) or []
                    except Exception:
                        files_prefix = []
                    for file_key in files_prefix:
                        if file_key.lower().endswith(".parquet"):
                            parquet_file_key = file_key
                            self._logger.debug(f"Found Parquet file for schema inference: {parquet_file_key}")
                            break
                    if parquet_file_key:
                        break

            if not parquet_file_key:
                log_and_raise_error(
                    self._logger,
                    f"No .parquet file found in '{relative_path}' for schema inference.",
                    FileNotFoundError,
                )

            parquet_key_for_path = parquet_file_key
            if parquet_key_for_path.startswith(f"{working_s3.bucket}/"):
                parquet_key_for_path = parquet_key_for_path[len(working_s3.bucket) + 1 :]
            parquet_s3_path = working_s3.get_path(parquet_key_for_path)
            df_schema = pq.read_schema(parquet_s3_path)
            columns_with_types = self._convert_pyarrow_schema_to_sql(
                df_schema, partition_defs=None, column_types=column_types
            )
            column_names = [col for col, _ in columns_with_types]
            self._check_redshift_reserved_words(column_names)

            fully_qualified_table = f"{schema}.{table}"
            create_table_query = f"CREATE TABLE {fully_qualified_table} (\n"

            # Add identity column first if requested
            if identity_column:
                id_col_name = "id" if identity_column is True else identity_column

                # Check if identity column name already exists in Parquet columns
                if id_col_name.lower() in [col.lower() for col in column_names]:
                    self._logger.warning(
                        f"Column '{id_col_name}' already exists in Parquet schema. "
                        f"Skipping identity column to avoid duplicate. "
                        f"Use a different identity_column name or remove '{id_col_name}' from your DataFrame."
                    )
                else:
                    create_table_query += f"  {id_col_name} BIGINT IDENTITY(1,1),\n"
                    self._logger.debug(f"Adding identity column '{id_col_name}' to table {fully_qualified_table}")

            create_table_query += ",\n".join([f"  {col} {dtype}" for col, dtype in columns_with_types])
            create_table_query += "\n);"

            self.execute_sql(create_table_query)

            self._logger.debug(f"Successfully created table {fully_qualified_table} from Parquet schema")

        except Exception as e:
            log_and_raise_error(self._logger, f"Failed to create table from Parquet schema: {e}")

    def _find_parquet_file(self, s3_connector, relative_path: str, known_files: list[str] = None) -> str:
        """
        Find a Parquet file in the given path for schema reading.

        Returns the S3 key of a parquet file.
        """
        if relative_path:
            relative_path = relative_path.strip().lstrip("/")

        # If it's a direct file path ending in .parquet, use it
        if relative_path.lower().endswith(".parquet"):
            return relative_path

        # Otherwise it's a directory, find a parquet file in it
        if not relative_path.endswith("/"):
            relative_path_for_listing = relative_path + "/"
        else:
            relative_path_for_listing = relative_path

        # Use known files if provided
        if known_files:
            for file_key in known_files:
                if file_key.lower().endswith(".parquet"):
                    return file_key

        # Fall back to S3 listing
        candidates = [relative_path_for_listing, relative_path_for_listing.lstrip("/"), relative_path]
        for prefix_candidate in candidates:
            try:
                files_prefix = s3_connector.list_files_in_prefix(prefix=prefix_candidate) or []
            except Exception:
                files_prefix = []
            for file_key in files_prefix:
                if file_key.lower().endswith(".parquet"):
                    return file_key

        log_and_raise_error(
            self._logger,
            f"No .parquet file found in '{relative_path}' for schema reading.",
            FileNotFoundError,
        )

    def _create_table_from_column_types(
        self,
        table: str,
        schema: str,
        column_types: dict[str, str],
        column_order: list[str] = None,
        identity_column: str | bool = None,
    ):
        """
        Create a table directly from column_types specification.

        This is used when explicit column types are provided and we don't want to infer from Parquet.

        Parameters
        ----------
        table : str
            Table name
        schema : str
            Schema name
        column_types : dict[str, str]
            Dictionary mapping column names to SQL types
        column_order : list[str], optional
            Order of columns. If None, uses sorted order of column_types keys
        identity_column : str | bool, optional
            Add an auto-incrementing IDENTITY column as the first column.
        """
        try:
            if column_order is None:
                column_order = sorted(column_types.keys())

            # Check for reserved words
            self._check_redshift_reserved_words(column_order)

            fully_qualified_table = f"{schema}.{table}"
            create_table_query = f"CREATE TABLE {fully_qualified_table} (\n"

            # Add identity column first if requested
            if identity_column:
                id_col_name = "id" if identity_column is True else identity_column

                # Check if identity column name already exists in column_types
                if id_col_name.lower() in [col.lower() for col in column_order]:
                    self._logger.warning(
                        f"Column '{id_col_name}' already exists in column_types. "
                        f"Skipping identity column to avoid duplicate. "
                        f"Use a different identity_column name or remove '{id_col_name}' from column_types."
                    )
                else:
                    create_table_query += f"  {id_col_name} BIGINT IDENTITY(1,1),\n"
                    self._logger.debug(f"Adding identity column '{id_col_name}' to table {fully_qualified_table}")

            create_table_query += ",\n".join([f"  {col} {column_types[col]}" for col in column_order])
            create_table_query += "\n);"

            self.execute_sql(create_table_query)

            self._logger.debug(f"Successfully created table {fully_qualified_table} from column_types")

        except Exception as e:
            log_and_raise_error(self._logger, f"Failed to create table from column_types: {e}")

    @staticmethod
    def _check_redshift_reserved_words(column_names):
        reserved_words = {
            "aes128",
            "aes256",
            "all",
            "allowoverwrite",
            "analyse",
            "analyze",
            "and",
            "any",
            "array",
            "as",
            "asc",
            "authorization",
            "backup",
            "between",
            "binary",
            "blanksasnull",
            "both",
            "by",
            "bzip2",
            "case",
            "cast",
            "check",
            "collate",
            "column",
            "constraint",
            "create",
            "credentials",
            "cross",
            "current_date",
            "current_time",
            "current_timestamp",
            "current_user",
            "current_user_id",
            "default",
            "deferrable",
            "deflate",
            "defrag",
            "delta",
            "delta32k",
            "desc",
            "disable",
            "distinct",
            "do",
            "else",
            "emptyasnull",
            "enable",
            "encode",
            "encrypt",
            "encryption",
            "end",
            "except",
            "explicit",
            "false",
            "for",
            "foreign",
            "freeze",
            "from",
            "full",
            "globaldict256",
            "globaldict64k",
            "grant",
            "group",
            "gzip",
            "having",
            "identity",
            "ignore",
            "ilike",
            "in",
            "initially",
            "inner",
            "intersect",
            "into",
            "is",
            "isnull",
            "join",
            "leading",
            "left",
            "like",
            "limit",
            "localtime",
            "localtimestamp",
            "lun",
            "luns",
            "lzo",
            "minus",
            "mostly13",
            "mostly32",
            "mostly8",
            "natural",
            "new",
            "not",
            "notnull",
            "null",
            "nulls",
            "off",
            "offline",
            "offset",
            "oid",
            "old",
            "on",
            "only",
            "open",
            "or",
            "order",
            "outer",
            "overlaps",
            "parallel",
            "partition",
            "percent",
            "permissions",
            "placing",
            "primary",
            "raw",
            "readratio",
            "recover",
            "references",
            "respect",
            "rejectlog",
            "resort",
            "restore",
            "right",
            "select",
            "session_user",
            "similar",
            "some",
            "sysdate",
            "system",
            "table",
            "tag",
            "tdes",
            "text255",
            "text32k",
            "then",
            "timestamp",
            "to",
            "top",
            "trailing",
            "true",
            "truncate",
            "unload",
            "user",
            "using",
            "verbose",
            "wallet",
            "when",
            "where",
            "with",
        }
        reserved_found = [col for col in column_names if col.lower() in reserved_words]
        if reserved_found:
            raise ValueError(
                f"The following column names are reserved words in Redshift and cannot be used: {reserved_found}"
            )

    def _cast_null_columns_for_parquet(
        self, df: pd.DataFrame | pl.DataFrame, column_types: dict[str, str]
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Cast null-only columns to appropriate types based on column_types specification.

        This ensures the Parquet file schema matches the table schema when using column_types.
        PyArrow's 'null' type is incompatible with Redshift Spectrum, so we cast null columns
        to typed null columns (e.g., int64 with nulls, date with nulls).
        """
        # Map Redshift SQL types to pandas/polars types
        type_mapping = {
            "DATE": "datetime64[ns]",
            "TIMESTAMP": "datetime64[ns]",
            "BIGINT": "Int64",
            "INT": "Int32",
            "SMALLINT": "Int16",
            "DECIMAL": "float64",  # Pandas doesn't have native decimal, use float
            "DOUBLE PRECISION": "float64",
            "REAL": "float32",
            "BOOLEAN": "bool",
            "VARCHAR": "object",
            "TEXT": "object",
        }

        if isinstance(df, pl.DataFrame):
            # Polars
            polars_type_mapping = {
                "DATE": pl.Date,
                "TIMESTAMP": pl.Datetime,
                "BIGINT": pl.Int64,
                "INT": pl.Int32,
                "SMALLINT": pl.Int16,
                "DECIMAL": pl.Float64,
                "DOUBLE PRECISION": pl.Float64,
                "REAL": pl.Float32,
                "BOOLEAN": pl.Boolean,
                "VARCHAR": pl.Utf8,
                "TEXT": pl.Utf8,
            }

            null_cols = [col for col in df.columns if df[col].dtype == pl.Null and col in column_types]
            if null_cols:
                self._logger.info(f"Casting {len(null_cols)} null column(s) to proper types: {null_cols}")
                for col in null_cols:
                    sql_type = column_types[col]
                    # Extract base type (e.g., "DECIMAL(10,2)" -> "DECIMAL", "VARCHAR(50)" -> "VARCHAR")
                    base_type = sql_type.split("(")[0].strip()
                    target_type = polars_type_mapping.get(base_type, pl.Utf8)
                    df = df.with_columns(pl.col(col).cast(target_type).alias(col))
        else:
            # Pandas
            null_cols = [col for col in df.columns if df[col].isna().all() and col in column_types]
            if null_cols:
                self._logger.info(f"Casting {len(null_cols)} null column(s) to proper types: {null_cols}")
                for col in null_cols:
                    sql_type = column_types[col]
                    # Extract base type
                    base_type = sql_type.split("(")[0].strip()
                    target_type = type_mapping.get(base_type, "object")

                    # Handle special cases
                    if base_type in ["DATE", "TIMESTAMP"]:
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(target_type)

        return df

    def _convert_null_columns_to_string(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame | pl.DataFrame:
        """
        Convert columns with all null values to string type.

        When a column contains only null values, PyArrow infers it as 'null' type,
        which Redshift Spectrum cannot read from Parquet files. This method converts
        such columns to string type to ensure compatibility.

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            The DataFrame to process.

        Returns
        -------
        pd.DataFrame | pl.DataFrame
            The DataFrame with null-type columns converted to string.
        """
        if isinstance(df, pl.DataFrame):
            # For Polars, check for Null dtype
            null_columns = [col for col in df.columns if df[col].dtype == pl.Null]
            if null_columns:
                self._logger.info(f"Converting {len(null_columns)} all-null column(s) to String type: {null_columns}")
                df = df.with_columns([pl.col(col).cast(pl.String).alias(col) for col in null_columns])
        else:
            # For Pandas, check for columns where all values are null
            # and the inferred PyArrow type would be 'null'
            null_columns = []
            for col in df.columns:
                if df[col].isna().all():
                    # Check if PyArrow would infer this as null type
                    try:
                        arrow_type = pa.Array.from_pandas(df[col]).type
                        if pa.types.is_null(arrow_type):
                            null_columns.append(col)
                    except Exception:
                        # If we can't determine the type, check if it's object with all None
                        if df[col].dtype == object or pd.api.types.is_object_dtype(df[col]):
                            null_columns.append(col)

            if null_columns:
                self._logger.info(f"Converting {len(null_columns)} all-null column(s) to string type: {null_columns}")
                for col in null_columns:
                    df[col] = df[col].astype("string")

        return df

    def _convert_pyarrow_schema_to_sql(
        self,
        arrow_schema: pa.Schema,
        partition_defs: list[tuple[str, str]] = None,
        column_types: dict[str, str] = None,
    ) -> list[tuple[str, str]]:
        """
        Convert PyArrow schema to SQL column definitions.

        Parameters
        ----------
        arrow_schema : pa.Schema
            PyArrow schema to convert
        partition_defs : list[tuple[str, str]], optional
            List of partition column definitions (name, type)
        column_types : dict[str, str], optional
            Override types for specific columns. Useful for null-only columns
            or when you want specific types. Example: {'date_col': 'DATE', 'status': 'VARCHAR(50)'}
        """
        columns_with_types = []
        partition_column_names = [p_name for p_name, p_type in partition_defs] if partition_defs else []
        column_type_overrides = column_types or {}

        for field in arrow_schema:
            col_name = field.name
            arrow_type = field.type

            if col_name in partition_column_names:
                self._logger.debug(f"Column '{col_name}' is a partition column, skipping from main schema definition.")
                continue

            sql_type = None
            if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
                sql_type = "VARCHAR(65535)"
            elif pa.types.is_int64(arrow_type):
                sql_type = "BIGINT"
            elif pa.types.is_int32(arrow_type):
                sql_type = "INT"
            elif pa.types.is_int16(arrow_type) or pa.types.is_int8(arrow_type):
                sql_type = "SMALLINT"
            elif pa.types.is_float64(arrow_type):
                sql_type = "DOUBLE PRECISION"
            elif pa.types.is_float32(arrow_type) or pa.types.is_float16(arrow_type):
                sql_type = "REAL"
            elif pa.types.is_boolean(arrow_type):
                sql_type = "BOOLEAN"
            elif pa.types.is_date32(arrow_type) or pa.types.is_date64(arrow_type):
                sql_type = "DATE"
            elif pa.types.is_timestamp(arrow_type):
                sql_type = "TIMESTAMP"
            elif pa.types.is_decimal(arrow_type):
                sql_type = f"DECIMAL({arrow_type.precision}, {arrow_type.scale})"
            elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
                sql_type = "VARBYTE(65535)"
            elif pa.types.is_null(arrow_type):
                # Columns with only null values get 'null' type in PyArrow.
                # Check if user provided an override, otherwise default to VARCHAR.
                if col_name in column_type_overrides:
                    sql_type = column_type_overrides[col_name]
                    self._logger.info(
                        f"Column '{col_name}' has PyArrow 'null' type (all values are null). "
                        f"Using user-specified type: {sql_type}"
                    )
                else:
                    sql_type = "VARCHAR(65535)"
                    self._logger.info(
                        f"Column '{col_name}' has PyArrow 'null' type (all values are null). "
                        "Defaulting to VARCHAR(65535)."
                    )
            else:
                self._logger.warning(
                    f"Unsupported PyArrow type '{arrow_type}' for column '{col_name}'. Skipping column."
                )
                continue

            # Apply user override if provided (takes precedence over inferred type)
            if col_name in column_type_overrides and not pa.types.is_null(arrow_type):
                sql_type = column_type_overrides[col_name]
                self._logger.info(f"Column '{col_name}' type overridden by user: {sql_type}")

            columns_with_types.append((col_name, sql_type))

        if not columns_with_types and not partition_column_names and arrow_schema.names:
            log_and_raise_error(
                self._logger,
                "No columns could be derived from the Parquet schema. All columns might be of unsupported types "
                "or are partition columns.",
            )
        elif (
            not columns_with_types
            and arrow_schema.names
            and all(name in partition_column_names for name in arrow_schema.names)
        ):
            self._logger.info(
                "All columns in Parquet schema are partition columns. "
                "Table will be created with only partition columns if defined."
            )
        elif not columns_with_types and not arrow_schema.names:
            log_and_raise_error(self._logger, "Parquet schema is empty. Cannot create table.")
        return columns_with_types

    @staticmethod
    def infer_column_types_from_dataframe(
        df: pd.DataFrame | pl.DataFrame, overrides: dict[str, str] = None
    ) -> dict[str, str]:
        """
        Generate a column_types dictionary from a DataFrame for use with load/create methods.

        This helper method infers Redshift SQL types from DataFrame column types and provides
        sensible defaults for nullable columns. You can override specific columns as needed.

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            The DataFrame to infer column types from
        overrides : dict[str, str], optional
            Override specific column types. These take precedence over inferred types.
            Example: {'created_date': 'DATE', 'status': 'VARCHAR(50)'}

        Returns
        -------
        dict[str, str]
            Dictionary mapping column names to Redshift SQL types

        Examples
        --------
        # Basic usage - infer all types
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', None],
            'amount': [10.5, 20.0, 30.5],
            'created_at': pd.to_datetime(['2024-01-01', '2024-01-02', None])
        })
        column_types = DataConnector.infer_column_types_from_dataframe(df)
        # Returns: {'id': 'BIGINT', 'name': 'VARCHAR(65535)', 'amount': 'DOUBLE PRECISION',
        #           'created_at': 'TIMESTAMP'}

        # With overrides for null-only or specific columns
        df = pd.DataFrame({
            'id': [1, 2],
            'future_date': [None, None]  # All nulls
        })
        column_types = DataConnector.infer_column_types_from_dataframe(
            df,
            overrides={'future_date': 'DATE'}  # Specify what the type should be
        )
        # Returns: {'id': 'BIGINT', 'future_date': 'DATE'}

        # Use with create_table_from_dataframe
        dc = DataConnector()
        column_types = DataConnector.infer_column_types_from_dataframe(
            df,
            overrides={'birth_date': 'DATE', 'salary': 'DECIMAL(12,2)'}
        )
        dc.create_table_from_dataframe(df, 'employees', 'hr', column_types=column_types)
        """
        import numpy as np

        column_types = {}
        overrides_dict = overrides or {}

        # Determine if it's pandas or polars
        if isinstance(df, pd.DataFrame):
            for col in df.columns:
                # Check override first
                if col in overrides_dict:
                    column_types[col] = overrides_dict[col]
                    continue

                dtype = df[col].dtype

                # Pandas type mapping
                if pd.api.types.is_integer_dtype(dtype):
                    if dtype == np.int64 or dtype == "Int64":
                        column_types[col] = "BIGINT"
                    elif dtype == np.int32 or dtype == "Int32":
                        column_types[col] = "INT"
                    elif dtype == np.int16 or dtype == "Int16" or dtype == np.int8 or dtype == "Int8":
                        column_types[col] = "SMALLINT"
                    else:
                        column_types[col] = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    if dtype == np.float64:
                        column_types[col] = "DOUBLE PRECISION"
                    elif dtype == np.float32:
                        column_types[col] = "REAL"
                    else:
                        column_types[col] = "DOUBLE PRECISION"
                elif pd.api.types.is_bool_dtype(dtype):
                    column_types[col] = "BOOLEAN"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    column_types[col] = "TIMESTAMP"
                elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                    column_types[col] = "VARCHAR(65535)"
                elif str(dtype) == "category":
                    column_types[col] = "VARCHAR(65535)"
                else:
                    # Default for unknown types (including all-null columns)
                    column_types[col] = "VARCHAR(65535)"

        else:  # Polars DataFrame
            for col in df.columns:
                # Check override first
                if col in overrides_dict:
                    column_types[col] = overrides_dict[col]
                    continue

                dtype = df[col].dtype

                # Polars type mapping
                if dtype in [pl.Int64, pl.UInt64]:
                    column_types[col] = "BIGINT"
                elif dtype in [pl.Int32, pl.UInt32]:
                    column_types[col] = "INT"
                elif dtype in [pl.Int16, pl.UInt16, pl.Int8, pl.UInt8]:
                    column_types[col] = "SMALLINT"
                elif dtype == pl.Float64:
                    column_types[col] = "DOUBLE PRECISION"
                elif dtype == pl.Float32:
                    column_types[col] = "REAL"
                elif dtype == pl.Boolean:
                    column_types[col] = "BOOLEAN"
                elif dtype == pl.Date:
                    column_types[col] = "DATE"
                elif dtype in [pl.Datetime, pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]:
                    column_types[col] = "TIMESTAMP"
                elif dtype == pl.Utf8 or dtype == pl.String:
                    column_types[col] = "VARCHAR(65535)"
                elif dtype == pl.Categorical:
                    column_types[col] = "VARCHAR(65535)"
                elif dtype == pl.Null:
                    # All-null column - default to VARCHAR unless override provided
                    column_types[col] = "VARCHAR(65535)"
                else:
                    # Default for unknown types
                    column_types[col] = "VARCHAR(65535)"

        return column_types

    def create_table_from_dataframe(
        self,
        df: pd.DataFrame | pl.DataFrame,
        table: str,
        schema: str,
        drop_existing_table: bool = False,
        column_types: dict[str, str] = None,
        s3_connector: S3Connector = None,
        s3_path: str = None,
        file_name: str = None,
        delete_s3_files_after: bool = True,
        truncate_before_load: bool = False,
        identity_column: str | bool = None,
        stat_update: bool = False,
    ):
        """
        Create a Redshift table from a DataFrame by uploading to S3, then loading via COPY.

        By default, uses a temporary S3 path that is automatically cleaned up. You can also
        specify a custom S3 path (useful for appending data or keeping files for audit).

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            The DataFrame to load into Redshift.
        table : str
            The name of the target table (without schema).
        schema : str
            The schema name where the table will be created.
        drop_existing_table : bool, optional
            If True, drops the existing table before creating. Defaults to False.
        column_types : dict[str, str], optional
            Override specific column types when creating the table.
            Useful for columns with all NULL values or to enforce specific types.
            Example: {'created_date': 'DATE', 'status': 'VARCHAR(50)', 'amount': 'DECIMAL(10,2)'}
        s3_connector : S3Connector, optional
            An existing S3Connector instance to use. If provided along with s3_path,
            uses that connector's bucket and the specified path. Useful when you want
            to control where files are stored (e.g., for appending or audit trails).
        s3_path : str, optional
            Custom relative path in S3 where the parquet file will be saved.
            If not provided, uses a temporary path that gets cleaned up.
            When s3_connector is provided, this path is relative to the connector's s3_root.
            Example: 'data/my_table/' or 'staging/loads/2024/'
        file_name : str, optional
            Custom name for the parquet file (without extension).
            If not provided, defaults to '{table}_data'.
            Example: 'my_custom_file' will create 'my_custom_file.parquet'
        delete_s3_files_after : bool, optional
            If True, deletes the S3 files after loading into Redshift. Defaults to True.
            Set to False if you want to keep the files (e.g., for backup or appending).
        truncate_before_load : bool, optional
            If True, truncates the table before loading (keeps table structure).
            Defaults to False. Use this for appending with clean slate without dropping table.
        identity_column : str | bool, optional
            Add an auto-incrementing IDENTITY column to the table.
            - If True: Creates a column named 'id' as BIGINT IDENTITY(1,1)
            - If str: Creates a column with the specified name as BIGINT IDENTITY(1,1)
            - If None/False: No identity column (default)
            The identity column is added as the first column in the table.
            Example: identity_column=True or identity_column='row_id'
        stat_update : bool, optional
            If True, updates table statistics after the COPY operation (runs ANALYZE).
            Requires table/database owner privileges. Defaults to False.
            Set to True only if you have the necessary permissions and want updated statistics.
            Note: If you encounter permission errors like "only table or database owner can analyze",
            keep this as False.

        Returns
        -------
        None

        Examples
        --------
        # Simple usage - creates table from DataFrame (temp files auto-deleted)
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        dc.create_table_from_dataframe(df, table='my_table', schema='my_schema')

        # Drop and recreate table
        dc.create_table_from_dataframe(
            df,
            table='my_table',
            schema='my_schema',
            drop_existing_table=True
        )

        # Use custom S3 path and keep files (for audit/backup)
        dc.create_table_from_dataframe(
            df,
            table='my_table',
            schema='my_schema',
            s3_path='staging/my_table_loads/',
            delete_s3_files_after=False
        )

        # Use existing S3 connector with custom path (s3_root automatically prepended)
        s3 = S3Connector(bucket='my-bucket', s3_root='project/')
        dc.create_table_from_dataframe(
            df,
            table='my_table',
            schema='my_schema',
            s3_connector=s3,
            s3_path='loads/daily/',  # Will be saved to 'project/loads/daily/'
            delete_s3_files_after=False
        )

        # Use custom file name to keep organized data
        dc.create_table_from_dataframe(
            df,
            table='my_table',
            schema='my_schema',
            s3_path='partitioned/country=mx/',
            file_name='scores_20241216',  # Creates 'scores_20241216.parquet'
            delete_s3_files_after=False
        )

        # Append to existing table (truncate first, keep files)
        dc.create_table_from_dataframe(
            df,
            table='my_table',
            schema='my_schema',
            truncate_before_load=True,
            s3_path='data/incremental/',
            delete_s3_files_after=False
        )

        # Use polars DataFrame
        import polars as pl
        df_polars = pl.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        dc.create_table_from_dataframe(df_polars, table='polars_table', schema='my_schema')
        """
        import uuid
        from datetime import datetime

        self._ensure_connected()
        self._mark_activity()

        fully_qualified_table = f"{schema}.{table}"
        working_s3 = None
        working_path = None
        is_temp_path = False
        working_bucket = None

        try:
            # Determine S3 connector and path
            if s3_connector is not None:
                working_s3 = s3_connector
                working_bucket = s3_connector.bucket
            else:
                working_bucket = self._resolve_s3_bucket()
                working_s3 = self._get_s3_for_bucket(working_bucket)

            if s3_path is not None:
                # Use custom path provided by user
                working_path = s3_path.strip().lstrip("/")
                if not working_path.endswith("/"):
                    working_path += "/"
                # If s3_connector provided and has s3_root, prepend it to the path
                if s3_connector is not None and s3_connector.s3_root:
                    working_path = f"{s3_connector.s3_root}/{working_path}"
                is_temp_path = False
            else:
                # Generate a unique temporary path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                temp_path = f"_temp/dataframe_load/{timestamp}_{unique_id}/"
                # If s3_connector provided and has s3_root, prepend it
                if s3_connector is not None and s3_connector.s3_root:
                    working_path = f"{s3_connector.s3_root}/{temp_path}"
                else:
                    working_path = temp_path
                is_temp_path = True

            # Use custom file name if provided, otherwise default to {table}_data
            # Strip .parquet extension if user included it (will be added later)
            if file_name is not None:
                parquet_file_name = file_name
                if parquet_file_name.endswith(".parquet"):
                    parquet_file_name = parquet_file_name[:-8]
            else:
                parquet_file_name = f"{table}_data"

            # Handle null columns only if column_types is NOT provided
            # When column_types is provided, save_dataframe handles all type conversions
            if column_types is None:
                # No column_types: Convert all-null columns to string (old behavior)
                df = self._convert_null_columns_to_string(df)

            # Determine which columns will be in the Parquet file
            # When column_types is provided, ALL columns (including all-null) must be in Parquet
            # to match the table schema
            parquet_columns = None
            adjusted_column_types = column_types

            # Check if table already exists and if we're appending (not dropping)
            table_exists = self._table_exists(table_name=table, schema_name=schema)
            if table_exists and not drop_existing_table:
                # Table exists and we're appending - read actual table schema
                self._logger.info(f"Table {schema}.{table} exists. Reading existing schema to match Parquet file...")
                try:
                    # Use pg_catalog tables to get schema and detect identity columns
                    schema_query = f"""
                        SELECT a.attname as column_name, 
                               format_type(a.atttypid, a.atttypmod) as data_type,
                               CASE WHEN d.adsrc LIKE '%identity%' THEN true ELSE false END as is_identity
                        FROM pg_class c
                        JOIN pg_attribute a ON c.oid = a.attrelid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        LEFT JOIN pg_attrdef d ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                        WHERE c.relkind = 'r'
                            AND n.nspname = '{schema}'
                            AND c.relname = '{table}'
                            AND a.attnum > 0
                            AND NOT a.attisdropped
                        ORDER BY a.attnum
                    """
                    existing_schema = self.execute_sql(schema_query, fetch_all=True)

                    # Map Redshift types to our column_types format
                    # EXCLUDE identity columns - they are auto-generated and should not be in the DataFrame/Parquet
                    existing_column_types = {}
                    if existing_schema:
                        for row in existing_schema:
                            col_name = row[0]
                            data_type = row[1]
                            is_identity = row[2] if len(row) > 2 else False

                            # Skip identity columns - they shouldn't be in the Parquet file
                            if is_identity is True:
                                self._logger.debug(f"Excluding identity column '{col_name}' from Parquet file")
                                continue

                            # Convert Redshift types to standard format
                            if "character varying" in data_type or "varchar" in data_type:
                                existing_column_types[col_name] = "VARCHAR(65535)"
                            elif data_type == "bigint":
                                existing_column_types[col_name] = "BIGINT"
                            elif data_type == "integer":
                                existing_column_types[col_name] = "INT"
                            elif data_type == "double precision":
                                existing_column_types[col_name] = "DOUBLE PRECISION"
                            elif data_type == "date":
                                existing_column_types[col_name] = "DATE"
                            elif "timestamp" in data_type:
                                existing_column_types[col_name] = "TIMESTAMP"
                            else:
                                existing_column_types[col_name] = data_type.upper()

                    # Use existing table schema instead of user-provided column_types (if any)
                    self._logger.info(
                        "Using existing table schema for Parquet compatibility (excluding identity columns)"
                    )
                    adjusted_column_types = existing_column_types
                    column_order_for_table = list(existing_column_types.keys())
                except Exception as e:
                    self._logger.warning(f"Could not read existing table schema: {e}. Using column_types as-is.")
                    if column_types:
                        column_order_for_table = list(column_types.keys())
                    else:
                        # No column_types and couldn't read schema - use DataFrame columns
                        column_order_for_table = list(df.columns)
            elif column_types:
                # Table doesn't exist or we're dropping it - use column_types as-is
                column_order_for_table = list(column_types.keys())
            else:
                # No column_types and table doesn't exist - use DataFrame columns
                column_order_for_table = list(df.columns)

            # Important: We must include ALL columns in the Parquet file
            # Even all-null columns must be included so the Parquet file matches the table schema

            # Ensure DataFrame has all columns (add missing columns as NaN)
            for col in column_order_for_table:
                if col not in df.columns:
                    df[col] = None

            # Reorder DataFrame columns to match the order
            df = df[column_order_for_table]
            parquet_columns = column_order_for_table

            # If we haven't already set adjusted_column_types (from existing table), adjust for DECIMAL
            if adjusted_column_types is None:
                # No adjusted_column_types yet - use DataFrame columns as-is
                pass
            elif column_types and adjusted_column_types == column_types:
                # We have column_types but haven't adjusted them yet - check for DECIMAL
                # Adjust ALL column types for consistency:
                # DECIMAL types must be DOUBLE PRECISION (Redshift can't load Parquet double into numeric)
                adjusted_column_types = column_types.copy()
                for col, col_type in column_types.items():
                    base_type = col_type.split("(")[0].strip().upper()
                    if base_type == "DECIMAL":
                        adjusted_column_types[col] = "DOUBLE PRECISION"
                        self._logger.debug(f"Adjusted {col} from DECIMAL to DOUBLE PRECISION for Parquet compatibility")

            # Save DataFrame to S3
            # Pass column_types to ensure Parquet schema matches table schema
            # When s3_connector is provided and we've already prepended s3_root to working_path,
            # pass s3_root="" to save_dataframe to prevent double s3_root
            self._logger.debug(f"Saving DataFrame to Parquet with column_types: {adjusted_column_types}")
            self._logger.debug(f"Parquet columns that will be saved: {parquet_columns}")

            # Determine s3_root parameter for save_dataframe
            # If we used s3_connector and already prepended its s3_root, pass empty string
            save_s3_root = "" if (s3_connector is not None and s3_connector.s3_root) else None

            working_s3.save_dataframe(
                df,
                directory=working_path,
                file_name=parquet_file_name,
                file_format="parquet",
                column_types=adjusted_column_types,
                s3_root=save_s3_root,
            )

            # Load from S3 to Redshift
            # Use specific file path to avoid Redshift trying to read hidden files in directory
            parquet_file_path = f"{working_path}{parquet_file_name}.parquet"
            self._logger.debug(f"Loading Parquet from: s3://{working_bucket}/{parquet_file_path}")
            self._logger.debug(f"Adjusted column types for table: {adjusted_column_types}")
            # NOTE: column_list is NOT used for PARQUET format - Parquet columns must match table order
            self.load_from_s3(
                table=table,
                schema=schema,
                relative_path=parquet_file_path,
                s3_bucket=working_bucket,
                format="PARQUET",
                drop_existing_table=drop_existing_table,
                truncate_before_load=truncate_before_load,
                column_types=adjusted_column_types,  # Use adjusted types (DECIMAL -> DOUBLE PRECISION)
                column_list=None,  # Do NOT use column_list for PARQUET - columns must match table order
                identity_column=identity_column,
                stat_update=stat_update,
            )

            self._logger.info(f"Successfully loaded data into {fully_qualified_table}")

            # Clean up S3 files if requested (always delete temp paths on success)
            should_delete = delete_s3_files_after or is_temp_path
            if should_delete and working_s3 and working_path:
                try:
                    files_to_delete = working_s3.list_files(prefix=working_path, bucket=working_bucket)
                    for file_key in files_to_delete:
                        try:
                            working_s3.delete_file(file_key, bucket=working_bucket)
                        except Exception as delete_error:
                            self._logger.warning(f"Failed to delete file {file_key}: {delete_error}")
                except Exception as cleanup_error:
                    self._logger.warning(f"Failed to clean up S3 files: {cleanup_error}")

        except Exception as e:
            # On error, keep files for debugging
            self._logger.info(f"Keeping S3 files for debugging at: s3://{working_bucket}/{working_path}")
            log_and_raise_error(self._logger, f"Error creating table from DataFrame: {e}")
        finally:
            self._start_idle_timer()
