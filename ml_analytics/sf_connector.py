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
from .utils import (
    format_sql_ignoring_comments,
    get_logger,
    load_sql_query,
    log_and_raise_error,
    resolve_sql_query_paths,
)

# Cached Spark session shared across SFConnector instances. Populated lazily by
# get_spark(); never created at import time so the package stays importable
# without PySpark.
_spark_ctx = None


def get_spark():
    """
    Get or create a cached Spark session that works both locally and on Databricks.

    Neither PySpark nor Databricks Connect is a dependency of this package; both
    are imported lazily so the rest of the package stays usable without them.

    Resolution order:

    1. Reuse an active :class:`SparkSession` if one exists. This is the normal
       case inside a Databricks notebook/cluster, where ``spark`` is already
       provided, so we attach to it rather than spinning up a new one.
    2. Otherwise create a Databricks Connect session
       (``DatabricksSession.builder.getOrCreate()``). This is the local-dev case:
       it connects to a remote cluster/serverless using your Databricks config
       (profile / env vars), so no notebook boilerplate is needed.
    3. Otherwise fall back to a plain local ``SparkSession``.

    This means a single ``spark = get_spark()`` line behaves correctly whether the
    code runs locally via Databricks Connect or as a notebook on Databricks.
    """
    global _spark_ctx
    if _spark_ctx is not None:
        return _spark_ctx

    # 1. Reuse an active session (the normal case inside a Databricks notebook/cluster).
    try:
        from pyspark.sql import SparkSession

        active = SparkSession.getActiveSession()
        if active is not None:
            _spark_ctx = active
            return _spark_ctx
    except ImportError:
        # PySpark itself isn't installed; Databricks Connect (below) ships its own.
        pass

    # 2. Try Databricks Connect (local dev against a remote cluster/serverless).
    try:
        from databricks.connect import DatabricksSession

        _spark_ctx = DatabricksSession.builder.getOrCreate()
        return _spark_ctx
    except ImportError:
        pass

    # 3. Fall back to a plain local Spark session.
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ImportError(
            "SFConnector needs a Spark session but neither PySpark nor Databricks "
            "Connect is available. Run it on a Spark runtime (e.g. Databricks) or "
            "install one locally with `pip install databricks-connect`."
        ) from exc

    _spark_ctx = SparkSession.builder.appName("ml_analytics").getOrCreate()
    return _spark_ctx


def _snowflake_account_url(account: str) -> str:
    """Normalize a Snowflake account identifier into a full host URL."""
    account_url = account.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if account_url.endswith(".snowflakecomputing.com"):
        return account_url
    return f"{account_url}.snowflakecomputing.com"


def _quote_spark_identifier(column_name: str) -> str:
    """Quote a Spark column identifier so dots/backticks are treated literally."""
    return f"`{str(column_name).replace('`', '``')}`"


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
        """
        Resolve a query string.

        If it looks like a SQL file path, load it (applying ``**kwargs`` as
        ``str.format`` template variables). Otherwise it's an inline query: it is
        returned as-is, except that when ``**kwargs`` are provided they are applied
        too, so callers don't have to ``query.format(...)`` themselves. With no
        kwargs the string is untouched, so inline SQL containing literal ``{`` /
        ``}`` (JSON, OBJECT_CONSTRUCT, ...) is left alone.

        Substitution is comment- and string-aware: ``{placeholder}`` tokens are
        replaced only in actual SQL code, never inside ``--`` / ``/* ... */``
        comments or quoted string literals. A commented script can therefore keep
        real ``{start_date}`` placeholders while literal braces in comments (e.g.
        documented ``{tutor_id}`` URL patterns) or string payloads are preserved.
        See ``format_sql_ignoring_comments``.
        """
        if query and query.strip().endswith(".sql"):
            loaded = load_sql_query(query.strip(), **kwargs)
            if loaded is None:
                log_and_raise_error(self._logger, f"Could not load SQL file: {query}")
            self._logger.info(f"Loaded SQL from file: {query}")
            return loaded
        if query and kwargs:
            try:
                return format_sql_ignoring_comments(query, **kwargs)
            except (KeyError, IndexError, ValueError) as e:
                log_and_raise_error(
                    self._logger,
                    f"Error formatting inline SQL query with {sorted(kwargs)}: {e}. "
                    f"Escape literal braces as '{{{{' / '}}}}' if the SQL is not a template.",
                )
        return query

    @staticmethod
    def _wrap_query_for_connector(query: str) -> str:
        """
        Make a query safe for the Snowflake Spark connector without removing comments.

        The connector embeds the supplied query inside its own wrappers (for schema
        discovery and pushdown, e.g. ``SELECT * FROM (<query>) WHERE 1 = 0``). A
        leading or trailing single-line ``--`` comment in ``<query>`` then comments
        out the connector's surrounding tokens and raises a parse error.

        Rather than stripping comments (which changes the query), wrap it in an
        explicit subquery with surrounding newlines. This isolates any comments on
        their own lines so they are preserved and *not* executed, while the query
        runs exactly as written.
        """
        if not query or not query.strip():
            return query
        inner = query.strip()
        # A trailing statement terminator is illegal inside a subquery; drop it.
        if inner.endswith(";"):
            inner = inner[:-1].rstrip()
        return f"SELECT * FROM (\n{inner}\n) AS ml_analytics_query"

    @staticmethod
    def _lowercase_spark_columns(df):
        columns = getattr(df, "columns", None)
        if not isinstance(columns, list):
            return df

        lowered_columns = [str(column).lower() for column in columns]
        if lowered_columns == columns:
            return df
        return df.toDF(*lowered_columns)

    @staticmethod
    def _cast_decimal_columns_for_pandas_conversion(df):
        try:
            from pyspark.sql import functions as F
            from pyspark.sql.types import DecimalType
        except ImportError:
            return df

        fields = getattr(getattr(df, "schema", None), "fields", None)
        if not isinstance(fields, list | tuple):
            return df

        expressions = []
        has_decimal_columns = False
        for field in fields:
            column_name = field.name
            column = F.col(_quote_spark_identifier(column_name))
            if isinstance(field.dataType, DecimalType):
                cast_type = "long" if getattr(field.dataType, "scale", 0) == 0 else "double"
                column = column.cast(cast_type)
                has_decimal_columns = True
            expressions.append(column.alias(column_name))

        if not has_decimal_columns:
            return df
        return df.select(*expressions)

    def _normalize_result_dataframe(self, df):
        df = self._lowercase_spark_columns(df)
        return self._cast_decimal_columns_for_pandas_conversion(df)

    def sql(
        self,
        query: str,
        return_pandas: bool = False,
        save_table: bool = False,
        table: str = None,
        schema: str = None,
        catalog: str = None,
        mode: str = "overwrite",
        optimize: bool = True,
        zorder_by=None,
        merge_schema: bool = True,
        comment: str = None,
        drop_existing: bool = True,
        overwrite_schema: bool = True,
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
        optimize : bool, optional
            If saving to Unity Catalog, run ``OPTIMIZE`` after the write.
            Defaults to True.
        zorder_by : str or list[str], optional
            Optional columns for Delta ``ZORDER BY`` during optimize.
        merge_schema : bool, optional
            If saving to Unity Catalog, set Delta ``mergeSchema=true``. Defaults to True.
        comment : str, optional
            Optional table comment stored as a Unity Catalog table property.
        drop_existing : bool, optional
            If saving, drop the destination table before writing so it is fully
            recreated from the result. Defaults to True. Set False to preserve the
            table (required for ``mode='append'``).
        overwrite_schema : bool, optional
            If saving with ``mode='overwrite'``, replace the table schema (drop removed
            columns) via Delta ``overwriteSchema=true``. Defaults to True.
        **kwargs
            Template variables substituted into the SQL file using ``str.format()``.

        Returns
        -------
        DataFrame
            The query result. When ``save_table`` is True, the returned DataFrame
            reads from the saved Unity Catalog Delta table (fast), not the Snowflake
            source — so downstream actions don't re-run the Snowflake query.
        """
        query = self._wrap_query_for_connector(self._resolve_query(query, **kwargs))
        spark = self._get_spark()
        try:
            df = spark.read.format(self.source_format).options(**self.spark_options()).option("query", query).load()
        except Exception as e:
            log_and_raise_error(self._logger, f"Error reading from Snowflake: {e}")

        df = self._normalize_result_dataframe(df)

        if save_table:
            # Replace the Snowflake-backed DataFrame with one reading from the saved
            # Delta table, so downstream use scans Unity Catalog instead of re-querying.
            df = self.save_to_uc(
                df,
                table=table,
                schema=schema,
                catalog=catalog,
                mode=mode,
                optimize=optimize,
                zorder_by=zorder_by,
                merge_schema=merge_schema,
                comment=comment,
                drop_existing=drop_existing,
                overwrite_schema=overwrite_schema,
            )

        if return_pandas:
            return df.toPandas()
        return df

    def save_pipeline_to_uc(
        self,
        query_paths,
        *,
        pipeline: str | None = None,
        catalog: str = None,
        schema: str = None,
        tables: dict[str, str] = None,
        table_prefix: str = "",
        table_suffix: str = "",
        mode: str = "overwrite",
        modes: dict[str, str] = None,
        optimize: bool = True,
        zorder_by=None,
        merge_schema: bool = True,
        comment: str = None,
        comments: dict[str, str] = None,
        drop_existing: bool = True,
        overwrite_schema: bool = True,
        return_all: bool = False,
        **kwargs,
    ):
        """
        Run YAML-ordered Snowflake queries and save each result as a Unity Catalog table.

        This is a convenience wrapper around ``sql(..., save_table=True)``. It
        uses the same folder/YAML resolution as ``execute_sql_scripts``:
        ``steps`` define the SQL files to run and their order.

        Parameters
        ----------
        query_paths
            Folder, file, list, or ordered dict of SQL files.
        pipeline
            Optional YAML pipeline name.
        catalog, schema
            Default Unity Catalog destination for unqualified table names.
        tables
            Optional mapping of step name to destination table. Values may be
            unqualified (using ``catalog`` / ``schema``) or fully qualified.
        table_prefix, table_suffix
            Applied to step names when ``tables`` does not define a destination.
        mode
            Default Spark write mode for every table.
        modes
            Optional mapping of step name to Spark write mode.
        optimize
            If True, run ``OPTIMIZE`` after saving each Unity Catalog table.
        zorder_by
            Optional columns for Delta ``ZORDER BY``. Pass a dict to configure
            columns per step, or a string/list to use the same columns for every
            saved table.
        merge_schema
            If True, set Delta ``mergeSchema=true`` for every saved table.
        comment
            Optional table comment applied to every saved table.
        comments
            Optional mapping of step name to table comment.
        drop_existing
            If True, drop each destination table before writing so it is fully
            recreated from its query result. Defaults to True. Set False to preserve
            tables (required for append modes).
        overwrite_schema
            If True, on overwrite replace each table's schema (drop removed columns)
            via Delta ``overwriteSchema=true``. Defaults to True.
        return_all
            If True, return a dict of step name to Spark DataFrame. Otherwise
            return the last step's Spark DataFrame.
        **kwargs
            Template variables substituted into SQL files via ``str.format()``.
        """
        resolved_paths = resolve_sql_query_paths(query_paths, pipeline=pipeline)
        if not resolved_paths:
            log_and_raise_error(self._logger, "No SQL files found for pipeline.")

        tables = tables or {}
        modes = modes or {}
        comments = comments or {}
        results = {}
        last_df = None

        for name, query_path in resolved_paths.items():
            destination = tables.get(name) or f"{table_prefix}{name}{table_suffix}"
            if not destination:
                log_and_raise_error(self._logger, f"No Unity Catalog table configured for step '{name}'.")

            step_mode = modes.get(name, mode)
            step_zorder_by = zorder_by.get(name) if isinstance(zorder_by, dict) else zorder_by
            step_comment = comments.get(name, comment)
            self._logger.info(f"[{name}] saving to Unity Catalog table {destination} (mode={step_mode}) ...")
            last_df = self.sql(
                str(query_path),
                save_table=True,
                table=destination,
                schema=schema,
                catalog=catalog,
                mode=step_mode,
                optimize=optimize,
                zorder_by=step_zorder_by,
                merge_schema=merge_schema,
                comment=step_comment,
                drop_existing=drop_existing,
                overwrite_schema=overwrite_schema,
                **kwargs,
            )
            results[name] = last_df

        return results if return_all else last_df

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

    @staticmethod
    def _zorder_clause(zorder_by=None) -> str:
        """Build the optional Delta ZORDER BY clause."""
        if not zorder_by:
            return ""
        if isinstance(zorder_by, str):
            columns = [column.strip() for column in zorder_by.split(",")]
        else:
            columns = [str(column).strip() for column in zorder_by]
        columns = [column for column in columns if column]
        if not columns:
            return ""
        return f" ZORDER BY ({', '.join(columns)})"

    @staticmethod
    def _sql_string_literal(value: str) -> str:
        """Escape a value for use inside a single-quoted SQL string literal."""
        return str(value).replace("'", "''")

    def set_uc_table_comment(self, table: str, comment: str, schema: str = None, catalog: str = None, spark=None):
        """
        Set a Unity Catalog table comment using Databricks table properties.

        Parameters
        ----------
        table
            Table name. May be fully qualified.
        comment
            Comment text to store.
        schema, catalog
            Optional qualifiers when ``table`` is not fully qualified.
        spark
            Optional SparkSession to use. Defaults to this connector's Spark session.
        """
        full_name = self._qualified_uc_name(table, schema=schema, catalog=catalog)
        spark = spark or self._get_spark()
        escaped_comment = self._sql_string_literal(comment)
        try:
            spark.sql(f"ALTER TABLE {full_name} SET TBLPROPERTIES ('comment' = '{escaped_comment}')")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error setting comment for Unity Catalog table '{full_name}': {e}")
        self._logger.info(f"Comment set for Unity Catalog table '{full_name}'.")

    def optimize_uc_table(self, table: str, schema: str = None, catalog: str = None, zorder_by=None, spark=None):
        """
        Run Databricks Delta ``OPTIMIZE`` on a Unity Catalog table.

        Parameters
        ----------
        table
            Table name. May be fully qualified.
        schema, catalog
            Optional qualifiers when ``table`` is not fully qualified.
        zorder_by
            Optional column or columns for ``ZORDER BY``.
        spark
            Optional SparkSession to use. Defaults to this connector's Spark session.
        """
        full_name = self._qualified_uc_name(table, schema=schema, catalog=catalog)
        spark = spark or self._get_spark()
        optimize_sql = f"OPTIMIZE {full_name}{self._zorder_clause(zorder_by)}"
        try:
            spark.sql(optimize_sql)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error optimizing Unity Catalog table '{full_name}': {e}")
        self._logger.info(f"Table '{full_name}' optimized.")

    def save_to_uc(
        self,
        df,
        table: str,
        schema: str = None,
        catalog: str = None,
        mode: str = "overwrite",
        optimize: bool = True,
        zorder_by=None,
        merge_schema: bool = True,
        comment: str = None,
        drop_existing: bool = True,
        overwrite_schema: bool = True,
    ):
        """
        Write a Spark DataFrame to a Databricks Unity Catalog table.

        Uses Spark's native ``df.write.saveAsTable(...)`` (a managed UC table),
        not the Snowflake connector. By default, runs Delta ``OPTIMIZE`` after
        the write.

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
        optimize : bool, optional
            If True, run ``OPTIMIZE`` after saving. Defaults to True.
        zorder_by : str or list[str], optional
            Optional columns for Delta ``ZORDER BY`` during optimize.
        merge_schema : bool, optional
            If True, writes as Delta with ``mergeSchema=true`` (used for appends and
            whenever the schema is not being overwritten). Defaults to True. Ignored on
            an overwrite when ``overwrite_schema`` is True (the two are mutually
            exclusive in Delta).
        comment : str, optional
            Optional table comment stored as a Unity Catalog table property.
        drop_existing : bool, optional
            If True, ``DROP TABLE IF EXISTS`` the destination before writing so the
            table is fully recreated from this DataFrame (removed columns disappear,
            and table properties / grants / history are reset). Defaults to True.
            Set to False to preserve the existing table — required for ``mode='append'``,
            which would otherwise drop the table on every call.
        overwrite_schema : bool, optional
            If True and ``mode='overwrite'``, writes with Delta ``overwriteSchema=true``
            so the table schema is replaced (columns absent from this DataFrame are
            dropped) rather than merged. Defaults to True. Has no effect on non-overwrite
            modes.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame reading back from the saved Unity Catalog Delta table
            (``spark.table(full_name)``), not the source DataFrame. Downstream actions
            then scan the fast Delta table instead of re-running the original read.
        """
        if not table:
            log_and_raise_error(self._logger, "A destination table name is required.")

        full_name = self._qualified_uc_name(table, schema=schema, catalog=catalog)
        spark = getattr(df, "sparkSession", None) or self._spark

        if drop_existing:
            try:
                spark.sql(f"DROP TABLE IF EXISTS {full_name}")
            except Exception as e:
                log_and_raise_error(self._logger, f"Error dropping Unity Catalog table '{full_name}': {e}")
            self._logger.info(f"Dropped existing Unity Catalog table '{full_name}' before write.")

        try:
            writer = df.write.format("delta")
            # overwriteSchema and mergeSchema are mutually exclusive in Delta. On an
            # overwrite, replacing the schema (so removed columns are dropped) takes
            # precedence; mergeSchema is used otherwise (e.g. evolving on append).
            if mode == "overwrite" and overwrite_schema:
                writer = writer.option("overwriteSchema", "true")
            elif merge_schema:
                writer = writer.option("mergeSchema", "true")
            writer.mode(mode).saveAsTable(full_name)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error writing to Unity Catalog table '{full_name}': {e}")
        self._logger.info(f"Table '{full_name}' written to Unity Catalog (mode={mode}).")

        if comment is not None:
            self.set_uc_table_comment(full_name, comment, spark=spark)

        if optimize:
            self.optimize_uc_table(full_name, zorder_by=zorder_by, spark=spark)

        # Return a DataFrame backed by the just-written Delta table so callers read
        # from fast Unity Catalog storage rather than re-executing the Snowflake read.
        return spark.table(full_name)
