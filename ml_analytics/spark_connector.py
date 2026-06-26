"""
Generic Spark / Databricks Unity Catalog operations.

This module holds engine-agnostic Spark helpers that don't care where the data
came from: a shared :func:`get_spark` session factory and
:class:`SparkTableManager`, which writes/optimizes/comments/drops Databricks
Unity Catalog Delta tables given any Spark DataFrame.

:class:`ml_analytics.sf_connector.SFConnector` reads from Snowflake and then
delegates its table-management methods here, so the same logic backs both a
plain ``SparkTableManager()`` and ``SFConnector(...).save_to_uc(...)``.

PySpark is intentionally NOT a dependency of this package and is imported
lazily, only when a method that actually needs a Spark session is called. This
keeps the rest of the package usable in environments without Spark installed.
"""

from .utils import get_logger, log_and_raise_error

# Cached Spark session shared across SparkTableManager / SFConnector instances.
# Populated lazily by get_spark(); never created at import time so the package
# stays importable without PySpark.
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
            "A Spark session is needed but neither PySpark nor Databricks Connect "
            "is available. Run on a Spark runtime (e.g. Databricks) or install one "
            "locally with `pip install databricks-connect`."
        ) from exc

    _spark_ctx = SparkSession.builder.appName("ml_analytics").getOrCreate()
    return _spark_ctx


class SparkTableManager:
    """
    Read and manage Databricks Unity Catalog (Delta) tables through Spark.

    This class is source-agnostic: give it any Spark DataFrame and it writes,
    optimizes, comments, or drops the corresponding managed Unity Catalog table.
    It performs no Snowflake/Redshift reads of its own — pair it with
    :class:`ml_analytics.data_connector.DataConnector` or
    :class:`ml_analytics.sf_connector.SFConnector` to produce the DataFrame.

    Parameters
    ----------
    catalog, schema : str, optional
        Default Unity Catalog catalog/schema used to qualify unqualified table
        names. An explicit ``catalog`` / ``schema`` passed to a method, or a
        fully-qualified (dotted) table name, overrides these.
    spark : SparkSession, optional
        Existing Spark session to reuse. If omitted, the active session is used,
        or one is created on first use via :func:`get_spark`.
    logger : logging.Logger, optional
        Logger to use. Defaults to a ``"Spark Table Manager"`` logger.
    """

    def __init__(self, *, catalog=None, schema=None, spark=None, logger=None):
        self.catalog = catalog
        self.schema = schema
        self._spark = spark
        self._logger = logger or get_logger("Spark Table Manager")

    def _get_spark(self):
        """Return the Spark session: the one passed in, or the cached shared one."""
        if self._spark is not None:
            return self._spark
        self._spark = get_spark()
        return self._spark

    def to_spark(self, df, schema=None, spark=None):
        """
        Return a Spark DataFrame, converting from pandas or polars when needed.

        A pandas DataFrame is converted via ``spark.createDataFrame``; a polars
        DataFrame is first turned into pandas. Anything else (already a Spark
        DataFrame) is returned unchanged, so no Spark session is created for the
        passthrough case.

        Parameters
        ----------
        df
            A pandas, polars, or Spark DataFrame.
        schema
            Optional explicit Spark schema for the conversion, passed straight to
            ``spark.createDataFrame(..., schema=schema)``. Accepts a
            ``pyspark.sql.types.StructType`` or a DDL-style string (e.g.
            ``"id long, name string"``). Use this when Spark's automatic type
            inference is lossy (all-null columns, mixed dtypes, tz-aware datetimes).
            Ignored when ``df`` is already a Spark DataFrame.
        spark
            Optional SparkSession to use for the conversion. Defaults to this
            manager's Spark session.
        """
        import pandas as pd

        is_pandas = isinstance(df, pd.DataFrame)
        is_polars = False
        try:
            import polars as pl

            is_polars = isinstance(df, pl.DataFrame)
        except ImportError:
            pass

        if not (is_pandas or is_polars):
            # Assume it's already a Spark DataFrame; don't spin up a session.
            return df

        spark = spark or self._get_spark()
        pandas_df = df.to_pandas() if is_polars else df
        try:
            if schema is not None:
                return spark.createDataFrame(pandas_df, schema=schema)
            return spark.createDataFrame(pandas_df)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error converting DataFrame to Spark: {e}")

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

    def _resolve_uc_name(self, table: str, schema: str = None, catalog: str = None) -> str:
        """Qualify ``table``, falling back to the manager's default schema/catalog."""
        return self._qualified_uc_name(
            table,
            schema=schema if schema is not None else self.schema,
            catalog=catalog if catalog is not None else self.catalog,
        )

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
            Optional SparkSession to use. Defaults to this manager's Spark session.
        """
        full_name = self._resolve_uc_name(table, schema=schema, catalog=catalog)
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
            Optional SparkSession to use. Defaults to this manager's Spark session.
        """
        full_name = self._resolve_uc_name(table, schema=schema, catalog=catalog)
        spark = spark or self._get_spark()
        optimize_sql = f"OPTIMIZE {full_name}{self._zorder_clause(zorder_by)}"
        try:
            spark.sql(optimize_sql)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error optimizing Unity Catalog table '{full_name}': {e}")
        self._logger.info(f"Table '{full_name}' optimized.")

    def drop_table(self, table: str, schema: str = None, catalog: str = None, spark=None):
        """
        ``DROP TABLE IF EXISTS`` a Unity Catalog table.

        Parameters
        ----------
        table
            Table name. May be fully qualified.
        schema, catalog
            Optional qualifiers when ``table`` is not fully qualified.
        spark
            Optional SparkSession to use. Defaults to this manager's Spark session.
        """
        full_name = self._resolve_uc_name(table, schema=schema, catalog=catalog)
        spark = spark or self._get_spark()
        try:
            spark.sql(f"DROP TABLE IF EXISTS {full_name}")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error dropping Unity Catalog table '{full_name}': {e}")
        self._logger.info(f"Dropped Unity Catalog table '{full_name}'.")

    def sql(self, query: str, return_pandas: bool = False, spark=None):
        """
        Run a Spark SQL statement via ``spark.sql(query)`` and return the result.

        Use this for arbitrary Spark/Unity Catalog SQL (SELECT, DDL, MERGE, ...).
        Unlike :class:`ml_analytics.sf_connector.SFConnector.sql`, this runs the
        query on the Spark engine itself, not against Snowflake.

        Parameters
        ----------
        query
            The Spark SQL statement to execute.
        return_pandas
            If True, return a pandas DataFrame; otherwise return the Spark
            DataFrame. Defaults to False.
        spark
            Optional SparkSession to use. Defaults to this manager's Spark session.
        """
        spark = spark or self._get_spark()
        try:
            df = spark.sql(query)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error executing Spark SQL: {e}")
        return df.toPandas() if return_pandas else df

    def read_table(self, table: str, schema: str = None, catalog: str = None, spark=None):
        """
        Read a Unity Catalog table into a Spark DataFrame via ``spark.table(...)``.

        Parameters
        ----------
        table
            Table name. May be fully qualified.
        schema, catalog
            Optional qualifiers when ``table`` is not fully qualified.
        spark
            Optional SparkSession to use. Defaults to this manager's Spark session.
        """
        full_name = self._resolve_uc_name(table, schema=schema, catalog=catalog)
        spark = spark or self._get_spark()
        try:
            return spark.table(full_name)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error reading Unity Catalog table '{full_name}': {e}")

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
        spark_schema=None,
    ):
        """
        Write a Spark DataFrame to a Databricks Unity Catalog table.

        Uses Spark's native ``df.write.saveAsTable(...)`` (a managed UC table).
        By default, runs Delta ``OPTIMIZE`` after the write.

        Parameters
        ----------
        df : pandas, polars, or pyspark.sql.DataFrame
            DataFrame to write. pandas/polars frames are converted to Spark first
            via :meth:`to_spark`.
        table : str
            Destination table name. May be fully qualified
            (``catalog.schema.table``), in which case ``schema`` / ``catalog``
            are ignored.
        schema, catalog : str, optional
            Unity Catalog schema and catalog to qualify ``table`` with. Fall back
            to the manager's defaults.
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
        spark_schema : optional
            Explicit Spark schema used only when ``df`` is a pandas/polars frame that
            must be converted to Spark. Passed to :meth:`to_spark`. Accepts a
            ``StructType`` or a DDL string. Ignored when ``df`` is already a Spark
            DataFrame.

        Returns
        -------
        pyspark.sql.DataFrame
            A DataFrame reading back from the saved Unity Catalog Delta table
            (``spark.table(full_name)``), not the source DataFrame. Downstream actions
            then scan the fast Delta table instead of re-running the original read.
        """
        if not table:
            log_and_raise_error(self._logger, "A destination table name is required.")

        full_name = self._resolve_uc_name(table, schema=schema, catalog=catalog)
        spark = getattr(df, "sparkSession", None) or self._spark or self._get_spark()
        # Accept pandas/polars DataFrames too, converting them to Spark first.
        df = self.to_spark(df, schema=spark_schema, spark=spark)

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
        # from fast Unity Catalog storage rather than re-executing the source read.
        return spark.table(full_name)
