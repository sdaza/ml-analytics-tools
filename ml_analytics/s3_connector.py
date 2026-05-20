"""
Generic utility functions for S3 data operations.
"""

import io
import os
import re
from functools import singledispatchmethod

import boto3
import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
from botocore.exceptions import ClientError, NoCredentialsError

from .utils import get_logger, log_and_raise_error


class S3Connector:
    def __init__(
        self, bucket=None, s3_root=None, prefix=None, log_level="INFO", auto_sso_login=False, sso_profile=None
    ):
        """
        Initialize an S3Connector instance.

        Parameters
        ----------
        bucket : str, optional
            The name of the S3 bucket to use for file storage. If omitted, uses ML_ANALYTICS_S3_BUCKET.
        s3_root : str, optional
            The root directory in the S3 bucket where files will be stored. Defaults to an empty string.
        auto_sso_login : bool, optional
            If True, automatically attempt AWS SSO login when credentials are missing. Default is False.
        sso_profile : str, optional
            AWS SSO profile to use for auto-login.
        """

        self._s3_prefix = prefix or "s3://"
        configured_bucket = bucket or os.getenv("ML_ANALYTICS_S3_BUCKET")
        self._bucket = configured_bucket.rstrip("/").lstrip("/") if configured_bucket else None
        self._s3_root = s3_root or ""
        self._s3_root = self._s3_root.rstrip("/").lstrip("/")
        self._logger = get_logger("S3 Connector")
        self._logger.setLevel(log_level)
        self._auto_sso_login = auto_sso_login
        self._sso_profile = sso_profile
        self._duckdb_conn = None
        self._registered_tables = set()

        bucket_label = self._bucket or "<not configured>"
        self._logger.info(f"S3 set to {bucket_label}/{self.s3_root}")
        self.initialize_s3_client()

    def _resolve_bucket(self, bucket: str = None) -> str:
        resolved_bucket = bucket or self._bucket
        if not resolved_bucket:
            log_and_raise_error(
                self._logger,
                "No S3 bucket configured. Pass bucket=... or set ML_ANALYTICS_S3_BUCKET.",
            )
        return resolved_bucket.rstrip("/").lstrip("/")

    def initialize_s3_client(self):
        try:
            # Attempt to create an S3 client
            self.s3 = boto3.Session().client("s3")
            self._logger.info("S3 client initialized")
        except NoCredentialsError as e:
            if self._auto_sso_login:
                self._logger.warning("AWS credentials not found. Attempting SSO login...")
                from .aws_auth import ensure_aws_sso_login

                if ensure_aws_sso_login(self._sso_profile):
                    # Retry initialization after SSO login
                    try:
                        self.s3 = boto3.Session().client("s3")
                        self._logger.info("S3 client initialized after SSO login")
                        return
                    except Exception as retry_error:
                        log_and_raise_error(
                            self._logger, f"Error initializing S3 client after SSO login: {retry_error}"
                        )
                else:
                    log_and_raise_error(
                        self._logger,
                        "AWS SSO login failed. Please run 'aws sso login' manually or use ensure_aws_authenticated().",
                    )
            else:
                log_and_raise_error(
                    self._logger,
                    f"AWS credentials not found: {e}. Either run 'aws sso login' or set auto_sso_login=True.",
                )
        except Exception as e:
            log_and_raise_error(self._logger, f"Error initializing S3 client: {e}")

    @singledispatchmethod
    def save_dataframe(
        self, df, directory=None, file_name=None, file_format="parquet", bucket=None, s3_root=None, column_types=None
    ):
        """
        Save a DataFrame to S3 as CSV or Parquet.

        Parameters
        ----------
        df : pd.DataFrame or pl.DataFrame
            The DataFrame to save. MUST be passed as a positional argument (not df=df).
        directory : str, optional
            The directory path where the file will be saved.
        file_name : str
            The name of the file (without extension).
        file_format : str, optional
            File format to save ('csv' or 'parquet'). Default is 'parquet'.
        bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.
        s3_root : str, optional
            The root directory in the S3 bucket. Defaults to the instance's s3_root.
        column_types : dict[str, str], optional
            For parquet format: Redshift column types to ensure schema compatibility.
            Keys are column names, values are Redshift types.
            Example: {'hire_date': 'DATE', 'user_id': 'BIGINT', 'salary': 'DECIMAL(12,2)'}

        Examples
        --------
        >>> s3 = S3Connector()
        >>> df = pd.DataFrame({'id': [1, 2, 3], 'name': ['a', 'b', 'c']})
        >>> s3.save_dataframe(df, directory='data', file_name='test')  # ✓ Correct
        >>> # s3.save_dataframe(df=df, file_name='test')  # ✗ Wrong - will raise error
        >>>
        >>> # With Redshift column types for parquet
        >>> column_types = {'hire_date': 'DATE', 'salary': 'DECIMAL(12,2)', 'user_id': 'BIGINT'}
        >>> s3.save_dataframe(df, file_name='employees', column_types=column_types)

        Raises
        ------
        ValueError
            If the DataFrame type is unsupported.

        Notes
        -----
        The df parameter MUST be positional because save_dataframe uses @singledispatchmethod
        for type dispatch between pandas and polars DataFrames.
        """
        raise ValueError("Unsupported DataFrame type. Use pandas or polars.")

    @save_dataframe.register
    def _(
        self,
        df: pd.DataFrame,
        directory=None,
        file_name=None,
        file_format="parquet",
        bucket=None,
        s3_root=None,
        column_types=None,
    ):
        """
        Save a pandas DataFrame to an S3 bucket as a CSV or Parquet file.
        """
        if file_format not in ["csv", "parquet"]:
            log_and_raise_error(self._logger, "Invalid file format. Choose 'csv' or 'parquet'.")
        if df.shape[0] == 0:
            log_and_raise_error(self._logger, "DataFrame is empty!")
        bucket = self._resolve_bucket(bucket)
        if s3_root is None:
            s3_root = self._s3_root
        if file_name is None:
            log_and_raise_error(self._logger, "No file_name provided")
        if directory is None:
            directory = ""

        if file_format == "csv" and file_name.endswith(".csv"):
            file_name = file_name[:-4]
        if file_format == "parquet" and file_name.endswith(".parquet"):
            file_name = file_name[:-8]

        s3_path = f"s3://{bucket}/{s3_root}/{directory}/{file_name}.{file_format}"
        s3_path = self._normalize_s3_path(s3_path)

        try:
            if file_format == "csv":
                df.to_csv(s3_path, index=False)
            elif file_format == "parquet":
                import pyarrow.parquet as pq

                # Prepare DataFrame for Redshift using helper methods
                df_copy = self._prepare_pandas_for_redshift(df, column_types)

                # Build PyArrow table with Redshift-compatible schema
                table = self._build_redshift_pyarrow_schema(df_copy, column_types)

                # Write with pyarrow
                pq.write_table(table, s3_path, compression="snappy")
            self._logger.debug(f"Saved pandas DataFrame to {s3_path}")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error saving pandas DataFrame to S3: {e}")

    @save_dataframe.register
    def _(
        self,
        df: pl.DataFrame,
        directory=None,
        file_name=None,
        file_format="parquet",
        bucket=None,
        s3_root=None,
        column_types=None,
    ):
        """
        Save a polars DataFrame to an S3 bucket as a CSV or Parquet file.
        """
        if file_format not in ["csv", "parquet"]:
            log_and_raise_error(self._logger, "Invalid file format. Choose 'csv' or 'parquet'.")
        if df.height == 0:
            log_and_raise_error(self._logger, "DataFrame is empty!")
        bucket = self._resolve_bucket(bucket)
        if s3_root is None:
            s3_root = self._s3_root
        if file_name is None:
            log_and_raise_error(self._logger, "No file_name provided")
        if directory is None:
            directory = ""

        if file_format == "csv" and file_name.endswith(".csv"):
            file_name = file_name[:-4]
        if file_format == "parquet" and file_name.endswith(".parquet"):
            file_name = file_name[:-8]

        s3_path = f"s3://{bucket}/{s3_root}/{directory}/{file_name}.{file_format}"
        s3_path = self._normalize_s3_path(s3_path)

        try:
            if file_format == "csv":
                df.write_csv(s3_path)
            elif file_format == "parquet":
                # Prepare DataFrame for Redshift-compatible Parquet using helper method
                df_copy = self._prepare_polars_for_redshift(df, column_types)
                df_copy.write_parquet(s3_path)
            self._logger.debug(f"Saved polars DataFrame to {s3_path}")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error saving polars DataFrame to S3: {e}")

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse an S3 path (e.g. s3://bucket/key) into (bucket, key)."""
        path = re.sub(r"^s3[a-z]*://", "", s3_path)
        bucket, _, key = path.partition("/")
        return bucket, key

    def _download_s3_to_buffer(self, bucket: str, key: str) -> io.BytesIO:
        """Download an S3 object into a BytesIO buffer using the boto3 client."""
        response = self.s3.get_object(Bucket=bucket, Key=key)
        buffer = io.BytesIO(response["Body"].read())
        buffer.seek(0)
        return buffer

    def _list_parquet_keys(self, bucket: str, prefix: str) -> list[str]:
        """List all .parquet file keys under an S3 prefix using the boto3 client."""
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".parquet"):
                    keys.append(key)
        return keys

    def read_parquet(
        self, file_path: str, bucket: str = None, s3_root: str = None, to_polars=False
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Read a Parquet file from S3 and return it as a pandas or polars DataFrame.

        Parameters
        ----------

        file_path : str
            The S3 path to the Parquet file (should end with .parquet).
        bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.
        s3_root : str, optional
            The root directory in the S3 bucket where files are stored. Defaults to the instance's s3_root.
        to_polars : bool, optional
            If True, returns a polars DataFrame. If False, returns a pandas DataFrame. Defaults to False.

        Returns
        -------
        Union[pd.DataFrame, pl.DataFrame]
            The DataFrame read from the Parquet file.
        """

        if not file_path:
            log_and_raise_error(self._logger, "No file_path provided")

        bucket = self._resolve_bucket(bucket)
        if s3_root is None:
            s3_root = self._s3_root

        file_path = self.get_path(file_path, s3_root=s3_root, bucket=bucket)

        if file_path.endswith(".parquet/"):
            file_path = file_path[:-1]

        is_directory = not file_path.rstrip("/").lower().endswith(".parquet")

        # Use boto3 client to download data, avoiding PyArrow/Polars credential resolution
        bucket_name, key_prefix = self._parse_s3_path(file_path)

        try:
            import pyarrow.parquet as pq

            if is_directory:
                parquet_keys = self._list_parquet_keys(bucket_name, key_prefix)
                if not parquet_keys:
                    log_and_raise_error(self._logger, f"No parquet files found under {file_path}")

                tables = []
                for key in parquet_keys:
                    buf = self._download_s3_to_buffer(bucket_name, key)
                    tables.append(pq.read_table(buf))
                table = pa.concat_tables(tables)
                schema = table.schema
            else:
                buffer = self._download_s3_to_buffer(bucket_name, key_prefix)
                parquet_file = pq.ParquetFile(buffer)
                schema = parquet_file.schema_arrow

            # Check for pandas extension types in the schema
            has_extension_types = False

            # Check pandas metadata from the schema
            pandas_metadata = schema.pandas_metadata
            if pandas_metadata:
                for col_meta in pandas_metadata.get("columns", []):
                    numpy_type = col_meta.get("numpy_type", "")
                    # Period types show up in numpy_type as 'period[X]'
                    if numpy_type.startswith("period"):
                        has_extension_types = True
                        self._logger.debug(f"Found pandas period type: {numpy_type}")
                        break

            # Also check field types directly for extension types
            if not has_extension_types:
                for field in schema:
                    field_type = field.type
                    # Check if it has extension_name attribute (pandas extension types)
                    if hasattr(field_type, "extension_name") and field_type.extension_name:
                        has_extension_types = True
                        self._logger.debug(f"Found extension type in field {field.name}: {field_type.extension_name}")
                        break

            if has_extension_types:
                self._logger.debug("Detected pandas extension types, using PyArrow conversion")
                if not is_directory:
                    buffer.seek(0)
                    table = pq.read_table(buffer)
                df_pandas = table.to_pandas()

                # Convert Period columns to datetime
                for col in df_pandas.columns:
                    if isinstance(df_pandas[col].dtype, pd.PeriodDtype):
                        self._logger.debug(f"Converting Period column '{col}' to datetime")
                        df_pandas[col] = df_pandas[col].dt.to_timestamp()

                if to_polars:
                    return pl.from_pandas(df_pandas)
                else:
                    return df_pandas
            else:
                # No extension types detected, safe to use Polars
                if is_directory:
                    df_polars = pl.from_arrow(table)
                else:
                    buffer.seek(0)
                    df_polars = pl.read_parquet(buffer)
                if to_polars:
                    return df_polars
                else:
                    return df_polars.to_pandas()

        except Exception as e:
            # Final fallback: use PyArrow via boto3
            self._logger.warning(f"Error with initial read attempt, falling back to PyArrow: {e}")
            try:
                import pyarrow.parquet as pq

                if is_directory:
                    parquet_keys = self._list_parquet_keys(bucket_name, key_prefix)
                    tables = []
                    for key in parquet_keys:
                        buf = self._download_s3_to_buffer(bucket_name, key)
                        tables.append(pq.read_table(buf))
                    table = pa.concat_tables(tables)
                else:
                    fallback_buffer = self._download_s3_to_buffer(bucket_name, key_prefix)
                    table = pq.read_table(fallback_buffer)
                df_pandas = table.to_pandas()

                # Convert any Period columns
                for col in df_pandas.columns:
                    if isinstance(df_pandas[col].dtype, pd.PeriodDtype):
                        self._logger.debug(f"Converting Period column '{col}' to datetime")
                        df_pandas[col] = df_pandas[col].dt.to_timestamp()

                if to_polars:
                    return pl.from_pandas(df_pandas)
                else:
                    return df_pandas
            except Exception as final_error:
                log_and_raise_error(self._logger, f"Error reading Parquet file from S3: {final_error}")

    # ===== Redshift Type Validation Helper Methods =====

    def _get_base_redshift_type(self, type_string: str) -> str:
        """Extract base type from Redshift type string (e.g., 'DECIMAL(12,2)' -> 'DECIMAL')."""
        return type_string.split("(")[0].strip().upper()

    def _is_date_type(self, base_type: str) -> bool:
        """Check if type is a date/timestamp type."""
        return base_type in ("DATE", "TIMESTAMP", "TIMESTAMPTZ")

    def _is_integer_type(self, base_type: str) -> bool:
        """Check if type is an integer type."""
        return base_type in ("BIGINT", "INT", "INTEGER", "SMALLINT")

    def _is_float_type(self, base_type: str) -> bool:
        """Check if type is a float/decimal type."""
        return base_type in ("DOUBLE PRECISION", "DOUBLE", "REAL", "FLOAT", "NUMERIC", "DECIMAL")

    def _is_string_type(self, base_type: str) -> bool:
        """Check if type is a string type."""
        return base_type in ("VARCHAR", "CHAR", "TEXT")

    def _get_target_int_type(self, base_type: str) -> str:
        """Get target pandas integer type based on Redshift type."""
        return "int64" if base_type == "BIGINT" else "int32"

    def _prepare_pandas_for_redshift(self, df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
        """
        Prepare pandas DataFrame for Redshift by converting column types.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to prepare.
        column_types : dict
            Dictionary mapping column names to Redshift types.

        Returns
        -------
        pd.DataFrame
            DataFrame with types converted for Redshift compatibility.
        """
        df_copy = df.copy()

        if column_types:
            for col in df.columns:
                if col in column_types:
                    base_type = self._get_base_redshift_type(column_types[col])
                    col_dtype = df_copy[col].dtype

                    # Date/timestamp types
                    if self._is_date_type(base_type):
                        if df_copy[col].isna().all():
                            df_copy[col] = pd.NaT
                        else:
                            df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")

                    # Integer types
                    elif self._is_integer_type(base_type):
                        target_int_type = self._get_target_int_type(base_type)

                        if df_copy[col].isna().all():
                            df_copy[col] = pd.Series([None] * len(df_copy), dtype=object)
                        elif df_copy[col].isna().any():
                            df_copy[col] = df_copy[col].astype("float64")
                            self._logger.debug(
                                f"Column {col} has NULLs - converting to float64 temporarily for Redshift compatibility"
                            )
                        else:
                            df_copy[col] = df_copy[col].astype(target_int_type)

                    # Float/decimal types
                    elif self._is_float_type(base_type):
                        if col_dtype == "object":
                            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
                        else:
                            df_copy[col] = df_copy[col].astype("float64")

                    # String/VARCHAR types
                    elif self._is_string_type(base_type):
                        if col_dtype != "object":
                            df_copy[col] = df_copy[col].astype(str)
                            df_copy[col] = df_copy[col].replace("nan", None)
                        else:
                            df_copy[col] = df_copy[col].fillna("").infer_objects(copy=False).astype(str)
                            df_copy[col] = df_copy[col].replace("", None)

        # Fallback: handle categorical and nullable integer types
        else:
            for col in df_copy.columns:
                col_dtype = df_copy[col].dtype

                if isinstance(col_dtype, pd.CategoricalDtype):
                    df_copy[col] = df_copy[col].astype(str)
                    df_copy[col] = df_copy[col].replace("nan", None)
                elif hasattr(col_dtype, "name") and "Int" in str(col_dtype):
                    if df_copy[col].isna().any():
                        df_copy[col] = df_copy[col].astype("float64")
                    else:
                        df_copy[col] = df_copy[col].astype("int64")
                elif col_dtype == "object":
                    df_copy[col] = df_copy[col].fillna("").infer_objects(copy=False).astype(str)
                    df_copy[col] = df_copy[col].replace("", None)

        return df_copy

    def _build_redshift_pyarrow_schema(self, df: pd.DataFrame, column_types: dict) -> pa.Table:
        """
        Build PyArrow table with Redshift-compatible schema.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to convert.
        column_types : dict
            Dictionary mapping column names to Redshift types.

        Returns
        -------
        pa.Table
            PyArrow table with Redshift-compatible types.
        """
        self._logger.debug(f"DataFrame columns before PyArrow conversion: {list(df.columns)}")
        self._logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")

        table = pa.Table.from_pandas(df, preserve_index=False)

        self._logger.debug(f"PyArrow table columns: {table.column_names}")

        # Rebuild schema to enforce Redshift types
        new_fields = []
        for field in table.schema:
            if column_types and field.name in column_types:
                base_type = self._get_base_redshift_type(column_types[field.name])

                # Handle all-null columns
                if pa.types.is_null(field.type):
                    if self._is_string_type(base_type):
                        new_fields.append(pa.field(field.name, pa.string()))
                        self._logger.debug(f"Converting all-null {field.name} from null to string for {base_type}")
                    elif self._is_integer_type(base_type):
                        int_type = pa.int64() if base_type == "BIGINT" else pa.int32()
                        new_fields.append(pa.field(field.name, int_type))
                        self._logger.debug(f"Converting all-null {field.name} from null to {int_type} for {base_type}")
                    elif self._is_float_type(base_type):
                        new_fields.append(pa.field(field.name, pa.float64()))
                        self._logger.debug(f"Converting all-null {field.name} from null to float64 for {base_type}")
                    elif self._is_date_type(base_type):
                        if base_type == "DATE":
                            new_fields.append(pa.field(field.name, pa.date32()))
                        else:
                            new_fields.append(pa.field(field.name, pa.timestamp("us")))
                        self._logger.debug(f"Converting all-null {field.name} from null to {base_type.lower()}")
                    else:
                        new_fields.append(pa.field(field.name, pa.string()))
                        self._logger.debug(f"Converting all-null {field.name} from null to string (default)")

                # Handle DATE type
                elif base_type == "DATE":
                    new_fields.append(pa.field(field.name, pa.date32()))
                    self._logger.debug(f"Converting {field.name} from {field.type} to date32 for DATE")

                # Handle INT vs BIGINT
                elif self._is_integer_type(base_type):
                    if base_type == "BIGINT":
                        if pa.types.is_integer(field.type) and field.type != pa.int64():
                            new_fields.append(pa.field(field.name, pa.int64()))
                            self._logger.debug(f"Converting {field.name} from {field.type} to int64 for BIGINT")
                        else:
                            new_fields.append(field)
                    else:
                        if pa.types.is_integer(field.type) and field.type != pa.int32():
                            new_fields.append(pa.field(field.name, pa.int32()))
                            self._logger.debug(f"Converting {field.name} from {field.type} to int32 for {base_type}")
                        else:
                            new_fields.append(field)

                # Handle large_string/large_binary
                elif pa.types.is_large_string(field.type):
                    new_fields.append(pa.field(field.name, pa.string()))
                elif pa.types.is_large_binary(field.type):
                    new_fields.append(pa.field(field.name, pa.binary()))
                else:
                    new_fields.append(field)

            # No column_types - just fix large types
            elif pa.types.is_large_string(field.type):
                new_fields.append(pa.field(field.name, pa.string()))
            elif pa.types.is_large_binary(field.type):
                new_fields.append(pa.field(field.name, pa.binary()))
            else:
                new_fields.append(field)

        new_schema = pa.schema(new_fields)

        # Cast to new schema if changed
        if new_schema != table.schema:
            self._logger.debug("Converting schema to Redshift-compatible types")
            arrays = []
            for i, field in enumerate(new_schema):
                old_field = table.schema[i]
                if pa.types.is_null(old_field.type) and not pa.types.is_null(field.type):
                    arrays.append(pa.nulls(len(table), type=field.type))
                else:
                    arrays.append(table.column(i).cast(field.type))
            table = pa.Table.from_arrays(arrays, schema=new_schema)

        self._logger.debug(f"Final Parquet schema: {table.schema}")
        return table

    def _prepare_polars_for_redshift(self, df: pl.DataFrame, column_types: dict) -> pl.DataFrame:
        """
        Prepare polars DataFrame for Redshift by converting column types.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to prepare.
        column_types : dict
            Dictionary mapping column names to Redshift types.

        Returns
        -------
        pl.DataFrame
            DataFrame with types converted for Redshift compatibility.
        """
        df_copy = df.clone()

        if column_types:
            for col in df.columns:
                if col in column_types:
                    base_type = self._get_base_redshift_type(column_types[col])

                    try:
                        # Date types
                        if self._is_date_type(base_type):
                            df_copy = df_copy.with_columns(pl.col(col).str.strptime(pl.Date, "%Y-%m-%d"))

                        # Integer types
                        elif self._is_integer_type(base_type):
                            target_type = pl.Int64 if base_type == "BIGINT" else pl.Int32
                            df_copy = df_copy.with_columns(pl.col(col).cast(target_type))

                        # Float types
                        elif self._is_float_type(base_type):
                            df_copy = df_copy.with_columns(pl.col(col).cast(pl.Float64))

                        # String types
                        elif self._is_string_type(base_type):
                            df_copy = df_copy.with_columns(pl.col(col).cast(pl.Utf8))

                    except Exception as e:
                        self._logger.debug(f"Could not convert column {col} to {base_type}: {e}")

        return df_copy

    def list_files(self, prefix: str = None, bucket: str = None, s3_root: str = None) -> list[str]:
        """
        List files in the specified S3 bucket. Optionally filter by a prefix.

        Parameters
        ----------
        prefix: str, optional
            The prefix to filter files. Defaults to ''.
        bucket: str
            The S3 bucket name.
        s3_root: str, optional
            The root directory in the S3 bucket where files are stored. Defaults to the instance's s3_root.

        Returns
        -------
        list[str]
            A list of file keys in the specified bucket (and prefix).
        """

        if prefix is None:
            prefix = ""
            prefix = prefix.rstrip("/")
        bucket = self._resolve_bucket(bucket)
        if s3_root is None:
            s3_root = self._s3_root

        if not prefix.startswith(s3_root):
            prefix = f"{s3_root}/{prefix}" if s3_root else prefix

        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if "Contents" in response:
                files = [obj["Key"] for obj in response["Contents"]]
            else:
                files = []
            self._logger.debug("Found %d files in bucket %s with prefix '%s'", len(files), bucket, prefix)
            return files
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchBucket":
                self._logger.debug(
                    "Bucket %s does not exist while listing prefix '%s' — treating as no files.", bucket, prefix
                )
                return []
            # Otherwise re-raise via the common helper
            log_and_raise_error(self._logger, f"Error listing files from S3: {e}")
        except Exception as e:
            log_and_raise_error(self._logger, f"Error listing files from S3: {e}")

    def _normalize_s3_path(self, path: str) -> str:
        has_s3_prefix = path.startswith("s3://")
        if has_s3_prefix:
            path_wo_prefix = path[5:]
        else:
            path_wo_prefix = path
        # Replace multiple slashes with a single slash (excluding the s3:// part)
        path_wo_prefix = re.sub(r"/+", "/", path_wo_prefix)
        normalized = path_wo_prefix.lstrip("/")
        return f"s3://{normalized}" if has_s3_prefix else normalized

    def get_path(self, relative_path: str, s3_root: str = None, bucket: str = None) -> str:
        """
        Get the S3 path for a given file.

        Parameters
        ----------
        relative_path : str
            The directory path where the file is stored
        s3_root : str, optional
            The root directory in the S3 bucket where files are stored. Defaults to the instance's s3_root.
        bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.

        Returns
        -------
        str
            The S3 path for the specified file.
        """

        if relative_path is None:
            relative_path = ""
        bucket = self._resolve_bucket(bucket)
        if s3_root is None:
            s3_root = self._s3_root

        normalized = self._normalize_s3_path(f"{self._s3_prefix}{bucket}/{s3_root}/{relative_path}/")
        if normalized.endswith(".parquet/") | normalized.endswith(".csv/"):
            normalized = normalized[:-1]
        return normalized

    def delete_file(self, key=None, bucket=None, s3_root=None):
        """
        Delete a file from the specified S3 bucket.

        Parameters
        ----------
        key : str
            The key of the file to delete.
        bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.
        s3_root : str, optional
            The root directory in the S3 bucket where files are stored. Defaults to the instance's s3_root.

        Returns
        -------
        None
        """

        bucket = self._resolve_bucket(bucket)
        if s3_root is None:
            s3_root = self._s3_root

        if key is None:
            log_and_raise_error(self._logger, "No key provided for deletion")
        if not key.startswith(s3_root):
            key = f"{s3_root}/{key}" if s3_root else key
        if not key.startswith("/"):
            key = f"/{key}"
        key = self._normalize_s3_path(key)

        try:
            # AWS Delete is idempotent; attempting to delete a missing key
            # normally succeeds. We want to be tolerant: do not raise on
            # missing objects — only log a warning and return. Normalize the
            # key for AWS API calls, probe existence, and proceed to delete
            # only if the object is present.
            aws_key = key
            if aws_key.startswith("s3://"):
                aws_key = aws_key[5:]
            if aws_key.startswith(f"{bucket}/"):
                aws_key = aws_key[len(bucket) + 1 :]
            aws_key = aws_key.lstrip("/")

            try:
                self.s3.head_object(Bucket=bucket, Key=aws_key)
            except ClientError as ce:
                code = ce.response.get("Error", {}).get("Code", "")
                # Treat missing object/bucket as a non-exceptional condition
                # for delete: log a warning and return to the caller.
                if code in ("404", "NoSuchKey", "NotFound"):
                    self._logger.warning("File not found for deletion: %s (bucket=%s)", key, bucket)
                    return
                if code == "NoSuchBucket":
                    self._logger.warning("Bucket not found while deleting %s: %s", key, bucket)
                    return
                # Non-trivial ClientError: escalate
                log_and_raise_error(self._logger, f"Error checking object existence: {ce}")

            # Proceed to delete now that head_object confirmed the object exists
            self.s3.delete_object(Bucket=bucket, Key=aws_key)
            # Keep a clear confirmation for successful deletions
            self._logger.debug("Deleted file %s from bucket %s", key, bucket)
        except Exception as e:
            log_and_raise_error(self._logger, f"Error deleting file from S3: {e}")

    def list_partition_paths(self, prefix: str, depth: int, bucket: str = None) -> list[str]:
        """
        Lists S3 "directory" paths that could represent partitions.
        This method iteratively lists common prefixes (simulating directories)
        up to the specified depth.

        Parameters
        ----------
        prefix : str
            The S3 prefix to start searching from (e.g., 'my_table_data/').
            Ensure it ends with a '/' if it's a base path.
        depth : int
            The number of partition levels to discover (e.g., 2 for year=/month=).
        bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.

        Returns
        -------
        list[str]
            A list of S3 prefixes that are potential partition paths.
            Example: ['my_table_data/year=2023/month=01/', 'my_table_data/year=2023/month=02/']
        """
        bucket = self._resolve_bucket(bucket)
        prefix = self._normalize_s3_path(prefix)
        if not prefix.endswith("/") and prefix:
            prefix += "/"

        if depth <= 0:
            return []

        current_level_prefixes = [prefix]
        discovered_partition_paths = []

        for current_depth in range(depth):
            next_level_prefixes = []
            for current_prefix in current_level_prefixes:
                try:
                    paginator = self.s3.get_paginator("list_objects_v2")
                    for page in paginator.paginate(Bucket=bucket, Prefix=current_prefix, Delimiter="/"):
                        for common_prefix in page.get("CommonPrefixes", []):
                            path = common_prefix.get("Prefix")
                            if path:
                                path_segment = path.rstrip("/").split("/")[-1]
                                if "=" in path_segment:
                                    if current_depth == depth - 1:
                                        discovered_partition_paths.append(path)
                                    else:
                                        next_level_prefixes.append(path)
                                else:
                                    self._logger.debug(f"Skipping non-partition-like path: {path}")
                except Exception as e:
                    self._logger.error(f"Error listing S3 objects for prefix {current_prefix} in bucket {bucket}: {e}")
                    continue
            current_level_prefixes = next_level_prefixes
            if not current_level_prefixes:
                break

        return discovered_partition_paths

    def list_files_in_prefix(self, prefix: str, bucket: str = None) -> list[str]:
        """
        List all files (objects) directly under a given S3 prefix, non-recursively.
        This is useful for finding a file within a specific "directory" for schema inference.

        Parameters
        ----------
        prefix : str
            The S3 prefix (directory) to list files from. Should end with '/'.
        bucket : str, optional
            The S3 bucket name. Defaults to the instance's bucket.

        Returns
        -------
        list[str]
            A list of file keys (full paths from bucket root) found directly under the prefix.
        """
        bucket = self._resolve_bucket(bucket)
        prefix = self._normalize_s3_path(prefix)
        if not prefix.endswith("/") and prefix:
            prefix += "/"

        files = []
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
                for obj in page.get("Contents", []):
                    if obj.get("Key") != prefix:
                        files.append(obj.get("Key"))
        except Exception as e:
            log_and_raise_error(self._logger, f"Error listing files in prefix {prefix} from S3: {e}")
        return files

    def set_bucket(self, bucket: str):
        """
        Set the S3 bucket for the connector.

        Parameters
        ----------
        bucket : str
            The name of the S3 bucket to use.
        """
        self._bucket = self._resolve_bucket(bucket)
        self._logger.info(f"S3 bucket set to {self._bucket}")

    def set_s3_root(self, s3_root: str):
        """
        Set the S3 root directory for the connector.

        Parameters
        ----------
        s3_root : str
            The root directory in the S3 bucket where files will be stored.
        """
        self._s3_root = s3_root
        self._logger.info(f"S3 root set to {self._s3_root}")

    def set_s3_prefix(self, prefix: str):
        """
        Set the S3 prefix for the connector.

        Parameters
        ----------
        prefix : str
            The prefix to use for S3 paths.
        """
        self._s3_prefix = prefix
        self._logger.info(f"S3 prefix set to {self._s3_prefix}")

    @property
    def bucket(self):
        """
        Get the current S3 bucket name.

        Returns
        -------
        str
            The name of the S3 bucket.
        """
        return self._bucket

    @property
    def s3_root(self):
        """
        Get the current S3 root directory.

        Returns
        -------
        str
            The root directory in the S3 bucket.
        """
        return self._s3_root

    @property
    def s3_prefix(self):
        """
        Get the current S3 prefix.

        Returns
        -------
        str
            The prefix used for S3 paths.
        """
        return self._s3_prefix

    def query(
        self,
        query: str,
        to_polars: bool = False,
        local_dataframes: dict[str, pd.DataFrame | pl.DataFrame] = None,
        use_persistent: bool = True,
    ):
        """
        Execute a SQL query using DuckDB on S3 Parquet files and/or local DataFrames.

        This unified method handles both one-off queries and persistent connection scenarios.
        By default, uses a persistent connection that is reused across multiple queries for
        better performance. Set use_persistent=False for a temporary connection.

        Parameters
        ----------
        query : str
            SQL query to execute. Can reference:
            - S3 Parquet files using read_parquet('s3://bucket/path/*.parquet')
            - Local DataFrames by their dictionary keys (when using local_dataframes parameter)
            - Registered DataFrames by name (when using register_dataframe)
            - Any DuckDB SQL features (joins, aggregations, window functions, CTEs, etc.)

        to_polars : bool, optional
            If True, returns a Polars DataFrame. If False, returns a Pandas DataFrame.
            Defaults to False (Pandas).

        local_dataframes : dict[str, pd.DataFrame | pl.DataFrame], optional
            Dictionary mapping table names to DataFrames (pandas or polars).
            These DataFrames will be registered for this query and can be referenced
            by their dictionary keys in the SQL query.

        use_persistent : bool, optional
            If True (default), uses a persistent DuckDB connection that remains open for subsequent
            queries. This is more efficient for multiple queries. Use register_dataframe() to register
            DataFrames that persist across queries, and close_duckdb_connection() when done.
            If False, creates a temporary connection just for this query.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            Query results as a DataFrame.

        Examples
        --------
        # Simple query on S3 data (uses persistent connection by default)
        df = s3.query(
            "SELECT * FROM read_parquet('s3://bucket/data/*.parquet') WHERE date > '2024-01-01'"
        )

        # Query with local DataFrames
        filters = pd.DataFrame({'product_id': [100, 200, 300]})
        df = s3.query(
            "SELECT s.*, f.category FROM read_parquet('s3://bucket/sales/*.parquet') s JOIN filters f ON s.product_id = f.product_id",
            local_dataframes={'filters': filters}
        )

        # Multi-query workflow with registered DataFrames
        # Register DataFrames once
        s3.register_dataframe('customers', customers_df)
        s3.register_dataframe('regions', regions_df)

        # Run multiple queries (persistent connection used by default)
        result1 = s3.query("SELECT * FROM customers WHERE segment = 'Premium'")
        result2 = s3.query("SELECT c.*, r.region FROM customers c JOIN regions r ON c.id = r.id")

        # Combine registered DataFrames with S3 data
        result3 = s3.query(
            "SELECT s.*, c.name FROM read_parquet('s3://bucket/sales/*.parquet') s JOIN customers c ON s.customer_id = c.id"
        )

        # Close persistent connection when done
        s3.close_duckdb_connection()

        # Use temporary connection for one-off query
        df = s3.query(
            "SELECT * FROM read_parquet('s3://bucket/data/*.parquet')",
            use_persistent=False
        )

        # Return as Polars DataFrame
        df = s3.query("SELECT * FROM read_parquet('s3://bucket/data/*.parquet')", to_polars=True)

        Notes
        -----
        - Uses persistent connection by default for better performance
        - DuckDB's httpfs extension is automatically installed and loaded
        - AWS credentials are automatically retrieved from boto3 session
        - Supports all DuckDB SQL features (aggregations, joins, window functions, CTEs, etc.)
        - More efficient than loading entire files into memory first
        - Can query across multiple Parquet files in a single query
        - Local DataFrames (Polars or Pandas) are automatically handled
        - Persistent connection by default for better performance
        - Connection is automatically cleaned up when S3Connector is destroyed
        """  # noqa: E501
        if use_persistent:
            # Use persistent connection and previously registered DataFrames
            conn = self.get_duckdb_connection(persistent=True)

            # Register any additional local DataFrames for this query
            if local_dataframes:
                self._logger.debug(f"Registering {len(local_dataframes)} local DataFrames")
                for table_name, df in local_dataframes.items():
                    if isinstance(df, pl.DataFrame):
                        df_to_register = df.to_pandas()
                        self._logger.debug(f"Converted Polars DataFrame '{table_name}' to Pandas")
                    else:
                        df_to_register = df
                    conn.register(table_name, df_to_register)
                    self._registered_tables.add(table_name)
                    self._logger.debug(f"Registered DataFrame '{table_name}' with shape {df_to_register.shape}")

            try:
                self._logger.info("Executing query on persistent DuckDB connection")
                self._logger.debug(f"Query: {query}")
                result = conn.execute(query).fetchdf()

                if to_polars:
                    result = pl.from_pandas(result)

                self._logger.info(f"Query completed. Result shape: {result.shape}")
                return result

            except Exception as e:
                log_and_raise_error(self._logger, f"Error executing query: {e}")

        else:
            # Create temporary connection for this query only
            try:
                conn = duckdb.connect(":memory:")
                conn.execute("INSTALL httpfs")
                conn.execute("LOAD httpfs")

                # Register local DataFrames if provided
                if local_dataframes:
                    self._logger.debug(f"Registering {len(local_dataframes)} local DataFrames")
                    for table_name, df in local_dataframes.items():
                        if isinstance(df, pl.DataFrame):
                            df_to_register = df.to_pandas()
                        else:
                            df_to_register = df
                        conn.register(table_name, df_to_register)

                # Configure AWS credentials
                session = boto3.Session()
                credentials = session.get_credentials()

                if credentials:
                    if credentials.token:
                        conn.execute(f"""
                            CREATE SECRET (
                                TYPE S3,
                                KEY_ID '{credentials.access_key}',
                                SECRET '{credentials.secret_key}',
                                SESSION_TOKEN '{credentials.token}',
                                REGION '{session.region_name or "us-east-1"}'
                            )
                        """)
                    else:
                        conn.execute(f"""
                            CREATE SECRET (
                                TYPE S3,
                                KEY_ID '{credentials.access_key}',
                                SECRET '{credentials.secret_key}',
                                REGION '{session.region_name or "us-east-1"}'
                            )
                        """)
                    self._logger.debug("AWS credentials configured for DuckDB")
                else:
                    self._logger.info("No explicit credentials found, using credential chain")
                    conn.execute("""
                        CREATE SECRET (
                            TYPE S3,
                            PROVIDER credential_chain
                        )
                    """)

                self._logger.info("Executing DuckDB query")
                self._logger.debug(f"Query: {query}")

                result = conn.execute(query).fetchdf()

                if to_polars:
                    result = pl.from_pandas(result)

                self._logger.info(f"Query completed. Result shape: {result.shape}")

                # Close temporary connection
                conn.close()

                return result

            except Exception as e:
                log_and_raise_error(self._logger, f"Error executing DuckDB query: {e}")

    def get_duckdb_connection(self, persistent: bool = True):
        """
        Get or create a DuckDB connection configured for S3 access.

        This method returns a persistent DuckDB connection that can be reused across
        multiple queries. The connection is configured with AWS credentials and httpfs
        extension for S3 access.

        Parameters
        ----------
        persistent : bool, optional
            If True (default), returns a persistent connection stored in the instance.
            If False, creates a new temporary connection.

        Returns
        -------
        duckdb.DuckDBPyConnection
            A DuckDB connection object ready for querying.

        Examples
        --------
        # Get persistent connection and run custom queries
        conn = s3.get_duckdb_connection()
        result = conn.execute("SELECT * FROM read_parquet('s3://bucket/data.parquet')").fetchdf()

        # Run multiple queries on the same connection
        conn.execute("CREATE TABLE temp_table AS SELECT * FROM read_parquet('s3://...')")
        result = conn.execute("SELECT COUNT(*) FROM temp_table").fetchdf()

        # When done, close the connection
        s3.close_duckdb_connection()

        Notes
        -----
        - The connection is automatically configured with AWS credentials
        - httpfs extension is installed and loaded for S3 access
        - Use close_duckdb_connection() when you're done to free resources
        - Persistent connections remain available until explicitly closed
        """
        if persistent and self._duckdb_conn is not None:
            return self._duckdb_conn

        try:
            conn = duckdb.connect(":memory:" if persistent else None)

            # Install and load httpfs for S3 access
            conn.execute("INSTALL httpfs")
            conn.execute("LOAD httpfs")

            # Configure AWS credentials
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials:
                if credentials.token:
                    conn.execute(f"""
                        CREATE SECRET (
                            TYPE S3,
                            KEY_ID '{credentials.access_key}',
                            SECRET '{credentials.secret_key}',
                            SESSION_TOKEN '{credentials.token}',
                            REGION '{session.region_name or "us-east-1"}'
                        )
                    """)
                else:
                    conn.execute(f"""
                        CREATE SECRET (
                            TYPE S3,
                            KEY_ID '{credentials.access_key}',
                            SECRET '{credentials.secret_key}',
                            REGION '{session.region_name or "us-east-1"}'
                        )
                    """)
                self._logger.debug("AWS credentials configured for DuckDB")
            else:
                self._logger.info("No explicit credentials found, using credential chain")
                conn.execute("""
                    CREATE SECRET (
                        TYPE S3,
                        PROVIDER credential_chain
                    )
                """)

            if persistent:
                self._duckdb_conn = conn
                self._logger.info("Persistent DuckDB connection created")
            else:
                self._logger.debug("Temporary DuckDB connection created")

            return conn

        except Exception as e:
            log_and_raise_error(self._logger, f"Error creating DuckDB connection: {e}")

    def register_dataframe(self, name: str, df: pd.DataFrame | pl.DataFrame):
        """
        Register a DataFrame in the persistent DuckDB connection.

        This allows you to reference the DataFrame by name in subsequent SQL queries.
        The DataFrame will remain registered until the connection is closed or the
        table is explicitly unregistered.

        Parameters
        ----------
        name : str
            The table name to use for the DataFrame in SQL queries.
        df : pd.DataFrame or pl.DataFrame
            The DataFrame to register.

        Examples
        --------
        # Register a DataFrame
        customers = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        s3.register_dataframe('customers', customers)

        # Now use it in queries
        result = s3.query("SELECT * FROM customers WHERE id > 1")

        # Register multiple DataFrames
        s3.register_dataframe('orders', orders_df)
        s3.register_dataframe('products', products_df)
        result = s3.query(
            "SELECT o.*, p.name as product_name FROM orders o JOIN products p ON o.product_id = p.id"
        )

        Notes
        -----
        - Polars DataFrames are automatically converted to pandas for registration
        - Registered tables persist across multiple queries
        - Use unregister_dataframe() or close_duckdb_connection() to clear registrations
        """
        conn = self.get_duckdb_connection(persistent=True)

        try:
            # Convert Polars to Pandas if needed
            if isinstance(df, pl.DataFrame):
                df_to_register = df.to_pandas()
                self._logger.debug(f"Converted Polars DataFrame '{name}' to Pandas for registration")
            else:
                df_to_register = df

            conn.register(name, df_to_register)
            self._registered_tables.add(name)
            self._logger.info(f"Registered DataFrame '{name}' with shape {df_to_register.shape}")

        except Exception as e:
            log_and_raise_error(self._logger, f"Error registering DataFrame '{name}': {e}")

    def unregister_dataframe(self, name: str):
        """
        Unregister a DataFrame from the DuckDB connection.

        Parameters
        ----------
        name : str
            The table name to unregister.

        Examples
        --------
        s3.unregister_dataframe('customers')
        """
        if self._duckdb_conn is None:
            self._logger.warning("No DuckDB connection exists")
            return

        try:
            self._duckdb_conn.unregister(name)
            self._registered_tables.discard(name)
            self._logger.info(f"Unregistered DataFrame '{name}'")
        except Exception as e:
            self._logger.warning(f"Error unregistering DataFrame '{name}': {e}")

    def close_duckdb_connection(self):
        """
        Close the persistent DuckDB connection and clear registered tables.

        Call this method when you're done with DuckDB queries to free resources.

        Examples
        --------
        # After finishing all queries
        s3.close_duckdb_connection()
        """
        if self._duckdb_conn is not None:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass
            self._duckdb_conn = None
            self._registered_tables.clear()
            try:
                self._logger.info("DuckDB connection closed")
            except Exception:
                pass

    def __del__(self):
        """Cleanup: close DuckDB connection when S3Connector is destroyed."""
        try:
            self.close_duckdb_connection()
        except Exception:
            pass
