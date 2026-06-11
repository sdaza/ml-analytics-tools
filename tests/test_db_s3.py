# %%
import builtins
import gc
import threading
import weakref
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from ml_analytics.data_connector import DataConnector
from ml_analytics.s3_connector import S3Connector
from ml_analytics.utils import _is_select_statement


class _DatabricksSecretsMock:
    def __init__(self, values):
        self._values = values

    def get(self, scope, key):
        value = self._values.get((scope, key))
        if value is None:
            raise KeyError(f"{scope}/{key}")
        return value


def _clear_snowflake_env(monkeypatch):
    for name in [
        "DATABRICKS_SECRET_SCOPE",
        "ML_ANALYTICS_DATABASE_ENGINE",
        "ML_ANALYTICS_DB_ENGINE",
        "ML_ANALYTICS_SNOWFLAKE_SECRET_SCOPE",
        "PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_ACCESS_TOKEN",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_AUTHENTICATOR",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_OAUTH_TOKEN",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_PRIVATE_KEY",
        "SNOWFLAKE_PRIVATE_KEY_FILE",
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
        "SNOWFLAKE_PRIVATE_KEY_PATH",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_SCHEMA",
        "SNOWFLAKE_SECRET_SCOPE",
        "SNOWFLAKE_TOKEN",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE",
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mock_s3():
    with patch("ml_analytics.s3_connector.boto3") as mock_boto3:
        mock_s3_client = MagicMock()
        mock_boto3.Session.return_value.client.return_value = mock_s3_client
        yield mock_s3_client


@pytest.fixture
def mock_db():
    with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_redshift.connect.return_value = mock_connection
        yield mock_cursor


@pytest.fixture
def mock_credentials(monkeypatch):
    monkeypatch.setenv("ML_ANALYTICS_S3_BUCKET", "test-bucket")
    with patch("ml_analytics.data_connector.get_credential_value") as mock_get_credential_value:
        mock_get_credential_value.side_effect = lambda key: f"mock_{key.lower()}"
        yield mock_get_credential_value


def test_snowflake_externalbrowser_params(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_USER", "your.name@example.com")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "example-account")
    monkeypatch.setenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "ANALYTICS_S_WH")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "ANALYTICS_DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC")

    dc = DataConnector(engine="snowflake")

    assert dc._db_params["user"] == "your.name@example.com"
    assert dc._db_params["account"] == "example-account"
    assert dc._db_params["authenticator"] == "externalbrowser"
    assert dc._db_params["warehouse"] == "ANALYTICS_S_WH"
    assert dc._db_params["database"] == "ANALYTICS_DB"
    assert dc._db_params["schema"] == "PUBLIC"
    assert "private_key" not in dc._db_params


def test_snowflake_key_pair_params_from_databricks_scope(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_USER", "your.name@example.com")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "example-account")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "ANALYTICS_S_WH")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "ANALYTICS_DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC")

    pem = "-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----"
    dbutils = MagicMock()
    dbutils.secrets = _DatabricksSecretsMock(
        {
            ("user-your.name@example.com", "snowflake_key"): pem,
            ("user-your.name@example.com", "snowflake_key_pass"): "key-pass",
        }
    )
    monkeypatch.setattr(builtins, "dbutils", dbutils, raising=False)

    with patch("ml_analytics.data_connector._load_private_key_der", return_value=b"der-key") as mock_load_key:
        dc = DataConnector(engine="snowflake")

    assert dc._snowflake_secret_scope == "user-your.name@example.com"
    assert dc._db_params["authenticator"] == "SNOWFLAKE_JWT"
    assert dc._db_params["private_key"] == b"der-key"
    assert "password" not in dc._db_params
    mock_load_key.assert_called_once_with(
        private_key=pem,
        private_key_path=None,
        passphrase="key-pass",
    )


def test_snowflake_spark_options_use_key_pair_secrets(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_USER", "your.name@example.com")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "example-account")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "ANALYTICS_S_WH")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "ANALYTICS_DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "PUBLIC")

    pem = "-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----"
    dbutils = MagicMock()
    dbutils.secrets = _DatabricksSecretsMock(
        {
            ("user-your.name@example.com", "snowflake_key"): pem,
            ("user-your.name@example.com", "snowflake_key_pass"): "key-pass",
        }
    )
    monkeypatch.setattr(builtins, "dbutils", dbutils, raising=False)

    with (
        patch("ml_analytics.data_connector._load_private_key_der", return_value=b"der-key"),
        patch(
            "ml_analytics.data_connector._load_private_key_pem_for_spark",
            return_value="spark-key",
        ) as mock_spark_key,
    ):
        dc = DataConnector(engine="snowflake")
        options = dc.snowflake_spark_options()

    assert options == {
        "sfURL": "example-account.snowflakecomputing.com",
        "sfUser": "your.name@example.com",
        "sfDatabase": "ANALYTICS_DB",
        "sfSchema": "PUBLIC",
        "sfWarehouse": "ANALYTICS_S_WH",
        "pem_private_key": "spark-key",
    }
    mock_spark_key.assert_called_once_with(
        private_key=pem,
        private_key_path=None,
        passphrase="key-pass",
    )


def test_snowflake_oauth_token_params(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_USER", "your.name@example.com")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "example-account")
    monkeypatch.setenv("SNOWFLAKE_TOKEN", "oauth-token")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "ignored-password")

    dc = DataConnector(engine="snowflake")

    assert dc._db_params["authenticator"] == "oauth"
    assert dc._db_params["token"] == "oauth-token"
    assert "password" not in dc._db_params


def test_s3_and_db_operations(mock_s3, mock_db, mock_credentials):
    # Initialize connectors with mocked credentials
    dc = DataConnector()
    s3 = S3Connector(bucket="test-bucket")

    # Mock S3 list files
    mock_s3.list_objects_v2.return_value = {"Contents": [{"Key": "ml-bi-projects/testing/test.parquet"}]}
    files = s3.list_files(prefix="ml-bi-projects")
    assert files == ["ml-bi-projects/testing/test.parquet"]

    # Mock SQL query
    mock_db.fetch_dataframe.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    query = "SELECT * FROM analytics.customer_features limit 10"
    df = dc.sql(query)
    assert not df.empty

    # Mock deleting a file from S3
    s3.delete_file(key="ml-bi-projects/testing/test.parquet")
    mock_s3.delete_object.assert_called_with(Bucket=s3.bucket, Key="ml-bi-projects/testing/test.parquet")


def test_s3_save_dataframe_with_pandas(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with patch("pyarrow.parquet.write_table") as mock_write_table:
        s3.save_dataframe(df, directory="ml-bi-projects/testing", file_name="test")
        mock_write_table.assert_called_once()


def test_s3_save_dataframe_with_polars(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    with patch("polars.DataFrame.write_parquet") as mock_write_parquet:
        s3.save_dataframe(df, directory="ml-bi-projects/testing", file_name="test")
        mock_write_parquet.assert_called_once()


def test_s3_read_parquet_with_period_column(mock_s3, tmp_path):
    import io

    s3 = S3Connector(bucket="test-bucket")

    # Create a DataFrame with a Period column (freq='M')
    df_with_period = pd.DataFrame(
        {"id": [1, 2, 3], "value": [100, 200, 300], "month": pd.period_range("2024-01", periods=3, freq="M")}
    )

    # Save to a temporary parquet file and read bytes for the mock buffer
    parquet_file = tmp_path / "test_period.parquet"
    df_with_period.to_parquet(parquet_file, engine="pyarrow")
    parquet_bytes = parquet_file.read_bytes()

    # Mock get_path to return an S3 path and _download_s3_to_buffer to return file content
    with (
        patch.object(s3, "get_path", return_value="s3://test-bucket/test_period.parquet"),
        patch.object(s3, "_download_s3_to_buffer", return_value=io.BytesIO(parquet_bytes)),
    ):
        result = s3.read_parquet("test_period.parquet")

        # Check the result is a pandas DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "month" in result.columns

        # Check that Period column was converted to datetime
        # The implementation detects extension types and uses PyArrow directly
        assert pd.api.types.is_datetime64_any_dtype(result["month"])

        # Verify the conversion worked correctly - the timestamps should match
        # the original period start times
        expected_timestamps = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
        pd.testing.assert_series_equal(result["month"], expected_timestamps, check_names=False)


# === Tests for _is_select_statement ===


class TestIsSelectStatement:
    def test_simple_select(self):
        assert _is_select_statement("SELECT * FROM users") is True

    def test_with_cte(self):
        assert _is_select_statement("WITH cte AS (SELECT 1) SELECT * FROM cte") is True

    def test_create_table(self):
        assert _is_select_statement("CREATE TABLE foo (id INT)") is False

    def test_insert(self):
        assert _is_select_statement("INSERT INTO foo VALUES (1)") is False

    def test_drop(self):
        assert _is_select_statement("DROP TABLE foo") is False

    def test_select_with_leading_comment(self):
        assert _is_select_statement("-- get all users\nSELECT * FROM users") is True

    def test_select_with_multiline_comment(self):
        assert _is_select_statement("/* fetch data */\nSELECT * FROM users") is True

    def test_create_with_leading_comment(self):
        assert _is_select_statement("-- setup\nCREATE TEMP TABLE tmp AS SELECT 1") is False

    def test_empty_string(self):
        assert _is_select_statement("") is False

    def test_whitespace_only(self):
        assert _is_select_statement("   \n  ") is False

    def test_select_lowercase(self):
        assert _is_select_statement("select * from users") is True

    def test_with_lowercase(self):
        assert _is_select_statement("with cte as (select 1) select * from cte") is True

    def test_select_into_should_be_false(self):
        # SELECT INTO creates a table, not a query that returns data
        assert _is_select_statement("SELECT * INTO analytics.model_output FROM users") is False

    def test_select_into_uppercase(self):
        assert _is_select_statement("SELECT col1, col2 INTO schema.new_table FROM old_table") is False

    def test_select_into_with_where(self):
        assert _is_select_statement("SELECT * INTO temp_table FROM users WHERE id > 100") is False

    def test_select_without_into(self):
        # Regular SELECT should still return True
        assert _is_select_statement("SELECT * FROM users WHERE name = 'into'") is True

    def test_with_select_into(self):
        # WITH ... SELECT INTO creates a table, not a query that returns data
        assert _is_select_statement("WITH cte AS (SELECT 1) SELECT * INTO new_table FROM cte") is False

    def test_with_multiple_ctes_select_into(self):
        # Complex WITH statement ending with SELECT INTO
        query = """WITH 
            cte1 AS (SELECT * FROM table1),
            cte2 AS (SELECT * FROM table2)
        SELECT col1, col2 INTO analytics.model_output FROM cte1 JOIN cte2"""
        assert _is_select_statement(query) is False

    def test_select_into_temp_table(self):
        # SELECT INTO TEMP creates a temporary table, not a query that returns data
        assert _is_select_statement("SELECT * INTO TEMP temp_table FROM users") is False

    def test_select_into_temporary_table(self):
        # SELECT INTO TEMPORARY creates a temporary table
        assert _is_select_statement("SELECT col1, col2 INTO TEMPORARY my_temp FROM source") is False

    def test_create_temp_table(self):
        # CREATE TEMP TABLE is a DDL statement, not a SELECT query
        assert _is_select_statement("CREATE TEMP TABLE foo AS SELECT * FROM bar") is False

    def test_create_temporary_table(self):
        # CREATE TEMPORARY TABLE is a DDL statement
        assert _is_select_statement("CREATE TEMPORARY TABLE foo AS SELECT * FROM bar") is False

    def test_with_select_into_temp(self):
        # WITH ... SELECT INTO TEMP
        assert _is_select_statement("WITH cte AS (SELECT 1) SELECT * INTO TEMP results FROM cte") is False

    def test_select_into_on_newline(self):
        # SELECT INTO where INTO is on its own line (common formatting)
        query = """SELECT
            col1,
            col2
        INTO analytics.model_output
        FROM source_table"""
        assert _is_select_statement(query) is False

    def test_with_select_into_multiline(self):
        # WITH statement with SELECT INTO across multiple lines (real-world case)
        query = """WITH
            ub AS (
                SELECT source_id FROM table1
            ),
            ua AS (
                SELECT source_id FROM table2
            )
        SELECT
            ub.*,
            ua.field
        INTO analytics.model_output
        FROM ub
            LEFT JOIN ua ON ua.source_id = ub.source_id"""
        assert _is_select_statement(query) is False

    def test_into_in_string_literal(self):
        # INTO appearing in a string literal should not trigger false positive
        assert _is_select_statement("SELECT * FROM users WHERE description = 'insert into table'") is True

    def test_into_in_column_name(self):
        # INTO as part of a column name should not trigger false positive
        assert _is_select_statement("SELECT into_field, into_date FROM users") is True

    def test_into_in_table_name(self):
        # INTO as part of a table name should not trigger false positive
        assert _is_select_statement("SELECT * FROM into_table WHERE id > 0") is True

    def test_into_in_comment(self):
        # INTO in a comment should not trigger false positive
        query = """-- This query looks into the data
        SELECT * FROM users"""
        assert _is_select_statement(query) is True

    def test_into_in_multiline_comment(self):
        # INTO in a multiline comment should not trigger false positive
        query = """/* 
        This query delves into the users table
        to find active records
        */
        SELECT * FROM users"""
        assert _is_select_statement(query) is True

    def test_into_as_column_alias(self):
        # INTO used in an alias should not trigger false positive
        assert _is_select_statement("SELECT field AS into_value FROM users") is True

    def test_into_in_where_clause(self):
        # INTO in a WHERE clause value should not trigger false positive
        assert _is_select_statement("SELECT * FROM logs WHERE action LIKE '%into%'") is True

    def test_substring_into(self):
        # Words containing 'into' should not trigger false positive
        assert _is_select_statement("SELECT pointer, into_something FROM table1") is True

    def test_insert_into_statement(self):
        # INSERT INTO is not a SELECT statement
        assert _is_select_statement("INSERT INTO users VALUES (1, 'test')") is False

    def test_select_into_vs_insert_into(self):
        # Make sure we distinguish SELECT INTO from INSERT INTO
        assert _is_select_statement("SELECT * INTO backup_users FROM users") is False
        assert _is_select_statement("INSERT INTO users SELECT * FROM staging") is False


# === Tests for multi-statement SQL in sql() ===


class TestSqlMultiStatement:
    def test_sql_multi_statement_last_is_select(self, mock_db, mock_credentials):
        dc = DataConnector()
        mock_db.fetch_dataframe.return_value = pd.DataFrame({"id": [1, 2]})

        multi_sql = "CREATE TEMP TABLE tmp AS SELECT 1; SELECT * FROM tmp"
        result = dc.sql(multi_sql)

        assert not result.empty
        # The CREATE should be executed first, then the SELECT
        assert mock_db.execute.call_count >= 1

    def test_sql_single_statement(self, mock_db, mock_credentials):
        dc = DataConnector()
        mock_db.fetch_dataframe.return_value = pd.DataFrame({"id": [1]})

        result = dc.sql("SELECT 1")
        assert not result.empty


# === Tests for SQL file path resolution ===


class TestSqlFilePathResolution:
    def test_sql_loads_from_sql_file(self, mock_db, mock_credentials, tmp_path):
        """sql() should load query from a .sql file when path ends with .sql"""
        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT * FROM users")

        dc = DataConnector()
        mock_db.fetch_dataframe.return_value = pd.DataFrame({"id": [1]})

        with patch("ml_analytics.data_connector.load_sql_query", return_value="SELECT * FROM users") as mock_load:
            dc.sql(str(sql_file))
            mock_load.assert_called_once_with(str(sql_file))

    def test_sql_passes_kwargs_to_load_sql_query(self, mock_db, mock_credentials):
        """sql() should forward kwargs for template substitution in SQL files"""
        dc = DataConnector()
        mock_db.fetch_dataframe.return_value = pd.DataFrame({"id": [1]})

        with patch(
            "ml_analytics.data_connector.load_sql_query",
            return_value="SELECT * FROM users WHERE status = 'active'",
        ) as mock_load:
            dc.sql("queries/test.sql", status="active")
            mock_load.assert_called_once_with("queries/test.sql", status="active")

    def test_sql_does_not_load_file_for_regular_query(self, mock_db, mock_credentials):
        """sql() should not attempt file loading for regular SQL strings"""
        dc = DataConnector()
        mock_db.fetch_dataframe.return_value = pd.DataFrame({"id": [1]})

        with patch("ml_analytics.data_connector.load_sql_query") as mock_load:
            dc.sql("SELECT 1")
            mock_load.assert_not_called()

    def test_execute_sql_loads_from_sql_file(self, mock_db, mock_credentials):
        """execute_sql() should load query from a .sql file"""
        dc = DataConnector()

        with patch(
            "ml_analytics.data_connector.load_sql_query",
            return_value="CREATE TABLE test (id INT)",
        ) as mock_load:
            dc.execute_sql("queries/create.sql")
            mock_load.assert_called_once_with("queries/create.sql")
            mock_db.execute.assert_called_once_with("CREATE TABLE test (id INT)")

    def test_execute_sql_passes_kwargs_to_load_sql_query(self, mock_db, mock_credentials):
        """execute_sql() should forward kwargs for template substitution"""
        dc = DataConnector()

        with patch(
            "ml_analytics.data_connector.load_sql_query",
            return_value="DROP TABLE users",
        ) as mock_load:
            dc.execute_sql("queries/drop.sql", table="users")
            mock_load.assert_called_once_with("queries/drop.sql", table="users")

    def test_sql_raises_when_file_not_found(self, mock_db, mock_credentials):
        """sql() should raise an error when .sql file cannot be loaded"""
        dc = DataConnector()

        with patch("ml_analytics.data_connector.load_sql_query", return_value=None):
            with pytest.raises(Exception, match="Could not load SQL file"):
                dc.sql("nonexistent/query.sql")


# === Tests for connection cleanup (__del__) ===


class TestConnectionCleanup:
    def test_del_closes_open_connection(self, mock_credentials):
        """__del__ should close the connection when the instance is garbage-collected."""
        with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
            mock_connection = MagicMock()
            mock_connection.closed = False  # simulate open connection
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_redshift.connect.return_value = mock_connection

            dc = DataConnector()
            dc.connect()

            dc.__del__()

            mock_connection.close.assert_called_once()

    def test_del_safe_when_no_connection(self, mock_db, mock_credentials):
        """__del__ should not raise when no connection was ever opened."""
        dc = DataConnector()
        # Should not raise
        dc.__del__()

    def test_del_safe_when_already_closed(self, mock_credentials):
        """__del__ should not raise when the connection is already closed."""
        with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
            mock_connection = MagicMock()
            mock_connection.closed = False
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_redshift.connect.return_value = mock_connection

            dc = DataConnector()
            dc.connect()
            dc.close_redshift_connection()

            # Should not raise even if called again
            dc.__del__()

    def test_gc_triggers_cleanup(self, mock_credentials):
        """Garbage collection should trigger connection cleanup."""
        with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
            mock_connection = MagicMock()
            mock_connection.closed = False
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_redshift.connect.return_value = mock_connection

            dc = DataConnector()
            dc.connect()
            # Cancel the idle timer so it doesn't hold a reference cycle
            dc._cancel_idle_timer()
            ref = weakref.ref(dc)

            del dc
            gc.collect()

            assert ref() is None
            mock_connection.close.assert_called_once()


# === Tests for thread-safe connect ===


class TestThreadSafeConnect:
    def test_concurrent_ensure_connected_creates_single_connection(self, mock_credentials):
        """Concurrent _ensure_connected calls should create only one connection."""
        with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
            mock_connection = MagicMock()
            mock_connection.closed = False  # simulate open connection
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_redshift.connect.return_value = mock_connection

            dc = DataConnector()
            barrier = threading.Barrier(4)
            errors = []

            def connect_thread():
                try:
                    barrier.wait(timeout=5)
                    dc._ensure_connected()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=connect_thread) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Threads raised errors: {errors}"
            # Only one connection should have been created
            assert mock_redshift.connect.call_count == 1

    def test_connect_skips_when_already_open(self, mock_credentials):
        """connect() should not create a new connection when one is already open."""
        with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
            mock_connection = MagicMock()
            mock_connection.closed = False
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_redshift.connect.return_value = mock_connection

            dc = DataConnector()
            dc.connect()
            dc.connect()  # Second call should be a no-op

            assert mock_redshift.connect.call_count == 1


# === Tests for multi-statement SQL in unload_to_s3 ===


class TestUnloadMultiStatement:
    def test_unload_multi_statement_executes_preceding(self, mock_s3, mock_db, mock_credentials):
        dc = DataConnector()

        with patch("ml_analytics.data_connector.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_creds = MagicMock()
            mock_creds.access_key = "fake_key"
            mock_creds.secret_key = "fake_secret"
            mock_creds.token = "fake_token"
            mock_session.get_credentials.return_value = mock_creds
            mock_boto3.Session.return_value = mock_session

            mock_s3.list_objects_v2.return_value = {}

            multi_sql = "CREATE TEMP TABLE tmp AS SELECT * FROM users; SELECT * FROM tmp"
            dc.unload_to_s3(
                query=multi_sql,
                relative_path="exports/test/",
                file_prefix="test",
            )

            # Should have executed the CREATE statement + the UNLOAD
            calls = mock_db.execute.call_args_list
            assert len(calls) >= 2
            # First call should be the CREATE TEMP TABLE
            first_stmt = calls[0][0][0]
            assert "CREATE TEMP TABLE" in first_stmt
            # Last call should be the UNLOAD with SELECT * FROM tmp
            last_stmt = calls[-1][0][0]
            assert "UNLOAD" in last_stmt
            assert "SELECT * FROM tmp" in last_stmt

    def test_unload_sql_file(self, mock_s3, mock_db, mock_credentials):
        dc = DataConnector()

        with (
            patch("ml_analytics.data_connector.boto3") as mock_boto3,
            patch("ml_analytics.data_connector.load_sql_query") as mock_load,
        ):
            mock_session = MagicMock()
            mock_creds = MagicMock()
            mock_creds.access_key = "fake_key"
            mock_creds.secret_key = "fake_secret"
            mock_creds.token = "fake_token"
            mock_session.get_credentials.return_value = mock_creds
            mock_boto3.Session.return_value = mock_session

            mock_s3.list_objects_v2.return_value = {}
            mock_load.return_value = "CREATE TEMP TABLE tmp AS SELECT 1;\nSELECT * FROM tmp"

            dc.unload_to_s3(
                query="sql/my_query.sql",
                relative_path="exports/test/",
                file_prefix="test",
            )

            mock_load.assert_called_once_with("sql/my_query.sql")
            calls = mock_db.execute.call_args_list
            assert len(calls) >= 2
            first_stmt = calls[0][0][0]
            assert "CREATE TEMP TABLE" in first_stmt
            last_stmt = calls[-1][0][0]
            assert "UNLOAD" in last_stmt


# %%
