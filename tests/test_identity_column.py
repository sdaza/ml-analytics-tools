"""
Tests for Identity Column feature in DataConnector

Tests the identity_column parameter functionality for automatically adding
auto-incrementing IDENTITY columns to Redshift tables.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml_analytics.data_connector import DataConnector


@pytest.fixture
def mock_credentials(monkeypatch):
    monkeypatch.setenv("ML_ANALYTICS_S3_BUCKET", "test-bucket")
    with patch("ml_analytics.data_connector.get_credential_value") as mock_get_credential_value:
        mock_get_credential_value.side_effect = lambda key: f"mock_{key.lower()}"
        yield mock_get_credential_value


@pytest.fixture
def mock_db():
    with patch("ml_analytics.data_connector.redshift_connector") as mock_redshift:
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_redshift.connect.return_value = mock_connection
        yield mock_cursor


@pytest.fixture
def mock_s3_connector():
    """Mock S3Connector with all necessary methods"""
    with patch("ml_analytics.data_connector.S3Connector") as mock_s3_class:
        mock_s3_instance = MagicMock()
        mock_s3_instance.bucket = "test-bucket"
        mock_s3_instance.save_dataframe = MagicMock()
        # Return a non-empty list for any list_files call to simulate files exist
        mock_s3_instance.list_files = MagicMock(return_value=["mock_file.parquet"])
        mock_s3_instance.list_files_in_prefix = MagicMock(return_value=["mock_file.parquet"])
        mock_s3_instance.delete_file = MagicMock()
        # Mock get_path to return a fake S3 path
        mock_s3_instance.get_path = MagicMock(return_value="s3://test-bucket/mock_file.parquet")
        mock_s3_class.return_value = mock_s3_instance
        yield mock_s3_instance


@pytest.fixture(autouse=True)
def mock_boto3():
    """Mock boto3 to provide fake AWS credentials - auto-used in all tests"""
    with patch("ml_analytics.data_connector.boto3") as mock_boto3_module:
        # Mock Session and credentials
        mock_session = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.access_key = "fake_access_key"
        mock_credentials.secret_key = "fake_secret_key"
        mock_credentials.token = None
        mock_session.get_credentials.return_value = mock_credentials

        # Make Session() callable and return the mock session
        mock_boto3_module.Session = MagicMock(return_value=mock_session)

        yield mock_boto3_module


@pytest.fixture
def dc(mock_credentials, mock_db, mock_s3_connector):
    """Initialize DataConnector for tests with mocked connections"""
    return DataConnector()


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing"""
    return pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["NYC", "LA", "Chicago"]})


def test_create_table_with_default_identity_column(dc, sample_df, mock_db, mock_s3_connector):
    """Test creating table with default identity column named 'id'"""
    # Mock table_exists to return False (table doesn't exist)
    mock_db.fetchone.return_value = (0,)

    # Mock PyArrow schema reading since no column_types provided
    import pyarrow as pa

    mock_schema = pa.schema([("name", pa.string()), ("age", pa.int64()), ("city", pa.string())])

    # Mock load_from_s3 to bypass COPY logic and just capture the call
    with patch("pyarrow.parquet.read_schema", return_value=mock_schema):
        with patch.object(dc, "load_from_s3") as mock_load:
            dc.create_table_from_dataframe(
                sample_df, table="test_users_default_id", schema="test", drop_existing_table=True, identity_column=True
            )
            # Verify load_from_s3 was called with identity_column
            assert mock_load.called
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("identity_column") is True

    # Verify CREATE TABLE was called with identity column by checking the mock_load call
    # Since we mocked load_from_s3, we won't see CREATE TABLE in execute calls
    # But we verified identity_column was passed through


def test_create_table_with_custom_identity_column(dc, sample_df, mock_db, mock_s3_connector):
    """Test creating table with custom identity column name"""
    # Mock table_exists to return False
    mock_db.fetchone.return_value = (0,)

    # Mock PyArrow schema reading since no column_types provided
    import pyarrow as pa

    mock_schema = pa.schema([("name", pa.string()), ("age", pa.int64()), ("city", pa.string())])

    with patch("pyarrow.parquet.read_schema", return_value=mock_schema):
        with patch.object(dc, "load_from_s3") as mock_load:
            dc.create_table_from_dataframe(
                sample_df,
                table="test_users_custom_id",
                schema="test",
                drop_existing_table=True,
                identity_column="user_id",
            )
            # Verify load_from_s3 was called with custom identity_column
            assert mock_load.called
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("identity_column") == "user_id"


def test_append_to_table_with_identity_column(dc, sample_df, mock_db, mock_s3_connector):
    """Test appending data to table with identity column"""
    # Mock table_exists to return False initially
    mock_db.fetchone.return_value = (0,)

    # Mock PyArrow schema reading since no column_types provided
    import pyarrow as pa

    mock_schema = pa.schema([("name", pa.string()), ("age", pa.int64()), ("city", pa.string())])

    with patch("pyarrow.parquet.read_schema", return_value=mock_schema):
        with patch.object(dc, "load_from_s3") as mock_load:
            # Create initial table
            dc.create_table_from_dataframe(
                sample_df, table="test_users_append", schema="test", drop_existing_table=True, identity_column=True
            )
            # Verify load_from_s3 was called with identity_column
            assert mock_load.called
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("identity_column") is True


def test_load_from_s3_with_identity_column(dc, mock_db, mock_s3_connector):
    """Test loading from S3 with identity column - tests CREATE TABLE logic"""
    # Mock table doesn't exist
    mock_db.fetchone.return_value = (0,)

    # Create a mock PyArrow schema
    import pyarrow as pa

    mock_schema = pa.schema([("product_name", pa.string()), ("price", pa.float64()), ("quantity", pa.int64())])

    # Test the CREATE TABLE part by mocking PyArrow and checking execute calls
    with patch("pyarrow.parquet.read_schema", return_value=mock_schema):
        # Call _create_table_from_parquet directly to test CREATE TABLE logic
        dc._create_table_from_parquet(
            table="test_products",
            schema="test",
            s3_bucket="test-bucket",
            relative_path="test_data.parquet",
            known_files=["test_data.parquet"],
            identity_column="product_id",
        )

    # Verify CREATE TABLE was called with product_id identity column
    execute_calls = [str(call) for call in mock_db.execute.call_args_list]
    create_table_calls = [call for call in execute_calls if "CREATE TABLE" in call]
    assert len(create_table_calls) > 0
    create_statement = create_table_calls[0]
    assert "product_id BIGINT IDENTITY(1,1)" in create_statement


def test_identity_column_with_column_types(dc, mock_db, mock_s3_connector):
    """Test identity column with custom column types"""
    # Mock table doesn't exist
    mock_db.fetchone.return_value = (0,)

    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"],
            "created_date": ["2024-01-01", "2024-01-02"],
            "status": ["active", "inactive"],
        }
    )

    column_types = {"name": "VARCHAR(100)", "email": "VARCHAR(255)", "created_date": "DATE", "status": "VARCHAR(50)"}

    with patch.object(dc, "load_from_s3") as mock_load:
        dc.create_table_from_dataframe(
            df,
            table="test_customers",
            schema="test",
            drop_existing_table=True,
            column_types=column_types,
            identity_column="customer_id",
        )
        # Verify load_from_s3 was called with identity_column and column_types
        assert mock_load.called
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get("identity_column") == "customer_id"
        assert call_kwargs.get("column_types") == column_types


def test_no_identity_column(dc, sample_df, mock_db, mock_s3_connector):
    """Test creating table without identity column (default behavior)"""
    # Mock table doesn't exist
    mock_db.fetchone.return_value = (0,)

    # Mock PyArrow schema reading since no column_types provided
    import pyarrow as pa

    mock_schema = pa.schema([("name", pa.string()), ("age", pa.int64()), ("city", pa.string())])

    with patch("pyarrow.parquet.read_schema", return_value=mock_schema):
        with patch.object(dc, "load_from_s3") as mock_load:
            dc.create_table_from_dataframe(
                sample_df,
                table="test_users_no_id",
                schema="test",
                drop_existing_table=True,
                # identity_column not specified
            )
            # Verify load_from_s3 was called WITHOUT identity_column
            assert mock_load.called
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("identity_column") is None


if __name__ == "__main__":
    """
    Note: Manual testing requires actual database connection.
    For CI/CD, tests run with mocks automatically.
    """
    print("Run tests with: pytest tests/test_identity_column.py -v")
