"""
Test suite for S3Connector Redshift validation refactoring.

This module tests the refactored Redshift type validation logic
to ensure pandas and polars DataFrames are properly converted
for Redshift compatibility.
"""

import os

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest

from ml_analytics.s3_connector import S3Connector


@pytest.fixture
def s3_connector():
    """Create an S3Connector instance for testing."""
    return S3Connector(bucket=os.getenv("S3_BUCKET", "test-bucket"), s3_root="test-data")


class TestPandasRedshiftValidation:
    """Test Redshift type validation for pandas DataFrames."""

    def test_all_null_date_columns(self, s3_connector, tmp_path):
        """Test handling of all-null DATE columns in pandas."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "hire_date": [None, None, None],
                "birth_date": [None, None, None],
            }
        )

        column_types = {"id": "BIGINT", "hire_date": "DATE", "birth_date": "DATE"}

        # Save to local parquet file
        output_file = str(tmp_path / "test_dates.parquet")

        # Prepare using the helper method directly
        df_prepared = s3_connector._prepare_pandas_for_redshift(df, column_types)

        # Build schema using helper method
        table = s3_connector._build_redshift_pyarrow_schema(df_prepared, column_types)

        # Write and verify
        pq.write_table(table, output_file)

        # Read back and check schema
        result_table = pq.read_table(output_file)

        # Verify DATE columns are date32
        assert str(result_table.schema[1].type) == "date32[day]", (
            f"hire_date should be date32, got {result_table.schema[1].type}"
        )
        assert str(result_table.schema[2].type) == "date32[day]", (
            f"birth_date should be date32, got {result_table.schema[2].type}"
        )

    def test_all_null_numeric_columns(self, s3_connector, tmp_path):
        """Test handling of all-null BIGINT and DECIMAL columns."""
        df = pd.DataFrame({"id": [1, 2, 3], "employee_id": [None, None, None], "salary": [None, None, None]})

        column_types = {"id": "BIGINT", "employee_id": "BIGINT", "salary": "DECIMAL(12,2)"}

        output_file = str(tmp_path / "test_numeric.parquet")

        df_prepared = s3_connector._prepare_pandas_for_redshift(df, column_types)
        table = s3_connector._build_redshift_pyarrow_schema(df_prepared, column_types)
        pq.write_table(table, output_file)

        result_table = pq.read_table(output_file)

        # All should be int64 for BIGINT
        assert str(result_table.schema[0].type) == "int64"
        assert str(result_table.schema[1].type) == "int64"
        # DECIMAL should be float64
        assert str(result_table.schema[2].type) == "double"

    def test_mixed_null_and_values(self, s3_connector, tmp_path):
        """Test columns with both NULL and actual values."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", None],
                "hire_date": ["2020-01-01", None, None],
                "salary": [50000.0, None, 55000.0],
            }
        )

        column_types = {"id": "BIGINT", "name": "VARCHAR(100)", "hire_date": "DATE", "salary": "DECIMAL(12,2)"}

        output_file = str(tmp_path / "test_mixed.parquet")

        df_prepared = s3_connector._prepare_pandas_for_redshift(df, column_types)
        table = s3_connector._build_redshift_pyarrow_schema(df_prepared, column_types)
        pq.write_table(table, output_file)

        result_table = pq.read_table(output_file)

        # Verify types
        assert str(result_table.schema[0].type) == "int64"  # BIGINT
        assert str(result_table.schema[1].type) == "string"  # VARCHAR
        assert str(result_table.schema[2].type) == "date32[day]"  # DATE
        assert str(result_table.schema[3].type) == "double"  # DECIMAL

    def test_int_vs_bigint_types(self, s3_connector, tmp_path):
        """Test INT vs BIGINT type handling."""
        df = pd.DataFrame({"small_id": [1, 2, 3], "medium_id": [100, 200, 300], "big_id": [1000000, 2000000, 3000000]})

        column_types = {"small_id": "SMALLINT", "medium_id": "INT", "big_id": "BIGINT"}

        output_file = str(tmp_path / "test_int_types.parquet")

        df_prepared = s3_connector._prepare_pandas_for_redshift(df, column_types)
        table = s3_connector._build_redshift_pyarrow_schema(df_prepared, column_types)
        pq.write_table(table, output_file)

        result_table = pq.read_table(output_file)

        # SMALLINT and INT should be int32
        assert str(result_table.schema[0].type) == "int32"
        assert str(result_table.schema[1].type) == "int32"
        # BIGINT should be int64
        assert str(result_table.schema[2].type) == "int64"

    def test_varchar_with_numbers(self, s3_connector, tmp_path):
        """Test VARCHAR columns that contain numeric data."""
        df = pd.DataFrame({"id": [1, 2, 3], "phone": ["5551234", "5555678", "5559999"]})

        column_types = {"id": "BIGINT", "phone": "VARCHAR(20)"}

        output_file = str(tmp_path / "test_varchar.parquet")

        df_prepared = s3_connector._prepare_pandas_for_redshift(df, column_types)
        table = s3_connector._build_redshift_pyarrow_schema(df_prepared, column_types)
        pq.write_table(table, output_file)

        result_table = pq.read_table(output_file)

        # Verify phone is string type
        assert str(result_table.schema[1].type) == "string"

        # Read back and verify values are preserved
        result_df = result_table.to_pandas()
        assert result_df["phone"].tolist() == ["5551234", "5555678", "5559999"]


class TestPolarsRedshiftValidation:
    """Test Redshift type validation for polars DataFrames."""

    def test_all_null_date_columns_polars(self, s3_connector, tmp_path):
        """Test handling of all-null DATE columns in polars."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "hire_date": [None, None, None],
                "birth_date": [None, None, None],
            }
        )

        column_types = {"id": "BIGINT", "hire_date": "DATE", "birth_date": "DATE"}

        output_file = str(tmp_path / "test_polars_dates.parquet")

        # Prepare using polars helper method
        df_prepared = s3_connector._prepare_polars_for_redshift(df, column_types)
        df_prepared.write_parquet(output_file)

        # Read back and verify
        result_table = pq.read_table(output_file)

        # Verify DATE columns exist and have appropriate type
        assert "hire_date" in result_table.column_names
        assert "birth_date" in result_table.column_names

    def test_mixed_null_and_values_polars(self, s3_connector, tmp_path):
        """Test polars with mixed null and values."""
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", None], "salary": [50000.0, None, 55000.0]})

        column_types = {"id": "BIGINT", "name": "VARCHAR(100)", "salary": "DECIMAL(12,2)"}

        output_file = str(tmp_path / "test_polars_mixed.parquet")

        df_prepared = s3_connector._prepare_polars_for_redshift(df, column_types)
        df_prepared.write_parquet(output_file)

        # Read and verify data is preserved
        result = pl.read_parquet(output_file)
        assert result.shape == df.shape
        assert result["id"].to_list() == [1, 2, 3]


class TestHelperMethods:
    """Test individual helper methods."""

    def test_get_base_redshift_type(self, s3_connector):
        """Test extraction of base Redshift type."""
        assert s3_connector._get_base_redshift_type("VARCHAR(100)") == "VARCHAR"
        assert s3_connector._get_base_redshift_type("DECIMAL(12,2)") == "DECIMAL"
        assert s3_connector._get_base_redshift_type("BIGINT") == "BIGINT"
        assert s3_connector._get_base_redshift_type("  DATE  ") == "DATE"

    def test_type_classification_methods(self, s3_connector):
        """Test type classification helper methods."""
        # Date types
        assert s3_connector._is_date_type("DATE") == True  # noqa: E712
        assert s3_connector._is_date_type("TIMESTAMP") == True  # noqa: E712
        assert s3_connector._is_date_type("VARCHAR") == False  # noqa: E712

        # Integer types
        assert s3_connector._is_integer_type("BIGINT") == True  # noqa: E712
        assert s3_connector._is_integer_type("INT") == True  # noqa: E712
        assert s3_connector._is_integer_type("SMALLINT") == True  # noqa: E712
        assert s3_connector._is_integer_type("VARCHAR") == False  # noqa: E712

        # Float types
        assert s3_connector._is_float_type("DOUBLE PRECISION") == True  # noqa: E712
        assert s3_connector._is_float_type("DECIMAL") == True  # noqa: E712
        assert s3_connector._is_float_type("FLOAT") == True  # noqa: E712
        assert s3_connector._is_float_type("INT") == False  # noqa: E712

        # String types
        assert s3_connector._is_string_type("VARCHAR") == True  # noqa: E712
        assert s3_connector._is_string_type("CHAR") == True  # noqa: E712
        assert s3_connector._is_string_type("TEXT") == True  # noqa: E712
        assert s3_connector._is_string_type("INT") == False  # noqa: E712

    def test_get_target_int_type(self, s3_connector):
        """Test integer type selection logic."""
        assert s3_connector._get_target_int_type("BIGINT") == "int64"
        assert s3_connector._get_target_int_type("INT") == "int32"
        assert s3_connector._get_target_int_type("INTEGER") == "int32"
        assert s3_connector._get_target_int_type("SMALLINT") == "int32"


class TestRealWorldScenarios:
    """Test real-world scenarios adapted from interactive.py."""

    def test_employee_data_with_nulls(self, s3_connector, tmp_path):
        """Test scenario from interactive.py lines 340-360."""
        df1 = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "hire_date": [None, None, None],
                "birth_date": [None, None, None],
                "salary": [50000.0, 60000.0, 55000.0],
            }
        )

        df2 = pd.DataFrame(
            {
                "id": [4, 5, 6],
                "name": ["David", "Eva", "Frank"],
                "hire_date": ["2015-01-01", None, None],
                "birth_date": [None, None, "2015-01-01"],
                "salary": [None, None, None],
            }
        )

        column_types = {
            "id": "BIGINT",
            "name": "VARCHAR(100)",
            "hire_date": "DATE",
            "birth_date": "DATE",
            "salary": "DECIMAL(12,2)",
        }

        # Process first DataFrame
        output_file1 = str(tmp_path / "employees1.parquet")
        df_prepared1 = s3_connector._prepare_pandas_for_redshift(df1, column_types)
        table1 = s3_connector._build_redshift_pyarrow_schema(df_prepared1, column_types)
        pq.write_table(table1, output_file1)

        # Process second DataFrame
        output_file2 = str(tmp_path / "employees2.parquet")
        df_prepared2 = s3_connector._prepare_pandas_for_redshift(df2, column_types)
        table2 = s3_connector._build_redshift_pyarrow_schema(df_prepared2, column_types)
        pq.write_table(table2, output_file2)

        # Both should have same schema
        result1 = pq.read_table(output_file1)
        result2 = pq.read_table(output_file2)

        assert result1.schema == result2.schema, "Schema should be identical for both DataFrames"

        # Verify DATE types
        assert str(result1.schema[2].type) == "date32[day]"
        assert str(result1.schema[3].type) == "date32[day]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
