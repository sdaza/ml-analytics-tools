from unittest.mock import MagicMock

import pytest

import ml_analytics.spark_connector as spark_module
from ml_analytics.spark_connector import SparkTableManager


@pytest.fixture(autouse=True)
def _reset_spark_ctx():
    """Keep the module-level cached Spark session from leaking between tests."""
    spark_module._spark_ctx = None
    yield
    spark_module._spark_ctx = None


def _manager(spark=None):
    spark = spark or MagicMock()
    df = MagicMock()
    spark.sql.return_value = df
    return SparkTableManager(spark=spark), spark, df


def test_sql_runs_inline_query():
    tm, spark, df = _manager()
    result = tm.sql("SELECT 1")
    spark.sql.assert_called_once_with("SELECT 1")
    assert result is df


def test_sql_formats_inline_query_with_kwargs():
    tm, spark, _ = _manager()
    tm.sql("SELECT {n} AS n", n=3)
    spark.sql.assert_called_once_with("SELECT 3 AS n")


def test_sql_preserves_comment_braces_when_formatting_inline():
    tm, spark, _ = _manager()
    tm.sql("-- campaign: exp-{tutor_id}\nSELECT {n}", n=1)
    spark.sql.assert_called_once_with("-- campaign: exp-{tutor_id}\nSELECT 1")


def test_sql_loads_sql_file(monkeypatch, tmp_path):
    sql_file = tmp_path / "q.sql"
    sql_file.write_text("SELECT {n} AS n")
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *a, **k: tmp_path)
    tm, spark, df = _manager()
    result = tm.sql("q.sql", n=5)
    spark.sql.assert_called_once_with("SELECT 5 AS n")
    assert result is df


def test_sql_missing_file_raises_before_spark(monkeypatch, tmp_path):
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *a, **k: tmp_path)
    tm, spark, _ = _manager()
    with pytest.raises(ValueError, match="Could not load SQL file"):
        tm.sql("missing.sql")
    spark.sql.assert_not_called()


def test_sql_return_pandas():
    tm, spark, df = _manager()
    pandas_df = object()
    df.toPandas.return_value = pandas_df
    result = tm.sql("SELECT 1", return_pandas=True)
    spark.sql.assert_called_once_with("SELECT 1")
    assert result is pandas_df


def test_sql_spark_kwarg_is_not_treated_as_template_var():
    tm, spark, _ = _manager()
    other = MagicMock()
    other.sql.return_value = MagicMock()
    tm.sql("SELECT {n} AS n", n=2, spark=other)
    other.sql.assert_called_once_with("SELECT 2 AS n")
    spark.sql.assert_not_called()
