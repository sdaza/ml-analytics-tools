import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import ml_analytics.sf_connector as sf_module
from ml_analytics import SFConnector


@pytest.fixture(autouse=True)
def _reset_spark_ctx():
    """Keep the module-level cached Spark session from leaking between tests."""
    sf_module._spark_ctx = None
    yield
    sf_module._spark_ctx = None

SNOWFLAKE_ENV = [
    "DATABRICKS_SECRET_SCOPE",
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
    "SNOWFLAKE_URL",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_WAREHOUSE",
]


class _DatabricksSecretsMock:
    def __init__(self, values):
        self._values = values

    def get(self, scope, key):
        value = self._values.get((scope, key))
        if value is None:
            raise KeyError(f"{scope}/{key}")
        return value


def _clear_snowflake_env(monkeypatch):
    for name in SNOWFLAKE_ENV:
        monkeypatch.delenv(name, raising=False)


def _mock_spark():
    """Spark double whose read chain returns a DataFrame mock."""
    spark = MagicMock()
    df = MagicMock()
    df.sparkSession = spark
    reader = spark.read.format.return_value
    reader.options.return_value.option.return_value.load.return_value = df
    reader.options.return_value.load.return_value = df
    return spark, df


def test_key_pair_options_from_secrets(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    monkeypatch.setenv("SNOWFLAKE_USER", "your.name@example.com")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "dr06406.eu-west-1")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "ANALYTICS_S_WH")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "DEEP_PURPLE")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "CDS")
    monkeypatch.setenv("SNOWFLAKE_ROLE", "SNOWFLAKE_STND_DATA")

    pem = "-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----"
    dbutils = MagicMock()
    dbutils.secrets = _DatabricksSecretsMock(
        {
            ("user-your.name@example.com", "snowflake_key"): pem,
            ("user-your.name@example.com", "snowflake_key_pass"): "key-pass",
        }
    )
    monkeypatch.setattr(builtins, "dbutils", dbutils, raising=False)

    with patch(
        "ml_analytics.sf_connector._load_private_key_pem_for_spark",
        return_value="spark-key",
    ) as mock_spark_key:
        options = SFConnector().spark_options()

    assert options == {
        "sfUrl": "dr06406.eu-west-1.snowflakecomputing.com",
        "sfUser": "your.name@example.com",
        "sfDatabase": "DEEP_PURPLE",
        "sfSchema": "CDS",
        "sfWarehouse": "ANALYTICS_S_WH",
        "sfRole": "SNOWFLAKE_STND_DATA",
        "pem_private_key": "spark-key",
    }
    mock_spark_key.assert_called_once_with(
        private_key=pem,
        private_key_path=None,
        passphrase="key-pass",
    )


def test_secret_scope_inferred_from_databricks_user(monkeypatch):
    """With no env vars or args, the scope is inferred from the Databricks user."""
    _clear_snowflake_env(monkeypatch)

    scope = "user-your.name@example.com"
    pem = "-----BEGIN ENCRYPTED PRIVATE KEY-----\nabc\n-----END ENCRYPTED PRIVATE KEY-----"
    dbutils = MagicMock()
    dbutils.secrets = _DatabricksSecretsMock(
        {
            (scope, "SNOWFLAKE_ACCOUNT"): "dr06406.eu-west-1",
            (scope, "snowflake_user"): "your.name@example.com",
            (scope, "SNOWFLAKE_DATABASE"): "DEEP_PURPLE",
            (scope, "SNOWFLAKE_SCHEMA"): "CDS",
            (scope, "SNOWFLAKE_WAREHOUSE"): "ANALYTICS_S_WH",
            (scope, "SNOWFLAKE_ROLE"): "SNOWFLAKE_STND_DATA",
            (scope, "snowflake_key"): pem,
            (scope, "snowflake_key_pass"): "key-pass",
        }
    )
    # dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    ctx = dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value
    ctx.userName.return_value.get.return_value = "your.name@example.com"
    monkeypatch.setattr(builtins, "dbutils", dbutils, raising=False)

    with patch(
        "ml_analytics.sf_connector._load_private_key_pem_for_spark",
        return_value="spark-key",
    ):
        options = SFConnector().spark_options()

    assert options == {
        "sfUrl": "dr06406.eu-west-1.snowflakecomputing.com",
        "sfUser": "your.name@example.com",
        "sfDatabase": "DEEP_PURPLE",
        "sfSchema": "CDS",
        "sfWarehouse": "ANALYTICS_S_WH",
        "sfRole": "SNOWFLAKE_STND_DATA",
        "pem_private_key": "spark-key",
    }


def test_full_url_is_not_double_suffixed(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="https://acct.eu-west-1.snowflakecomputing.com/", user="u")
    assert sf.spark_options()["sfUrl"] == "acct.eu-west-1.snowflakecomputing.com"


def test_password_options(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u", password="secret")
    options = sf.spark_options()
    assert options["sfPassword"] == "secret"
    assert "pem_private_key" not in options


def test_oauth_token_options(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u", token="tok")
    options = sf.spark_options()
    assert options["sfToken"] == "tok"
    assert options["sfAuthenticator"] == "oauth"


def test_extra_options_override(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u", role="A", extra_options={"sfRole": "B", "sfTimezone": "UTC"})
    options = sf.spark_options()
    assert options["sfRole"] == "B"
    assert options["sfTimezone"] == "UTC"


def test_externalbrowser_authenticator_raises(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u", authenticator="externalbrowser")
    with pytest.raises(ValueError, match="externalbrowser"):
        sf.spark_options()


def test_resolve_query_inline_passthrough(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    assert sf._resolve_query("SELECT 1") == "SELECT 1"


def test_resolve_query_loads_sql_file(monkeypatch, tmp_path):
    _clear_snowflake_env(monkeypatch)
    sql_file = tmp_path / "q.sql"
    sql_file.write_text("SELECT {n} AS n")
    monkeypatch.setattr("ml_analytics.sf_connector.find_project_root", lambda *a, **k: tmp_path, raising=False)
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *a, **k: tmp_path)
    sf = SFConnector(account="acct", user="u")
    assert sf._resolve_query("q.sql", n=5) == "SELECT 5 AS n"


def test_resolve_query_preserves_sql_file_comments(monkeypatch, tmp_path):
    _clear_snowflake_env(monkeypatch)
    sql_file = tmp_path / "q.sql"
    sql_file.write_text(
        """
-- leading comment
SELECT '-- keep string literal' AS value, {n} AS n -- inline comment
/* block comment */
"""
    )
    monkeypatch.setattr("ml_analytics.sf_connector.find_project_root", lambda *a, **k: tmp_path, raising=False)
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *a, **k: tmp_path)
    sf = SFConnector(account="acct", user="u")

    resolved = sf._resolve_query("q.sql", n=5)
    # Comments are preserved, not stripped; template substitution still applies.
    assert "-- leading comment" in resolved
    assert "-- inline comment" in resolved
    assert "/* block comment */" in resolved
    assert "SELECT '-- keep string literal' AS value, 5 AS n" in resolved


def test_wrap_query_for_connector_isolates_comments():
    sf = SFConnector(account="acct", user="u")
    wrapped = sf._wrap_query_for_connector("-- lead\nSELECT 1 AS n -- trail")
    # Comments stay; they're isolated on their own lines so the connector's own
    # wrapping cannot be commented out.
    assert wrapped == "SELECT * FROM (\n-- lead\nSELECT 1 AS n -- trail\n) AS ml_analytics_query"


def test_wrap_query_for_connector_handles_empty():
    sf = SFConnector(account="acct", user="u")
    assert sf._wrap_query_for_connector("") == ""
    assert sf._wrap_query_for_connector("   ") == "   "


def test_resolve_query_formats_inline_query_with_kwargs(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    resolved = sf._resolve_query("SELECT * FROM t WHERE d = '{date}' AND id IN ({ids})", date="2025-01-01", ids="1, 2")
    assert resolved == "SELECT * FROM t WHERE d = '2025-01-01' AND id IN (1, 2)"


def test_resolve_query_inline_without_kwargs_left_untouched(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    # No kwargs: literal braces (e.g. JSON / OBJECT_CONSTRUCT) must not be touched.
    query = "SELECT OBJECT_CONSTRUCT('a', 1) WHERE x = '{not_a_placeholder}'"
    assert sf._resolve_query(query) == query


def test_resolve_query_inline_with_comments_skips_format(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    # Comment carries a literal {tutor_id} doc placeholder that is NOT a template var.
    query = (
        "-- campaign: exp-target-raf-pilot-{tutor_id}_0_bau\n"
        "SELECT * FROM t WHERE d = '{date}'"
    )
    # Even with kwargs, a commented query is returned verbatim (no str.format).
    assert sf._resolve_query(query, date="2025-01-01") == query


def test_resolve_query_inline_block_comment_skips_format(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    query = "/* docs: url ?campaign={tutor_id} */\nSELECT 1"
    assert sf._resolve_query(query, tutor_id=99) == query


def test_resolve_query_inline_bad_template_raises(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    with pytest.raises(ValueError, match="formatting inline SQL"):
        sf._resolve_query("SELECT '{missing}'", date="2025-01-01")


def test_resolve_query_missing_file_raises(monkeypatch, tmp_path):
    _clear_snowflake_env(monkeypatch)
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *a, **k: tmp_path)
    sf = SFConnector(account="acct", user="u")
    with pytest.raises(ValueError, match="Could not load SQL file"):
        sf._resolve_query("missing.sql")


def test_qualified_uc_name_parts():
    assert SFConnector._qualified_uc_name("t", schema="s", catalog="c") == "c.s.t"
    assert SFConnector._qualified_uc_name("t", schema="s") == "s.t"
    assert SFConnector._qualified_uc_name("t") == "t"


def test_qualified_uc_name_already_qualified():
    # A dotted table name is treated as fully qualified; schema/catalog ignored.
    assert SFConnector._qualified_uc_name("cat.sch.tbl", schema="x", catalog="y") == "cat.sch.tbl"


def test_save_to_uc_uses_saveastable(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark = MagicMock()
    sf = SFConnector(account="acct", user="u", spark=spark)

    calls = {}

    class _Writer:
        def format(self, fmt):
            calls["format"] = fmt
            return self

        def option(self, key, value):
            calls["option"] = (key, value)
            return self

        def mode(self, m):
            calls["mode"] = m
            return self

        def saveAsTable(self, name):
            calls["name"] = name

    class _DF:
        write = _Writer()
        sparkSession = spark

    sf.save_to_uc(_DF(), table="tbl", schema="sch", catalog="cat", mode="append", drop_existing=False)
    assert calls == {
        "format": "delta",
        "option": ("mergeSchema", "true"),
        "mode": "append",
        "name": "cat.sch.tbl",
    }
    spark.sql.assert_called_once_with("OPTIMIZE cat.sch.tbl")


def test_save_to_uc_can_zorder_comment_or_skip_optimize(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark = MagicMock()
    sf = SFConnector(account="acct", user="u", spark=spark)

    df = MagicMock()
    df.sparkSession = spark

    sf.save_to_uc(
        df,
        table="tbl",
        schema="sch",
        catalog="cat",
        zorder_by=["customer_id", "event_date"],
        comment="Tutor's metrics",
        drop_existing=False,
    )
    assert [call.args[0] for call in spark.sql.call_args_list] == [
        "ALTER TABLE cat.sch.tbl SET TBLPROPERTIES ('comment' = 'Tutor''s metrics')",
        "OPTIMIZE cat.sch.tbl ZORDER BY (customer_id, event_date)",
    ]

    spark.reset_mock()
    sf.save_to_uc(df, table="tbl", schema="sch", catalog="cat", optimize=False, drop_existing=False)
    spark.sql.assert_not_called()


def test_save_to_uc_drops_and_overwrites_schema_by_default(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark = MagicMock()
    sf = SFConnector(account="acct", user="u", spark=spark)

    calls = {}

    class _Writer:
        def format(self, fmt):
            calls["format"] = fmt
            return self

        def option(self, key, value):
            calls["option"] = (key, value)
            return self

        def mode(self, m):
            calls["mode"] = m
            return self

        def saveAsTable(self, name):
            calls["name"] = name

    class _DF:
        write = _Writer()
        sparkSession = spark

    sf.save_to_uc(_DF(), table="tbl", schema="sch", catalog="cat", optimize=False)

    # Defaults: drop the table first, then overwrite the schema (not merge).
    assert calls == {
        "format": "delta",
        "option": ("overwriteSchema", "true"),
        "mode": "overwrite",
        "name": "cat.sch.tbl",
    }
    spark.sql.assert_called_once_with("DROP TABLE IF EXISTS cat.sch.tbl")


def test_save_to_uc_can_skip_drop(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark = MagicMock()
    sf = SFConnector(account="acct", user="u", spark=spark)

    df = MagicMock()
    df.sparkSession = spark

    sf.save_to_uc(df, table="tbl", schema="sch", catalog="cat", optimize=False, drop_existing=False)

    assert not any("DROP TABLE" in call.args[0] for call in spark.sql.call_args_list)


def test_save_to_uc_requires_table(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    sf = SFConnector(account="acct", user="u")
    with pytest.raises(ValueError, match="table name is required"):
        sf.save_to_uc(object(), table="")


def test_missing_account_raises(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    with pytest.raises(ValueError):
        SFConnector(user="u").spark_options()


def test_missing_user_raises(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    with pytest.raises(ValueError):
        SFConnector(account="acct").spark_options()


def test_get_spark_raises_clear_error_without_pyspark(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pyspark"):
            raise ImportError("No module named 'pyspark'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sf = SFConnector(account="acct", user="u")
    with pytest.raises(ImportError, match="PySpark"):
        sf._get_spark()


def test_get_spark_is_cached_and_shared(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    fake_session = MagicMock(name="active-session")
    fake_spark_sql = MagicMock()
    fake_spark_sql.SparkSession.getActiveSession.return_value = fake_session

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyspark.sql":
            return fake_spark_sql
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    first = sf_module.get_spark()
    second = sf_module.get_spark()

    assert first is fake_session
    assert second is fake_session
    # Active session resolved only once; second call returns the cached value.
    fake_spark_sql.SparkSession.getActiveSession.assert_called_once()


def test_sql_builds_reader_chain(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark, df = _mock_spark()
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    result = sf.sql("select 1")

    assert result is df
    spark.read.format.assert_called_once_with("net.snowflake.spark.snowflake")
    spark.read.format.return_value.options.return_value.option.assert_called_once_with(
        "query", "SELECT * FROM (\nselect 1\n) AS ml_analytics_query"
    )


def test_sql_lowercases_spark_columns(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark, df = _mock_spark()
    df.columns = ["USER_ID", "NPC"]
    normalized_df = MagicMock()
    df.toDF.return_value = normalized_df
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    result = sf.sql("select 1")

    assert result is normalized_df
    df.toDF.assert_called_once_with("user_id", "npc")


def test_sql_return_pandas(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark, df = _mock_spark()
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    sf.sql("select 1", return_pandas=True)

    df.toPandas.assert_called_once()


def test_sql_return_pandas_uses_normalized_dataframe(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark, df = _mock_spark()
    df.columns = ["USER_ID"]
    normalized_df = MagicMock()
    df.toDF.return_value = normalized_df
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    sf.sql("select 1", return_pandas=True)

    normalized_df.toPandas.assert_called_once()
    df.toPandas.assert_not_called()


def test_decimal_columns_are_cast_to_primitives(monkeypatch):
    class FakeDecimalType:
        def __init__(self, precision, scale):
            self.precision = precision
            self.scale = scale

    class FakeColumn:
        def __init__(self, name):
            self.name = name
            self.cast_type = None
            self.alias_name = None

        def cast(self, cast_type):
            self.cast_type = cast_type
            return self

        def alias(self, alias_name):
            self.alias_name = alias_name
            return self

    functions_module = ModuleType("pyspark.sql.functions")
    functions_module.col = lambda name: FakeColumn(name)
    types_module = ModuleType("pyspark.sql.types")
    types_module.DecimalType = FakeDecimalType
    sql_module = ModuleType("pyspark.sql")
    sql_module.functions = functions_module
    sql_module.types = types_module
    pyspark_module = ModuleType("pyspark")
    pyspark_module.sql = sql_module
    monkeypatch.setitem(sys.modules, "pyspark", pyspark_module)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql_module)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions_module)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", types_module)

    df = MagicMock()
    df.schema.fields = [
        SimpleNamespace(name="user_id", dataType=FakeDecimalType(38, 0)),
        SimpleNamespace(name="score", dataType=FakeDecimalType(10, 4)),
        SimpleNamespace(name="label", dataType=object()),
    ]
    normalized_df = MagicMock()
    df.select.return_value = normalized_df

    result = SFConnector._cast_decimal_columns_for_pandas_conversion(df)

    assert result is normalized_df
    selected_columns = df.select.call_args.args
    assert [(column.name, column.cast_type, column.alias_name) for column in selected_columns] == [
        ("`user_id`", "long", "user_id"),
        ("`score`", "double", "score"),
        ("`label`", None, "label"),
    ]


def test_save_pipeline_to_uc_uses_yaml_order_and_file_stem_tables(monkeypatch, tmp_path):
    _clear_snowflake_env(monkeypatch)
    folder = tmp_path / "queries"
    folder.mkdir()
    (folder / "base.sql").write_text("SELECT '{run_date}' AS run_date;")
    (folder / "features.sql").write_text("SELECT 1 AS feature;")
    (folder / "daily.yaml").write_text(
        """
steps:
  - features
  - base
"""
    )
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *args, **kwargs: tmp_path)

    spark, df = _mock_spark()
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    result = sf.save_pipeline_to_uc(
        "queries",
        pipeline="daily",
        catalog="prod",
        schema="analytics",
        run_date="2026-06-17",
    )

    # Returned DataFrame reads from the saved Delta table, not the Snowflake source.
    assert result is spark.table.return_value
    query_calls = spark.read.format.return_value.options.return_value.option.call_args_list
    assert [call.args for call in query_calls] == [
        ("query", "SELECT * FROM (\nSELECT 1 AS feature\n) AS ml_analytics_query"),
        ("query", "SELECT * FROM (\nSELECT '2026-06-17' AS run_date\n) AS ml_analytics_query"),
    ]
    save_calls = df.write.format.return_value.option.return_value.mode.return_value.saveAsTable.call_args_list
    assert [call.args[0] for call in save_calls] == [
        "prod.analytics.features",
        "prod.analytics.base",
    ]
    # drop_existing defaults to True, so each table is dropped before its write.
    assert [call.args[0] for call in spark.sql.call_args_list] == [
        "DROP TABLE IF EXISTS prod.analytics.features",
        "OPTIMIZE prod.analytics.features",
        "DROP TABLE IF EXISTS prod.analytics.base",
        "OPTIMIZE prod.analytics.base",
    ]


def test_save_pipeline_to_uc_allows_table_and_mode_overrides(monkeypatch, tmp_path):
    _clear_snowflake_env(monkeypatch)
    folder = tmp_path / "queries"
    folder.mkdir()
    (folder / "base.sql").write_text("SELECT 1 AS col_1;")
    (folder / "final.sql").write_text("SELECT 2 AS col_2;")
    monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *args, **kwargs: tmp_path)

    spark, df = _mock_spark()
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    result = sf.save_pipeline_to_uc(
        "queries",
        schema="analytics",
        catalog="prod",
        tables={"final": "churn_daily"},
        table_prefix="stg_",
        modes={"final": "append"},
        zorder_by={"final": "customer_id"},
        drop_existing=False,
        return_all=True,
    )

    assert result == {"base": spark.table.return_value, "final": spark.table.return_value}
    mode_calls = df.write.format.return_value.option.return_value.mode.call_args_list
    assert [call.args[0] for call in mode_calls] == ["overwrite", "append"]
    save_calls = df.write.format.return_value.option.return_value.mode.return_value.saveAsTable.call_args_list
    assert [call.args[0] for call in save_calls] == [
        "prod.analytics.stg_base",
        "prod.analytics.churn_daily",
    ]
    assert [call.args[0] for call in spark.sql.call_args_list] == [
        "OPTIMIZE prod.analytics.stg_base",
        "OPTIMIZE prod.analytics.churn_daily ZORDER BY (customer_id)",
    ]
