import builtins
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
    sf = SFConnector(account="acct", user="u")

    calls = {}

    class _Writer:
        def mode(self, m):
            calls["mode"] = m
            return self

        def saveAsTable(self, name):
            calls["name"] = name

    class _DF:
        write = _Writer()

    sf.save_to_uc(_DF(), table="tbl", schema="sch", catalog="cat", mode="append")
    assert calls == {"mode": "append", "name": "cat.sch.tbl"}


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
    spark.read.format.return_value.options.return_value.option.assert_called_once_with("query", "select 1")


def test_sql_return_pandas(monkeypatch):
    _clear_snowflake_env(monkeypatch)
    spark, df = _mock_spark()
    sf = SFConnector(account="acct", user="u", password="p", spark=spark)

    sf.sql("select 1", return_pandas=True)

    df.toPandas.assert_called_once()
