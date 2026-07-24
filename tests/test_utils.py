from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from ml_analytics.utils import (
    execute_sql_scripts,
    format_sql_ignoring_comments,
    get_sql_files,
    load_sql_query,
    strip_sql_comments,
)

SQL_FOLDER_NAME = "queries"


@pytest.fixture()
def project_root(tmp_path):
    """Temporary project root with a queries/ subfolder containing .sql files."""
    folder = tmp_path / SQL_FOLDER_NAME
    folder.mkdir()
    for name in ["alpha", "beta", "gamma"]:
        (folder / f"{name}.sql").write_text(f"SELECT '{name}';")
    return tmp_path


def _patch_root(project_root):
    """Return a context manager that makes find_project_root return project_root."""
    return patch("ml_analytics.utils.find_project_root", return_value=project_root)


class TestGetSqlFilesFallback:
    def test_returns_alphabetical_order_without_yaml(self, project_root):
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["alpha", "beta", "gamma"]

    def test_values_are_paths(self, project_root):
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert all(isinstance(v, Path) for v in result.values())

    def test_empty_folder_returns_empty_dict(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with _patch_root(tmp_path):
            result = get_sql_files("empty")
        assert result == {}


class TestGetSqlFilesYamlMode:
    def test_yaml_defines_order(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(
            dedent("""\
                name: test_pipeline
                steps:
                  - gamma
                  - alpha
                  - beta
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["gamma", "alpha", "beta"]

    def test_yaml_only_includes_listed_steps(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(
            dedent("""\
                name: test_pipeline
                steps:
                  - gamma
                  - alpha
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["gamma", "alpha"]
        assert "beta" not in result

    def test_missing_sql_file_is_skipped_with_warning(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(
            dedent("""\
                name: test_pipeline
                steps:
                  - alpha
                  - nonexistent
                  - beta
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["alpha", "beta"]
        assert "nonexistent" not in result

    def test_malformed_yaml_falls_back_to_alphabetical(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(":: this is not valid yaml ::")
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["alpha", "beta", "gamma"]

    def test_yaml_without_steps_key_falls_back_to_alphabetical(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(
            dedent("""\
                name: test_pipeline
                description: no steps key here
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["alpha", "beta", "gamma"]

    def test_yaml_with_empty_steps_falls_back_to_alphabetical(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(
            dedent("""\
                name: test_pipeline
                steps: []
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["alpha", "beta", "gamma"]

    def test_paths_resolve_to_correct_sql_files(self, project_root):
        (project_root / SQL_FOLDER_NAME / "pipeline.yaml").write_text(
            dedent("""\
                name: test_pipeline
                steps:
                  - beta
                  - alpha
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        sql_folder = project_root / SQL_FOLDER_NAME
        assert result["beta"] == sql_folder / "beta.sql"
        assert result["alpha"] == sql_folder / "alpha.sql"


class TestLoadSqlQuery:
    def test_relative_path_falls_back_to_cwd_without_project_root(self, monkeypatch, tmp_path):
        sql_file = tmp_path / "sql" / "experiment.sql"
        sql_file.parent.mkdir()
        sql_file.write_text("SELECT {n} AS n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *args, **kwargs: None)
        monkeypatch.setattr("ml_analytics.utils._databricks_notebook_dir", lambda: None)

        assert load_sql_query("sql/experiment.sql", n=7) == "SELECT 7 AS n"

    def test_relative_path_falls_back_to_databricks_notebook_dir(self, monkeypatch, tmp_path):
        driver_dir = tmp_path / "driver"
        notebook_dir = tmp_path / "Workspace" / "Users" / "me" / "exploring"
        sql_file = notebook_dir / "sql" / "experiment.sql"
        driver_dir.mkdir()
        sql_file.parent.mkdir(parents=True)
        sql_file.write_text("SELECT 1")

        monkeypatch.chdir(driver_dir)
        monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *args, **kwargs: None)
        monkeypatch.setattr("ml_analytics.utils._databricks_notebook_dir", lambda: notebook_dir)

        assert load_sql_query("sql/experiment.sql") == "SELECT 1"

    def test_strip_comments_before_template_substitution(self, monkeypatch, tmp_path):
        sql_file = tmp_path / "sql" / "experiment.sql"
        sql_file.parent.mkdir()
        sql_file.write_text("-- comment with {missing_placeholder}\nSELECT {n} AS n")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *args, **kwargs: None)
        monkeypatch.setattr("ml_analytics.utils._databricks_notebook_dir", lambda: None)

        assert load_sql_query("sql/experiment.sql", strip_comments=True, n=7) == "SELECT 7 AS n"

    def test_substitutes_placeholders_in_commented_script(self, monkeypatch, tmp_path):
        # A commented script keeps real placeholders; braces in the comment are left alone.
        sql_file = tmp_path / "sql" / "experiment.sql"
        sql_file.parent.mkdir()
        sql_file.write_text("-- docs: see {tutor_id} url pattern\nSELECT * FROM t WHERE d = '{start_date}'")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("ml_analytics.utils.find_project_root", lambda *args, **kwargs: None)
        monkeypatch.setattr("ml_analytics.utils._databricks_notebook_dir", lambda: None)

        result = load_sql_query("sql/experiment.sql", start_date="2026-01-01")
        assert result == "-- docs: see {tutor_id} url pattern\nSELECT * FROM t WHERE d = '2026-01-01'"


class TestFormatSqlIgnoringComments:
    def test_no_kwargs_returns_verbatim(self):
        sql = "-- {x}\nSELECT {y}"
        assert format_sql_ignoring_comments(sql) == sql

    def test_substitutes_code_placeholders(self):
        assert format_sql_ignoring_comments("SELECT {n} AS n", n=7) == "SELECT 7 AS n"

    def test_skips_single_line_comment(self):
        sql = "SELECT {n}  -- keep {not_a_var}\nFROM t"
        assert format_sql_ignoring_comments(sql, n=1) == "SELECT 1  -- keep {not_a_var}\nFROM t"

    def test_skips_block_comment(self):
        sql = "/* {leave} me */ SELECT {n}"
        assert format_sql_ignoring_comments(sql, n=1) == "/* {leave} me */ SELECT 1"

    def test_substitutes_placeholder_inside_string_literal(self):
        # The common pattern: a placeholder wrapped in single quotes must substitute.
        sql = "SELECT * FROM t WHERE d BETWEEN '{start}' AND '{end}'"
        assert format_sql_ignoring_comments(sql, start="2026-01-01", end="2026-03-31") == (
            "SELECT * FROM t WHERE d BETWEEN '2026-01-01' AND '2026-03-31'"
        )

    def test_comment_token_inside_string_is_not_a_comment(self):
        # A '--' inside a string literal must not start a comment; {n} after it substitutes.
        sql = "SELECT 'a -- b' AS v, {n} AS n"
        assert format_sql_ignoring_comments(sql, n=1) == "SELECT 'a -- b' AS v, 1 AS n"

    def test_handles_doubled_quote_escape_in_string(self):
        sql = "SELECT 'it''s here' AS v, {n} AS n"
        assert format_sql_ignoring_comments(sql, n=3) == "SELECT 'it''s here' AS v, 3 AS n"

    def test_literal_braces_in_string_must_be_escaped(self):
        # JSON-style braces in a string get formatted too, so they must be escaped as {{ }}.
        sql = "SELECT '{{\"a\": 1}}' AS j, {n} AS n"
        assert format_sql_ignoring_comments(sql, n=2) == 'SELECT \'{"a": 1}\' AS j, 2 AS n'

    def test_escaped_braces_in_code(self):
        # Doubled braces in a code region are unescaped by str.format as usual.
        assert format_sql_ignoring_comments("SELECT {n} AS {{lit}}", n=1) == "SELECT 1 AS {lit}"

    def test_missing_placeholder_raises_keyerror(self):
        with pytest.raises(KeyError):
            format_sql_ignoring_comments("SELECT {missing}", n=1)


class TestStripSqlComments:
    def test_strips_line_and_block_comments(self):
        sql = dedent("""\
            -- leading comment
            SELECT 1 AS id, -- inline comment
                   2 AS value
            /* block comment */
            FROM table
        """)

        assert strip_sql_comments(sql) == "SELECT 1 AS id,\n       2 AS value\n\nFROM table"

    def test_preserves_comment_tokens_inside_strings(self):
        sql = "SELECT '-- not a comment' AS value, '/* also not */' AS other -- trailing"

        assert strip_sql_comments(sql) == "SELECT '-- not a comment' AS value, '/* also not */' AS other"


# ---------------------------------------------------------------------------
# execute_sql_scripts — folder path inputs
# ---------------------------------------------------------------------------


def _make_sql_folder(tmp_path, names=("alpha", "beta")):
    """Create tmp_path/queries/ with simple DDL .sql files and return the folder."""
    folder = tmp_path / SQL_FOLDER_NAME
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / f"{name}.sql").write_text(f"DROP TABLE IF EXISTS {name}_table")
    return folder


def _mock_dc():
    """Return a MagicMock that satisfies the DataConnector protocol used inside execute_sql_scripts."""
    dc = MagicMock()
    dc.cursor = MagicMock()
    dc.cursor.execute = MagicMock()
    return dc


class TestExecuteSqlScriptsFolderInput:
    def test_str_folder_path_resolves_and_executes(self, tmp_path):
        _make_sql_folder(tmp_path)
        dc = _mock_dc()
        with _patch_root(tmp_path):
            execute_sql_scripts(SQL_FOLDER_NAME, data_connector=dc)
        assert dc.cursor.execute.called

    def test_path_directory_resolves_and_executes(self, tmp_path):
        folder = _make_sql_folder(tmp_path)
        dc = _mock_dc()
        with _patch_root(tmp_path):
            execute_sql_scripts(folder, data_connector=dc)
        assert dc.cursor.execute.called

    def test_str_folder_respects_yaml_order(self, tmp_path):
        folder = _make_sql_folder(tmp_path, names=["alpha", "beta"])
        (folder / "pipeline.yaml").write_text(
            dedent("""\
                name: test
                steps:
                  - beta
                  - alpha
            """)
        )
        dc = _mock_dc()
        executed_order = []
        dc.cursor.execute.side_effect = lambda stmt: executed_order.append(stmt)

        with _patch_root(tmp_path):
            execute_sql_scripts(SQL_FOLDER_NAME, data_connector=dc)

        assert executed_order[0] == "DROP TABLE IF EXISTS beta_table"
        assert executed_order[1] == "DROP TABLE IF EXISTS alpha_table"

    def test_str_empty_folder_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with _patch_root(tmp_path):
            with pytest.raises(ValueError, match="No SQL files found"):
                execute_sql_scripts("empty")

    def test_single_file_path_still_works(self, tmp_path):
        folder = _make_sql_folder(tmp_path, names=["alpha"])
        dc = _mock_dc()
        with _patch_root(tmp_path):
            execute_sql_scripts(folder / "alpha.sql", data_connector=dc)
        assert dc.cursor.execute.called

    def test_dict_with_string_values_executes(self, tmp_path):
        folder = _make_sql_folder(tmp_path, names=["alpha", "beta"])
        dc = _mock_dc()
        explicit = {
            "alpha": str(folder / "alpha.sql"),
            "beta": str(folder / "beta.sql"),
        }
        with _patch_root(tmp_path):
            execute_sql_scripts(explicit, data_connector=dc)
        assert dc.cursor.execute.called

    def test_dict_with_path_values_executes(self, tmp_path):
        folder = _make_sql_folder(tmp_path, names=["alpha", "beta"])
        dc = _mock_dc()
        explicit = {
            "alpha": folder / "alpha.sql",
            "beta": folder / "beta.sql",
        }
        with _patch_root(tmp_path):
            execute_sql_scripts(explicit, data_connector=dc)
        assert dc.cursor.execute.called

    def test_list_of_paths_executes_in_order(self, tmp_path):
        folder = _make_sql_folder(tmp_path, names=["alpha", "beta"])
        dc = _mock_dc()
        executed_order = []
        dc.cursor.execute.side_effect = lambda stmt: executed_order.append(stmt)
        path_list = [folder / "beta.sql", folder / "alpha.sql"]
        with _patch_root(tmp_path):
            execute_sql_scripts(path_list, data_connector=dc)
        assert executed_order[0] == "DROP TABLE IF EXISTS beta_table"
        assert executed_order[1] == "DROP TABLE IF EXISTS alpha_table"

    def test_list_of_strings_executes(self, tmp_path):
        folder = _make_sql_folder(tmp_path, names=["alpha"])
        dc = _mock_dc()
        with _patch_root(tmp_path):
            execute_sql_scripts([str(folder / "alpha.sql")], data_connector=dc)
        assert dc.cursor.execute.called

    def test_str_single_sql_file_path_executes(self, tmp_path):
        _make_sql_folder(tmp_path, names=["alpha"])
        dc = _mock_dc()
        relative_path = f"{SQL_FOLDER_NAME}/alpha.sql"
        with _patch_root(tmp_path):
            execute_sql_scripts(relative_path, data_connector=dc)
        assert dc.cursor.execute.called


# ---------------------------------------------------------------------------
# Named pipeline support — layout A (separate file) and layout B (named sections)
# ---------------------------------------------------------------------------


class TestGetSqlFilesNamedPipeline:
    """Tests for get_sql_files(pipeline=...) named pipeline selection."""

    def test_layout_a_separate_yaml_file(self, project_root):
        """pipeline='daily' resolves from daily.yaml (layout A)."""
        (project_root / SQL_FOLDER_NAME / "daily.yaml").write_text(
            dedent("""\
                name: daily
                steps:
                  - gamma
                  - alpha
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME, pipeline="daily")
        assert list(result.keys()) == ["gamma", "alpha"]

    def test_layout_a_only_includes_listed_steps(self, project_root):
        """Layout A excludes SQL files not listed in its steps."""
        (project_root / SQL_FOLDER_NAME / "daily.yaml").write_text(
            dedent("""\
                name: daily
                steps:
                  - beta
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME, pipeline="daily")
        assert list(result.keys()) == ["beta"]
        assert "alpha" not in result
        assert "gamma" not in result

    def test_layout_b_named_section_in_any_yaml_file(self, project_root):
        """pipeline='weekly' resolves from pipelines.weekly.steps in any YAML (layout B)."""
        (project_root / SQL_FOLDER_NAME / "my_config.yaml").write_text(
            dedent("""\
                pipelines:
                  daily:
                    steps:
                      - alpha
                      - beta
                  weekly:
                    steps:
                      - gamma
                      - beta
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME, pipeline="weekly")
        assert list(result.keys()) == ["gamma", "beta"]

    def test_layout_b_different_section_same_file(self, project_root):
        """Two different named sections from the same YAML file resolve independently."""
        (project_root / SQL_FOLDER_NAME / "my_config.yaml").write_text(
            dedent("""\
                pipelines:
                  daily:
                    steps:
                      - alpha
                      - beta
                  weekly:
                    steps:
                      - gamma
            """)
        )
        with _patch_root(project_root):
            daily = get_sql_files(SQL_FOLDER_NAME, pipeline="daily")
            weekly = get_sql_files(SQL_FOLDER_NAME, pipeline="weekly")
        assert list(daily.keys()) == ["alpha", "beta"]
        assert list(weekly.keys()) == ["gamma"]

    def test_layout_a_takes_precedence_over_layout_b(self, project_root):
        """When both daily.yaml and a pipelines.daily section exist, daily.yaml wins."""
        (project_root / SQL_FOLDER_NAME / "daily.yaml").write_text(
            dedent("""\
                steps:
                  - gamma
            """)
        )
        (project_root / SQL_FOLDER_NAME / "all_pipelines.yaml").write_text(
            dedent("""\
                pipelines:
                  daily:
                    steps:
                      - alpha
                      - beta
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME, pipeline="daily")
        assert list(result.keys()) == ["gamma"]

    def test_unknown_pipeline_name_falls_back_to_alphabetical(self, project_root):
        """A pipeline name that does not exist in any YAML falls back to alphabetical order."""
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME, pipeline="nonexistent")
        assert list(result.keys()) == ["alpha", "beta", "gamma"]

    def test_layout_b_list_of_objects_format(self, project_root):
        """pipelines as a list of {name, steps} objects is supported."""
        (project_root / SQL_FOLDER_NAME / "pipelines.yaml").write_text(
            dedent("""\
                pipelines:
                  - name: daily
                    steps:
                      - alpha
                      - beta
                  - name: weekly
                    steps:
                      - gamma
            """)
        )
        with _patch_root(project_root):
            daily = get_sql_files(SQL_FOLDER_NAME, pipeline="daily")
            weekly = get_sql_files(SQL_FOLDER_NAME, pipeline="weekly")
        assert list(daily.keys()) == ["alpha", "beta"]
        assert list(weekly.keys()) == ["gamma"]

    def test_single_arbitrary_yaml_auto_discovered(self, project_root):
        """A single YAML with any name is auto-discovered when pipeline= is not given."""
        (project_root / SQL_FOLDER_NAME / "etl_config.yaml").write_text(
            dedent("""\
                steps:
                  - gamma
                  - alpha
            """)
        )
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["gamma", "alpha"]

    def test_multiple_yamls_without_pipeline_falls_back_to_alphabetical(self, project_root):
        """Multiple YAMLs with no pipeline= selection warns and falls back to alphabetical."""
        (project_root / SQL_FOLDER_NAME / "daily.yaml").write_text("steps:\n  - gamma\n")
        (project_root / SQL_FOLDER_NAME / "weekly.yaml").write_text("steps:\n  - alpha\n")
        with _patch_root(project_root):
            result = get_sql_files(SQL_FOLDER_NAME)
        assert list(result.keys()) == ["alpha", "beta", "gamma"]

    def test_execute_sql_scripts_passes_pipeline_kwarg(self, tmp_path):
        """execute_sql_scripts(pipeline=...) threads the name through to get_sql_files."""
        folder = _make_sql_folder(tmp_path, names=["alpha", "beta"])
        (folder / "daily.yaml").write_text(
            dedent("""\
                steps:
                  - beta
                  - alpha
            """)
        )
        dc = _mock_dc()
        executed_order = []
        dc.cursor.execute.side_effect = lambda stmt: executed_order.append(stmt)

        with _patch_root(tmp_path):
            execute_sql_scripts(SQL_FOLDER_NAME, data_connector=dc, pipeline="daily")

        assert executed_order[0] == "DROP TABLE IF EXISTS beta_table"
        assert executed_order[1] == "DROP TABLE IF EXISTS alpha_table"


class TestDatabricksCurrentUser:
    """databricks_current_user resolution, including the databricks-sdk fallback."""

    @pytest.fixture(autouse=True)
    def _reset_caches(self, monkeypatch):
        from ml_analytics import utils

        monkeypatch.setattr(utils, "_SDK_CURRENT_USER", None)
        monkeypatch.setattr(utils, "_SDK_CURRENT_USER_RESOLVED", False)
        monkeypatch.setattr(utils, "_DBUTILS_RESOLVED", True)

    def test_returns_none_without_dbutils(self, monkeypatch):
        from ml_analytics import utils

        monkeypatch.setattr(utils, "_DBUTILS", None)
        assert utils.databricks_current_user() is None

    def test_notebook_context_user_wins(self, monkeypatch):
        from ml_analytics import utils

        dbutils = MagicMock()
        dbutils.notebook.entry_point.getDbutils.return_value.notebook.return_value.getContext.return_value.userName.return_value.get.return_value = "notebook.user@example.com"  # noqa: E501
        monkeypatch.setattr(utils, "_DBUTILS", dbutils)

        assert utils.databricks_current_user() == "notebook.user@example.com"

    @staticmethod
    def _install_fake_sdk(monkeypatch, workspace_client):
        """
        Register a fake ``databricks.sdk`` module exposing ``workspace_client``.

        databricks-sdk is not a dependency of this package (the fallback import
        is best-effort), so the real module may be absent — e.g. in CI. Faking
        it in sys.modules keeps these tests runnable everywhere.
        """
        import sys
        import types

        sdk_module = types.ModuleType("databricks.sdk")
        sdk_module.WorkspaceClient = workspace_client
        package_module = types.ModuleType("databricks")
        package_module.sdk = sdk_module
        monkeypatch.setitem(sys.modules, "databricks", package_module)
        monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_module)

    def test_falls_back_to_workspace_client_and_caches(self, monkeypatch):
        from ml_analytics import utils

        # A dbutils without a notebook context (the remote databricks-sdk case).
        monkeypatch.setattr(utils, "_DBUTILS", object())

        mock_ws = MagicMock()
        mock_ws.return_value.current_user.me.return_value.user_name = "sdk.user@example.com"
        self._install_fake_sdk(monkeypatch, mock_ws)

        assert utils.databricks_current_user() == "sdk.user@example.com"
        assert utils.databricks_current_user() == "sdk.user@example.com"
        mock_ws.assert_called_once()

    def test_workspace_client_failure_returns_none(self, monkeypatch):
        from ml_analytics import utils

        monkeypatch.setattr(utils, "_DBUTILS", object())
        self._install_fake_sdk(monkeypatch, MagicMock(side_effect=RuntimeError("no config")))

        assert utils.databricks_current_user() is None

    def test_missing_sdk_returns_none(self, monkeypatch):
        import sys

        from ml_analytics import utils

        monkeypatch.setattr(utils, "_DBUTILS", object())
        # Simulate databricks-sdk not being installed at all (e.g. CI):
        # a None entry in sys.modules makes the import raise ImportError.
        monkeypatch.setitem(sys.modules, "databricks", None)
        monkeypatch.setitem(sys.modules, "databricks.sdk", None)

        assert utils.databricks_current_user() is None


class TestDisplay:
    def test_uses_runtime_display_when_present(self, monkeypatch):
        from ml_analytics import utils

        calls = []

        def fake_display(obj):
            calls.append(obj)
            return "shown"

        monkeypatch.setattr(utils, "_resolve_runtime_display", lambda: fake_display)
        assert utils.display({"a": 1}) == "shown"
        assert calls == [{"a": 1}]

    def test_spark_like_show_fallback(self, monkeypatch):
        from ml_analytics import utils

        monkeypatch.setattr(utils, "_resolve_runtime_display", lambda: None)
        obj = MagicMock()
        obj.show = MagicMock(return_value=None)
        # MagicMock has toPandas by default via attribute access — remove it
        del obj.toPandas

        utils.display(obj, n=10)
        obj.show.assert_called_once_with(10)

    def test_topandas_fallback(self, monkeypatch, capsys):
        from ml_analytics import utils

        monkeypatch.setattr(utils, "_resolve_runtime_display", lambda: None)
        obj = MagicMock(spec=["limit", "toPandas"])
        limited = MagicMock()
        limited.toPandas.return_value = "preview"
        obj.limit.return_value = limited

        utils.display(obj, n=5)
        obj.limit.assert_called_once_with(5)
        assert "preview" in capsys.readouterr().out

    def test_print_fallback(self, monkeypatch, capsys):
        from ml_analytics import utils

        monkeypatch.setattr(utils, "_resolve_runtime_display", lambda: None)
        utils.display("hello")
        assert "hello" in capsys.readouterr().out

    def test_resolve_skips_own_display(self, monkeypatch):
        """Importing our display into builtins must not recurse."""
        import builtins
        import sys

        from ml_analytics import utils

        monkeypatch.setattr(builtins, "display", utils.display, raising=False)
        monkeypatch.setitem(sys.modules, "databricks", None)
        monkeypatch.setitem(sys.modules, "databricks.sdk", None)
        monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", None)
        import __main__

        monkeypatch.setattr(__main__, "display", utils.display, raising=False)
        monkeypatch.setitem(sys.modules, "IPython", None)

        assert utils._resolve_runtime_display() is None

    def test_resolve_skips_sdk_runtime_off_cluster(self, monkeypatch):
        """Off-cluster must not import databricks.sdk.runtime (can hang under uv run)."""
        import builtins
        import sys

        from ml_analytics import utils

        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.delattr(builtins, "display", raising=False)
        import __main__

        monkeypatch.delattr(__main__, "display", raising=False)
        monkeypatch.setitem(sys.modules, "IPython", None)

        imported = {"runtime": False}

        class _BoomFinder:
            def find_spec(self, fullname, path, target=None):
                if fullname == "databricks.sdk.runtime" or fullname.startswith("databricks.sdk.runtime."):
                    imported["runtime"] = True
                    raise AssertionError("databricks.sdk.runtime must not be imported off-cluster")
                return None

        monkeypatch.setattr(sys, "meta_path", [_BoomFinder(), *sys.meta_path])
        for key in list(sys.modules):
            if key == "databricks" or key.startswith("databricks."):
                monkeypatch.delitem(sys.modules, key, raising=False)

        assert utils._resolve_runtime_display() is None
        assert imported["runtime"] is False

    def test_resolve_uses_sdk_runtime_on_cluster(self, monkeypatch):
        """On Databricks runtime, databricks.sdk.runtime.display is eligible."""
        import builtins
        import sys
        import types

        from ml_analytics import utils

        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        monkeypatch.delattr(builtins, "display", raising=False)
        import __main__

        monkeypatch.delattr(__main__, "display", raising=False)
        monkeypatch.setitem(sys.modules, "IPython", None)

        fake_display = lambda obj: ("sdk", obj)
        fake_display.__module__ = "databricks.sdk.runtime"

        runtime_mod = types.ModuleType("databricks.sdk.runtime")
        runtime_mod.display = fake_display
        sdk_mod = types.ModuleType("databricks.sdk")
        sdk_mod.runtime = runtime_mod
        package_mod = types.ModuleType("databricks")
        package_mod.sdk = sdk_mod
        monkeypatch.setitem(sys.modules, "databricks", package_mod)
        monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_mod)
        monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", runtime_mod)

        assert utils._resolve_runtime_display() is fake_display


class TestGetDbutils:
    """_get_dbutils must not import databricks.sdk.runtime off-cluster."""

    def test_skips_sdk_runtime_off_cluster(self, monkeypatch):
        import builtins
        import sys

        from ml_analytics import utils

        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.setattr(utils, "_DBUTILS", None)
        monkeypatch.setattr(utils, "_DBUTILS_RESOLVED", False)
        monkeypatch.delattr(builtins, "dbutils", raising=False)
        import __main__

        monkeypatch.delattr(__main__, "dbutils", raising=False)
        monkeypatch.setitem(sys.modules, "IPython", None)

        imported = {"runtime": False}

        class _BoomFinder:
            def find_spec(self, fullname, path, target=None):
                if fullname == "databricks.sdk.runtime" or fullname.startswith("databricks.sdk.runtime."):
                    imported["runtime"] = True
                    raise AssertionError("databricks.sdk.runtime must not be imported off-cluster")
                return None

        monkeypatch.setattr(sys, "meta_path", [_BoomFinder(), *sys.meta_path])
        for key in list(sys.modules):
            if key == "databricks" or key.startswith("databricks."):
                monkeypatch.delitem(sys.modules, key, raising=False)

        assert utils._get_dbutils() is None
        assert imported["runtime"] is False

    def test_uses_sdk_runtime_on_cluster(self, monkeypatch):
        import builtins
        import sys
        import types

        from ml_analytics import utils

        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        monkeypatch.setattr(utils, "_DBUTILS", None)
        monkeypatch.setattr(utils, "_DBUTILS_RESOLVED", False)
        monkeypatch.delattr(builtins, "dbutils", raising=False)
        import __main__

        monkeypatch.delattr(__main__, "dbutils", raising=False)
        monkeypatch.setitem(sys.modules, "IPython", None)

        fake_dbutils = object()
        runtime_mod = types.ModuleType("databricks.sdk.runtime")
        runtime_mod.dbutils = fake_dbutils
        sdk_mod = types.ModuleType("databricks.sdk")
        sdk_mod.runtime = runtime_mod
        package_mod = types.ModuleType("databricks")
        package_mod.sdk = sdk_mod
        monkeypatch.setitem(sys.modules, "databricks", package_mod)
        monkeypatch.setitem(sys.modules, "databricks.sdk", sdk_mod)
        monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", runtime_mod)

        assert utils._get_dbutils() is fake_dbutils


class TestShow:
    def test_closes_under_agg(self, monkeypatch):
        import sys
        import types

        from ml_analytics import utils

        mock_plt = MagicMock()
        mock_matplotlib = types.ModuleType("matplotlib")
        mock_matplotlib.get_backend = MagicMock(return_value="Agg")
        monkeypatch.setitem(sys.modules, "matplotlib", mock_matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)

        utils.show()
        mock_plt.close.assert_called_once()
        mock_plt.show.assert_not_called()

    def test_shows_on_interactive_backend(self, monkeypatch):
        import sys
        import types

        from ml_analytics import utils

        mock_plt = MagicMock()
        mock_matplotlib = types.ModuleType("matplotlib")
        mock_matplotlib.get_backend = MagicMock(return_value="MacOSX")
        monkeypatch.setitem(sys.modules, "matplotlib", mock_matplotlib)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)

        utils.show()
        mock_plt.show.assert_called_once()
        mock_plt.close.assert_not_called()
