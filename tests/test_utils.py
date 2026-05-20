from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

from ml_analytics.utils import execute_sql_scripts, get_sql_files

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
