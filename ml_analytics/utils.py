"""
Collection of helper methods. These should be fully generic and make no
assumptions about the format of input data.
"""

import glob
import logging
import os
import time
from pathlib import Path

import yaml

_PROJECT_ROOT = None


def get_logger(name: str) -> logging.Logger:
    """ "
    Get a logger with the specified name, ensuring handlers are not duplicated.

    :param name: The name of the logger.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check the logger's own handlers, not hasHandlers(): hasHandlers() walks up
    # to the root logger, which is pre-configured in environments like Databricks,
    # leaving our messages with the root handler's raw "LEVEL:name:message" format.
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def log_and_raise_error(logger: logging.Logger, message: str, exception_type: type[Exception] = ValueError) -> None:
    """ "
    Logs an error message and raises an exception of the specified type.

    :param logger: The logger instance to use. # Keep this signature
    :param message: The error message to log and raise.
    :param exception_type: The type of exception to raise (default is ValueError).
    """
    logger.error(message)
    raise exception_type(message)


_DBUTILS = None
_DBUTILS_RESOLVED = False


def _get_dbutils():
    """
    Return a Databricks ``dbutils`` handle if running on Databricks, else None.

    Resolved lazily and cached. Imposes no hard dependency on Databricks: tries
    ``databricks.sdk.runtime`` (preinstalled on the Databricks runtime, works in
    both notebooks and jobs), then falls back to a notebook-injected ``dbutils``
    global. Returns None anywhere else (e.g. local or non-Spark environments),
    so callers can simply skip the secret-scope lookup.
    """
    global _DBUTILS, _DBUTILS_RESOLVED
    if _DBUTILS_RESOLVED:
        return _DBUTILS

    _DBUTILS_RESOLVED = True
    try:
        import builtins

        _DBUTILS = getattr(builtins, "dbutils", None)
        if _DBUTILS is not None:
            return _DBUTILS
    except Exception:
        pass

    try:
        import __main__

        _DBUTILS = getattr(__main__, "dbutils", None)
        if _DBUTILS is not None:
            return _DBUTILS
    except Exception:
        pass

    try:
        from databricks.sdk.runtime import dbutils

        _DBUTILS = dbutils
        return _DBUTILS
    except Exception:
        pass

    # Fallback: dbutils injected into the notebook's interactive namespace.
    try:
        import IPython

        ip = IPython.get_ipython()
        if ip is not None:
            _DBUTILS = ip.user_ns.get("dbutils")
    except Exception:
        _DBUTILS = None

    return _DBUTILS


def _databricks_notebook_dir() -> Path | None:
    """
    Return the active Databricks notebook's workspace filesystem directory.

    Databricks sets Python's cwd to the driver directory in many notebook/job
    contexts, so relative workspace files need the notebook path as an anchor.
    """
    dbutils = _get_dbutils()
    if dbutils is None:
        return None

    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        notebook_path = ctx.notebookPath().get()
    except Exception:
        return None

    if not notebook_path:
        return None

    return (Path("/Workspace") / str(notebook_path).lstrip("/")).parent


def get_credential_value(name, scope="ml"):
    """
    Get the value of the credential or variable called "name" in different environments.

    Resolution order (first hit wins):
        1. Environment variable ``name`` — works everywhere; primary path for
           local/non-Spark environments and allows overriding other sources.
        2. Databricks secret scope via ``dbutils.secrets.get(scope, name)`` —
           used automatically when running on Databricks; skipped elsewhere.
        3. SecretProvider mount file at ``/mnt/<scope>/<name>`` (legacy).

    :param name: The name of the variable or credential to load
    :param scope: The Databricks secret scope / SecretProvider mount scope (default: "ml")
    :return: The value of the variable or credential
    """
    value = os.getenv(name)
    if value is not None:
        return value

    # Databricks secret scope (no-op outside Databricks).
    dbutils = _get_dbutils()
    if dbutils is not None:
        try:
            secret = dbutils.secrets.get(scope=scope, key=name)
            if secret:
                return secret
        except Exception:
            # Scope/key missing or secrets API unavailable; fall through.
            pass

    file_path = f"/mnt/{scope}/{name}"
    try:
        with open(file_path) as file:
            value = file.read().strip()
            if value:
                return value
    except FileNotFoundError:
        pass
    except OSError as e:
        raise Exception(f"Error reading the file {file_path}: {e}") from e

    raise Exception(
        f"Credential or variable '{name}' not found in environment variable, "
        f"Databricks secret scope '{scope}', or SecretProvider mount file."
    )


def find_project_root(marker_files: list[str] = None, required: bool = True) -> Path | None:
    """
    Finds the project root directory by searching upwards for marker files/dirs.

    Tries searching from the script's directory first, then falls back to the
    current working directory. Caches the result.

    Args:
        marker_files: A list of filenames or directory names that indicate the project root.
        required: If False, return None instead of logging an error and raising
            when no project root is found (e.g. when the package is installed in
            site-packages on Databricks).

    Returns:
        The absolute path to the project root as a Path object, or None when
        ``required=False`` and no root is found.

    Raises:
        FileNotFoundError: If the project root cannot be determined and ``required=True``.
    """

    logger = get_logger("ml_analytics.utils.find_project_root")

    if marker_files is None:
        marker_files = [".git", "pyproject.toml", ".env"]
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    start_dirs = []
    try:
        script_dir = Path(__file__).parent.resolve()
        start_dirs.append(script_dir)
        logger.debug(f"find_project_root: Trying start dir from __file__: {script_dir}")
    except NameError:
        logger.debug("find_project_root: __file__ not defined.")
        pass

    cwd = Path.cwd().resolve()
    if not start_dirs or cwd not in [d.resolve() for d in start_dirs]:
        start_dirs.append(cwd)
    logger.debug(f"find_project_root: Trying start dir from cwd: {cwd}")

    for start_dir in start_dirs:
        current_dir = start_dir
        for _ in range(10):
            logger.debug(f"find_project_root: Searching in: {current_dir}")
            for marker in marker_files:
                if (current_dir / marker).exists():
                    _PROJECT_ROOT = current_dir
                    logger.info(f"Project root found at: {_PROJECT_ROOT}")
                    return _PROJECT_ROOT
            parent_dir = current_dir.parent
            if parent_dir == current_dir:
                break
            current_dir = parent_dir

    if not required:
        # Expected when installed as a dependency (e.g. on Databricks); stay quiet.
        return None
    error_message = f"Could not find project root. Searched upwards from {start_dirs} for markers: {marker_files}"
    log_and_raise_error(logger, error_message, FileNotFoundError)


def _load_pipeline_yaml(folder: Path, pipeline: str | None = None) -> tuple[list[str], Path] | tuple[None, None]:
    """
    Return (steps, source_path) for the requested pipeline, or (None, None) if absent/invalid.

    When pipeline is None (default):
        - Scans the folder for *.yaml files.
        - If exactly one is found, reads its top-level `steps` list.
        - If multiple are found, warns and falls back to alphabetical order.

    When pipeline is a name (e.g. "daily"):
        1. Tries `<name>.yaml` → top-level `steps` list  (one file per pipeline).
        2. Falls back to searching all *.yaml files for `pipelines.<name>.steps`
           (named sections layout — both dict and list-of-objects shapes supported).
        3. Returns (None, None) if neither is found or valid.
    """
    logger = get_logger("ml_analytics.utils._load_pipeline_yaml")

    def _extract_steps(data: object, yaml_path: Path, context: str) -> list[str] | None:
        if not isinstance(data, dict):
            logger.warning(
                f"'{yaml_path}' did not parse as a mapping ({context}) — falling back to alphabetical order."
            )
            return None
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            logger.warning(f"'{yaml_path}' has no valid 'steps' list ({context}) — falling back to alphabetical order.")
            return None
        return [str(s) for s in steps]

    def _yaml_files(folder: Path) -> list[Path]:
        return sorted(folder.glob("*.yaml"))

    if pipeline is None:
        yaml_files = _yaml_files(folder)
        if not yaml_files:
            return None, None
        if len(yaml_files) > 1:
            names = ", ".join(p.name for p in yaml_files)
            logger.warning(
                f"Multiple YAML files found in '{folder}' ({names}) — "
                f"specify pipeline= to select one. Falling back to alphabetical order."
            )
            return None, None
        yaml_path = yaml_files[0]
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            steps = _extract_steps(data, yaml_path, "top-level steps")
            return (steps, yaml_path) if steps is not None else (None, None)
        except Exception as e:
            logger.warning(f"Failed to parse '{yaml_path}': {e} — falling back to alphabetical order.")
            return None, None

    # Named pipeline — layout A: <pipeline>.yaml
    named_yaml = folder / f"{pipeline}.yaml"
    if named_yaml.exists():
        try:
            with open(named_yaml) as f:
                data = yaml.safe_load(f)
            steps = _extract_steps(data, named_yaml, f"pipeline='{pipeline}'")
            if steps is not None:
                return steps, named_yaml
        except Exception as e:
            logger.warning(f"Failed to parse '{named_yaml}': {e} — searching other YAML files.")

    # Named pipeline — layout B: search all *.yaml for pipelines.<name>.steps
    # Supports two shapes for the `pipelines` key:
    #   dict:  pipelines: {training: {steps: [...]}, ...}
    #   list:  pipelines: [{name: training, steps: [...]}, ...]
    for yaml_path in _yaml_files(folder):
        if yaml_path == named_yaml:
            continue
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            pipelines_section = data.get("pipelines")
            if isinstance(pipelines_section, dict) and pipeline in pipelines_section:
                steps = _extract_steps(pipelines_section[pipeline], yaml_path, f"pipelines.{pipeline}")
                if steps is not None:
                    return steps, yaml_path
            if isinstance(pipelines_section, list):
                for entry in pipelines_section:
                    if isinstance(entry, dict) and entry.get("name") == pipeline:
                        steps = _extract_steps(entry, yaml_path, f"pipelines[name={pipeline}]")
                        if steps is not None:
                            return steps, yaml_path
        except Exception as e:
            logger.warning(f"Failed to parse '{yaml_path}': {e} — skipping.")

    logger.warning(
        f"Pipeline '{pipeline}' not found in any YAML file in '{folder}' — falling back to alphabetical order."
    )
    return None, None


def get_sql_files(relative_folder: str, pipeline: str | None = None) -> dict[str, Path]:
    """
    Gets an ordered dictionary of .sql files in a folder relative to the project root.
    Keys are the file names without extension, values are Path objects.

    If a pipeline.yaml (or a named <pipeline>.yaml) exists in the folder, its `steps`
    list defines both the set and the execution order of SQL files. Otherwise files are
    ordered alphabetically.

    Args:
        relative_folder: Folder path relative to the project root.
        pipeline: Optional pipeline name to select. Supports two layouts:
            - A separate `<name>.yaml` file with a top-level `steps` list.
            - A named section inside `pipeline.yaml` under `pipelines.<name>.steps`.
            When None (default), reads the top-level `steps` from `pipeline.yaml`.
    """
    logger = get_logger("ml_analytics.utils.get_sql_files")

    sql_files_dict = {}
    try:
        project_root = find_project_root()
        absolute_folder_path = project_root / relative_folder

        steps, yaml_source_path = _load_pipeline_yaml(absolute_folder_path, pipeline=pipeline)

        if steps is not None:
            yaml_source = yaml_source_path.name
            for step in steps:
                sql_path = absolute_folder_path / f"{step}.sql"
                if not sql_path.exists():
                    logger.warning(f"Step '{step}' listed in {yaml_source} but '{sql_path}' not found — skipping.")
                    continue
                sql_files_dict[step] = sql_path
            if not sql_files_dict:
                logger.warning(f"{yaml_source} in '{absolute_folder_path}' resolved to zero existing .sql files.")
                return {}
            order_str = " → ".join(sql_files_dict.keys())
            logger.info(f"Pipeline order (from {yaml_source}): {order_str}")
            return sql_files_dict

        path_pattern = str(absolute_folder_path / "*.sql")
        files = glob.glob(path_pattern)
        logger.debug(f"Globbing '{path_pattern}' → {len(files)} file(s) found.")

        if not files:
            logger.warning(
                f"No .sql files found in '{absolute_folder_path}'. "
                f"Project root resolved to: '{project_root}'. "
                f"Folder exists: {absolute_folder_path.exists()}."
            )
            return {}

        files.sort(key=lambda x: Path(x).name)

        for file_path_str in files:
            file_path = Path(file_path_str)
            query_name = file_path.stem
            sql_files_dict[query_name] = file_path

        return sql_files_dict
    except FileNotFoundError:
        logger.error("Could not locate SQL files because project root was not found.")
        return {}
    except Exception as e:
        logger.error(f"Error getting SQL files from {relative_folder}: {e}", exc_info=True)
        return {}


def strip_sql_comments(sql_content: str) -> str:
    """
    Remove SQL comments while preserving quoted string literals.

    This handles single-line ``--`` comments and block ``/* ... */`` comments.
    It is intentionally conservative and keeps newlines from removed comments so
    surrounding SQL tokens do not get accidentally joined together.
    """
    cleaned = []
    in_single_line_comment = False
    in_multi_line_comment = False
    in_string = False
    string_delimiter = None
    i = 0

    while i < len(sql_content):
        char = sql_content[i]
        next_chars = sql_content[i : i + 2]

        if in_single_line_comment:
            if char == "\n":
                in_single_line_comment = False
                cleaned.append(char)
            i += 1
            continue

        if in_multi_line_comment:
            if next_chars == "*/":
                in_multi_line_comment = False
                i += 2
                continue
            if char == "\n":
                cleaned.append(char)
            i += 1
            continue

        if not in_string and next_chars == "--":
            in_single_line_comment = True
            i += 2
            continue

        if not in_string and next_chars == "/*":
            in_multi_line_comment = True
            i += 2
            continue

        if char in ("'", '"'):
            if not in_string:
                in_string = True
                string_delimiter = char
            elif char == string_delimiter:
                if i + 1 < len(sql_content) and sql_content[i + 1] == char:
                    cleaned.append(char)
                    cleaned.append(sql_content[i + 1])
                    i += 2
                    continue
                in_string = False
                string_delimiter = None

        cleaned.append(char)
        i += 1

    return "\n".join(line.rstrip() for line in "".join(cleaned).splitlines()).strip()


def load_sql_query(query_path: str, strip_comments: bool = False, **kwargs) -> str | None:
    """
    Load a SQL query from a file.

    Relative paths are resolved from the project root first, then the current
    working directory, then the active Databricks notebook directory when
    available.

    When ``strip_comments`` is True, SQL comments are removed before template
    substitution. This is useful for connectors that wrap queries internally and
    can mis-handle leading ``--`` comments.
    """
    logger = get_logger("ml_analytics.utils.load_sql_query")

    try:
        path = Path(query_path).expanduser()
        if path.is_absolute():
            candidate_paths = [path]
        else:
            candidate_paths = []
            project_root = find_project_root(required=False)
            if project_root is not None:
                candidate_paths.append(project_root / path)
            candidate_paths.append(Path.cwd() / path)
            notebook_dir = _databricks_notebook_dir()
            if notebook_dir is not None:
                candidate_paths.append(notebook_dir / path)

        checked_paths = []
        for absolute_file_path in dict.fromkeys(candidate_paths):
            checked_paths.append(str(absolute_file_path))
            if not absolute_file_path.exists():
                continue
            with open(absolute_file_path) as file:
                sql_content = file.read()
                if strip_comments:
                    sql_content = strip_sql_comments(sql_content)
                return sql_content.format(**kwargs)

        logger.error(f"SQL file '{query_path}' not found. Checked: {checked_paths}.")
        return None
    except Exception as e:
        logger.error(f"Error loading SQL query {query_path}: {e}", exc_info=True)
        return None


def _split_sql_statements(sql_content: str) -> list[str]:
    """
    Split SQL content into individual statements, respecting comments and string literals.

    Handles:
    - Single-line comments (-- comment)
    - Multi-line comments (/* comment */)
    - String literals with ' and " (including escaped quotes)

    Args:
        sql_content: The full SQL content to split

    Returns:
        A list of SQL statements (stripped, non-empty)
    """
    statements = []
    current_statement = []
    in_single_line_comment = False
    in_multi_line_comment = False
    in_string = False
    string_delimiter = None
    i = 0

    while i < len(sql_content):
        char = sql_content[i]

        # Check for multi-line comment start
        if not in_string and not in_single_line_comment and not in_multi_line_comment:
            if i + 1 < len(sql_content) and sql_content[i : i + 2] == "/*":
                in_multi_line_comment = True
                current_statement.append(char)
                i += 1
                if i < len(sql_content):
                    current_statement.append(sql_content[i])
                i += 1
                continue

        # Check for multi-line comment end
        if in_multi_line_comment:
            current_statement.append(char)
            if i + 1 < len(sql_content) and sql_content[i : i + 2] == "*/":
                in_multi_line_comment = False
                i += 1
                if i < len(sql_content):
                    current_statement.append(sql_content[i])
            i += 1
            continue

        # Check for single-line comment start
        if not in_string and not in_single_line_comment and not in_multi_line_comment:
            if i + 1 < len(sql_content) and sql_content[i : i + 2] == "--":
                in_single_line_comment = True
                current_statement.append(char)
                i += 1
                continue

        # Check for single-line comment end (newline)
        if in_single_line_comment:
            current_statement.append(char)
            if char == "\n":
                in_single_line_comment = False
            i += 1
            continue

        # Handle string literals (basic support for ' and ")
        if not in_single_line_comment and not in_multi_line_comment:
            if char in ("'", '"'):
                if not in_string:
                    in_string = True
                    string_delimiter = char
                elif char == string_delimiter:
                    # Check for escaped quote (doubled quote in SQL)
                    if i + 1 < len(sql_content) and sql_content[i + 1] == char:
                        current_statement.append(char)
                        current_statement.append(char)
                        i += 2
                        continue
                    else:
                        in_string = False
                        string_delimiter = None

        # Handle semicolon as statement separator (only when not in comment or string)
        if char == ";" and not in_string and not in_single_line_comment and not in_multi_line_comment:
            # End of statement
            stmt = "".join(current_statement).strip()
            if stmt:
                statements.append(stmt)
            current_statement = []
            i += 1
            continue

        # Regular character
        current_statement.append(char)
        i += 1

    # Add any remaining statement
    stmt = "".join(current_statement).strip()
    if stmt:
        statements.append(stmt)

    return statements


def _is_select_statement(statement: str) -> bool:
    """
    Check whether a SQL statement is a SELECT query (SELECT or WITH ... SELECT).

    Skips leading comments (single-line ``--`` and multi-line ``/* */``) and
    blank lines before inspecting the first meaningful SQL keyword.

    Note: SELECT INTO statements are excluded as they are DDL operations that
    create tables rather than returning data.

    Args:
        statement: A single SQL statement string.

    Returns:
        True if the statement is a SELECT or WITH query that returns data, False otherwise.
    """
    if not statement:
        return False

    # Strip leading whitespace
    text = statement.strip()

    # Remove leading comments (both single-line and multi-line)
    while text:
        if text.startswith("--"):
            # Skip to end of line
            newline_idx = text.find("\n")
            if newline_idx == -1:
                return False
            text = text[newline_idx + 1 :].lstrip()
        elif text.startswith("/*"):
            end_idx = text.find("*/")
            if end_idx == -1:
                return False
            text = text[end_idx + 2 :].lstrip()
        else:
            break

    first_word = text.split()[0].upper() if text.split() else ""

    # Check if it starts with SELECT or WITH
    if first_word not in ("SELECT", "WITH"):
        return False

    # Check for SELECT INTO pattern in the entire statement (DDL, not a query)
    # This applies to both "SELECT ... INTO ..." and "WITH ... SELECT ... INTO ..."
    text_upper = text.upper()

    # Look for INTO keyword (can be surrounded by spaces, newlines, or tabs)
    # We use regex to handle different whitespace patterns
    import re

    into_match = re.search(r"\bINTO\b", text_upper)

    if into_match:  # INTO found in the statement
        into_idx = into_match.start()

        # Find the last SELECT in the statement (for WITH queries)
        last_select_match = None
        for match in re.finditer(r"\bSELECT\b", text_upper):
            last_select_match = match

        if last_select_match:
            last_select_idx = last_select_match.start()

            # Find the first FROM after the last SELECT
            from_match = re.search(r"\bFROM\b", text_upper[last_select_idx:])

            # If INTO appears after the last SELECT and before FROM (or no FROM), it's DDL
            if into_idx > last_select_idx:
                if from_match is None or into_idx < (last_select_idx + from_match.start()):
                    return False

    return True


def resolve_sql_query_paths(query_paths, pipeline: str | None = None) -> dict[str, Path]:
    """
    Normalize SQL pipeline input into an ordered mapping of query name to file path.

    Args:
        query_paths: one of:
            - str: relative folder path from project root; SQL files are discovered
              via get_sql_files() (respects pipeline.yaml if present).
            - Path pointing to a directory: same as str, resolved relative to project root.
            - Path pointing to a single .sql file: executes that file only.
            - list[str | Path]: ordered list of individual SQL file paths.
            - dict[str, str | Path]: explicit ordered mapping of name -> path; preserves insertion order.
        pipeline: Optional pipeline name passed to get_sql_files() when query_paths is a folder.

    Returns:
        Ordered dict[str, Path].
    """
    logger = get_logger("ml_analytics.utils.resolve_sql_query_paths")

    if isinstance(query_paths, str):
        try:
            project_root = find_project_root()
            candidate = project_root / query_paths
        except FileNotFoundError:
            candidate = Path(query_paths)
        if candidate.is_file():
            return {candidate.stem: candidate}
        resolved = get_sql_files(query_paths, pipeline=pipeline)
        if not resolved:
            log_and_raise_error(logger, f"No SQL files found for folder '{query_paths}'.")
        return resolved

    if isinstance(query_paths, Path):
        if query_paths.is_dir():
            try:
                project_root = find_project_root()
                relative = query_paths.relative_to(project_root)
            except ValueError:
                relative = query_paths
            resolved = get_sql_files(str(relative), pipeline=pipeline)
            if not resolved:
                log_and_raise_error(logger, f"No SQL files found in directory '{query_paths}'.")
            return resolved
        return {query_paths.stem: query_paths}

    if isinstance(query_paths, list):
        return {Path(p).stem: Path(p) for p in query_paths}

    if isinstance(query_paths, dict):
        return {k: Path(v) if isinstance(v, str) else v for k, v in query_paths.items()}

    log_and_raise_error(logger, f"Expected a folder path, list, or dict, got: {type(query_paths)}")
    return {}


def execute_sql_scripts(
    query_paths,
    data_connector=None,
    format: str = "pandas",
    pipeline: str | None = None,
    **kwargs,
):
    """
    Loads, splits, and executes SQL statements from multiple files.
    Stops execution immediately if any SQL statement fails.

    If the last statement of the last file is a SELECT/WITH query, it is
    executed as a fetch query and the result is returned as a DataFrame.
    All preceding statements are executed normally without fetching results.

    Args:
        query_paths: one of:
            - str: relative folder path from project root; SQL files are discovered
              via get_sql_files() (respects pipeline.yaml if present).
            - Path pointing to a directory: same as str, resolved relative to project root.
            - Path pointing to a single .sql file: executes that file only.
            - list[str | Path]: ordered list of individual SQL file paths.
            - dict[str, str | Path]: explicit ordered mapping of name → path; executed in insertion order.
        data_connector: Optional DataConnector instance to reuse. If None, a new one will be created.
        format: Output format when the last statement is a SELECT ('pandas' or 'polars'). Defaults to 'pandas'.
        pipeline: Optional pipeline name passed to get_sql_files() when query_paths is a folder.
            Selects a named pipeline from a dedicated `<name>.yaml` file or a named section
            inside `pipeline.yaml` under `pipelines.<name>.steps`.
        **kwargs: Additional keyword arguments for SQL query formatting.

    Returns:
        pandas or polars DataFrame if the last statement is a SELECT/WITH query,
        None otherwise.
    """
    logger = get_logger("ml_analytics.utils.execute_sql_scripts")

    from ml_analytics.data_connector import DataConnector

    query_paths = resolve_sql_query_paths(query_paths, pipeline=pipeline)

    def _run_scripts(dc):
        """Execute all scripts on the given DataConnector instance.

        Returns a DataFrame if the last statement is a SELECT, otherwise None.
        """
        dc._ensure_connected()
        if not getattr(dc, "cursor", None):
            log_and_raise_error(logger, "DataConnector has no cursor after connecting.")

        # Collect all (name, statements) so we can identify the very last statement
        all_scripts = []
        for name, query_path in query_paths.items():
            logger.debug(f"Loading SQL script: {name}")
            full_sql_content = load_sql_query(query_path, **kwargs)
            if full_sql_content is None:
                error_msg = f"Failed to load SQL script '{name}' - stopping execution for debugging"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            stmts = _split_sql_statements(full_sql_content)
            all_scripts.append((name, stmts))

        # Flatten into a single list of (name, index, statement) keeping order
        flat_statements = []
        for name, stmts in all_scripts:
            for i, stmt in enumerate(stmts):
                flat_statements.append((name, i, stmt))

        if not flat_statements:
            logger.warning("No SQL statements found in the provided scripts.")
            return None

        # Check if the last statement is a SELECT query
        last_name, last_idx, last_stmt = flat_statements[-1]
        last_is_select = _is_select_statement(last_stmt)

        # Execute all statements (or all except the last if it's a SELECT)
        statements_to_execute = flat_statements[:-1] if last_is_select else flat_statements

        current_script = None
        script_start = None
        for name, i, statement in statements_to_execute:
            if name != current_script:
                if current_script is not None:
                    elapsed = time.time() - script_start
                    logger.info(f"[{current_script}] done in {elapsed:.1f}s")
                current_script = name
                script_start = time.time()
                logger.info(f"[{name}] running ...")
            try:
                logger.debug(f"Executing statement {i + 1} from {name}: {statement[:100]}...")
                with dc._lock:
                    dc._cancel_idle_timer()
                    dc._last_activity = time.time()
                    dc.cursor.execute(statement)
                dc._start_idle_timer()
            except Exception as stmt_error:
                error_msg = f"SQL execution failed at statement {i + 1} in script '{name}': {stmt_error}"
                logger.error(error_msg, exc_info=True)
                logger.error(f"Failed statement: {statement}")
                raise RuntimeError(error_msg) from stmt_error

        # Log completion of the last executed script
        if current_script is not None and not last_is_select:
            elapsed = time.time() - script_start
            logger.info(f"[{current_script}] done in {elapsed:.1f}s")

        # If the last statement is a SELECT, fetch the result as a DataFrame
        if last_is_select:
            if current_script != last_name:
                logger.info(f"[{last_name}] running ...")
            fetch_start = time.time()
            result = dc.sql(last_stmt, format=format)
            elapsed = time.time() - fetch_start
            logger.info(f"[{last_name}] done in {elapsed:.1f}s")
            return result

        logger.info("All SQL scripts completed successfully.")
        return None

    # --- Main execution ---

    if data_connector is not None:
        logger.info("Using provided DataConnector instance")
        try:
            return _run_scripts(data_connector)
        except Exception:
            logger.error("SQL script execution stopped due to error.")
            raise
    else:
        dc = DataConnector()
        try:
            return _run_scripts(dc)
        except Exception as e:
            logger.error(f"SQL script execution stopped due to error: {e}")
            raise
        finally:
            try:
                dc.close_redshift_connection()
                logger.debug("Closed DataConnector connection")
            except Exception:
                logger.debug("Failed to cleanly close DataConnector connection", exc_info=True)
