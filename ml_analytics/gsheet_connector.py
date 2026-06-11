"""
Google Sheets connector for reading and writing data to Google Sheets.

Data Cleaning Features:
- Automatically handles missing values (NaN, None) by converting to empty strings
- Replaces infinity values (inf, -inf) with empty strings
- Normalizes null-like string values ('None', 'none', 'null', 'NULL')
- Converts object columns to strings to avoid type issues
- Pads rows with missing columns to match header length (handles trailing empty cells)
- Truncates rows that are longer than headers
- All cleaning is applied automatically during read/write operations
"""

import io
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .utils import get_credential_value, get_logger, log_and_raise_error


class GSheet:
    """
    A connector class for interacting with Google Sheets API.

    This class provides methods to read from and write to Google Sheets,
    with support for both service account and OAuth2 authentication.
    """

    @staticmethod
    def _dataframe_to_values(data: pd.DataFrame, include_headers: bool = True) -> list[list[Any]]:
        """
        Convert a DataFrame into JSON-serializable rows for the Sheets API.

        Handles Categorical, NaN/None/inf, datetime/Timestamp (including tz-aware),
        Period, and timedelta columns that would otherwise fail json.dumps.
        """
        from pandas.api import types as pdt

        data_clean = data.copy()
        for col in data_clean.columns:
            series = data_clean[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                data_clean[col] = series.astype(object)
                continue
            if pdt.is_datetime64_any_dtype(series):
                data_clean[col] = series.dt.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(series.dtype, pd.PeriodDtype) or pdt.is_timedelta64_dtype(series):
                data_clean[col] = series.astype(str)
        data_clean = data_clean.fillna("")
        data_clean = data_clean.replace([float("inf"), float("-inf")], "")
        for col in data_clean.columns:
            if data_clean[col].dtype == "object":
                data_clean[col] = data_clean[col].astype(str).replace("nan", "").replace("None", "")
        if include_headers:
            return [data_clean.columns.tolist()] + data_clean.values.tolist()
        return data_clean.values.tolist()

    @staticmethod
    def _format_sheet_name(sheet_name: str) -> str:
        """
        Format a sheet name for use in A1 notation.
        Adds single quotes around the name if it contains spaces or special characters.

        Parameters
        ----------
        sheet_name : str
            The sheet name to format.

        Returns
        -------
        str
            Properly formatted sheet name for A1 notation.
        """
        # If sheet name contains spaces or special characters, wrap in single quotes
        if any(char in sheet_name for char in [" ", "!", "'"]):
            # Escape any single quotes in the sheet name by doubling them
            escaped_name = sheet_name.replace("'", "''")
            return f"'{escaped_name}'"
        return sheet_name

    @staticmethod
    def _find_google_credentials_file() -> Path | None:
        """
        Search for a Google service account credentials JSON file starting from the project root,
        then current directory and parent directory.

        Returns
        -------
        Path | None
            Path to the credentials file if found, None otherwise.
        """
        from .utils import find_project_root

        directories_to_search = []

        # Try to find project root first
        try:
            project_root = find_project_root()
            directories_to_search.append(project_root)
        except FileNotFoundError:
            pass

        # Then check current directory and parent directory
        current_dir = Path.cwd()
        directories_to_search.extend([current_dir, current_dir.parent])

        for search_dir in directories_to_search:
            # Look for JSON files in directory
            for json_file in search_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        # Check if it's a Google service account file
                        if (
                            isinstance(data, dict)
                            and data.get("type") == "service_account"
                            and (
                                "googleapis.com" in str(data.get("auth_uri", ""))
                                or "googleapis.com" in str(data.get("token_uri", ""))
                                or "client_email" in data
                            )
                        ):
                            return json_file
                except (json.JSONDecodeError, Exception):
                    continue

        return None

    def __init__(
        self,
        credentials_path: str | Path = None,
        credentials_json: dict = None,
        scopes: list[str] = None,
        log_level: str = "INFO",
        scope: str = "ml",
        spreadsheet_id: str = None,
    ):
        """
        Initialize the Google Sheets connector.

        Parameters
        ----------
        credentials_path : str | Path, optional
            Path to the service account credentials JSON file.
            If not provided, will look for 'gsheet_credentials.json' in the current directory.
        credentials_json : dict, optional
            Service account credentials as a dictionary (alternative to credentials_path).
        scopes : list[str], optional
            Google API scopes to use. Defaults to read/write access to Google Sheets.
        log_level : str, optional
            Logging level. Default is "INFO".
        scope : str, optional
            Scope for mounted secrets (e.g., '/mnt/{scope}/GOOGLE_CREDENTIALS').
            Default is "ml".
        spreadsheet_id : str, optional
            Default spreadsheet ID used by any method that accepts a ``spreadsheet_id``
            argument. A ``spreadsheet_id`` passed to an individual method call always
            takes precedence over this default. When neither is provided, falls back
            to the ``GSHEET_SPREADSHEET_ID`` environment variable if set.

        Examples
        --------
        >>> # Using credentials file
        >>> gsheet = GSheet(credentials_path="path/to/credentials.json")
        >>>
        >>> # Using credentials dictionary
        >>> creds_dict = json.loads(os.environ['GOOGLE_CREDENTIALS'])
        >>> gsheet = GSheet(credentials_json=creds_dict)
        >>>
        >>> # Auto-load from default location
        >>> gsheet = GSheet()  # Looks for gsheet_credentials.json
        >>>
        >>> # Using mounted secrets with a custom scope
        >>> gsheet = GSheet(scope="custom-scope")
        >>>
        >>> # Bind a default spreadsheet ID so later calls can omit it
        >>> gsheet = GSheet(spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        >>> gsheet.read_sheet()  # uses the bound ID
        >>> gsheet.write_sheet(df, spreadsheet_id="other-id")  # per-call ID overrides
        """
        self._logger = get_logger("GSheet")
        self._logger.setLevel(log_level)
        self._scope = scope

        if scopes is None:
            self.scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/drive",
            ]
        else:
            self.scopes = scopes

        # Initialize credentials
        self.credentials = self._initialize_credentials(credentials_path, credentials_json)
        self.service_account_email = getattr(self.credentials, "service_account_email", None)

        # Build the service
        try:
            self.service = build("sheets", "v4", credentials=self.credentials)
        except Exception as e:
            log_and_raise_error(self._logger, f"Failed to initialize Google Sheets API service: {e}")

        # Build Drive API service for sharing capabilities
        try:
            self.drive_service = build("drive", "v3", credentials=self.credentials)
        except Exception as e:
            self._logger.warning(f"Failed to initialize Google Drive API service: {e}")
            self.drive_service = None

        # Single success message after all services initialized
        self._logger.info("Google API services initialized successfully")

        # Resolve default spreadsheet_id: explicit arg wins, else GSHEET_SPREADSHEET_ID env var
        if spreadsheet_id is None:
            spreadsheet_id = os.environ.get("GSHEET_SPREADSHEET_ID")
            if spreadsheet_id:
                self._logger.debug("Using GSHEET_SPREADSHEET_ID from environment")
        self.spreadsheet_id = spreadsheet_id

    def _assemble_credentials_from_components(self) -> dict | None:
        """
        Assemble Google service account credentials from individual Vault secrets.

        Supports credentials stored as separate fields:
        - GOOGLE_PROJECT_ID
        - GOOGLE_API_PKEY_ID
        - GOOGLE_API_PKEY
        - GOOGLE_CLIENT_EMAIL
        - GOOGLE_CLIENT_ID
        - GOOGLE_CERT_URL

        Returns
        -------
        dict | None
            Service account credentials dictionary if all required fields found, None otherwise.
        """
        try:
            # Fetch all required credential components
            project_id = get_credential_value("GOOGLE_PROJECT_ID", scope=self._scope)
            private_key_id = get_credential_value("GOOGLE_API_PKEY_ID", scope=self._scope)
            private_key = get_credential_value("GOOGLE_API_PKEY", scope=self._scope)
            client_email = get_credential_value("GOOGLE_CLIENT_EMAIL", scope=self._scope)
            client_id = get_credential_value("GOOGLE_CLIENT_ID", scope=self._scope)
            cert_url = get_credential_value("GOOGLE_CERT_URL", scope=self._scope)

            # Handle escaped newlines in private key (common in Vault)
            if "\\n" in private_key:
                private_key = private_key.replace("\\n", "\n")

            # Build the service account credentials dictionary
            credentials_dict = {
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": private_key_id,
                "private_key": private_key,
                "client_email": client_email,
                "client_id": client_id,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": cert_url,
            }

            self._logger.debug("Assembled Google credentials from individual Vault secrets")
            return credentials_dict

        except Exception as e:
            self._logger.debug(f"Could not assemble credentials from individual components: {e}")
            return None

    def _initialize_credentials(
        self, credentials_path: str | Path = None, credentials_json: dict = None
    ) -> service_account.Credentials:
        """
        Initialize Google API credentials from file or dictionary.

        Parameters
        ----------
        credentials_path : str | Path, optional
            Path to credentials JSON file.
        credentials_json : dict, optional
            Credentials dictionary.

        Returns
        -------
        service_account.Credentials
            Google service account credentials.
        """
        # Try to auto-load from default location if no credentials provided
        if credentials_path is None and credentials_json is None:
            # First try to get credentials from environment variable or mounted secret (single JSON)
            try:
                credentials_str = get_credential_value("GOOGLE_CREDENTIALS", scope=self._scope)
                credentials_json = json.loads(credentials_str)
                self._logger.debug("Using GOOGLE_CREDENTIALS from environment or mounted secret")
            except Exception:
                # Try assembling from individual Vault secrets
                credentials_json = self._assemble_credentials_from_components()

            if credentials_json is None:
                # OAuth fallback: only when OAuth env vars set and no SA creds found.
                if os.environ.get("GOOGLE_OAUTH_CLIENT_ID") and os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"):
                    return self._initialize_oauth_credentials()
                # Fall back to a JSON file in the project/current directory
                default_path = Path.cwd() / "gsheet_credentials.json"
                if default_path.exists():
                    credentials_path = default_path
                else:
                    # Search for any JSON file containing Google credentials
                    found_creds = GSheet._find_google_credentials_file()
                    if found_creds:
                        credentials_path = found_creds
                    else:
                        self._logger.error(
                            "No credentials provided and no Google credentials JSON file found in current directory."
                        )
                        self._logger.info(
                            "Please provide credentials via 'credentials_path' parameter or place a Google service account JSON file in the current directory."  # noqa: E501
                        )
                        log_and_raise_error(
                            self._logger,
                            "Either 'credentials_path' or 'credentials_json' must be provided, or a Google service account JSON file must exist in the current directory",  # noqa: E501
                        )

        try:
            if credentials_path is not None:
                credentials_path = Path(credentials_path)
                if not credentials_path.exists():
                    log_and_raise_error(
                        self._logger,
                        f"Credentials file not found at: {credentials_path}",
                    )
                credentials = service_account.Credentials.from_service_account_file(
                    str(credentials_path), scopes=self.scopes
                )
            else:
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_json, scopes=self.scopes
                )

            return credentials

        except Exception as e:
            log_and_raise_error(self._logger, f"Failed to initialize credentials: {e}")

    def _initialize_oauth_credentials(self) -> "OAuthCredentials":
        """
        Authenticate via the OAuth installed-app flow, caching the token.

        Uses a valid cached token, refreshes it if expired, else runs a one-time
        browser consent. Reads GOOGLE_OAUTH_CLIENT_ID/SECRET and optional
        GOOGLE_CLOUD_PROJECT; caches to GSHEET_TOKEN_PATH (default
        ~/.config/ml-analytics/gsheet_token.json).
        """
        client_id = os.environ["GOOGLE_OAUTH_CLIENT_ID"]
        client_secret = os.environ["GOOGLE_OAUTH_CLIENT_SECRET"]
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

        token_path_str = os.environ.get("GSHEET_TOKEN_PATH")
        if token_path_str:
            token_path = Path(token_path_str)
        else:
            token_path = Path.home() / ".config" / "ml-analytics" / "gsheet_token.json"

        creds = None
        if token_path.exists():
            creds = OAuthCredentials.from_authorized_user_file(str(token_path), self.scopes)

        if creds and creds.valid:
            self._logger.debug("Using cached OAuth token")
        elif creds and creds.expired and creds.refresh_token:
            self._logger.info("Refreshing expired OAuth token")
            creds.refresh(Request())
            self._save_oauth_token(token_path, creds)
        else:
            self._logger.info("No valid OAuth token found; launching browser consent flow")
            client_config = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "project_id": project_id,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": ["http://localhost"],
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, self.scopes)
            creds = flow.run_local_server(port=0)
            self._save_oauth_token(token_path, creds)

        self._logger.debug("OAuth credentials initialized")
        return creds

    def _save_oauth_token(self, token_path: Path, creds: "OAuthCredentials") -> None:
        """Persist OAuth credentials to ``token_path`` with 0600 permissions."""
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        try:
            os.chmod(token_path, 0o600)
        except OSError as e:
            self._logger.debug(f"Could not chmod token file: {e}")

    def _resolve_spreadsheet_id(self, spreadsheet_id: str | None) -> str | None:
        """Return the per-call spreadsheet_id if provided, else the instance default."""
        return spreadsheet_id if spreadsheet_id is not None else self.spreadsheet_id

    def read_sheet(
        self,
        spreadsheet_id: str = None,
        range_name: str = None,
        sheet_name: str = None,
        return_as: str = "dataframe",
    ) -> pd.DataFrame | list[list[Any]]:
        """
        Read data from a Google Sheet.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet to read from. Falls back to the instance
            default set via ``GSheet(spreadsheet_id=...)`` when omitted.
        range_name : str, optional
            The A1 notation range to read (e.g., 'Sheet1!A1:D10').
            If None and sheet_name is provided, reads entire sheet.
        sheet_name : str, optional
            The name of the sheet to read from. Used if range_name is not provided.
        return_as : str, optional
            Format to return data: 'dataframe' (default) or 'list'.

        Returns
        -------
        pd.DataFrame | list[list[Any]]
            The data from the sheet as a DataFrame or list of lists.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> df = gsheet.read_sheet("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        >>> df = gsheet.read_sheet("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        ...                        range_name="Sheet1!A1:D10")
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)
        if spreadsheet_id is None:
            log_and_raise_error(
                self._logger,
                "No spreadsheet_id provided and no default set on the GSheet instance",
            )

        # Build the range
        if range_name is None:
            if sheet_name is None:
                range_name = "A:ZZ"  # Read all columns
            else:
                formatted_sheet = self._format_sheet_name(sheet_name)
                range_name = f"{formatted_sheet}!A:ZZ"

        try:
            result = self.service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()

            values = result.get("values", [])

            if not values:
                self._logger.warning(f"No data found in range: {range_name}")
                return pd.DataFrame() if return_as == "dataframe" else []

            self._logger.info(f"Successfully read {len(values)} rows from spreadsheet {spreadsheet_id}")

            if return_as == "dataframe":
                # Use first row as headers
                if len(values) > 0:
                    headers = values[0]
                    data_rows = values[1:]

                    # Ensure all rows have the same length as headers by padding with empty strings
                    # This handles cases where rows have trailing empty cells that Google Sheets API omits
                    num_columns = len(headers)
                    normalized_rows = []
                    for row in data_rows:
                        if len(row) < num_columns:
                            # Pad row with empty strings to match header length
                            row = row + [""] * (num_columns - len(row))
                        elif len(row) > num_columns:
                            # Truncate row if it's longer than headers (rare but possible)
                            row = row[:num_columns]
                        normalized_rows.append(row)

                    df = pd.DataFrame(normalized_rows, columns=headers)
                    # Handle missing values: replace empty strings and None with empty string
                    df = df.fillna("")
                    # Replace any remaining None-like values that might come from sheets
                    df = df.replace([None, "None", "none", "null", "NULL"], "")
                    return df
                return pd.DataFrame()
            else:
                return values

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error reading from Google Sheet: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error reading from Google Sheet: {e}",
            )

    def write_sheet(
        self,
        data: pd.DataFrame | list[list[Any]],
        spreadsheet_id: str = None,
        spreadsheet_title: str = None,
        range_name: str = None,
        sheet_name: str = None,
        value_input_option: str = "USER_ENTERED",
        include_headers: bool = True,
        clear_before_write: bool = False,
        share_with: list[str] | str = None,
        role: str = "writer",
        autofit_columns: bool = True,
        column_padding: int = 30,
    ) -> dict | tuple[dict, str]:
        """
        Write data to a Google Sheet. Creates a new spreadsheet if it doesn't exist.

        Parameters
        ----------
        data : pd.DataFrame | list[list[Any]]
            The data to write. Can be a DataFrame or list of lists.
        spreadsheet_id : str, optional
            The ID of the spreadsheet to write to. Falls back to the instance
            default set via ``GSheet(spreadsheet_id=...)`` when omitted. If both
            are None, ``spreadsheet_title`` must be provided and a new
            spreadsheet will be created.
        spreadsheet_title : str, optional
            Title for a new spreadsheet. Used only if no spreadsheet_id is resolved.
            A new spreadsheet will be created with this title.
        range_name : str, optional
            The A1 notation range to write to (e.g., 'Sheet1!A1').
        sheet_name : str, optional
            The name of the sheet to write to. Used if range_name is not provided.
        value_input_option : str, optional
            How to interpret the input data. Options: 'RAW' or 'USER_ENTERED' (default).
        include_headers : bool, optional
            Whether to include DataFrame column names as headers. Default is True.
        clear_before_write : bool, optional
            Whether to clear the range before writing. Default is False.
        share_with : list[str] | str, optional
            Email address(es) to share the spreadsheet with (only used when creating new spreadsheet).
        role : str, optional
            Permission level when sharing: 'reader', 'writer', or 'owner'. Default is 'writer'.
        autofit_columns : bool, optional
            Whether to auto-resize column widths to fit content after writing.
            Default is True.
        column_padding : int, optional
            Extra pixels to add to each column width after auto-resize for readability.
            Default is 30. Set to 0 for a tight fit with no padding.

        Returns
        -------
        dict | tuple[dict, str]
            If spreadsheet exists: returns the API response containing update information.
            If spreadsheet created: returns tuple of (API response, spreadsheet_id).

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>>
        >>> # Write to existing spreadsheet
        >>> gsheet.write_sheet(df, spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        >>>
        >>> # Create new spreadsheet and write data
        >>> result, new_id = gsheet.write_sheet(
        ...     df,
        ...     spreadsheet_title="My Data",
        ...     share_with="user@example.com"
        ... )
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)

        # Create new spreadsheet if ID not provided
        created_new = False
        if spreadsheet_id is None:
            if spreadsheet_title is None:
                log_and_raise_error(self._logger, "Either 'spreadsheet_id' or 'spreadsheet_title' must be provided")

            # Determine sheet names from data if needed
            sheet_names_to_create = None
            if sheet_name:
                sheet_names_to_create = [sheet_name]

            spreadsheet_id = self.create_spreadsheet(
                title=spreadsheet_title,
                sheet_names=sheet_names_to_create,
                share_with=None,  # Don't share here, will be done after writing
                role=role,
            )
            created_new = True
            self._logger.info(f"Created new spreadsheet '{spreadsheet_title}' with ID: {spreadsheet_id}")

        # Build the range
        if range_name is None:
            if sheet_name is None:
                range_name = "Sheet1!A1"
            else:
                # Ensure the sheet exists (create if needed)
                if not created_new:
                    self._ensure_sheet_exists(spreadsheet_id, sheet_name)
                formatted_sheet = self._format_sheet_name(sheet_name)
                range_name = f"{formatted_sheet}!A1"

        # Convert DataFrame to list of lists if needed
        # Build the range
        if range_name is None:
            if sheet_name is None:
                range_name = "Sheet1!A1"
            else:
                formatted_sheet = self._format_sheet_name(sheet_name)
                range_name = f"{formatted_sheet}!A1"

        # Check if sheet exists and create it if needed (only when using sheet_name, not range_name)
        if sheet_name and not created_new:
            self._ensure_sheet_exists(spreadsheet_id, sheet_name)

        # Convert DataFrame to list of lists if needed
        if isinstance(data, pd.DataFrame):
            values = self._dataframe_to_values(data, include_headers=include_headers)
        else:
            values = data

        try:
            if clear_before_write:
                if "!" in range_name:
                    sheet_part = range_name.split("!")[0]
                    clear_range_full = f"{sheet_part}!A1:ZZZ100000"
                else:
                    clear_range_full = "A1:ZZZ100000"
                self.clear_range(spreadsheet_id, clear_range_full)

            # Write the data
            body = {"values": values}
            result = (
                self.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption=value_input_option,
                    body=body,
                )
                .execute()
            )

            updated_cells = result.get("updatedCells", 0)
            self._logger.info(f"Successfully wrote {updated_cells} cells to spreadsheet {spreadsheet_id}")

            # Share the spreadsheet if email addresses are provided (works for both new and existing)
            if share_with:
                self.share_spreadsheet(
                    spreadsheet_id=spreadsheet_id,
                    email_addresses=share_with,
                    role=role,
                    send_notification=True,
                )

            # Auto-fit column widths to content
            if autofit_columns:
                num_cols = len(values[0]) if values else None
                effective_sheet = sheet_name if sheet_name else "Sheet1"
                self._autofit_columns(spreadsheet_id, effective_sheet, num_cols, padding_pixels=column_padding)

            # Return spreadsheet_id if newly created, otherwise just the result
            if created_new:
                return result, spreadsheet_id
            return result

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error writing to Google Sheet: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error writing to Google Sheet: {e}",
            )

    def append_sheet(
        self,
        spreadsheet_id: str = None,
        data: pd.DataFrame | list[list[Any]] = None,
        range_name: str = None,
        sheet_name: str = None,
        value_input_option: str = "USER_ENTERED",
        include_headers: bool = False,
        autofit_columns: bool = False,
        column_padding: int = 30,
    ) -> dict:
        """
        Append data to a Google Sheet.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet to append to. Falls back to the instance
            default set via ``GSheet(spreadsheet_id=...)`` when omitted.
        data : pd.DataFrame | list[list[Any]]
            The data to append.
        range_name : str, optional
            The A1 notation range to append to (e.g., 'Sheet1!A1').
        sheet_name : str, optional
            The name of the sheet to append to. Used if range_name is not provided.
        value_input_option : str, optional
            How to interpret the input data. Options: 'RAW' or 'USER_ENTERED' (default).
        include_headers : bool, optional
            Whether to include DataFrame column names as headers. Default is False.
        autofit_columns : bool, optional
            Whether to auto-resize column widths to fit content after appending.
            Default is False.
        column_padding : int, optional
            Extra pixels to add to each column width after auto-resize for readability.
            Default is 30. Set to 0 for a tight fit with no padding.

        Returns
        -------
        dict
            The API response containing append information.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> df = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
        >>> gsheet.append_sheet("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms", df)
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)
        if spreadsheet_id is None:
            log_and_raise_error(
                self._logger,
                "No spreadsheet_id provided and no default set on the GSheet instance",
            )
        if data is None:
            log_and_raise_error(self._logger, "'data' is required for append_sheet")

        # Build the range
        if range_name is None:
            if sheet_name is None:
                range_name = "Sheet1!A1"
            else:
                # Ensure the sheet exists (create if needed)
                self._ensure_sheet_exists(spreadsheet_id, sheet_name)
                formatted_sheet = self._format_sheet_name(sheet_name)
                range_name = f"{formatted_sheet}!A1"

        # Convert DataFrame to list of lists if needed
        if isinstance(data, pd.DataFrame):
            values = self._dataframe_to_values(data, include_headers=include_headers)
        else:
            values = data

        try:
            body = {"values": values}
            result = (
                self.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption=value_input_option,
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )

            updated_cells = result.get("updates", {}).get("updatedCells", 0)
            self._logger.info(f"Successfully appended {updated_cells} cells to spreadsheet {spreadsheet_id}")

            # Auto-fit column widths to content
            if autofit_columns:
                num_cols = len(values[0]) if values else None
                effective_sheet = sheet_name if sheet_name else "Sheet1"
                self._autofit_columns(spreadsheet_id, effective_sheet, num_cols, padding_pixels=column_padding)

            return result

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error appending to Google Sheet: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error appending to Google Sheet: {e}",
            )

    def clear_range(self, spreadsheet_id: str = None, range_name: str = None) -> dict:
        """
        Clear values from a range in a Google Sheet.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet. Falls back to the instance default set
            via ``GSheet(spreadsheet_id=...)`` when omitted.
        range_name : str
            The A1 notation range to clear.

        Returns
        -------
        dict
            The API response.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> gsheet.clear_range("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        ...                    "Sheet1!A1:D10")
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)
        if spreadsheet_id is None:
            log_and_raise_error(
                self._logger,
                "No spreadsheet_id provided and no default set on the GSheet instance",
            )
        if range_name is None:
            log_and_raise_error(self._logger, "'range_name' is required for clear_range")

        try:
            result = (
                self.service.spreadsheets().values().clear(spreadsheetId=spreadsheet_id, range=range_name).execute()
            )

            self._logger.info(f"Successfully cleared range {range_name}")
            return result

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error clearing Google Sheet range: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error clearing Google Sheet range: {e}",
            )

    def create_spreadsheet(
        self,
        title: str,
        sheet_names: list[str] = None,
        share_with: list[str] | str = None,
        role: str = "writer",
        send_notification: bool = True,
    ) -> str:
        """
        Create a new Google Spreadsheet and optionally share it with specified email addresses.

        Parameters
        ----------
        title : str
            The title of the new spreadsheet.
        sheet_names : list[str], optional
            List of sheet names to create. If None, creates a single sheet named "Sheet1".
        share_with : list[str] | str, optional
            Email address(es) to share the spreadsheet with.
            Can be a single email string or a list of email strings.
        role : str, optional
            Permission level for shared users: 'reader', 'writer', or 'owner'.
            Default is 'writer'.
        send_notification : bool, optional
            Whether to send email notifications to users when sharing.
            Default is True.

        Returns
        -------
        str
            The spreadsheet ID of the newly created spreadsheet.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> # Create and share with one person
        >>> spreadsheet_id = gsheet.create_spreadsheet(
        ...     "My New Spreadsheet",
        ...     sheet_names=["Data", "Analysis"],
        ...     share_with="user@example.com"
        ... )
        >>>
        >>> # Create and share with multiple people
        >>> spreadsheet_id = gsheet.create_spreadsheet(
        ...     "Team Dashboard",
        ...     share_with=["alice@example.com", "bob@example.com"],
        ...     role="reader"
        ... )
        """
        if self.drive_service is None:
            log_and_raise_error(
                self._logger,
                "Drive API service is not initialized. Cannot create spreadsheet. "
                "Please ensure the Drive API is enabled in your Google Cloud project.",
            )

        try:
            # Use Drive API to create the spreadsheet file
            file_metadata = {"name": title, "mimeType": "application/vnd.google-apps.spreadsheet"}

            file = self.drive_service.files().create(body=file_metadata, fields="id").execute()

            spreadsheet_id = file.get("id")
            self._logger.info(f"Created new spreadsheet '{title}' with ID: {spreadsheet_id}")

            # If custom sheet names are specified, update the spreadsheet
            if sheet_names:
                try:
                    requests = []
                    # Delete the default "Sheet1" if we're creating custom sheets
                    requests.append(
                        {
                            "deleteSheet": {
                                "sheetId": 0  # Default sheet ID
                            }
                        }
                    )
                    # Add custom sheets
                    for i, name in enumerate(sheet_names):
                        requests.append({"addSheet": {"properties": {"sheetId": i + 1, "title": name}}})

                    batch_update_request = {"requests": requests}
                    self.service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id, body=batch_update_request
                    ).execute()
                    self._logger.info(f"Added custom sheets: {sheet_names}")
                except Exception as e:
                    self._logger.warning(f"Could not add custom sheets: {e}")

            # Share the spreadsheet if email addresses are provided
            if share_with:
                self.share_spreadsheet(
                    spreadsheet_id=spreadsheet_id,
                    email_addresses=share_with,
                    role=role,
                    send_notification=send_notification,
                )

            return spreadsheet_id

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error creating Google Spreadsheet: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error creating Google Spreadsheet: {e}",
            )

    def share_spreadsheet(
        self,
        spreadsheet_id: str = None,
        email_addresses: list[str] | str = None,
        role: str = "writer",
        send_notification: bool = True,
    ) -> list[dict]:
        """
        Share a Google Spreadsheet with one or more email addresses.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet to share. Falls back to the instance
            default set via ``GSheet(spreadsheet_id=...)`` when omitted.
        email_addresses : list[str] | str
            Email address(es) to share the spreadsheet with.
            Can be a single email string or a list of email strings.
        role : str, optional
            Permission level: 'reader', 'writer', or 'owner'. Default is 'writer'.
        send_notification : bool, optional
            Whether to send email notifications. Default is True.

        Returns
        -------
        list[dict]
            List of permission objects created.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> gsheet.share_spreadsheet(
        ...     "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        ...     "user@example.com",
        ...     role="writer"
        ... )
        >>>
        >>> # Share with multiple users
        >>> gsheet.share_spreadsheet(
        ...     "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        ...     ["alice@example.com", "bob@example.com"],
        ...     role="reader"
        ... )
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)
        if spreadsheet_id is None:
            log_and_raise_error(
                self._logger,
                "No spreadsheet_id provided and no default set on the GSheet instance",
            )
        if email_addresses is None:
            log_and_raise_error(self._logger, "'email_addresses' is required for share_spreadsheet")

        if self.drive_service is None:
            log_and_raise_error(
                self._logger,
                "Drive API service is not initialized. Cannot share spreadsheet.",
            )

        # Convert single email to list
        if isinstance(email_addresses, str):
            email_addresses = [email_addresses]

        permissions = []
        for email in email_addresses:
            try:
                permission = {
                    "type": "user",
                    "role": role,
                    "emailAddress": email,
                }

                result = (
                    self.drive_service.permissions()
                    .create(
                        fileId=spreadsheet_id,
                        body=permission,
                        sendNotificationEmail=send_notification,
                    )
                    .execute()
                )

                permissions.append(result)
                self._logger.info(f"Shared spreadsheet {spreadsheet_id} with {email} as {role}")

            except HttpError as e:
                self._logger.error(f"Failed to share with {email}: {e}")
            except Exception as e:
                self._logger.error(f"Error sharing with {email}: {e}")

        return permissions

    def get_service_account_email(self) -> str:
        """
        Get the service account email address.

        This email should be used to share Google Spreadsheets for programmatic access.

        Returns
        -------
        str
            The service account email address.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> email = gsheet.get_service_account_email()
        >>> print(f"Share your spreadsheet with: {email}")
        """
        return self.service_account_email

    def _ensure_sheet_exists(self, spreadsheet_id: str, sheet_name: str) -> bool:
        """
        Check if a sheet exists in the spreadsheet, create it if it doesn't.

        Parameters
        ----------
        spreadsheet_id : str
            The ID of the spreadsheet.
        sheet_name : str
            The name of the sheet to check/create.

        Returns
        -------
        bool
            True if sheet was created, False if it already existed.
        """
        try:
            # Get spreadsheet info to check existing sheets
            spreadsheet = self.get_spreadsheet_info(spreadsheet_id)
            existing_sheets = [sheet["properties"]["title"] for sheet in spreadsheet["sheets"]]

            if sheet_name in existing_sheets:
                return False  # Sheet already exists

            # Create the sheet
            requests = [{"addSheet": {"properties": {"title": sheet_name}}}]

            batch_update_request = {"requests": requests}
            self.service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=batch_update_request).execute()

            self._logger.info(f"Created new sheet '{sheet_name}' in spreadsheet {spreadsheet_id}")
            return True  # Sheet was created

        except HttpError as e:
            self._logger.warning(f"Could not check/create sheet '{sheet_name}': {e}")
            return False
        except Exception as e:
            self._logger.warning(f"Error checking/creating sheet '{sheet_name}': {e}")
            return False

    def _autofit_columns(
        self,
        spreadsheet_id: str,
        sheet_name: str = None,
        num_columns: int = None,
        padding_pixels: int = 30,
    ) -> None:
        """
        Auto-resize columns to fit their content, then add padding for readability.

        First uses Google Sheets autoResizeDimensions to fit content, then reads
        back the resulting column widths and adds extra padding so the data doesn't
        look cramped.

        Parameters
        ----------
        spreadsheet_id : str
            The ID of the spreadsheet.
        sheet_name : str, optional
            The name of the sheet. Defaults to "Sheet1".
        num_columns : int, optional
            Number of columns to resize. If None, resizes all columns in the sheet.
        padding_pixels : int, optional
            Extra pixels to add to each column width after auto-resize.
            Default is 30. Set to 0 to skip padding.
        """
        try:
            target_name = sheet_name or "Sheet1"
            info = self.get_spreadsheet_info(spreadsheet_id)
            sheet_id = None
            for sheet in info["sheets"]:
                if sheet["properties"]["title"] == target_name:
                    sheet_id = sheet["properties"]["sheetId"]
                    break

            if sheet_id is None:
                self._logger.warning(f"Sheet '{target_name}' not found for auto-fit; skipping column resize")
                return

            # Step 1: Auto-resize columns to fit content
            dimensions = {
                "sheetId": sheet_id,
                "dimension": "COLUMNS",
                "startIndex": 0,
            }
            if num_columns is not None:
                dimensions["endIndex"] = num_columns

            request = {"autoResizeDimensions": {"dimensions": dimensions}}
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"requests": [request]}
            ).execute()
            self._logger.debug(f"Auto-fit columns for sheet '{target_name}'")

            # Step 2: Add padding to each column for better readability
            if padding_pixels > 0:
                self._add_column_padding(spreadsheet_id, sheet_id, target_name, num_columns, padding_pixels)

        except Exception as e:
            self._logger.warning(f"Could not auto-fit columns: {e}")

    def _add_column_padding(
        self,
        spreadsheet_id: str,
        sheet_id: int,
        sheet_name: str,
        num_columns: int | None,
        padding_pixels: int,
    ) -> None:
        """
        Add extra padding to column widths after auto-resize.

        Reads back the current column widths from the sheet metadata and adds
        the specified padding to each column.

        Parameters
        ----------
        spreadsheet_id : str
            The ID of the spreadsheet.
        sheet_id : int
            The numeric sheet ID within the spreadsheet.
        sheet_name : str
            The sheet name (used for logging).
        num_columns : int | None
            Number of columns to pad. If None, pads all columns with metadata.
        padding_pixels : int
            Extra pixels to add to each column width.
        """
        try:
            fields = "sheets(properties(sheetId,title),data(columnMetadata(pixelSize)))"
            spreadsheet = self.service.spreadsheets().get(spreadsheetId=spreadsheet_id, fields=fields).execute()

            column_widths = []
            for sheet in spreadsheet.get("sheets", []):
                if sheet["properties"]["sheetId"] == sheet_id:
                    for data_section in sheet.get("data", []):
                        for col_meta in data_section.get("columnMetadata", []):
                            column_widths.append(col_meta.get("pixelSize", 100))
                    break

            if not column_widths:
                self._logger.debug("No column metadata found; skipping padding")
                return

            end_col = num_columns if num_columns is not None else len(column_widths)
            end_col = min(end_col, len(column_widths))

            padding_requests = []
            for i in range(end_col):
                new_width = column_widths[i] + padding_pixels
                padding_requests.append(
                    {
                        "updateDimensionProperties": {
                            "range": {
                                "sheetId": sheet_id,
                                "dimension": "COLUMNS",
                                "startIndex": i,
                                "endIndex": i + 1,
                            },
                            "properties": {"pixelSize": new_width},
                            "fields": "pixelSize",
                        }
                    }
                )

            if padding_requests:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id, body={"requests": padding_requests}
                ).execute()
                self._logger.debug(f"Added {padding_pixels}px padding to {end_col} columns in sheet '{sheet_name}'")

        except Exception as e:
            self._logger.warning(f"Could not add column padding: {e}")

    def format_columns_as_percent(
        self,
        spreadsheet_id: str = None,
        columns: list[str | int] = None,
        sheet_name: str = None,
        pattern: str = "0.0%",
        has_header: bool = True,
    ) -> dict:
        """
        Apply percent number formatting to one or more columns in a sheet.

        Values should be written as raw ratios (e.g. 0.143, not "14.3%"). Users
        will see "14.3%" in the UI while sorting and filtering remain numeric.

        Parameters
        ----------
        spreadsheet_id : str
            The ID of the spreadsheet to format.
        columns : list[str | int]
            Columns to format. Each entry is either a column name (matched
            against the header row) or a 0-based column index.
        sheet_name : str, optional
            The name of the sheet. Defaults to "Sheet1".
        pattern : str, optional
            Google Sheets number format pattern. Default is "0.0%".
            Examples: "0%", "0.00%", "0.0%;[red]-0.0%".
        has_header : bool, optional
            If True (default), the first row is treated as a header and left
            unformatted; formatting starts at row 2. If False, formatting
            starts at row 1.

        Returns
        -------
        dict
            The batchUpdate API response.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> gsheet.write_sheet(df, spreadsheet_id=sid, sheet_name="Summary")
        >>> gsheet.format_columns_as_percent(
        ...     spreadsheet_id=sid,
        ...     columns=["conversion_rate", "bounce_rate"],
        ...     sheet_name="Summary",
        ... )
        """
        return self._apply_number_format(
            spreadsheet_id=spreadsheet_id,
            columns=columns,
            number_format={"type": "PERCENT", "pattern": pattern},
            sheet_name=sheet_name,
            has_header=has_header,
        )

    def format_columns_as_number(
        self,
        spreadsheet_id: str = None,
        columns: list[str | int] = None,
        sheet_name: str = None,
        pattern: str = "#,##0.00",
        has_header: bool = True,
    ) -> dict:
        """
        Apply numeric formatting (e.g. thousands separators) to one or more columns.

        Values should be written as raw numbers (e.g. 6302320.01). With the
        default pattern, users will see "6,302,320.01" in the UI while sorting
        and filtering remain numeric.

        Parameters
        ----------
        spreadsheet_id : str
            The ID of the spreadsheet to format.
        columns : list[str | int]
            Columns to format. Each entry is either a column name (matched
            against the header row) or a 0-based column index.
        sheet_name : str, optional
            The name of the sheet. Defaults to "Sheet1".
        pattern : str, optional
            Google Sheets number format pattern. Default is "#,##0.00".
            Examples: "#,##0" (integer with thousands), "#,##0.00" (two decimals),
            "$#,##0.00" (currency-style prefix), "#,##0.00;[red]-#,##0.00".
        has_header : bool, optional
            If True (default), the first row is treated as a header and left
            unformatted; formatting starts at row 2. If False, formatting
            starts at row 1.

        Returns
        -------
        dict
            The batchUpdate API response.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> gsheet.write_sheet(df, spreadsheet_id=sid, sheet_name="Summary")
        >>> gsheet.format_columns_as_number(
        ...     spreadsheet_id=sid,
        ...     columns=["revenue", "cost"],
        ...     sheet_name="Summary",
        ...     pattern="#,##0.00",
        ... )
        """
        return self._apply_number_format(
            spreadsheet_id=spreadsheet_id,
            columns=columns,
            number_format={"type": "NUMBER", "pattern": pattern},
            sheet_name=sheet_name,
            has_header=has_header,
        )

    def format_columns_as_date(
        self,
        spreadsheet_id: str = None,
        columns: list[str | int] = None,
        sheet_name: str = None,
        pattern: str = "yyyy-mm-dd",
        has_header: bool = True,
        include_time: bool = False,
    ) -> dict:
        """
        Apply date (or date-time) formatting to one or more columns.

        Works with cells that Sheets has parsed as dates. When writing via
        ``write_sheet``/``append_sheet`` with ``value_input_option="USER_ENTERED"``
        (the default), the helper ``_dataframe_to_values`` converts pandas
        datetime columns to ``"YYYY-MM-DD HH:MM:SS"`` strings, which Sheets
        parses back into serial date values. Applying this format controls
        how they are displayed while preserving sort/filter semantics.

        Parameters
        ----------
        spreadsheet_id : str
            The ID of the spreadsheet to format.
        columns : list[str | int]
            Columns to format. Each entry is either a column name (matched
            against the header row) or a 0-based column index.
        sheet_name : str, optional
            The name of the sheet. Defaults to "Sheet1".
        pattern : str, optional
            Google Sheets date/time format pattern. Default is "yyyy-mm-dd".
            Examples: "yyyy-mm-dd", "dd/mm/yyyy", "mmm d, yyyy",
            "yyyy-mm-dd hh:mm:ss".
        has_header : bool, optional
            If True (default), the first row is treated as a header and left
            unformatted; formatting starts at row 2. If False, formatting
            starts at row 1.
        include_time : bool, optional
            If True, use the DATE_TIME number format type (so the cell is
            treated as a timestamp). If False (default), use the DATE type.
            The ``pattern`` still controls the exact display either way.

        Returns
        -------
        dict
            The batchUpdate API response.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> gsheet.write_sheet(df, spreadsheet_id=sid, sheet_name="Summary")
        >>> gsheet.format_columns_as_date(
        ...     spreadsheet_id=sid,
        ...     columns=["created_at", "updated_at"],
        ...     sheet_name="Summary",
        ...     pattern="yyyy-mm-dd hh:mm:ss",
        ...     include_time=True,
        ... )
        """
        fmt_type = "DATE_TIME" if include_time else "DATE"
        return self._apply_number_format(
            spreadsheet_id=spreadsheet_id,
            columns=columns,
            number_format={"type": fmt_type, "pattern": pattern},
            sheet_name=sheet_name,
            has_header=has_header,
        )

    def _apply_number_format(
        self,
        spreadsheet_id: str,
        columns: list[str | int],
        number_format: dict,
        sheet_name: str = None,
        has_header: bool = True,
    ) -> dict:
        """
        Apply a Google Sheets numberFormat dict to a set of columns.

        Shared implementation for format_columns_as_percent and
        format_columns_as_number. Resolves column names via the header row,
        looks up the sheetId, and issues a single batchUpdate with one
        repeatCell request per column.
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)
        if spreadsheet_id is None:
            log_and_raise_error(
                self._logger,
                "No spreadsheet_id provided and no default set on the GSheet instance",
            )
        if columns is None:
            log_and_raise_error(self._logger, "'columns' is required for column formatting")

        target_sheet = sheet_name or "Sheet1"

        info = self.get_spreadsheet_info(spreadsheet_id)
        sheet_id = None
        for sheet in info.get("sheets", []):
            if sheet["properties"]["title"] == target_sheet:
                sheet_id = sheet["properties"]["sheetId"]
                break
        if sheet_id is None:
            log_and_raise_error(
                self._logger,
                f"Sheet '{target_sheet}' not found in spreadsheet {spreadsheet_id}",
            )

        header = None
        column_indices = []
        for col in columns:
            if isinstance(col, bool) or not isinstance(col, int | str):
                log_and_raise_error(self._logger, f"Invalid column identifier: {col!r}")
            if isinstance(col, int):
                column_indices.append(col)
                continue
            if header is None:
                formatted_sheet = self._format_sheet_name(target_sheet)
                header_range = f"{formatted_sheet}!1:1"
                result = (
                    self.service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=header_range).execute()
                )
                header_values = result.get("values", [])
                header = header_values[0] if header_values else []
            try:
                column_indices.append(header.index(col))
            except ValueError:
                log_and_raise_error(
                    self._logger,
                    f"Column '{col}' not found in header row of sheet '{target_sheet}'",
                )

        start_row = 1 if has_header else 0
        requests = [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row,
                        "startColumnIndex": idx,
                        "endColumnIndex": idx + 1,
                    },
                    "cell": {"userEnteredFormat": {"numberFormat": number_format}},
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
            for idx in column_indices
        ]

        try:
            result = (
                self.service.spreadsheets()
                .batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": requests})
                .execute()
            )
            self._logger.info(
                f"Applied {number_format['type']} format '{number_format['pattern']}' "
                f"to {len(requests)} column(s) in sheet '{target_sheet}'"
            )
            return result

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error applying number format: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error applying number format: {e}",
            )

    def get_spreadsheet_info(self, spreadsheet_id: str = None) -> dict:
        """
        Get information about a spreadsheet.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet. Falls back to the instance default set
            via ``GSheet(spreadsheet_id=...)`` when omitted.

        Returns
        -------
        dict
            Spreadsheet metadata including sheets, properties, etc.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> info = gsheet.get_spreadsheet_info("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        >>> print(info['properties']['title'])
        """
        spreadsheet_id = self._resolve_spreadsheet_id(spreadsheet_id)
        if spreadsheet_id is None:
            log_and_raise_error(
                self._logger,
                "No spreadsheet_id provided and no default set on the GSheet instance",
            )
        try:
            spreadsheet = self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

            return spreadsheet

        except HttpError as e:
            log_and_raise_error(
                self._logger,
                f"HTTP error getting spreadsheet info: {e}",
            )
        except Exception as e:
            log_and_raise_error(
                self._logger,
                f"Error getting spreadsheet info: {e}",
            )

    def to_csv(self, spreadsheet_id: str = None, range_name: str = None, sheet_name: str = None) -> str:
        """
        Read data from Google Sheet and convert to CSV string.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet to read from. Falls back to the instance
            default set via ``GSheet(spreadsheet_id=...)`` when omitted.
        range_name : str, optional
            The A1 notation range to read.
        sheet_name : str, optional
            The name of the sheet to read from.

        Returns
        -------
        str
            CSV formatted string.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> csv_data = gsheet.to_csv("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        """
        df = self.read_sheet(spreadsheet_id, range_name, sheet_name, return_as="dataframe")
        return df.to_csv(index=False)

    def gsheet_to_s3(
        self,
        spreadsheet_id: str = None,
        file_name: str = None,
        directory: str = None,
        range_name: str = None,
        sheet_name: str = None,
        file_format: str = "csv",
        s3_connector=None,
        bucket: str = None,
    ) -> None:
        """
        Transfer data from Google Sheet to S3.

        Parameters
        ----------
        spreadsheet_id : str, optional
            The ID of the spreadsheet to read from. Falls back to the instance
            default set via ``GSheet(spreadsheet_id=...)`` when omitted.
        file_name : str
            The name of the file (without extension).
        directory : str, optional
            The directory path where the file will be saved.
        range_name : str, optional
            The A1 notation range to read.
        sheet_name : str, optional
            The name of the sheet to read from.
        file_format : str, optional
            File format to save: 'csv' or 'parquet'. Default is 'csv'.
        s3_connector : S3Connector, optional
            Existing S3Connector instance. If provided, bucket parameter is ignored and bucket is taken from the connector.
        bucket : str, optional
            S3 bucket name. Only used if s3_connector is None. If both s3_connector and bucket are None, raises an error.

        Examples
        --------
        >>> gsheet = GSheet(credentials_path="creds.json")
        >>> s3 = S3Connector(bucket="my-bucket", s3_root="my-project")
        >>> gsheet.gsheet_to_s3(
        ...     spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        ...     s3_connector=s3,
        ...     directory="data",
        ...     file_name="output"
        ... )
        """  # noqa: E501
        from .s3_connector import S3Connector

        if file_name is None:
            log_and_raise_error(self._logger, "'file_name' is required for gsheet_to_s3")

        # Read data from Google Sheet (read_sheet resolves the spreadsheet_id)
        df = self.read_sheet(spreadsheet_id, range_name, sheet_name, return_as="dataframe")

        # Additional cleaning for S3 transfer
        df = df.fillna("")
        df = df.replace([float("inf"), float("-inf")], "")

        # Initialize S3 connector if not provided
        if s3_connector is None:
            if bucket is None:
                log_and_raise_error(
                    self._logger,
                    "Either 's3_connector' or 'bucket' parameter must be provided.",
                )
            s3_connector = S3Connector(bucket=bucket, auto_sso_login=True)

        # Get bucket and s3_root from s3_connector
        target_bucket = s3_connector.bucket
        s3_root = s3_connector.s3_root

        # Add file extension if not present
        file_extension = f".{file_format.lower()}"
        if not file_name.endswith(file_extension):
            file_name_with_ext = f"{file_name}{file_extension}"
        else:
            file_name_with_ext = file_name

        # Construct full S3 key with s3_root and directory
        if directory is None:
            directory = ""

        parts = [s3_root, directory, file_name_with_ext]
        full_s3_key = "/".join(part.strip("/") for part in parts if part).lstrip("/")

        # Convert to appropriate format
        if file_format.lower() == "csv":
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            body = buffer.getvalue()
        elif file_format.lower() == "parquet":
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            body = buffer.getvalue()
        else:
            log_and_raise_error(
                self._logger,
                f"Unsupported file format: {file_format}. Use 'csv' or 'parquet'.",
            )

        # Upload to S3
        s3_connector.s3.put_object(Bucket=target_bucket, Key=full_s3_key, Body=body)
        self._logger.info(f"Successfully transferred data to s3://{target_bucket}/{full_s3_key}")
