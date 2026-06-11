"""
Tests for GSheet connector credential loading from vault services.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.oauth2.credentials import Credentials as RealOAuthCredentials

from ml_analytics.gsheet_connector import GSheet


@pytest.fixture(autouse=True)
def clear_oauth_env_vars(monkeypatch):
    """Clear OAuth env vars so service-account tests aren't disturbed by a loaded .env."""
    for var in ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_CLOUD_PROJECT", "GSHEET_TOKEN_PATH"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_google_service_account():
    """Mock Google service account credentials."""
    with patch("ml_analytics.gsheet_connector.service_account.Credentials") as mock_creds:
        mock_credentials_instance = MagicMock()
        mock_credentials_instance.service_account_email = "mock-service@mock-project.iam.gserviceaccount.com"
        mock_creds.from_service_account_info.return_value = mock_credentials_instance
        mock_creds.from_service_account_file.return_value = mock_credentials_instance
        yield mock_creds


@pytest.fixture
def mock_google_api_services():
    """Mock Google Sheets and Drive API services."""
    with patch("ml_analytics.gsheet_connector.build") as mock_build:
        mock_sheets_service = MagicMock()
        mock_drive_service = MagicMock()

        def build_service(service_name, version, credentials):
            if service_name == "sheets":
                return mock_sheets_service
            elif service_name == "drive":
                return mock_drive_service

        mock_build.side_effect = build_service
        yield {"build": mock_build, "sheets": mock_sheets_service, "drive": mock_drive_service}


@pytest.fixture
def sample_credentials_dict():
    """Sample Google service account credentials dictionary."""
    return {
        "type": "service_account",
        "project_id": "mock-project-id",
        "private_key_id": "mock-private-key-id-12345",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK_KEY_DATA\n-----END PRIVATE KEY-----",
        "client_email": "mock-service@mock-project.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mock-service%40mock-project.iam.gserviceaccount.com",
    }


class TestGSheetCredentialLoading:
    """Test credential loading from vault service."""

    def test_load_credentials_from_single_json_vault(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test loading credentials from GOOGLE_CREDENTIALS environment/vault variable."""
        credentials_json_str = json.dumps(sample_credentials_dict)

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            # Mock returning the full JSON string from vault
            mock_get_cred.return_value = credentials_json_str

            # Initialize GSheet connector
            gsheet = GSheet(scope="ml")

            # Verify get_credential_value was called with correct parameters
            mock_get_cred.assert_called_once_with("GOOGLE_CREDENTIALS", scope="ml")

            # Verify service account credentials were created from the JSON
            mock_google_service_account.from_service_account_info.assert_called_once()
            call_args = mock_google_service_account.from_service_account_info.call_args
            assert call_args[0][0] == sample_credentials_dict

            # Verify GSheet instance was initialized correctly
            assert gsheet.service_account_email == "mock-service@mock-project.iam.gserviceaccount.com"

    def test_load_credentials_from_individual_vault_components(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test assembling credentials from individual vault secrets."""

        def mock_get_credential_side_effect(name, scope="ml"):
            """Mock individual credential components from vault."""
            credentials_map = {
                "GOOGLE_CREDENTIALS": None,  # Simulate GOOGLE_CREDENTIALS not available
                "GOOGLE_PROJECT_ID": "mock-project-id",
                "GOOGLE_API_PKEY_ID": "mock-private-key-id-12345",
                "GOOGLE_API_PKEY": "-----BEGIN PRIVATE KEY-----\\nMOCK_KEY_DATA\\n-----END PRIVATE KEY-----",
                "GOOGLE_CLIENT_EMAIL": "mock-service@mock-project.iam.gserviceaccount.com",
                "GOOGLE_CLIENT_ID": "123456789012345678901",
                "GOOGLE_CERT_URL": "https://www.googleapis.com/robot/v1/metadata/x509/mock-service%40mock-project.iam.gserviceaccount.com",
            }

            if name == "GOOGLE_CREDENTIALS":
                # Simulate GOOGLE_CREDENTIALS not found in vault
                raise Exception("Credential not found")

            if name not in credentials_map:
                raise Exception(f"Credential {name} not found")

            return credentials_map[name]

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            # Initialize GSheet connector
            _ = GSheet(scope="ml")

            # Verify get_credential_value was called for GOOGLE_CREDENTIALS first
            assert mock_get_cred.call_count >= 7  # GOOGLE_CREDENTIALS + 6 components

            # Verify all individual components were requested
            expected_calls = [
                "GOOGLE_CREDENTIALS",
                "GOOGLE_PROJECT_ID",
                "GOOGLE_API_PKEY_ID",
                "GOOGLE_API_PKEY",
                "GOOGLE_CLIENT_EMAIL",
                "GOOGLE_CLIENT_ID",
                "GOOGLE_CERT_URL",
            ]

            actual_calls = [call[0][0] for call in mock_get_cred.call_args_list]
            for expected_key in expected_calls:
                assert expected_key in actual_calls, f"Expected {expected_key} to be requested from vault"

            # Verify service account credentials were created
            mock_google_service_account.from_service_account_info.assert_called_once()

            # Verify the assembled credentials have correct structure
            call_args = mock_google_service_account.from_service_account_info.call_args
            assembled_creds = call_args[0][0]
            assert assembled_creds["type"] == "service_account"
            assert assembled_creds["project_id"] == "mock-project-id"
            assert assembled_creds["client_email"] == "mock-service@mock-project.iam.gserviceaccount.com"

            # Verify newline handling in private key (\\n should be converted to \n)
            assert "\\n" not in assembled_creds["private_key"]
            assert "\n" in assembled_creds["private_key"]

    def test_load_credentials_with_custom_scope(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test loading credentials with custom vault scope."""
        credentials_json_str = json.dumps(sample_credentials_dict)

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.return_value = credentials_json_str

            # Initialize with custom scope
            _ = GSheet(scope="custom-scope")

            # Verify get_credential_value was called with custom scope
            mock_get_cred.assert_called_once_with("GOOGLE_CREDENTIALS", scope="custom-scope")

    def test_fallback_to_file_when_vault_unavailable(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test fallback to file-based credentials when vault secrets unavailable."""

        def mock_get_credential_side_effect(name, scope="ml"):
            """Simulate vault credentials not available."""
            raise Exception(f"Credential {name} not found in vault")

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            # Mock file existence and content
            _ = Path.cwd() / "gsheet_credentials.json"
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True

                # Initialize GSheet connector
                _ = GSheet()

                # Verify vault was attempted first
                assert mock_get_cred.called

                # Verify file-based credentials were used
                mock_google_service_account.from_service_account_file.assert_called_once()

    def test_error_when_no_credentials_available(self, mock_google_service_account, mock_google_api_services):
        """Test error handling when no credentials are available anywhere."""

        def mock_get_credential_side_effect(name, scope="ml"):
            """Simulate vault credentials not available."""
            raise Exception(f"Credential {name} not found")

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            # Mock no credential files exist
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = False

                with patch("ml_analytics.gsheet_connector.GSheet._find_google_credentials_file") as mock_find:
                    mock_find.return_value = None

                    # Should raise error when no credentials available
                    with pytest.raises(Exception) as exc_info:
                        GSheet()

                    assert "must be provided" in str(exc_info.value).lower()

    def test_private_key_newline_conversion(self, mock_google_service_account, mock_google_api_services):
        """Test that escaped newlines in private key are correctly converted."""

        def mock_get_credential_side_effect(name, scope="ml"):
            """Return credentials with escaped newlines in private key."""
            credentials_map = {
                "GOOGLE_CREDENTIALS": None,
                "GOOGLE_PROJECT_ID": "test-project",
                "GOOGLE_API_PKEY_ID": "key-id",
                "GOOGLE_API_PKEY": "-----BEGIN PRIVATE KEY-----\\nLINE1\\nLINE2\\n-----END PRIVATE KEY-----",
                "GOOGLE_CLIENT_EMAIL": "test@test.iam.gserviceaccount.com",
                "GOOGLE_CLIENT_ID": "123456",
                "GOOGLE_CERT_URL": "https://example.com/cert",
            }

            if name == "GOOGLE_CREDENTIALS":
                raise Exception("Not found")

            return credentials_map[name]

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            _ = GSheet()

            # Get the assembled credentials
            call_args = mock_google_service_account.from_service_account_info.call_args
            assembled_creds = call_args[0][0]

            # Verify escaped newlines were converted to actual newlines
            expected_key = "-----BEGIN PRIVATE KEY-----\nLINE1\nLINE2\n-----END PRIVATE KEY-----"
            assert assembled_creds["private_key"] == expected_key
            assert "\\n" not in assembled_creds["private_key"]


class TestGSheetCredentialPriority:
    """Test credential loading priority order."""

    def test_explicit_credentials_json_parameter(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that explicit credentials_json parameter takes highest priority."""

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            # This should not be called when explicit credentials_json is provided
            mock_get_cred.return_value = '{"invalid": "should not be used"}'

            # Initialize with explicit credentials_json
            _ = GSheet(credentials_json=sample_credentials_dict)

            # Verify vault was NOT called when explicit credentials provided
            mock_get_cred.assert_not_called()

            # Verify credentials were created from the provided dict
            mock_google_service_account.from_service_account_info.assert_called_once()
            call_args = mock_google_service_account.from_service_account_info.call_args
            assert call_args[0][0] == sample_credentials_dict

    def test_explicit_credentials_path_parameter(self, mock_google_service_account, mock_google_api_services):
        """Test that explicit credentials_path parameter takes priority over vault."""

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            # This should not be called when explicit credentials_path is provided
            mock_get_cred.return_value = '{"invalid": "should not be used"}'

            with patch("pathlib.Path.exists", return_value=True):
                # Initialize with explicit credentials_path
                _ = GSheet(credentials_path="custom_path.json")

                # Verify vault was NOT called when explicit path provided
                mock_get_cred.assert_not_called()

                # Verify file-based credentials were used
                mock_google_service_account.from_service_account_file.assert_called_once()

    def test_vault_single_json_before_components(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that single JSON vault credential is tried before individual components."""
        credentials_json_str = json.dumps(sample_credentials_dict)

        def mock_get_credential_side_effect(name, scope="ml"):
            """Return single JSON on first call for GOOGLE_CREDENTIALS."""
            if name == "GOOGLE_CREDENTIALS":
                return credentials_json_str
            # These component calls should never happen
            raise AssertionError(f"Should not request {name} when GOOGLE_CREDENTIALS is available")

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            # Initialize GSheet
            _ = GSheet()

            # Verify only GOOGLE_CREDENTIALS was requested, not individual components
            assert mock_get_cred.call_count == 1
            mock_get_cred.assert_called_once_with("GOOGLE_CREDENTIALS", scope="ml")

    def test_components_before_file_fallback(self, mock_google_service_account, mock_google_api_services):
        """Test that individual vault components are tried before file fallback."""

        call_sequence = []

        def mock_get_credential_side_effect(name, scope="ml"):
            """Track call sequence and return component credentials."""
            call_sequence.append(name)

            credentials_map = {
                "GOOGLE_PROJECT_ID": "test-project",
                "GOOGLE_API_PKEY_ID": "key-id",
                "GOOGLE_API_PKEY": "-----BEGIN PRIVATE KEY-----\\nKEY\\n-----END PRIVATE KEY-----",
                "GOOGLE_CLIENT_EMAIL": "test@test.iam.gserviceaccount.com",
                "GOOGLE_CLIENT_ID": "123456",
                "GOOGLE_CERT_URL": "https://example.com/cert",
            }

            if name == "GOOGLE_CREDENTIALS":
                raise Exception("Not found")

            return credentials_map[name]

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            # Initialize GSheet
            _ = GSheet()

            # Verify vault components were tried (GOOGLE_CREDENTIALS should be first in sequence)
            assert call_sequence[0] == "GOOGLE_CREDENTIALS"
            assert "GOOGLE_PROJECT_ID" in call_sequence

            # Verify service account was created from components, not from file
            mock_google_service_account.from_service_account_info.assert_called_once()
            mock_google_service_account.from_service_account_file.assert_not_called()


class TestGSheetCredentialAssembly:
    """Test credential assembly from individual components."""

    def test_all_required_fields_present_in_assembled_credentials(
        self, mock_google_service_account, mock_google_api_services
    ):
        """Test that assembled credentials contain all required Google service account fields."""

        def mock_get_credential_side_effect(name, scope="ml"):
            credentials_map = {
                "GOOGLE_PROJECT_ID": "test-project-123",
                "GOOGLE_API_PKEY_ID": "key-id-456",
                "GOOGLE_API_PKEY": "-----BEGIN PRIVATE KEY-----\\nTEST_KEY\\n-----END PRIVATE KEY-----",
                "GOOGLE_CLIENT_EMAIL": "service@test-project.iam.gserviceaccount.com",
                "GOOGLE_CLIENT_ID": "987654321",
                "GOOGLE_CERT_URL": "https://www.googleapis.com/robot/v1/metadata/x509/service%40test-project.iam.gserviceaccount.com",
            }

            if name == "GOOGLE_CREDENTIALS":
                raise Exception("Not found")

            return credentials_map[name]

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            _ = GSheet()

            # Get the assembled credentials
            call_args = mock_google_service_account.from_service_account_info.call_args
            assembled_creds = call_args[0][0]

            # Verify all required fields are present
            required_fields = [
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
                "client_id",
                "auth_uri",
                "token_uri",
                "auth_provider_x509_cert_url",
                "client_x509_cert_url",
            ]

            for field in required_fields:
                assert field in assembled_creds, f"Missing required field: {field}"

            # Verify field values
            assert assembled_creds["type"] == "service_account"
            assert assembled_creds["project_id"] == "test-project-123"
            assert assembled_creds["private_key_id"] == "key-id-456"
            assert assembled_creds["client_email"] == "service@test-project.iam.gserviceaccount.com"
            assert assembled_creds["client_id"] == "987654321"
            assert assembled_creds["auth_uri"] == "https://accounts.google.com/o/oauth2/auth"
            assert assembled_creds["token_uri"] == "https://oauth2.googleapis.com/token"

    def test_partial_component_credentials_returns_none(self, mock_google_service_account, mock_google_api_services):
        """Test that missing any component credential returns None and falls back to file."""

        def mock_get_credential_side_effect(name, scope="ml"):
            """Missing GOOGLE_CLIENT_ID to simulate partial credentials."""
            credentials_map = {
                "GOOGLE_PROJECT_ID": "test-project",
                "GOOGLE_API_PKEY_ID": "key-id",
                "GOOGLE_API_PKEY": "-----BEGIN PRIVATE KEY-----\\nKEY\\n-----END PRIVATE KEY-----",
                "GOOGLE_CLIENT_EMAIL": "test@test.iam.gserviceaccount.com",
                # GOOGLE_CLIENT_ID is missing
                "GOOGLE_CERT_URL": "https://example.com/cert",
            }

            if name == "GOOGLE_CREDENTIALS":
                raise Exception("Not found")

            if name == "GOOGLE_CLIENT_ID":
                raise Exception("GOOGLE_CLIENT_ID not found")

            if name in credentials_map:
                return credentials_map[name]

            raise Exception(f"{name} not found")

        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.side_effect = mock_get_credential_side_effect

            # Mock file existence for fallback
            with patch("pathlib.Path.exists", return_value=True):
                _ = GSheet()

                # Verify file-based credentials were used as fallback
                mock_google_service_account.from_service_account_file.assert_called_once()


class TestGSheetAutofitColumns:
    """Test column auto-fit behavior in write and append operations."""

    def _create_gsheet_instance(self, sample_credentials_dict, mock_google_service_account, mock_google_api_services):
        """Helper to create a GSheet instance with mocked credentials."""
        credentials_json_str = json.dumps(sample_credentials_dict)
        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.return_value = credentials_json_str
            gsheet = GSheet(scope="ml")
        return gsheet

    def _mock_sheet_with_column_metadata(self, mock_sheets, column_widths=None):
        """Helper to set up mocks for autofit with column metadata for padding."""
        if column_widths is None:
            column_widths = [80, 60]

        # Mock get spreadsheet info (first call: find sheet_id, second call: column metadata)
        mock_sheets.spreadsheets().get().execute.return_value = {
            "sheets": [
                {
                    "properties": {"sheetId": 0, "title": "Sheet1"},
                    "data": [{"columnMetadata": [{"pixelSize": w} for w in column_widths]}],
                }
            ]
        }

        # Mock batchUpdate for autofit and padding
        mock_sheets.spreadsheets().batchUpdate().execute.return_value = {}

    def test_write_sheet_autofit_enabled_by_default(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that write_sheet calls autofit by default."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().update() chain
        mock_sheets.spreadsheets().values().update().execute.return_value = {"updatedCells": 6}

        # Mock sheet metadata with column widths for padding
        self._mock_sheet_with_column_metadata(mock_sheets, column_widths=[80, 60])

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        gsheet.write_sheet(df, spreadsheet_id="test-id-123")

        # Verify batchUpdate was called (for autofit + padding)
        mock_sheets.spreadsheets().batchUpdate.assert_called()

    def test_write_sheet_autofit_disabled(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that write_sheet skips autofit when disabled."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().update() chain
        mock_sheets.spreadsheets().values().update().execute.return_value = {"updatedCells": 6}

        # Reset batchUpdate call tracking
        mock_sheets.spreadsheets().batchUpdate.reset_mock()

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        gsheet.write_sheet(df, spreadsheet_id="test-id-123", autofit_columns=False)

        # Verify batchUpdate was NOT called for autofit
        mock_sheets.spreadsheets().batchUpdate.assert_not_called()

    def test_append_sheet_autofit_disabled_by_default(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that append_sheet does NOT call autofit by default."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().append() chain
        mock_sheets.spreadsheets().values().append().execute.return_value = {"updates": {"updatedCells": 6}}

        # Reset batchUpdate call tracking
        mock_sheets.spreadsheets().batchUpdate.reset_mock()

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        gsheet.append_sheet("test-id-123", df)

        # Verify batchUpdate was NOT called (autofit defaults to False for append)
        mock_sheets.spreadsheets().batchUpdate.assert_not_called()

    def test_append_sheet_autofit_enabled(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that append_sheet calls autofit when explicitly enabled."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().append() chain
        mock_sheets.spreadsheets().values().append().execute.return_value = {"updates": {"updatedCells": 6}}

        # Mock sheet metadata with column widths for padding
        self._mock_sheet_with_column_metadata(mock_sheets, column_widths=[80, 60])

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        gsheet.append_sheet("test-id-123", df, autofit_columns=True)

        # Verify batchUpdate was called (for autofit + padding)
        mock_sheets.spreadsheets().batchUpdate.assert_called()

    def test_write_sheet_custom_column_padding(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that write_sheet passes custom column_padding to autofit."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().update() chain
        mock_sheets.spreadsheets().values().update().execute.return_value = {"updatedCells": 6}

        # Mock sheet metadata with column widths
        self._mock_sheet_with_column_metadata(mock_sheets, column_widths=[80, 60])

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        # Use custom padding
        gsheet.write_sheet(df, spreadsheet_id="test-id-123", column_padding=50)

        # Verify batchUpdate was called (autofit + padding)
        mock_sheets.spreadsheets().batchUpdate.assert_called()

    def test_write_sheet_zero_padding_skips_padding_step(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that column_padding=0 only does auto-resize without extra padding."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().update() chain
        mock_sheets.spreadsheets().values().update().execute.return_value = {"updatedCells": 6}

        # Mock get spreadsheet info (only for auto-resize, no column metadata needed)
        mock_sheets.spreadsheets().get().execute.return_value = {
            "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1"}}]
        }

        # Mock batchUpdate for autofit only
        mock_sheets.spreadsheets().batchUpdate().execute.return_value = {}

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        gsheet.write_sheet(df, spreadsheet_id="test-id-123", column_padding=0)

        # Verify batchUpdate was called once (for autofit only, no padding)
        mock_sheets.spreadsheets().batchUpdate.assert_called()

    def test_autofit_gracefully_handles_missing_column_metadata(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Test that autofit with padding doesn't fail if column metadata is unavailable."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]

        # Mock the values().update() chain
        mock_sheets.spreadsheets().values().update().execute.return_value = {"updatedCells": 6}

        # Mock get spreadsheet info WITHOUT column metadata (simulates metadata not returned)
        mock_sheets.spreadsheets().get().execute.return_value = {
            "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1"}}]
        }

        # Mock batchUpdate
        mock_sheets.spreadsheets().batchUpdate().execute.return_value = {}

        import pandas as pd

        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]})

        # Should not raise even though column metadata is missing
        gsheet.write_sheet(df, spreadsheet_id="test-id-123")

        # Verify auto-resize batchUpdate was still called
        mock_sheets.spreadsheets().batchUpdate.assert_called()


class TestGSheetFormatColumnsAsPercent:
    """Test percent-format helper for columns."""

    def _create_gsheet_instance(self, sample_credentials_dict, mock_google_service_account, mock_google_api_services):
        credentials_json_str = json.dumps(sample_credentials_dict)
        with patch("ml_analytics.gsheet_connector.get_credential_value") as mock_get_cred:
            mock_get_cred.return_value = credentials_json_str
            gsheet = GSheet(scope="ml")
        return gsheet

    def _setup_mocks(self, mock_sheets, sheet_id=0, sheet_title="Sheet1", header=None):
        """Configure sheet-info, header-read, and batchUpdate responses."""
        mock_sheets.spreadsheets().get().execute.return_value = {
            "sheets": [{"properties": {"sheetId": sheet_id, "title": sheet_title}}]
        }
        if header is not None:
            mock_sheets.spreadsheets().values().get().execute.return_value = {"values": [header]}
        mock_sheets.spreadsheets().batchUpdate().execute.return_value = {"replies": []}

    def _get_batch_update_body(self, mock_sheets):
        """Return the last body kwarg passed to batchUpdate (the real call, not the setup chain)."""
        for call in reversed(mock_sheets.spreadsheets().batchUpdate.call_args_list):
            if "body" in call.kwargs:
                return call.kwargs["body"]
        raise AssertionError("batchUpdate was never called with a body kwarg")

    def test_format_by_column_names(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Column names are resolved via the header row."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets, header=["name", "visits", "conversion_rate", "bounce_rate"])

        gsheet.format_columns_as_percent(
            spreadsheet_id="test-id",
            columns=["conversion_rate", "bounce_rate"],
            sheet_name="Sheet1",
        )

        body = self._get_batch_update_body(mock_sheets)
        requests = body["requests"]
        assert len(requests) == 2
        # Column indices resolved from header
        assert requests[0]["repeatCell"]["range"]["startColumnIndex"] == 2
        assert requests[0]["repeatCell"]["range"]["endColumnIndex"] == 3
        assert requests[1]["repeatCell"]["range"]["startColumnIndex"] == 3
        assert requests[1]["repeatCell"]["range"]["endColumnIndex"] == 4
        # Default pattern and header skip
        fmt = requests[0]["repeatCell"]["cell"]["userEnteredFormat"]["numberFormat"]
        assert fmt == {"type": "PERCENT", "pattern": "0.0%"}
        assert requests[0]["repeatCell"]["range"]["startRowIndex"] == 1

    def test_format_by_column_indices_skips_header_lookup(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Integer column identifiers do not trigger a header read."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets)

        # Track calls to the header-read chain
        mock_sheets.spreadsheets().values().get.reset_mock()

        gsheet.format_columns_as_percent(
            spreadsheet_id="test-id",
            columns=[2, 5],
        )

        body = self._get_batch_update_body(mock_sheets)
        indices = [r["repeatCell"]["range"]["startColumnIndex"] for r in body["requests"]]
        assert indices == [2, 5]
        # Header lookup should never have been invoked
        mock_sheets.spreadsheets().values().get.assert_not_called()

    def test_format_mixed_names_and_indices(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Names and indices can be mixed in the same call."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets, header=["a", "b", "c", "d"])

        gsheet.format_columns_as_percent(
            spreadsheet_id="test-id",
            columns=[0, "c"],
        )

        body = self._get_batch_update_body(mock_sheets)
        indices = [r["repeatCell"]["range"]["startColumnIndex"] for r in body["requests"]]
        assert indices == [0, 2]

    def test_format_custom_pattern_and_no_header(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Custom pattern is passed through and has_header=False keeps row 0."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets)

        gsheet.format_columns_as_percent(
            spreadsheet_id="test-id",
            columns=[0],
            pattern="0.00%",
            has_header=False,
        )

        body = self._get_batch_update_body(mock_sheets)
        req = body["requests"][0]["repeatCell"]
        assert req["range"]["startRowIndex"] == 0
        assert req["cell"]["userEnteredFormat"]["numberFormat"]["pattern"] == "0.00%"

    def test_format_resolves_custom_sheet_name(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Non-default sheet names resolve to the correct sheetId."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        mock_sheets.spreadsheets().get().execute.return_value = {
            "sheets": [
                {"properties": {"sheetId": 0, "title": "Sheet1"}},
                {"properties": {"sheetId": 42, "title": "Summary"}},
            ]
        }
        mock_sheets.spreadsheets().batchUpdate().execute.return_value = {"replies": []}

        gsheet.format_columns_as_percent(
            spreadsheet_id="test-id",
            columns=[1],
            sheet_name="Summary",
        )

        body = self._get_batch_update_body(mock_sheets)
        assert body["requests"][0]["repeatCell"]["range"]["sheetId"] == 42

    def test_format_raises_when_sheet_not_found(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Unknown sheet name raises an error and never calls batchUpdate."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        mock_sheets.spreadsheets().get().execute.return_value = {
            "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1"}}]
        }
        mock_sheets.spreadsheets().batchUpdate.reset_mock()

        with pytest.raises(Exception) as exc_info:
            gsheet.format_columns_as_percent(
                spreadsheet_id="test-id",
                columns=[0],
                sheet_name="Missing",
            )
        assert "not found" in str(exc_info.value).lower()
        mock_sheets.spreadsheets().batchUpdate.assert_not_called()

    def test_format_raises_when_column_name_missing(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Column name missing from header row raises an error."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets, header=["name", "visits"])
        mock_sheets.spreadsheets().batchUpdate.reset_mock()

        with pytest.raises(Exception) as exc_info:
            gsheet.format_columns_as_percent(
                spreadsheet_id="test-id",
                columns=["conversion_rate"],
            )
        assert "conversion_rate" in str(exc_info.value)
        mock_sheets.spreadsheets().batchUpdate.assert_not_called()

    def test_format_columns_as_number_default_pattern(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """format_columns_as_number emits NUMBER type with thousands-separator pattern."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets, header=["name", "revenue", "cost"])

        gsheet.format_columns_as_number(
            spreadsheet_id="test-id",
            columns=["revenue", "cost"],
        )

        body = self._get_batch_update_body(mock_sheets)
        assert len(body["requests"]) == 2
        fmt = body["requests"][0]["repeatCell"]["cell"]["userEnteredFormat"]["numberFormat"]
        assert fmt == {"type": "NUMBER", "pattern": "#,##0.00"}
        # Column indices resolved from header (revenue=1, cost=2)
        indices = [r["repeatCell"]["range"]["startColumnIndex"] for r in body["requests"]]
        assert indices == [1, 2]

    def test_format_columns_as_number_custom_pattern(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """Custom pattern is forwarded to the API unchanged."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets)

        gsheet.format_columns_as_number(
            spreadsheet_id="test-id",
            columns=[0],
            pattern="$#,##0.00",
        )

        body = self._get_batch_update_body(mock_sheets)
        fmt = body["requests"][0]["repeatCell"]["cell"]["userEnteredFormat"]["numberFormat"]
        assert fmt == {"type": "NUMBER", "pattern": "$#,##0.00"}

    def test_format_columns_as_date_default_pattern(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """format_columns_as_date emits DATE type with the default pattern."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets, header=["name", "created_at", "updated_at"])

        gsheet.format_columns_as_date(
            spreadsheet_id="test-id",
            columns=["created_at", "updated_at"],
        )

        body = self._get_batch_update_body(mock_sheets)
        assert len(body["requests"]) == 2
        fmt = body["requests"][0]["repeatCell"]["cell"]["userEnteredFormat"]["numberFormat"]
        assert fmt == {"type": "DATE", "pattern": "yyyy-mm-dd"}
        indices = [r["repeatCell"]["range"]["startColumnIndex"] for r in body["requests"]]
        assert indices == [1, 2]

    def test_format_columns_as_date_with_include_time(
        self, sample_credentials_dict, mock_google_service_account, mock_google_api_services
    ):
        """include_time=True switches the format type to DATE_TIME."""
        gsheet = self._create_gsheet_instance(
            sample_credentials_dict, mock_google_service_account, mock_google_api_services
        )
        mock_sheets = mock_google_api_services["sheets"]
        self._setup_mocks(mock_sheets)

        gsheet.format_columns_as_date(
            spreadsheet_id="test-id",
            columns=[0],
            pattern="yyyy-mm-dd hh:mm:ss",
            include_time=True,
        )

        body = self._get_batch_update_body(mock_sheets)
        fmt = body["requests"][0]["repeatCell"]["cell"]["userEnteredFormat"]["numberFormat"]
        assert fmt == {"type": "DATE_TIME", "pattern": "yyyy-mm-dd hh:mm:ss"}


class TestGSheetOAuth:
    """Test OAuth installed-app authentication path."""

    def _set_oauth_env(self, monkeypatch, token_path):
        # no service-account creds present
        for var in (
            "GOOGLE_CREDENTIALS",
            "GOOGLE_PROJECT_ID",
            "GOOGLE_API_PKEY_ID",
            "GOOGLE_API_PKEY",
            "GOOGLE_CLIENT_EMAIL",
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CERT_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
        monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "GOCSPX-secret")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "preply-gworkspace-cli")
        monkeypatch.setenv("GSHEET_TOKEN_PATH", str(token_path))

    def test_oauth_runs_flow_when_no_token(self, monkeypatch, tmp_path, mock_google_api_services):
        token_file = tmp_path / "sub" / "token.json"  # missing parent dir -> exercises mkdir
        self._set_oauth_env(monkeypatch, token_file)

        mock_creds = MagicMock(spec=RealOAuthCredentials)
        mock_creds.to_json.return_value = '{"new": true}'

        with patch("ml_analytics.gsheet_connector.InstalledAppFlow") as mock_flow, \
             patch("ml_analytics.gsheet_connector.OAuthCredentials") as mock_oauth:
            flow_instance = MagicMock()
            flow_instance.run_local_server.return_value = mock_creds
            mock_flow.from_client_config.return_value = flow_instance

            gsheet = GSheet()

            mock_flow.from_client_config.assert_called_once()
            flow_instance.run_local_server.assert_called_once_with(port=0)
            mock_oauth.from_authorized_user_file.assert_not_called()
            assert gsheet.credentials is mock_creds
            assert token_file.exists()
            assert token_file.read_text() == '{"new": true}'

    def test_oauth_uses_valid_cached_token(self, monkeypatch, tmp_path, mock_google_api_services):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")  # exists -> cache branch
        self._set_oauth_env(monkeypatch, token_file)

        mock_creds = MagicMock(spec=RealOAuthCredentials)
        mock_creds.valid = True

        with patch("ml_analytics.gsheet_connector.OAuthCredentials") as mock_oauth, \
             patch("ml_analytics.gsheet_connector.InstalledAppFlow") as mock_flow:
            mock_oauth.from_authorized_user_file.return_value = mock_creds

            gsheet = GSheet()

            mock_oauth.from_authorized_user_file.assert_called_once_with(str(token_file), gsheet.scopes)
            mock_flow.from_client_config.assert_not_called()
            mock_creds.refresh.assert_not_called()
            assert gsheet.credentials is mock_creds

    def test_oauth_refreshes_expired_token(self, monkeypatch, tmp_path, mock_google_api_services):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")
        self._set_oauth_env(monkeypatch, token_file)

        mock_creds = MagicMock(spec=RealOAuthCredentials)
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh-token-value"
        mock_creds.to_json.return_value = '{"refreshed": true}'

        with patch("ml_analytics.gsheet_connector.OAuthCredentials") as mock_oauth, \
             patch("ml_analytics.gsheet_connector.InstalledAppFlow") as mock_flow, \
             patch("ml_analytics.gsheet_connector.Request") as _mock_request:
            mock_oauth.from_authorized_user_file.return_value = mock_creds

            gsheet = GSheet()

            mock_creds.refresh.assert_called_once()
            mock_flow.from_client_config.assert_not_called()
            assert gsheet.credentials is mock_creds
            assert token_file.read_text() == '{"refreshed": true}'

    def test_oauth_falls_back_to_flow_on_corrupt_token(self, monkeypatch, tmp_path, mock_google_api_services):
        token_file = tmp_path / "token.json"
        token_file.write_text("not-json")  # corrupt cache -> treat as miss
        self._set_oauth_env(monkeypatch, token_file)

        mock_creds = MagicMock(spec=RealOAuthCredentials)
        mock_creds.to_json.return_value = '{"new": true}'

        with patch("ml_analytics.gsheet_connector.OAuthCredentials") as mock_oauth, \
             patch("ml_analytics.gsheet_connector.InstalledAppFlow") as mock_flow:
            mock_oauth.from_authorized_user_file.side_effect = ValueError("bad token")
            flow_instance = MagicMock()
            flow_instance.run_local_server.return_value = mock_creds
            mock_flow.from_client_config.return_value = flow_instance

            gsheet = GSheet()

            mock_flow.from_client_config.assert_called_once()
            assert gsheet.credentials is mock_creds
            assert token_file.read_text() == '{"new": true}'


class TestGSheetDataFrameToValues:
    """Test DataFrame normalization used before writing to Sheets."""

    def test_datetime_columns_are_serialized_as_strings(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2025-01-02 03:04:05", "2025-06-07 08:09:10"]),
                "n": [1, 2],
            }
        )
        values = GSheet._dataframe_to_values(df, include_headers=True)
        # Headers + 2 rows
        assert values[0] == ["ts", "n"]
        assert values[1][0] == "2025-01-02 03:04:05"
        assert values[2][0] == "2025-06-07 08:09:10"
        # Must be JSON-serializable (this is what the Sheets API does internally)
        json.dumps(values)

    def test_tz_aware_datetime_is_serializable(self):
        import pandas as pd

        df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-02 03:04:05"], utc=True)})
        values = GSheet._dataframe_to_values(df, include_headers=False)
        json.dumps(values)
        assert values[0][0].startswith("2025-01-02")

    def test_period_and_timedelta_are_serializable(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "p": pd.period_range("2025-01", periods=2, freq="M"),
                "d": pd.to_timedelta(["1 days", "2 days"]),
            }
        )
        values = GSheet._dataframe_to_values(df, include_headers=False)
        json.dumps(values)

    def test_nan_inf_and_categorical_still_handled(self):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b", None]),
                "x": [1.0, np.inf, np.nan],
            }
        )
        values = GSheet._dataframe_to_values(df, include_headers=True)
        json.dumps(values)
        # Row layout: [header, row0, row1, row2]
        assert values[2][1] == ""  # inf in row 1
        assert values[3][0] == ""  # None in categorical, row 2
        assert values[3][1] == ""  # NaN in row 2
