#!/usr/bin/env python3
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from ml_analytics.aws_auth import (
    _do_sso_login,
    ensure_aws_authenticated,
    ensure_aws_sso_login,
    run_uv_command,
)


class TestDoSsoLogin(unittest.TestCase):
    """Tests for _do_sso_login helper."""

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_login_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(_do_sso_login())

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_login_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        self.assertFalse(_do_sso_login())

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_login_with_profile(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(_do_sso_login(profile="my-profile"))
        cmd = mock_run.call_args[0][0]
        self.assertIn("--profile", cmd)
        self.assertIn("my-profile", cmd)

    @patch("ml_analytics.aws_auth.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300))
    def test_login_timeout(self, mock_run):
        self.assertFalse(_do_sso_login())

    @patch("ml_analytics.aws_auth.subprocess.run", side_effect=FileNotFoundError)
    def test_login_aws_cli_not_found(self, mock_run):
        self.assertFalse(_do_sso_login())


class TestEnsureAwsSsoLogin(unittest.TestCase):
    """Tests for ensure_aws_sso_login."""

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_already_authenticated(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(ensure_aws_sso_login())
        # Should only call get-caller-identity, not sso login
        self.assertEqual(mock_run.call_count, 1)
        cmd = mock_run.call_args[0][0]
        self.assertIn("get-caller-identity", cmd)

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_not_authenticated_then_login_succeeds(self, mock_run):
        # First call (get-caller-identity) fails, second (sso login) succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1),  # get-caller-identity fails
            MagicMock(returncode=0),  # sso login succeeds
        ]
        self.assertTrue(ensure_aws_sso_login())
        self.assertEqual(mock_run.call_count, 2)

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_not_authenticated_then_login_fails(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=1),  # get-caller-identity fails
            MagicMock(returncode=1),  # sso login fails
        ]
        self.assertFalse(ensure_aws_sso_login())

    @patch("ml_analytics.aws_auth._do_sso_login")
    def test_force_skips_credential_check(self, mock_do_login):
        mock_do_login.return_value = True
        self.assertTrue(ensure_aws_sso_login(force=True))
        mock_do_login.assert_called_once_with(None)

    @patch("ml_analytics.aws_auth._do_sso_login")
    def test_force_with_profile(self, mock_do_login):
        mock_do_login.return_value = True
        self.assertTrue(ensure_aws_sso_login(profile="dev", force=True))
        mock_do_login.assert_called_once_with("dev")

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_with_profile(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(ensure_aws_sso_login(profile="prod"))
        cmd = mock_run.call_args[0][0]
        self.assertIn("--profile", cmd)
        self.assertIn("prod", cmd)


class TestEnsureAwsAuthenticated(unittest.TestCase):
    """Tests for ensure_aws_authenticated convenience function."""

    @patch("ml_analytics.aws_auth.ensure_aws_sso_login", return_value=True)
    def test_success(self, mock_sso):
        self.assertTrue(ensure_aws_authenticated())
        mock_sso.assert_called_once_with(None)

    @patch("ml_analytics.aws_auth.ensure_aws_sso_login", return_value=False)
    def test_sso_fails(self, mock_sso):
        self.assertFalse(ensure_aws_authenticated())

    @patch("ml_analytics.aws_auth.ensure_aws_sso_login", return_value=True)
    def test_with_profile(self, mock_sso):
        self.assertTrue(ensure_aws_authenticated(sso_profile="prod"))
        mock_sso.assert_called_once_with("prod")


class TestRunUvCommand(unittest.TestCase):
    """Tests for run_uv_command."""

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
        self.assertTrue(run_uv_command("uv sync"))
        mock_run.assert_called_once_with("uv sync", shell=True, capture_output=True, text=True, timeout=300)

    @patch("ml_analytics.aws_auth.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        self.assertFalse(run_uv_command("uv sync"))

    @patch("ml_analytics.aws_auth.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300))
    def test_timeout(self, mock_run):
        self.assertFalse(run_uv_command("uv sync"))


if __name__ == "__main__":
    unittest.main()
