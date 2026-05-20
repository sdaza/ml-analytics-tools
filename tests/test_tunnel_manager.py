#!/usr/bin/env python3
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import ml_analytics.tunnel_manager as tunnel_manager

# Import from the new tunnel_manager module
from ml_analytics.tunnel_manager import TunnelManager


class TestTunnelManager(unittest.TestCase):
    """Test cases for TunnelManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.default_tunnel_name = "test-tunnel"
        self.default_code_path = "/usr/local/bin/code"
        self.default_restart_time = "03:00"

    def test_initialization_with_defaults(self):
        """Test TunnelManager initialization with default values"""
        manager = TunnelManager()

        self.assertEqual(manager.tunnel_name, "general")
        self.assertEqual(manager.code_path, os.path.expanduser("~/code"))
        self.assertEqual(manager.restart_time, "04:00")
        self.assertIsNone(manager.tunnel_process)

    def test_initialization_with_custom_values(self):
        """Test TunnelManager initialization with custom values"""
        manager = TunnelManager(
            tunnel_name=self.default_tunnel_name,
            code_path=self.default_code_path,
            restart_time=self.default_restart_time,
        )

        self.assertEqual(manager.tunnel_name, self.default_tunnel_name)
        self.assertEqual(manager.code_path, self.default_code_path)
        self.assertEqual(manager.restart_time, self.default_restart_time)

    def test_parse_restart_time_daily(self):
        """Test _parse_restart_time with HH:MM format"""
        manager = TunnelManager(restart_time="08:00")
        schedule_type, hours = manager._parse_restart_time()
        self.assertEqual(schedule_type, "daily")
        self.assertIsNone(hours)

    def test_parse_restart_time_interval(self):
        """Test _parse_restart_time with Xh interval format"""
        manager = TunnelManager(restart_time="3h")
        schedule_type, hours = manager._parse_restart_time()
        self.assertEqual(schedule_type, "interval")
        self.assertEqual(hours, 3)

    def test_parse_restart_time_invalid_interval(self):
        """Test _parse_restart_time raises on bad Xh value"""
        manager = TunnelManager(restart_time="0h")
        with self.assertRaises(ValueError):
            manager._parse_restart_time()

    def test_parse_restart_time_invalid_format(self):
        """Test _parse_restart_time raises on unrecognised format"""
        manager = TunnelManager(restart_time="daily")
        with self.assertRaises(ValueError):
            manager._parse_restart_time()

    @patch("os.path.exists")
    @patch("os.access")
    @patch("builtins.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    def test_start_tunnel_success(self, mock_popen, mock_file, mock_access, mock_exists):
        """Test successful tunnel start"""
        # Mock path validation
        mock_exists.return_value = True
        mock_access.return_value = True

        # Setup mock process
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        manager = TunnelManager(tunnel_name=self.default_tunnel_name, code_path=self.default_code_path)

        manager.start_tunnel()

        # Verify subprocess was called with correct arguments
        expected_cmd = [self.default_code_path, "tunnel", "--name", self.default_tunnel_name]
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        self.assertEqual(args[0], expected_cmd)
        self.assertEqual(kwargs["text"], True)
        self.assertEqual(kwargs["bufsize"], 1)

        # Verify tunnel_process was set
        self.assertEqual(manager.tunnel_process, mock_process)

    @patch("builtins.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    def test_start_tunnel_with_exception(self, mock_popen, mock_file):
        """Test tunnel start with exception handling"""
        mock_popen.side_effect = Exception("Popen failed")

        manager = TunnelManager()

        # Should not raise exception
        manager.start_tunnel()

        # Process should still be None
        self.assertIsNone(manager.tunnel_process)

    def test_restart_tunnel_when_running(self):
        """Test restarting tunnel when process is running"""
        manager = TunnelManager()

        # Mock a running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.stdin = MagicMock()
        manager.tunnel_process = mock_process

        manager.restart_tunnel()

        # Verify restart command was sent
        mock_process.stdin.write.assert_called_once_with("r\n")
        mock_process.stdin.flush.assert_called_once()

    @patch.object(TunnelManager, "start_tunnel")
    def test_restart_tunnel_when_not_running(self, mock_start):
        """Test restarting tunnel when process is not running"""
        manager = TunnelManager()

        # Mock a dead process
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process has exited
        manager.tunnel_process = mock_process

        manager.restart_tunnel()

        # Verify start_tunnel was called
        mock_start.assert_called_once()

    @patch.object(TunnelManager, "start_tunnel")
    def test_restart_tunnel_when_process_none(self, mock_start):
        """Test restarting tunnel when process is None"""
        manager = TunnelManager()
        manager.tunnel_process = None

        manager.restart_tunnel()

        # Verify start_tunnel was called
        mock_start.assert_called_once()

    @patch.object(TunnelManager, "start_tunnel")
    def test_restart_tunnel_with_exception(self, mock_start):
        """Test restart tunnel with exception handling"""
        manager = TunnelManager()

        # Mock a process that raises exception on stdin.write
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin.write.side_effect = Exception("Write failed")
        manager.tunnel_process = mock_process

        manager.restart_tunnel()

        # Verify start_tunnel was called as fallback
        mock_start.assert_called_once()

    @patch.object(TunnelManager, "start_tunnel")
    def test_check_tunnel_health_when_dead(self, mock_start):
        """Test health check when tunnel process is dead"""
        manager = TunnelManager()

        # Mock a dead process
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process has exited
        manager.tunnel_process = mock_process

        manager.check_tunnel_health()

        # Verify start_tunnel was called
        mock_start.assert_called_once()

    def test_check_tunnel_health_when_alive(self):
        """Test health check when tunnel process is alive"""
        manager = TunnelManager()

        # Mock a running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        manager.tunnel_process = mock_process

        with patch.object(manager, "start_tunnel") as mock_start:
            manager.check_tunnel_health()

            # Verify start_tunnel was NOT called
            mock_start.assert_not_called()

    def test_check_tunnel_health_when_none(self):
        """Test health check when tunnel process is None"""
        manager = TunnelManager()
        manager.tunnel_process = None

        with patch.object(manager, "start_tunnel") as mock_start:
            manager.check_tunnel_health()

            # Verify start_tunnel was NOT called (no process to check)
            mock_start.assert_not_called()

    @patch("schedule.every")
    @patch("schedule.run_pending")
    @patch("time.sleep")
    @patch.object(TunnelManager, "start_tunnel")
    def test_run_initialization(self, mock_start, mock_sleep, mock_run_pending, mock_every):
        """Test run method initialization"""
        # Make sleep raise KeyboardInterrupt after first call to exit loop
        mock_sleep.side_effect = KeyboardInterrupt()

        manager = TunnelManager()
        manager.tunnel_process = MagicMock()

        try:
            manager.run()
        except KeyboardInterrupt:
            pass

        # Verify start_tunnel was called
        mock_start.assert_called_once()

        # Verify schedules were set up
        self.assertTrue(mock_every.called)

    @patch("schedule.every")
    @patch("schedule.run_pending")
    @patch("time.sleep")
    @patch.object(TunnelManager, "start_tunnel")
    @patch.object(TunnelManager, "stop_tunnel")
    def test_run_keyboard_interrupt_cleanup(self, mock_stop, mock_start, mock_sleep, mock_run_pending, mock_every):
        """Test run method cleanup on keyboard interrupt"""
        mock_sleep.side_effect = KeyboardInterrupt()

        manager = TunnelManager()

        try:
            manager.run()
        except KeyboardInterrupt:
            pass

        # Verify stop_tunnel was called
        mock_stop.assert_called_once()


class TestTunnelManagerMain(unittest.TestCase):
    """Test cases for main CLI function"""

    @patch("argparse.ArgumentParser.parse_args")
    @patch.object(TunnelManager, "run")
    def test_main_with_default_args(self, mock_run, mock_parse_args):
        """Test main function with default arguments"""
        mock_args = MagicMock()
        mock_args.name = "general"
        mock_args.code_path = None
        mock_args.restart_time = "04:00"
        mock_parse_args.return_value = mock_args

        tunnel_manager.main()

        # Verify run was called
        mock_run.assert_called_once()

    @patch("argparse.ArgumentParser.parse_args")
    @patch.object(TunnelManager, "run")
    def test_main_with_custom_args(self, mock_run, mock_parse_args):
        """Test main function with custom arguments"""
        mock_args = MagicMock()
        mock_args.name = "custom-tunnel"
        mock_args.code_path = "/custom/path/code"
        mock_args.restart_time = "02:00"
        mock_parse_args.return_value = mock_args

        tunnel_manager.main()

        # Verify run was called
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
