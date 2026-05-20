#!/usr/bin/env python3
"""
VS Code Tunnel Manager

Manages VS Code tunnels with automatic restarts and health checks.
"""

import os
import subprocess
import threading
import time

import schedule

from .utils import get_logger

logger = get_logger(__name__)


class TunnelManager:
    """Manages VS Code tunnel lifecycle with scheduled restarts and health monitoring."""

    def __init__(
        self,
        tunnel_name: str = "general",
        code_path: str | None = None,
        restart_time: str = "04:00",
        health_check_minutes: int = 15,
    ):
        """
        Initialize the Tunnel Manager.

        Args:
            tunnel_name: Name for the VS Code tunnel
            code_path: Path to the code executable (defaults to ~/code)
            restart_time: Restart schedule — either HH:MM for a daily fixed time (e.g. "08:00")
                          or Xh for a recurring interval (e.g. "3h" = every 3 hours)
            health_check_minutes: Interval in minutes for health checks (default: 15)
        """
        self.tunnel_name = tunnel_name
        self.code_path = code_path or os.path.expanduser("~/code")
        self.restart_time = restart_time
        self.health_check_minutes = health_check_minutes
        self.tunnel_process = None
        self.log_file = None
        self._monitor_thread = None
        self._monitor_stop = threading.Event()

    def __del__(self):
        """Cleanup resources on deletion."""
        self._close_log_file()

    def _close_log_file(self):
        """Close the log file if open."""
        log_file = getattr(self, "log_file", None)
        if log_file and not log_file.closed:
            try:
                log_file.close()
            except Exception as e:
                logger.warning(f"Error closing log file: {e}")

    def _validate_code_path(self) -> bool:
        """
        Validate that the code executable exists.

        Returns:
            True if code path is valid, False otherwise
        """
        if not os.path.exists(self.code_path):
            logger.error(f"❌ Code executable not found at: {self.code_path}")
            return False

        if not os.access(self.code_path, os.X_OK):
            logger.error(f"❌ Code executable not executable: {self.code_path}")
            return False

        return True

    def _monitor_output(self):
        """Read tunnel stdout, write to log file, and surface auth prompts to the terminal."""
        auth_keywords = ("github.com/login/device", "use code")
        try:
            for line in self.tunnel_process.stdout:
                if self._monitor_stop.is_set():
                    break
                if self.log_file and not self.log_file.closed:
                    self.log_file.write(line)
                    self.log_file.flush()
                lower = line.lower()
                if any(k in lower for k in auth_keywords):
                    logger.info(f"🔑 AUTH REQUIRED — {line.rstrip()}")
        except Exception as e:
            logger.debug(f"Output monitor stopped: {e}")

    def start_tunnel(self):
        """Start the VS Code tunnel."""
        try:
            # Validate code path
            if not self._validate_code_path():
                logger.error("Cannot start tunnel: invalid code path")
                return

            # Stop previous monitor thread
            self._monitor_stop.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=3)

            # Clean up old process if it exists
            if self.tunnel_process is not None:
                try:
                    if self.tunnel_process.poll() is None:
                        logger.warning("⚠️  Terminating existing tunnel process before starting new one...")
                        self.tunnel_process.terminate()
                        self.tunnel_process.wait(timeout=3)
                except Exception as e:
                    logger.warning(f"Error cleaning up old process: {e}")

            # Close previous log file if exists
            self._close_log_file()

            # Open new log file
            log_path = os.path.expanduser("~/vscode-tunnel.log")
            self.log_file = open(log_path, "a")

            cmd = [self.code_path, "tunnel", "--name", self.tunnel_name]

            self.tunnel_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Start background thread to monitor output and surface auth prompts
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_output, daemon=True)
            self._monitor_thread.start()

            logger.info(f"✅ Started VS Code tunnel '{self.tunnel_name}' (PID: {self.tunnel_process.pid})")
            logger.info(f"🌐 Access at: https://vscode.dev/tunnel/{self.tunnel_name}")

        except Exception as e:
            logger.error(f"❌ Error starting tunnel: {e}")
            self._close_log_file()

    def restart_tunnel(self):
        """Restart the tunnel by sending 'r' command."""
        try:
            if self.tunnel_process and self.tunnel_process.poll() is None:
                logger.info("🔄 Sending restart command to tunnel...")
                self.tunnel_process.stdin.write("r\n")
                self.tunnel_process.stdin.flush()
                logger.info("✅ Restart command sent")
            else:
                logger.warning("⚠️  Tunnel process not running, starting new one...")
                self.start_tunnel()
        except Exception as e:
            logger.error(f"❌ Error restarting tunnel: {e}")
            logger.info("Attempting to start new tunnel...")
            self.start_tunnel()

    def check_tunnel_health(self):
        """Check if tunnel process is still alive."""
        if self.tunnel_process and self.tunnel_process.poll() is not None:
            logger.warning("⚠️  Tunnel process died! Restarting...")
            self.restart_tunnel()

    def stop_tunnel(self):
        """Stop the tunnel gracefully."""
        if self.tunnel_process and self.tunnel_process.poll() is None:
            try:
                logger.info("🛑 Stopping tunnel...")
                self.tunnel_process.stdin.write("x\n")
                self.tunnel_process.stdin.flush()
                self.tunnel_process.wait(timeout=5)
                logger.info("✅ Tunnel stopped")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Tunnel didn't stop gracefully, terminating...")
                self.tunnel_process.terminate()
                try:
                    self.tunnel_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.error("❌ Force killing tunnel process")
                    self.tunnel_process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping tunnel: {e}")
            finally:
                self._monitor_stop.set()
                self._close_log_file()

    def _parse_restart_time(self) -> tuple[str, int | None]:
        """
        Parse restart_time into a schedule type and value.

        Returns:
            ('interval', hours) for Xh format, or ('daily', None) for HH:MM format.
        """
        value = self.restart_time.strip()
        if value.endswith("h"):
            try:
                hours = int(value[:-1])
                if hours < 1:
                    raise ValueError
                return ("interval", hours)
            except ValueError:
                raise ValueError(f"Invalid restart interval '{value}'. Use format like '3h'.") from None
        else:
            try:
                time.strptime(value, "%H:%M")
                return ("daily", None)
            except ValueError:
                raise ValueError(
                    f"Invalid restart time '{value}'. Use HH:MM (e.g. '08:00') or Xh (e.g. '3h')."
                ) from None

    def run(self):
        """Run the manager with scheduled restarts."""
        logger.info("🚀 Starting VS Code Tunnel Manager")

        schedule_type, interval_hours = self._parse_restart_time()
        if schedule_type == "interval":
            logger.info(f"⏰ Restart scheduled every {interval_hours} hour(s)")
            schedule.every(interval_hours).hours.do(self.restart_tunnel)
        else:
            logger.info(f"⏰ Daily restart scheduled at: {self.restart_time}")
            schedule.every().day.at(self.restart_time).do(self.restart_tunnel)

        logger.info(f"🏥 Health check interval: {self.health_check_minutes} minutes")

        # Start tunnel immediately
        self.start_tunnel()

        # Schedule health check with configurable interval
        schedule.every(self.health_check_minutes).minutes.do(self.check_tunnel_health)

        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping Tunnel Manager")
            self.stop_tunnel()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="VS Code Tunnel Manager with automatic restarts and health monitoring")
    parser.add_argument("--name", default="general", help="Tunnel name (default: general)")
    parser.add_argument("--code-path", help="Path to code executable (default: ~/code)")
    parser.add_argument(
        "--restart-time",
        default="04:00",
        help="Restart schedule: HH:MM for daily at a fixed time (e.g. '08:00') or Xh for every X hours (e.g. '3h'). Default: 04:00",  # noqa: E501
    )
    parser.add_argument(
        "--health-check-interval", type=int, default=15, help="Health check interval in minutes (default: 15)"
    )

    args = parser.parse_args()

    manager = TunnelManager(
        tunnel_name=args.name,
        code_path=args.code_path,
        restart_time=args.restart_time,
        health_check_minutes=args.health_check_interval,
    )

    manager.run()


if __name__ == "__main__":
    main()
