"""
AWS authentication utilities.
"""

import subprocess
import sys

from .utils import get_logger

logger = get_logger("aws_auth")


def _do_sso_login(profile: str = None) -> bool:
    """
    Performs the interactive SSO login flow.

    Parameters
    ----------
    profile : str, optional
        AWS profile name to use.

    Returns
    -------
    bool
        True if login successful, False otherwise.
    """
    try:
        logger.info("AWS SSO login required - starting authentication...")
        login_cmd = ["aws", "sso", "login"]
        if profile:
            login_cmd.extend(["--profile", profile])

        # Redirect stdout to stderr so user sees prompts but eval doesn't execute them
        login_result = subprocess.run(login_cmd, stdout=sys.stderr, timeout=300)

        if login_result.returncode == 0:
            logger.info("✓ AWS SSO login successful")
            return True
        else:
            logger.error("✗ AWS SSO login failed")
            return False

    except subprocess.TimeoutExpired:
        logger.error("AWS SSO login timed out")
        return False
    except FileNotFoundError:
        logger.error("AWS CLI not found. Please install AWS CLI.")
        return False
    except Exception as e:
        logger.error(f"Error during AWS SSO login: {e}")
        return False


def ensure_aws_sso_login(profile: str = None, force: bool = False) -> bool:
    """
    Ensures AWS SSO is authenticated. If not, prompts user to login.

    Parameters
    ----------
    profile : str, optional
        AWS profile name to use. If None, uses default profile.
    force : bool, optional
        If True, skip the cached credential check and force a fresh SSO login.

    Returns
    -------
    bool
        True if authenticated successfully, False otherwise.
    """
    if force:
        return _do_sso_login(profile)

    try:
        # Check if already logged in by attempting to get caller identity
        cmd = ["aws", "sts", "get-caller-identity"]
        if profile:
            cmd.extend(["--profile", profile])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Already authenticated - don't log to reduce noise
            return True

        # Not logged in, attempt SSO login
        return _do_sso_login(profile)

    except subprocess.TimeoutExpired:
        logger.error("AWS SSO login timed out")
        return False
    except FileNotFoundError:
        logger.error("AWS CLI not found. Please install AWS CLI.")
        return False
    except Exception as e:
        logger.error(f"Error during AWS SSO login: {e}")
        return False


def ensure_aws_authenticated(sso_profile: str = None, print_exports: bool = False) -> bool:
    """
    Convenience function that ensures AWS SSO is authenticated.

    Parameters
    ----------
    sso_profile : str, optional
        AWS SSO profile to use
    print_exports : bool, optional
        Kept for backward-compatible CLI calls. No shell exports are required.

    Returns
    -------
    bool
        True if AWS SSO authentication succeeded, False otherwise.

    Example
    -------
    >>> from ml_analytics.aws_auth import ensure_aws_authenticated
    >>> ensure_aws_authenticated()
    """
    del print_exports
    logger.info("Ensuring AWS authentication...")

    if not ensure_aws_sso_login(sso_profile):
        return False

    logger.info("✓ AWS authentication complete")
    return True


def run_uv_command(command: str) -> bool:
    """
    Runs a UV command and returns whether it succeeded.

    Parameters
    ----------
    command : str
        The UV command to run (e.g., "uv sync", "uv add package")

    Returns
    -------
    bool
        True if the command executed successfully, False otherwise.

    Example
    -------
    >>> from ml_analytics.aws_auth import run_uv_command
    >>> run_uv_command("uv sync")
    """
    try:
        logger.info(f"Running UV command: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info("✓ UV command completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            logger.error(f"✗ UV command failed: {result.stderr}")
            if result.stderr:
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"UV command timed out: {command}")
        return False
    except Exception as e:
        logger.error(f"Error running UV command '{command}': {e}")
        return False
