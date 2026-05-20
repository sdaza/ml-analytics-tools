"""
Command-line interface for ml-analytics-tools
"""

import sys

from .aws_auth import ensure_aws_authenticated, ensure_aws_sso_login
from .utils import get_logger

logger = get_logger("ml_analytics_cli")


def aws_auth_command():
    """
    CLI command to authenticate with AWS SSO.
    Can be run as: ml-analytics-auth
    """
    try:
        if ensure_aws_authenticated(print_exports=True):
            return 0
        else:
            print("Authentication failed", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def aws_sso_command():
    """
    CLI command to authenticate with AWS SSO only.
    Can be run as: ml-analytics-sso
    """
    try:
        if ensure_aws_sso_login():
            return 0
        else:
            print("SSO login failed", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point for CLI commands."""
    # This is a placeholder for potential future CLI expansion
    aws_auth_command()


if __name__ == "__main__":
    sys.exit(main())
