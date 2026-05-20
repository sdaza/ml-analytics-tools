# CLI Commands

The package includes a small set of command-line helpers.

## `ml-analytics-auth`

Checks AWS credentials and starts AWS SSO login if the current credentials are missing or expired.

```bash
ml-analytics-auth
```

Use when S3 or Redshift workflows need AWS credentials.

## `ml-analytics-sso`

Runs the AWS SSO credential check/login flow directly.

```bash
ml-analytics-sso
```

For named profiles, use the AWS CLI directly and export the profile before running your Python code:

```bash
aws sso login --profile my-profile
export AWS_PROFILE=my-profile
```

## `ml-analytics-tunnel`

Runs the VS Code tunnel helper.

```bash
ml-analytics-tunnel
```

See [Tunnel Manager](TUNNEL_MANAGER.md) for details.

## Related Docs

- [AWS Authentication Guide](AWS_AUTHENTICATION.md)
- [Google Sheets Connector Usage Guide](GSHEET_CONNECTOR_USAGE.md)
- [Slack Connector Usage Guide](SLACK_CONNECTOR_USAGE.md)
