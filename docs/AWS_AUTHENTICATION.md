# AWS Authentication Guide

This package can use AWS credentials for S3 operations and Redshift unload/copy workflows.
It does not assume any private package registry or organization-specific AWS account.

## Requirements

- AWS CLI installed and configured.
- Access to the AWS account, role, or SSO profile that owns your S3 buckets.
- An S3 bucket passed explicitly to `S3Connector`/`DataConnector` or configured with `ML_ANALYTICS_S3_BUCKET`.

## CLI Usage

Use the package CLI to check the current credentials and run AWS SSO login when needed:

```bash
ml-analytics-auth
```

For SSO-only usage, this command is equivalent:

```bash
ml-analytics-sso
```

To use a named profile outside the package CLI, run:

```bash
aws sso login --profile my-profile
export AWS_PROFILE=my-profile
```

## Python Usage

```python
from ml_analytics import ensure_aws_authenticated

ensure_aws_authenticated()
```

For a specific SSO profile:

```python
from ml_analytics import ensure_aws_sso_login

ensure_aws_sso_login(profile="my-profile")
```

`S3Connector` can also attempt SSO login when credentials are missing:

```python
from ml_analytics import S3Connector

s3 = S3Connector(
    bucket="my-analytics-bucket",
    s3_root="exports",
    auto_sso_login=True,
    sso_profile="my-profile",
)
```

## Troubleshooting

If authentication fails, verify the AWS CLI sees your identity:

```bash
aws sts get-caller-identity
```

If you use a named profile:

```bash
aws sts get-caller-identity --profile my-profile
```

If S3 methods fail with a missing bucket message, pass `bucket=...` or set:

```bash
export ML_ANALYTICS_S3_BUCKET=my-analytics-bucket
```
