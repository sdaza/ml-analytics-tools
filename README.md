# ML Analytics Tools

[![CI](https://github.com/sdaza/ml-analytics-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/sdaza/ml-analytics-tools/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/sdaza/ml-analytics-tools)](https://github.com/sdaza/ml-analytics-tools/releases)
[![PyPI](https://img.shields.io/pypi/v/ml-analytics-tools.svg)](https://pypi.org/project/ml-analytics-tools/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Utilities for common analytics and machine learning workflows: Redshift, S3,
Google Sheets, Slack, MLflow, model evaluation, and SQL pipelines.

The package is intentionally infrastructure-neutral. Buckets, credentials,
MLflow hosts, and tokens are provided by your environment or by explicit
arguments.

## What Is Included

- `DataConnector`: run Redshift or Snowflake SQL, load SQL files, unload/load data through S3, and create Redshift tables from DataFrames.
- `SFConnector`: read Snowflake through Spark and save results to Unity Catalog tables (Databricks). PySpark is imported lazily, so the rest of the package works without it.
- `S3Connector`: read, write, list, delete, and query S3 data with DuckDB.
- `GSheet`: read, write, share, and export Google Sheets data.
- `SlackConnector`: send messages, upload files, and manage simple Slack interactions.
- `ModelManager`: create MLflow experiments, log models, register versions, manage aliases, and handle permissions.
- `model_tools`: classification, regression, survival analysis, CatBoost helpers, plotting, and reporting utilities.
- `utils`: project-root discovery, SQL file loading, logging, credentials, and YAML SQL pipelines.

## Install

From PyPI, after a release is available:

```bash
uv add ml-analytics-tools
```

Directly from GitHub:

```bash
uv add git+https://github.com/sdaza/ml-analytics-tools
```

For local development:

```bash
uv sync --all-groups
```

## Configuration

The package loads a `.env` file from the project root when it is imported.
Only configure the services you use.

```bash
# Redshift
BI_REDSHIFT_HOST=redshift-cluster.example.com
BI_REDSHIFT_DB=analytics
BI_REDSHIFT_USER=analytics_user
BI_REDSHIFT_PASSWORD=secret
BI_REDSHIFT_PORT=5439

# Snowflake
SNOWFLAKE_USER=your.name@example.com
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_WAREHOUSE=ANALYTICS_S_WH
SNOWFLAKE_DATABASE=ANALYTICS_DB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_ROLE=SNOWFLAKE_STND_DATA
SNOWFLAKE_AUTHENTICATOR=externalbrowser

# Browser-free Snowflake auth for local or Databricks jobs
SNOWFLAKE_PRIVATE_KEY_PATH=~/.snowflake/rsa_key.p8
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=secret

# S3
ML_ANALYTICS_S3_BUCKET=my-analytics-bucket

# MLflow
MLFLOW_TRACKING_URI=https://mlflow.example.com
MLFLOW_TRACKING_USERNAME=user@example.com
MLFLOW_TRACKING_PASSWORD=secret

# Google Sheets
GSHEET_SPREADSHEET_ID=optional-default-sheet-id
GOOGLE_CREDENTIALS='{"type":"service_account", ...}'

# Slack
SLACK_BOT_TOKEN=xoxb-your-token
```

S3 buckets are never hard-coded. Pass `bucket=...` or `s3_bucket=...`, or set
`ML_ANALYTICS_S3_BUCKET`.

## AWS Authentication

Use the CLI helper for AWS SSO:

```bash
ml-analytics-auth
```

You can also call it from Python:

```python
from ml_analytics import ensure_aws_authenticated

ensure_aws_authenticated()
```

See [AWS Authentication](docs/AWS_AUTHENTICATION.md) and
[CLI Commands](docs/CLI_COMMANDS.md) for details.

## Quick Examples

### Query Redshift

```python
from ml_analytics import DataConnector

dc = DataConnector()

df = dc.sql("SELECT * FROM analytics.customer_features LIMIT 100")
df_polars = dc.sql("queries/features.sql", format="polars", country="es")
```

### Query Snowflake

```python
from ml_analytics import DataConnector

dc = DataConnector(engine="snowflake")

df = dc.sql("SELECT 1 AS col_1")
```

For local interactive work, `SNOWFLAKE_AUTHENTICATOR=externalbrowser` is supported.
SSO tokens are cached in the OS keychain, so the browser login only happens once
per token lifetime. (Note: `externalbrowser` works with `DataConnector` only;
`SFConnector` rejects it, since Spark jobs block on the interactive browser SSO.)
For Databricks and Spark jobs, use key-pair auth instead. The connector reads
default Databricks personal-scope secrets automatically:

```bash
databricks secrets put-secret user-your.name@example.com snowflake_key --bytes-value """$(cat rsa_key.p8)"""
databricks secrets put-secret user-your.name@example.com snowflake_key_pass --string-value """<password>"""
```

Then build Spark connector options without opening a browser:

```python
from ml_analytics import DataConnector

dc = DataConnector(engine="snowflake", secret_scope="user-your.name@example.com")
options = dc.snowflake_spark_options()

df = (
    spark.read.format("net.snowflake.spark.snowflake")
    .options(**options)
    .option("query", "SELECT 1 AS col_1")
    .load()
)
```

### Query Snowflake With Spark (`SFConnector`)

On Databricks, `SFConnector` reads Snowflake directly as Spark DataFrames and can
persist results into Unity Catalog tables. It reuses the same `SNOWFLAKE_*`
settings and key-pair secrets as `DataConnector`, and only imports PySpark when a
query/write method runs.

```python
from ml_analytics import SFConnector

sf = SFConnector()  # reads SNOWFLAKE_* env vars; secret scope inferred from SNOWFLAKE_USER

# Spark DataFrame
df = sf.sql("SELECT * FROM cds.dim_tutor LIMIT 1000")

# pandas DataFrame
pdf = sf.sql("SELECT 1 AS col_1", return_pandas=True)

# run a query from a .sql file (relative to project root), with templating
df = sf.sql("queries/experiment.sql", days=14)

# pull and save the result to a Unity Catalog table in one call
sf.sql("queries/experiment.sql", save_table=True, schema="analytics", table="exp")

# or save any Spark DataFrame to Unity Catalog
sf.save_to_uc(df, table="exp", schema="analytics", catalog="prod")

# save a YAML-ordered folder of SQL queries as Unity Catalog tables
df = sf.save_pipeline_to_uc(
    "queries/churn_pipeline",
    pipeline="daily",
    catalog="prod",
    schema="analytics",
)
```

Credentials resolve per field as: explicit argument → `SNOWFLAKE_*` environment
variable → Databricks secret. See the
[Snowflake Spark Connector](docs/SF_CONNECTOR_USAGE.md) guide for credential
setup and all options.

### Create A Redshift Table From A DataFrame

```python
dc.create_table_from_dataframe(
    df,
    table="model_scores",
    schema="analytics",
    drop_existing_table=True,
)
```

### Work With S3

```python
from ml_analytics import S3Connector

s3 = S3Connector(bucket="my-analytics-bucket", s3_root="projects/churn")

s3.save_dataframe(df, directory="outputs", file_name="scores")

summary = s3.query(
    """
    SELECT segment, count(*) AS rows
    FROM read_parquet('s3://my-analytics-bucket/projects/churn/outputs/*.parquet')
    GROUP BY segment
    """
)
```

### Read And Write Google Sheets

```python
from ml_analytics import GSheet

gsheet = GSheet(credentials_path="gsheet_credentials.json")

df = gsheet.read_sheet(spreadsheet_id="...", sheet_name="Input")
gsheet.write_sheet(df, spreadsheet_id="...", sheet_name="Results")
```

#### OAuth authentication (alternative to a service account)

`GSheet` can authenticate as your own Google account using OAuth installed-app
credentials. Set these env vars and the connector uses OAuth automatically when
no service-account credentials are found:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_OAUTH_CLIENT_ID` | yes | OAuth client id (`...apps.googleusercontent.com`) |
| `GOOGLE_OAUTH_CLIENT_SECRET` | yes | OAuth client secret (`GOCSPX-...`) |
| `GOOGLE_CLOUD_PROJECT` | optional | GCP project id (e.g. `my-gcp-project`) |
| `GSHEET_TOKEN_PATH` | optional | Token cache path (default `~/.config/ml-analytics/gsheet_token.json`) |

The first run opens a browser for one-time consent; the cached refresh token
makes later runs non-interactive. Under OAuth, `get_service_account_email()`
returns `None`.

### Log To MLflow

```python
from ml_analytics import ModelManager

manager = ModelManager(model_name="churn-model", user="user@example.com")

manager.start_run("training")
manager.log_metric("auc", 0.91)
manager.end_run()
```

### Send A Slack Message

```python
from ml_analytics import SlackConnector

slack = SlackConnector()
slack.send_message(channel="#ml-alerts", text="Training finished")
```

## Detailed Guides

| Guide | Use It For |
| --- | --- |
| [AWS Authentication](docs/AWS_AUTHENTICATION.md) | AWS SSO setup and Python helpers |
| [CLI Commands](docs/CLI_COMMANDS.md) | Available console commands |
| [Snowflake Spark Connector](docs/SF_CONNECTOR_USAGE.md) | `SFConnector` credential setup, reads, and writes on Spark/Databricks |
| [Google Sheets](docs/GSHEET_CONNECTOR_USAGE.md) | Sheets setup, sharing, exports, and examples |
| [Slack](docs/SLACK_CONNECTOR_USAGE.md) | Slack token setup and message/file examples |
| [Tunnel Manager](docs/TUNNEL_MANAGER.md) | SSH tunnel configuration and CLI usage |

## Development

Run the standard checks before opening a PR:

```bash
uv run ruff check
uv run pytest
```

CI runs Ruff and pytest on Python 3.11 and 3.12.

## Releases

This repository uses Release Please. Conventional commits on `main` create or
update a release PR with the next version and changelog. When that PR is merged,
the release workflow builds the package and publishes it to PyPI through Trusted
Publishing using the `pypi` GitHub environment.

## Contributing

Keep changes small, covered by tests when behavior changes, and free of
environment-specific defaults. Prefer explicit configuration over hidden
infrastructure assumptions.
