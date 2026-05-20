# ML Analytics Tools

A Python library for analytics and machine learning workflows, providing connectors for AWS S3 and Redshift, Google Sheets, Slack, MLflow model management, and general ML utilities.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
  - [Environment Variables](#environment-variables)
  - [AWS Authentication](#aws-authentication)
  - [Installation](#installation)
- [Data Connector](#data-connector)
  - [Initialization](#initialization)
  - [Executing SQL Queries](#executing-sql-queries)
  - [Copying Tables](#copying-tables)
  - [Unloading Data to S3](#unloading-data-to-s3)
  - [Loading Data from S3 to Redshift](#loading-data-from-s3-to-redshift)
  - [Creating Redshift Tables from DataFrames](#creating-redshift-tables-from-dataframes)
  - [Creating and Syncing Spectrum Tables](#creating-and-syncing-spectrum-tables)
- [S3 Connector](#s3-connector)
  - [Initialization](#initialization-1)
  - [Saving Data to S3](#saving-data-to-s3)
  - [Listing and Deleting S3 Files](#listing-and-deleting-s3-files)
  - [Querying Parquet Files with DuckDB](#querying-parquet-files-with-duckdb)
- [Google Sheets Connector](#google-sheets-connector)
  - [Setup](#setup-1)
  - [Reading from Google Sheets](#reading-from-google-sheets)
  - [Writing to Google Sheets](#writing-to-google-sheets)
  - [Transferring Data to S3](#transferring-data-to-s3)
- [Slack Connector](#slack-connector)
- [Model Manager](#model-manager)
  - [Initialization](#initialization-2)
  - [Logging a Model](#logging-a-model)
  - [Managing Access](#managing-access)
  - [Managing Model Versions and Loading Models](#managing-model-versions-and-loading-models)
- [Model Tools](#model-tools)
- [Utilities](#utilities)
  - [SQL Pipeline YAML](#sql-pipeline-yaml)
- [Additional Guides](#additional-guides)
- [Contributing](#contributing)

---

## Overview

The ML Analytics Tools library offers tools to facilitate:

- **Data connectors**: Connect to Redshift, execute SQL queries, unload/load data between Redshift and S3.
- **S3 connector**: Query S3 Parquet files with DuckDB, save/read DataFrames, manage files.
- **Google Sheets connector**: Read and write Google Sheets with service account authentication.
- **Slack connector**: Send messages, reactions, and upload files to Slack channels.
- **Model Manager**: Register, log, version, and delete ML models via MLflow.
- **Model Tools**: Feature engineering, performance evaluation, survival analysis, and visualization utilities.

---

## Setup

### Environment Variables

Create a `.env` file with your credentials:

```
BI_REDSHIFT_HOST=redshift-cluster.example.com
BI_REDSHIFT_DB=analytics
BI_REDSHIFT_USER=analytics_user
BI_REDSHIFT_PASSWORD=secret
BI_REDSHIFT_PORT=5439
ML_ANALYTICS_S3_BUCKET=my-analytics-bucket

MLFLOW_TRACKING_USERNAME=user@example.com
MLFLOW_TRACKING_PASSWORD=secret
MLFLOW_TRACKING_URI=https://mlflow.example.io
```

### AWS Authentication

For AWS SSO authentication, use the built-in CLI command:

```bash
ml-analytics-auth
```

For full details, see the [AWS Authentication Guide](docs/AWS_AUTHENTICATION.md) and [CLI Commands Reference](docs/CLI_COMMANDS.md).

### Installation

Install from your package index or directly from GitHub:

```bash
uv add ml-analytics-tools
uv add git+https://github.com/<your-org>/ml-analytics-tools
```

---

## Data Connector

### Initialization

```python
from ml_analytics import DataConnector

dc = DataConnector()
```

### Executing SQL Queries

Both `sql()` and `execute_sql()` accept a SQL string **or** a path to a `.sql` file (relative to project root). File paths support `**kwargs` for template substitution via `str.format()`.

```python
# Inline query
df = dc.sql("SELECT * FROM analytics.customer_features LIMIT 10")

# From a .sql file
df = dc.sql("notebooks/iv-analysis/causal_analysis_features.sql")

# From a .sql file with template variables
df = dc.sql("queries/features.sql", status="active")
```

For DDL/DML operations or fetching single results without DataFrame conversion:

```python
# Execute DDL
dc.execute_sql("CREATE TABLE test (id INT, name VARCHAR(100))")

# Execute from a .sql file
dc.execute_sql("queries/create_table.sql")

# Fetch single result
result = dc.execute_sql("SELECT COUNT(*) FROM test", fetch_result=True)
count = result[0]

# Fetch multiple rows
rows = dc.execute_sql("SELECT * FROM test LIMIT 10", fetch_all=True)
for row in rows:
    print(row)
```

### Copying Tables

```python
dc.copy_table(
    source_table='external_schema.model_scores',
    destination_table='analytics.model_scores',
    drop_destination_table=True
)
```

### Unloading Data to S3

Export query results directly to S3 using Redshift's UNLOAD command:

```python
# Simple unload to S3
dc.unload_to_s3(
    query='SELECT * FROM analytics.customer_weekly_features',
    relative_path='exports/customers/',
    file_prefix='customers'
)

# Control file size
dc.unload_to_s3(
    query='SELECT * FROM analytics.large_table',
    relative_path='exports/large_table/',
    file_prefix='large',
    max_file_size='500 MB'
)

# Clean output directory before unload
dc.unload_to_s3(
    query='SELECT * FROM analytics.daily_report',
    relative_path='exports/daily/',
    file_prefix='daily_report',
    drop_existing_files=True
)
```

> **Note**: When using `parallel=True` (default), Redshift creates one file per cluster slice (typically 2–32 files). Use `parallel=False` for single-file output, especially with partitioning.
>
> **Tip**: `drop_existing_files=True` removes all files matching the prefix before the UNLOAD, ensuring a clean output directory for scheduled exports.

### Loading Data from S3 to Redshift

Load data from S3 into Redshift tables using the COPY command.

> **Important**: Some parameters are format-specific:
> - **CSV only**: `blank_as_null`, `empty_as_null`, `null_as`
> - **CSV and JSON only**: `date_format`, `time_format`, `accept_inv_chars`
> - **Parquet**: Does not support text-oriented options (NULL handling, delimiters, date/time format strings)

```python
# Simple Parquet load
dc.load_from_s3(
    table='my_table',
    schema='my_schema',
    relative_path='imports/my_table/'
)

# Drop and recreate table from Parquet
dc.load_from_s3(
    table='my_table',
    schema='my_schema',
    relative_path='imports/data/',
    drop_existing_table=True
)

# Load specific columns
dc.load_from_s3(
    table='my_table',
    schema='my_schema',
    relative_path='imports/data/',
    column_list=['id', 'name', 'created_at']
)
```

### Creating Redshift Tables from DataFrames

Create Redshift tables directly from pandas or polars DataFrames:

```python
import pandas as pd
from ml_analytics import DataConnector

dc = DataConnector()
df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

# Auto-creates if not exists, appends if exists
dc.create_table_from_dataframe(df, table='my_table', schema='my_schema')

# Drop and recreate
dc.create_table_from_dataframe(df, table='my_table', schema='my_schema', drop_existing_table=True)

# Truncate then load (keeps table structure)
dc.create_table_from_dataframe(df, table='my_table', schema='my_schema', truncate_before_load=True)

# Keep S3 staging files after loading
dc.create_table_from_dataframe(
    df,
    table='my_table',
    schema='my_schema',
    s3_path='staging/loads/',
    delete_s3_files_after=False
)
```

Temporary S3 staging files are automatically cleaned up after loading.

#### Auto-incrementing Identity Columns

```python
# Add default 'id BIGINT IDENTITY(1,1)' as first column
dc.create_table_from_dataframe(
    df,
    table='users',
    schema='public',
    drop_existing_table=True,
    identity_column=True
)

# Custom identity column name
dc.load_from_s3(
    table='products',
    schema='public',
    relative_path='data/products/',
    drop_existing_table=True,
    identity_column='product_id'
)
```

Identity columns are auto-detected when appending data and excluded from COPY commands.

### Creating and Syncing Spectrum Tables

Create and sync Redshift Spectrum tables. This method creates a `DataConnector` with `s3_root = ''`, so `relative_path` is relative to the bucket root.

```python
# Create a table without partitions (syncs data by default)
dc.create_spectrum_table(
    table='test',
    schema='external_schema',
    s3_bucket='my-analytics-bucket',
    relative_path='projects/model-scores/test/',
    force_table_creation=True,
    sync_partitions_on_creation=True,
)
```

Create tables with partitions. Avoid column names that are SQL reserved words (e.g., `group`). Assumes data was saved as `s3.save_dataframe(df, directory='test/start_date=2026-04-01/test=6', file_name='test')`:

```python
partitions = [('start_date', 'DATE'), ('test', 'INT')]
dc.create_spectrum_table(
    table='test',
    schema='external_schema',
    s3_bucket='my-analytics-bucket',
    relative_path='projects/model-scores/test/',
    partitions=partitions,
    force_table_creation=True,
    sync_partitions_on_creation=True,
)
```

Sync a specific set of partitions:

```python
dc.sync_spectrum_data(
    table='test',
    schema='external_schema',
    s3_bucket='my-analytics-bucket',
    relative_path='projects/model-scores/test/',
    partition_values={'start_date': '2026-04-01', 'test': 6}
)
```

Sync all new partitions automatically:

```python
dc.sync_spectrum_partitions(
    table='test',
    schema='external_schema',
    s3_bucket='my-analytics-bucket',
    relative_path='projects/model-scores/test/',
    partitions_columns=['start_date', 'test']
)
```

---

## S3 Connector

### Initialization

```python
from ml_analytics import S3Connector

s3 = S3Connector(bucket="my-analytics-bucket", s3_root='projects/testing')
```

### Saving Data to S3

```python
# Save as Parquet (default)
s3.save_dataframe(df, directory='testing', file_name='test', file_format='parquet')

# Read back (returns pandas DataFrame by default; use to_polars=True for Polars)
df = s3.read_parquet('testing/test.parquet')
```

### Listing and Deleting S3 Files

```python
s3.list_files('testing')
s3.delete_file('testing/test.parquet')
```

### Querying Parquet Files with DuckDB

Run SQL queries directly on Parquet files in S3 without loading entire datasets into memory. Combine S3 data with local DataFrames in a single query.

#### Basic S3 Queries

```python
# Query a single file
df = s3.query(
    "SELECT * FROM read_parquet('s3://my-bucket/data/file.parquet') WHERE date > '2024-01-01'"
)

# Query multiple files with wildcard
df = s3.query(
    "SELECT COUNT(*) as total, AVG(amount) as avg_amount FROM read_parquet('s3://my-bucket/sales/*.parquet')"
)

# Use the get_path() helper to construct paths
path = s3.get_path('data/events/')
df = s3.query(f"SELECT * FROM read_parquet('{path}*.parquet') LIMIT 100")

# Window functions
df = s3.query('''
    SELECT
        customer_id,
        date,
        amount,
        SUM(amount) OVER (PARTITION BY customer_id ORDER BY date) as running_total
    FROM read_parquet('s3://bucket/transactions/*.parquet')
    WHERE date BETWEEN '2024-01-01' AND '2024-12-31'
''')

# Return as Polars DataFrame
df = s3.query(
    "SELECT * FROM read_parquet('s3://bucket/data/*.parquet')",
    to_polars=True
)
```

#### Querying with Local DataFrames

Pass local DataFrames (pandas or polars) directly into your queries:

```python
import pandas as pd

product_filter = pd.DataFrame({
    'product_id': [100, 200, 300],
    'category': ['Electronics', 'Clothing', 'Books']
})

df = s3.query(
    '''
    SELECT s.*, f.category
    FROM read_parquet('s3://bucket/sales/*.parquet') s
    JOIN product_filter f ON s.product_id = f.product_id
    WHERE s.date >= '2024-01-01'
    ''',
    local_dataframes={'product_filter': product_filter}
)
```

#### Persistent Connection and Registered DataFrames

For better performance across multiple queries, use the persistent DuckDB connection:

```python
# Register DataFrames once and reuse across queries
customers = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'segment': ['Premium', 'Standard', 'Premium']})
regions = pd.DataFrame({'id': [1, 2, 3], 'region': ['North', 'South', 'West']})

s3.register_dataframe('customers', customers)
s3.register_dataframe('regions', regions)

result1 = s3.query("SELECT * FROM customers WHERE segment = 'Premium'")
result2 = s3.query("SELECT c.*, r.region FROM customers c JOIN regions r ON c.id = r.id")

# Combine registered DataFrames with S3 data
result3 = s3.query(
    "SELECT s.*, c.name FROM read_parquet('s3://bucket/sales/*.parquet') s JOIN customers c ON s.customer_id = c.id"
)

# Close persistent connection when done (optional — auto-cleaned up on destruction)
s3.close_duckdb_connection()
```

For advanced use, access the DuckDB connection directly:

```python
conn = s3.get_duckdb_connection()
conn.execute("CREATE TABLE temp_summary AS SELECT segment, COUNT(*) as count FROM customers GROUP BY segment")
result = conn.execute("SELECT * FROM temp_summary").fetchdf()
s3.close_duckdb_connection()
```

**Benefits of DuckDB querying:**
- Memory efficient — processes data without loading entire files
- Fast columnar processing on Parquet format
- Full SQL: joins, aggregations, window functions
- Query across multiple Parquet files seamlessly
- Combine S3 data with local DataFrames in a single query
- Works with both pandas and polars

---

## Google Sheets Connector

### Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create or select a project.
2. Enable the **Google Sheets API** and **Google Drive API**.
3. Create a service account and download the JSON credentials file.
4. Share your spreadsheet with the service account email.

For detailed setup instructions, see the [Google Sheets Connector Usage Guide](docs/GSHEET_CONNECTOR_USAGE.md).

```python
from ml_analytics import GSheet

gsheet = GSheet()
```

### Reading from Google Sheets

```python
# Read entire sheet as DataFrame
df = gsheet.read_sheet(spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")

# Read specific range
df = gsheet.read_sheet(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    range_name="Sheet1!A1:D10"
)

# Read by sheet name
df = gsheet.read_sheet(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    sheet_name="DataSheet"
)
```

### Writing to Google Sheets

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [30, 25, 35],
    'City': ['New York', 'Paris', 'London']
})

gsheet.write_sheet(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    data=df,
    sheet_name="Results"
)

gsheet.append_sheet(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    data=df
)
```

### Transferring Data to S3

```python
from ml_analytics import S3Connector

s3 = S3Connector(bucket="my-data-bucket", s3_root="data/exports")

gsheet.gsheet_to_s3(
    spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    s3_connector=s3,
    file_name="sheet_data",
    file_format="parquet"
)
```

---

## Slack Connector

Send messages, reactions, and upload files to Slack channels by name or ID.

See the [Slack Connector Usage Guide](docs/SLACK_CONNECTOR_USAGE.md) for setup and examples.

---

## Model Manager

Manage ML models via MLflow: register, log, version, grant access, and delete.

### Initialization

When registering a new model, specify all project tags:

```python
from ml_analytics import ModelManager

model_manager = ModelManager(
    model_name="model-manager-testing",
    task="classification",
    project="test",
    description="test",
    team="test",
    user="user@example.com",
    create_registered_model=True,
    start_initial_run=True,
    run_name='test_run'
)
```

To create experiments without registering a model, set `create_registered_model=False`. This creates only an experiment bucket.

If the registered model already exists, only `model_name` and `user` are required:

```python
model_manager = ModelManager(
    model_name="model-manager-testing",
    user="user@example.com"
)
```

### Logging a Model

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

input_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
predictions = np.array([0, 1, 0])

model = RandomForestClassifier()
model.fit(input_data, predictions)

model_manager.log_model(
    model,
    input_data=input_data,
    predictions=predictions,
    register_model=True,
    flavor="sklearn",
    description="A RandomForest model for classification",
    tags={'status': 'testing'}
)
```

Log metrics, params, and artifacts:

```python
model_manager.log_metric('precision', 0.85)
model_manager.log_param('n_estimators', 100)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot([0, 1, 2], [0, 1, 0])
plt.savefig('example.png')
model_manager.log_artifact('example.png', artifact_path='models')

model_manager.end_run()
```

### Managing Access

Assign `READ`, `EDIT`, `MANAGE`, or `NO_PERMISSIONS` to a user:

```python
model_manager.grant_experiment_permission(username="user@example.com", permission="EDIT")
model_manager.grant_registered_model_permission(username="user@example.com", permission="EDIT")
```

### Managing Model Versions and Loading Models

```python
# Set an alias for a specific version
model_manager.set_model_alias(alias="test", version=5)

# Load the latest model version
my_model = model_manager.load_latest_model()
my_model.predict(input_data)

# Load by alias
my_model = model_manager.load_model(alias="test")

# Load by version
my_model = model_manager.load_model(version=5)

# Load by model URI
model_uri = model_manager.get_model_uri(version=4)
my_model = model_manager.load_model(model_uri=model_uri)

# Load from experiment run
my_model = model_manager.load_model_from_experiment(run_name='bustling-shrike-818')

# Register a model from an experiment run
model_manager.register_model(
    run_name='dazzling-owl-923',
    tags={"status": "testing"},
    description="test model"
)

# Delete by version or alias
model_manager.delete_model(version=4)
model_manager.delete_model(alias="test")
```

---

## Model Tools

The `ml_analytics.model_tools` module provides utility functions for the ML model development lifecycle.

### CatBoost Data Preparation

- **`prepare_catboost_data(df, cat_features, feature_list=None)`**: Prepares a DataFrame for CatBoost by converting categorical columns to `str` (avoiding `None`/`NaN` errors). Returns a cleaned copy.
- **`make_catboost_pool(df, feature_list, cat_features, label=None, **pool_kwargs)`**: Builds a CatBoost `Pool` with automatic preprocessing via `prepare_catboost_data`.

### Feature Engineering

- **`get_features(df, target_col=None)`**: Extracts categorical and numerical feature names from a DataFrame. Pass `target_col` to exclude it from the feature lists.

### Classification Metrics

- **`get_balanced_accuracy(y_true, y_pred)`**: Balanced accuracy — average recall across classes. Useful for imbalanced datasets.
- **`pr_auc_score(y_true, y_pred_proba)`**: Area Under the Precision-Recall Curve (PR AUC).
- **`brier_score(y_true, y_pred_proba)`**: Brier score for probabilistic predictions. Lower is better (0 = perfect, 0.25 = worst).
- **`expected_calibration_error(y_true, y_pred_proba, n_bins=10)`**: Expected Calibration Error (ECE) — measures how well predicted probabilities match actual outcomes across confidence bins.
- **`mcc_score(y_true, y_pred)`**: Matthews Correlation Coefficient for binary classification.
- **`get_metrics(y_obs, y_pred, y_prob, prefix=None)`**: Dictionary of common classification metrics (balanced accuracy, MCC, precision, recall, F1, PR AUC, ROC AUC, Brier score, ECE).

### Performance Evaluation

- **`get_performance(data, target_col, pred_col, prob_col, grouping_cols=None, test_size_minimum_per_group=50)`**: DataFrame with detailed metrics for data subgroups: target rate, score stats, rank correlation, TP/TN/FP/FN percentages, decile analysis.

### Survival Analysis Metrics

- **`survival_mae(observed_time, event_indicator, predicted_time)`**: MAE for survival data, accounting for censored observations.
- **`get_metrics_surv(observed_time, event_indicator, predicted_time, prefix=None)`**: Dictionary with MAE and Concordance Index (C-Index).
- **`get_performance_surv(data, time_col, event_col, pred_col, grouping_cols=None, test_size_minimum_per_group=50)`**: DataFrame with survival metrics per subgroup: observed/predicted times, Kaplan-Meier median survival, event rate, MAE, C-Index.

### Data Splitting

- **`time_split(data, date_column, test_ratio=0.3, max_date=None)`**: Splits data into train/test sets using a quantile split on sorted dates. `test_ratio` controls the test set proportion (default 0.3). `max_date` filters data before splitting.

### Feature Selection

- **`catboost_feature_selection(...)`**: Feature selection using CatBoost's built-in algorithms (e.g., `RecursiveByShapValues`).
  - Key parameters: `model`, `algorithm`, `train_df`, `feature_list`, `target_column`, `num_features_to_select`, `force_to_include`.

### Visualization

- **`plot_score_bins(df, prob_col, target_col, bins=10, ...)`**: Bar plot of average target rate per score bin. Helps visualize model calibration and lift.
- **`shap_plot(pipeline, data, features, output_path="shap_summary.png", ...)`**: Generates and saves a SHAP summary plot for a scikit-learn pipeline.
- **`plot_trend(data, x_col, y_col, hue_col, ...)`**: Line plot for time series data grouped by a hue column. Supports vertical reference lines (`vline_date`), grid, custom labels, and saving to file.

---

## Utilities

The `ml_analytics.utils` module contains general-purpose helpers used across the library.

- **`get_logger(name)`**: Returns a `logging.Logger` with the given name, ensuring handlers are not duplicated across calls.
- **`log_and_raise_error(logger, message, exception_type=ValueError)`**: Logs an error and raises the specified exception type.
- **`get_credential_value(name, scope="ml")`**: Retrieves a credential value by name from the configured secrets backend.
- **`find_project_root(marker_files=None)`**: Searches upward from the script's directory for marker files (e.g., `.git`, `pyproject.toml`) to locate the project root. Result is cached.
- **`load_sql_query(query_path, **kwargs)`**: Loads a SQL file and supports `.format(**kwargs)` for dynamic query templating.
- **`get_sql_files(relative_folder, pipeline=None)`**: Returns an ordered `{name: Path}` dict of `.sql` files in the given folder. See [SQL Pipeline YAML](#sql-pipeline-yaml) below for how YAML files control ordering and selection.
- **`execute_sql_scripts(query_paths, pipeline=None, **kwargs)`**: Loads and executes SQL statements from multiple files sequentially via `DataConnector`. If the last statement of the last file is a `SELECT`/`WITH` query it is returned as a DataFrame. Accepts:
  - `str` — relative path from project root to a folder (discovers files via `get_sql_files()`) or a single `.sql` file.
  - `Path` to a directory — same as a folder `str`, resolved relative to project root.
  - `Path` to a single `.sql` file — executes that file only.
  - `list[str | Path]` — ordered list of individual SQL file paths; executed in list order.
  - `dict[str, str | Path]` — explicit ordered mapping of name → path; executed in insertion order.

---

### SQL Pipeline YAML

A YAML file in the SQL folder controls which files are included and their execution order. The file can have **any name** — no fixed naming convention is required.

#### Single pipeline (one YAML file)

If the folder contains exactly one `.yaml` file, it is picked up automatically:

```yaml
# queries/etl_config.yaml  (any filename works)
steps:
  - scope_leads
  - contact_dimensions
  - final_assembly
```

```python
# Runs scope_leads → contact_dimensions → final_assembly
execute_sql_scripts("queries")
get_sql_files("queries")
```

Without a YAML file, SQL files are ordered alphabetically.

#### Multiple pipelines — separate files (layout A)

Name each YAML file after the pipeline and select it with `pipeline=`:

```
queries/
  daily.yaml
  weekly.yaml
  scope_leads.sql
  contact_dimensions.sql
  weekly_report.sql
```

```yaml
# queries/daily.yaml
steps:
  - scope_leads
  - contact_dimensions
```

```yaml
# queries/weekly.yaml
steps:
  - scope_leads
  - weekly_report
```

```python
execute_sql_scripts("queries", pipeline="daily")
execute_sql_scripts("queries", pipeline="weekly")
```

#### Multiple pipelines — named sections (layout B)

Define all pipelines as named sections inside a single YAML file:

```yaml
# queries/pipelines.yaml  (any filename works)
pipelines:
  daily:
    steps:
      - scope_leads
      - contact_dimensions
  weekly:
    steps:
      - scope_leads
      - weekly_report
```

```python
execute_sql_scripts("queries", pipeline="daily")
execute_sql_scripts("queries", pipeline="weekly")
```

#### SQL files in subdirectories

Step names can include subdirectory paths relative to the folder:

```yaml
steps:
  - staging/load_users       # → queries/staging/load_users.sql
  - transforms/agg_events    # → queries/transforms/agg_events.sql
  - final/report             # → queries/final/report.sql
```

#### Resolution rules summary

| Situation | Behaviour |
|---|---|
| No YAML in folder | Alphabetical order |
| One YAML, no `pipeline=` | Auto-discovered; uses its `steps` list |
| Multiple YAMLs, no `pipeline=` | Warns and falls back to alphabetical |
| `pipeline="name"`, `name.yaml` exists | Uses `name.yaml` → `steps` (layout A) |
| `pipeline="name"`, no `name.yaml` | Searches all `*.yaml` for `pipelines.name.steps` (layout B) |
| `pipeline="name"`, not found anywhere | Warns and falls back to alphabetical |

---

## Additional Guides

| Guide | Description |
|---|---|
| [AWS Authentication Guide](docs/AWS_AUTHENTICATION.md) | AWS SSO setup |
| [CLI Commands Reference](docs/CLI_COMMANDS.md) | Available CLI commands and usage |
| [Google Sheets Connector Usage Guide](docs/GSHEET_CONNECTOR_USAGE.md) | Google Sheets setup and examples |
| [Slack Connector Usage Guide](docs/SLACK_CONNECTOR_USAGE.md) | Slack setup and examples |
| [Tunnel Manager](docs/TUNNEL_MANAGER.md) | VS Code tunnel helper for remote development |

---

## Contributing

Contributions are welcome. Please follow the standard fork-and-pull-request workflow.
