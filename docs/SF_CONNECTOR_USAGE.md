# Snowflake Spark Connector

`SFConnector` reads from Snowflake through Spark
(`spark.read.format("snowflake")`) and can persist results into Databricks Unity
Catalog tables, which is the right tool on Databricks. It is distinct from
`DataConnector`, which talks to Snowflake/Redshift through the pure-Python
`snowflake-connector-python` / `redshift-connector` drivers.

PySpark is **not** a dependency of this package. `SFConnector` imports it lazily,
only when a method that needs a Spark session runs (`sql`, `save_to_uc`). You
can import and construct the connector anywhere; running a
query off a Spark runtime raises a clear `ImportError`.

## When To Use Which Connector

| | `SFConnector` | `DataConnector(engine="snowflake")` |
| --- | --- | --- |
| Backend | Spark (`net.snowflake.spark.snowflake`) | `snowflake-connector-python` |
| Returns | Spark DataFrame (or pandas) | pandas / polars |
| Best for | Databricks / Spark jobs, large reads & writes | Local work, scripts, notebooks |
| Needs PySpark | Yes (lazy import) | No |

## Setup

`SFConnector` reuses the same `SNOWFLAKE_*` settings as `DataConnector`. Each
field is resolved in this order:

1. Explicit constructor argument (e.g. `SFConnector(account=...)`)
2. `SNOWFLAKE_*` environment variable
3. Databricks secret (when a secret scope is set or inferred from the user email)

### Environment variables

```bash
SNOWFLAKE_ACCOUNT=dr06406.eu-west-1          # identifier or full *.snowflakecomputing.com URL
SNOWFLAKE_USER=your.name@example.com
SNOWFLAKE_DATABASE=DEEP_PURPLE
SNOWFLAKE_SCHEMA=CDS
SNOWFLAKE_WAREHOUSE=ANALYTICS_S_WH
SNOWFLAKE_ROLE=SNOWFLAKE_STND_DATA
```

### Authentication

The connector picks an auth method based on what it can resolve, in this order:

1. **Key-pair (`SNOWFLAKE_JWT`)** — when a private key is available. Recommended
   for Databricks and any non-interactive job.
2. **OAuth token** — when `SNOWFLAKE_TOKEN` is set.
3. **Password** — when `SNOWFLAKE_PASSWORD` is set.
4. **Authenticator** — a non-interactive authenticator. `externalbrowser` is
   **not supported** here: it is interactive and Spark jobs block on the browser
   SSO handshake, so `SFConnector` raises a clear error instead of hanging. Use
   key-pair / OAuth for Spark, or `DataConnector` for interactive local queries.

#### Key-pair on Databricks (recommended)

Store the private key and its passphrase as personal-scope secrets. The default
scope is inferred from your email as `user-<email>`:

```bash
databricks secrets put-secret user-your.name@example.com snowflake_key --bytes-value "$(cat rsa_key.p8)"
databricks secrets put-secret user-your.name@example.com snowflake_key_pass --string-value "<passphrase>"
```

The connector reads the `snowflake_key` / `snowflake_key_pass` secrets
automatically. **You normally do not pass `secret_scope`** — when
`SNOWFLAKE_USER` is set to your email, the scope is inferred as
`user-<email>`:

```python
from ml_analytics import SFConnector

sf = SFConnector()  # scope inferred from SNOWFLAKE_USER
df = sf.sql("SELECT * FROM cds.dim_tutor LIMIT 1000")
```

Pass `secret_scope=...` (or set `SNOWFLAKE_SECRET_SCOPE`) only when the scope is
not your `user-<email>` — for example a shared/team scope:

```python
sf = SFConnector(secret_scope="team-analytics")
```

The scope is resolved in this order: `secret_scope` argument →
`SNOWFLAKE_SECRET_SCOPE` → `ML_ANALYTICS_SNOWFLAKE_SECRET_SCOPE` →
`DATABRICKS_SECRET_SCOPE` → `user-<SNOWFLAKE_USER>` when the user is an email →
`user-<current Databricks user email>` (auto-detected from the notebook context).

The last fallback means that on Databricks you can store **everything**
(account, user, connection settings, key) as personal-scope secrets and call
`SFConnector()` with no arguments and no env vars — the scope is inferred from
whoever is running the notebook:

```python
from ml_analytics import SFConnector

sf = SFConnector()  # scope = user-<your Databricks email>, all values from secrets
df = sf.sql("SELECT * FROM cds.dim_tutor LIMIT 1000")
```

This requires the `account` and `user` values to be resolvable from secrets too,
e.g. `SNOWFLAKE_ACCOUNT` and `snowflake_user` stored in your scope.

#### Key-pair from a file or PEM string

```python
sf = SFConnector(
    account="dr06406.eu-west-1",
    user="your.name@example.com",
    private_key_path="~/.snowflake/rsa_key.p8",
    private_key_passphrase="<passphrase>",
    database="DEEP_PURPLE",
    schema="CDS",
    warehouse="ANALYTICS_S_WH",
    role="SNOWFLAKE_STND_DATA",
)
```

## Usage

```python
from ml_analytics import SFConnector

sf = SFConnector()  # secret scope inferred from SNOWFLAKE_USER

# Spark DataFrame
df = sf.sql("SELECT * FROM cds.dim_tutor LIMIT 1000")
df.display()

# pandas DataFrame
pdf = sf.sql("SELECT 1 AS col_1", return_pandas=True)
```

### Running a query from a `.sql` file

`sql()` accepts either inline SQL or a path to a `.sql` file (relative to the
project root); a path ending in `.sql` is loaded automatically. Pass keyword
arguments to substitute `{placeholders}` in the file via `str.format()`:

```python
df = sf.sql("queries/dim_tutor.sql")                 # load + run a file
df = sf.sql("queries/experiment.sql", days=14)       # with {days} templating
```

### Saving results to Unity Catalog

Persist a result into a Databricks Unity Catalog table with Spark's native
`saveAsTable`. Do it inline while pulling the data with `save_table=True`, or
call `save_to_uc()` on any Spark DataFrame:

```python
# inline: pull from Snowflake and write to analytics.cr_subs in one call
sf.sql(
    "queries/cr_to_subscribers.sql",
    save_table=True,
    schema="analytics",
    table="cr_subs",
    mode="overwrite",          # overwrite | append | ignore | error
)

# explicit save of an existing Spark DataFrame
sf.save_to_uc(df, table="cr_subs", schema="analytics", catalog="prod")

# a fully-qualified table name passes through untouched
sf.save_to_uc(df, table="prod.analytics.cr_subs")
```

`save_to_uc` writes a managed Unity Catalog table — it does **not** write back to
Snowflake.

On Databricks the active Spark session is detected and used automatically — you
do not need to create or pass one.

### Saving YAML-ordered queries to Unity Catalog

Use `save_pipeline_to_uc()` when you have a folder of Snowflake SQL files and
want each query result saved as a Spark / Unity Catalog table. The YAML controls
execution order. By default, each destination table uses the SQL file stem.

```python
from ml_analytics import SFConnector

sf = SFConnector()

df = sf.save_pipeline_to_uc(
    "queries/churn_pipeline",
    pipeline="daily",
    catalog="prod",
    schema="analytics",
    mode="overwrite",
    run_date="2026-06-17",
)
```

Example folder:

```text
queries/churn_pipeline/
  daily.yaml
  base.sql
  features.sql
  final.sql
```

```yaml
# daily.yaml
steps:
  - base
  - features
  - final
```

This creates:

```text
prod.analytics.base
prod.analytics.features
prod.analytics.final
```

Pass `tables` when you want friendlier destination names for specific steps:

```python
df = sf.save_pipeline_to_uc(
    "queries/churn_pipeline",
    pipeline="daily",
    catalog="prod",
    schema="analytics",
    tables={
        "base": "churn_base",
        "features": "churn_features",
        "final": "churn_daily",
    },
)
```

Each SQL file is read from Snowflake through Spark and saved to Unity Catalog
with `saveAsTable`. The returned value is the final Spark DataFrame; pass
`return_all=True` to get a dict of every step's DataFrame.

Tables are written as Delta with schema merge enabled by default:

```python
df.write.format("delta").option("mergeSchema", "true").mode(mode).saveAsTable(...)
```

Tables are optimized after each save by default:

```sql
OPTIMIZE prod.analytics.features
```

Pass `zorder_by="column_name"` (or a list of columns) to add `ZORDER BY`.
For per-step ZORDER columns, pass a dict:

```python
sf.save_pipeline_to_uc(
    "queries/churn_pipeline",
    pipeline="daily",
    catalog="prod",
    schema="analytics",
    zorder_by={
        "features": ["customer_id", "event_date"],
        "final": "customer_id",
    },
)
```

Pass `optimize=False` only when you explicitly want to skip Databricks Delta
optimization.

You can also set comments while saving:

```python
sf.save_pipeline_to_uc(
    "queries/churn_pipeline",
    pipeline="daily",
    catalog="prod",
    schema="analytics",
    comments={"final": "Individual tutor-level training metrics"},
)
```

### Extra Snowflake options

Pass any additional Snowflake Spark options through `extra_options`; they
override the resolved defaults and apply to every Snowflake read:

```python
sf = SFConnector(extra_options={"sfTimezone": "UTC"})
```

### Inspecting the resolved options

`spark_options()` returns the exact dict passed to Spark (useful for debugging).
By default the private key is included; pass `include_private_key=False` to omit
the key material:

```python
sf.spark_options(include_private_key=False)
```

## Constructor reference

| Argument | Description |
| --- | --- |
| `account` | Account identifier or full URL. Falls back to `SNOWFLAKE_ACCOUNT` / `SNOWFLAKE_URL`. |
| `user` | Snowflake user. Falls back to `SNOWFLAKE_USER`. |
| `database`, `schema`, `warehouse`, `role` | Standard connection settings; fall back to matching `SNOWFLAKE_*`. |
| `password` | Password authentication (ignored when a key or token is present). |
| `token` | OAuth token. Falls back to `SNOWFLAKE_TOKEN`. |
| `authenticator` | Snowflake authenticator (e.g. `oauth`). Interactive `externalbrowser` is rejected — it blocks Spark jobs. |
| `private_key`, `private_key_path`, `private_key_passphrase` | Key-pair material (PEM string, file path, passphrase). |
| `secret_scope` | Databricks secret scope. Inferred from the user email when omitted. |
| `source_format` | Spark data source name. Defaults to `net.snowflake.spark.snowflake`; pass `"snowflake"` for the Databricks short alias. |
| `extra_options` | Extra Snowflake Spark options merged into every Snowflake read. |
| `spark` | Optional `SparkSession`. Rarely needed — the active session is detected automatically on Databricks. |
