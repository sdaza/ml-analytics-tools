# Snowflake Spark Connector

`SFConnector` reads from and writes to Snowflake through Spark
(`spark.read.format("snowflake")`), which is the right tool on Databricks. It is
distinct from `DataConnector`, which talks to Snowflake/Redshift through the
pure-Python `snowflake-connector-python` / `redshift-connector` drivers.

PySpark is **not** a dependency of this package. `SFConnector` imports it lazily,
only when a method that needs a Spark session runs (`sql`, `save_table`). You
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
4. **Authenticator** — e.g. `SNOWFLAKE_AUTHENTICATOR=externalbrowser` (interactive;
   not suitable for Spark jobs — the connector logs a warning).

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
`DATABRICKS_SECRET_SCOPE` → `user-<SNOWFLAKE_USER>` when the user is an email.

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

# Write a Spark DataFrame back to Snowflake
sf.save_table(df, "cds.my_table", mode="overwrite")  # or mode="append"
```

On Databricks the active Spark session is detected and used automatically — you
do not need to create or pass one.

### Extra Snowflake options

Pass any additional Snowflake Spark options through `extra_options`; they
override the resolved defaults and apply to every read and write:

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
| `authenticator` | Snowflake authenticator (e.g. `externalbrowser`). |
| `private_key`, `private_key_path`, `private_key_passphrase` | Key-pair material (PEM string, file path, passphrase). |
| `secret_scope` | Databricks secret scope. Inferred from the user email when omitted. |
| `source_format` | Spark data source name. Defaults to `net.snowflake.spark.snowflake`; pass `"snowflake"` for the Databricks short alias. |
| `extra_options` | Extra Snowflake Spark options merged into every read/write. |
| `spark` | Optional `SparkSession`. Rarely needed — the active session is detected automatically on Databricks. |
