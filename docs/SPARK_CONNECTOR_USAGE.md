# Spark Table Manager

`SparkTableManager` provides **source-agnostic** Spark / Databricks Unity Catalog
operations: write a DataFrame to a managed Delta table, run `OPTIMIZE`, set a
table comment, drop or read a table, run arbitrary Spark SQL, and convert
pandas/polars frames to Spark.

It performs no Snowflake/Redshift reads of its own — pair it with
[`DataConnector`](../README.md#configuration) or
[`SFConnector`](SF_CONNECTOR_USAGE.md) to produce the DataFrame, or hand it a
plain pandas/polars frame you already have in memory.

`SFConnector` delegates its Unity Catalog methods (`save_to_uc`,
`optimize_uc_table`, `set_uc_table_comment`) to a `SparkTableManager` internally,
so the same logic backs both classes.

PySpark is **not** a dependency of this package. `SparkTableManager` imports it
lazily — only when a method that needs a Spark session actually runs. You can
import and construct it anywhere; running an operation off a Spark runtime raises
a clear `ImportError`.

## When To Use It

| Use `SparkTableManager` when you want to… | Method |
| --- | --- |
| Save a Spark / pandas / polars DataFrame to a Unity Catalog table | `save_to_uc` |
| Convert a pandas / polars DataFrame to Spark | `to_spark` |
| Read a Unity Catalog table as a Spark DataFrame | `read_table` |
| Run arbitrary Spark SQL (`SELECT`, DDL, `MERGE`, …) | `sql` |
| Run Delta `OPTIMIZE` (optionally `ZORDER BY`) | `optimize_uc_table` |
| Set a table comment | `set_uc_table_comment` |
| Drop a table | `drop_table` |

If you only need to read from Snowflake and persist the result, you can keep
using `SFConnector` directly — it exposes `save_to_uc` and friends already.

## The Spark Session

`SparkTableManager` resolves its Spark session through the shared `get_spark()`
helper, which:

1. reuses an active `SparkSession` if one exists (the normal case inside a
   Databricks notebook/cluster);
2. otherwise creates a Databricks Connect session (local dev against a remote
   cluster/serverless using your Databricks config);
3. otherwise falls back to a plain local `SparkSession`.

So on Databricks you don't need to create or pass a session. Locally, install a
runtime with `pip install databricks-connect`. You can also pass an explicit
session: `SparkTableManager(spark=spark)`.

## Quick Start

```python
from ml_analytics import SparkTableManager

# default catalog/schema qualify unqualified table names
tm = SparkTableManager(catalog="prod", schema="analytics")

# run Spark SQL and get a Spark DataFrame (or pandas with return_pandas=True)
df = tm.sql("SELECT * FROM prod.analytics.lessons WHERE country = 'US'")

# save it to prod.analytics.lessons_us, optimized, with a comment
tm.save_to_uc(df, table="lessons_us", comment="US lessons")

# read it back as a Spark DataFrame
again = tm.read_table("lessons_us")

# drop it
tm.drop_table("lessons_us")
```

A fully-qualified (dotted) table name overrides the manager defaults:

```python
tm.save_to_uc(df, table="prod.analytics.lessons_us")
```

## Saving a DataFrame

`save_to_uc` writes a **managed** Unity Catalog Delta table with Spark's native
`saveAsTable`, and (by default) runs `OPTIMIZE` afterwards. It returns a
DataFrame backed by the just-written table, so downstream work scans fast Delta
storage instead of re-running the original read.

```python
tm.save_to_uc(
    df,
    table="lessons_us",
    schema="analytics",        # optional; overrides the manager default
    catalog="prod",            # optional; overrides the manager default
    mode="overwrite",          # overwrite | append | ignore | error
    optimize=True,             # run OPTIMIZE after the write
    zorder_by=["country", "lesson_date"],
    comment="US lessons",
    drop_existing=True,        # DROP TABLE IF EXISTS before writing
    overwrite_schema=True,     # replace the schema on overwrite (drop removed columns)
)
```

Notes:

- `drop_existing=True` (the default) recreates the table from the DataFrame, so
  removed columns disappear and table properties/grants/history reset. Set it to
  `False` to preserve the table — **required for `mode="append"`**, which would
  otherwise drop the table on every call.
- `overwrite_schema` and `merge_schema` are mutually exclusive in Delta. On an
  overwrite, replacing the schema takes precedence; `merge_schema` is used
  otherwise (e.g. evolving on append).

## pandas and polars DataFrames

You don't need a Spark DataFrame to start. `save_to_uc` accepts pandas and polars
frames and converts them automatically; you can also convert explicitly with
`to_spark`:

```python
import pandas as pd

pdf = pd.DataFrame({"user_id": [1, 2], "country": ["US", "MX"]})

# convert to Spark
sdf = tm.to_spark(pdf)

# or save the pandas frame straight to a table
tm.save_to_uc(pdf, table="users")
```

A frame that is already a Spark DataFrame passes through `to_spark` untouched —
no session is spun up for the passthrough case. polars frames are converted via
`.to_pandas()` first.

### Explicit Spark schema

Spark infers types when converting from pandas, which can be lossy for edge cases
(all-null columns, mixed object dtypes, timezone-aware datetimes). Pass an
explicit schema — a `pyspark.sql.types.StructType` or a DDL string — to avoid
that:

```python
# on to_spark
sdf = tm.to_spark(pdf, schema="user_id long, country string")

# on save_to_uc (named spark_schema, to not clash with the Unity Catalog `schema`)
tm.save_to_uc(pdf, table="users", spark_schema="user_id long, country string")
```

## Maintenance Helpers

```python
# run OPTIMIZE (optionally ZORDER BY) on an existing table
tm.optimize_uc_table("lessons_us", zorder_by="country")

# set a table comment
tm.set_uc_table_comment("lessons_us", "US lessons, refreshed daily")

# drop a table
tm.drop_table("lessons_us")
```

Each accepts `schema` / `catalog` overrides and an optional `spark` session,
falling back to the manager defaults.

## Relationship to `SFConnector`

`SFConnector` reads from Snowflake through Spark and **delegates** its Unity
Catalog writes to a `SparkTableManager`. These are equivalent:

```python
# via SFConnector (reads from Snowflake, then persists)
sf.sql("queries/experiment.sql", save_table=True, schema="analytics", table="exp")

# the standalone manager, given any DataFrame you already have
tm.save_to_uc(df, table="exp", schema="analytics")
```

Use `SFConnector` when the data comes from Snowflake; reach for
`SparkTableManager` when you already have a DataFrame (from any source) and just
need Spark/Unity Catalog table operations.
