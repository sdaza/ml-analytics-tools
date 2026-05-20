# Google Sheets Connector

`GSheet` reads and writes Google Sheets using a Google service account. It can
also share spreadsheets and move sheet data to S3.

## Setup

1. Enable the Google Sheets API and Google Drive API in your Google Cloud project.
2. Create a service account and download its JSON key.
3. Share each spreadsheet with the service account email.
4. Provide credentials with one of these options:

```bash
# Recommended for local work
GOOGLE_CREDENTIALS='{"type":"service_account", ...}'

# Optional default spreadsheet
GSHEET_SPREADSHEET_ID=your-spreadsheet-id
```

You can also pass `credentials_path="gsheet_credentials.json"` or
`credentials_json={...}` directly to `GSheet`.

The connector checks credentials in this order:

1. `GOOGLE_CREDENTIALS`
2. Mounted secret at `/mnt/<scope>/GOOGLE_CREDENTIALS`, default scope `ml`
3. Individual credential variables such as `GOOGLE_PROJECT_ID` and `GOOGLE_CLIENT_EMAIL`
4. `gsheet_credentials.json`
5. Auto-discovered service account JSON files in the project directories

## Basic Usage

```python
from ml_analytics import GSheet

gsheet = GSheet(credentials_path="gsheet_credentials.json")

df = gsheet.read_sheet(
    spreadsheet_id="your-spreadsheet-id",
    sheet_name="Input",
)

gsheet.write_sheet(
    df,
    spreadsheet_id="your-spreadsheet-id",
    sheet_name="Results",
    clear_before_write=True,
)
```

If `GSHEET_SPREADSHEET_ID` is set, or if you pass it at initialization, later
calls can omit `spreadsheet_id`:

```python
gsheet = GSheet(spreadsheet_id="your-spreadsheet-id")

df = gsheet.read_sheet(sheet_name="Input")
gsheet.write_sheet(df, sheet_name="Results")
```

## Ranges And Lists

```python
# Read a specific A1 range
df = gsheet.read_sheet(range_name="Input!A1:D100")

# Return raw values instead of a DataFrame
values = gsheet.read_sheet(sheet_name="Input", return_as="list")
```

## Create Or Share A Spreadsheet

When no `spreadsheet_id` is provided, `write_sheet` can create a spreadsheet:

```python
result = gsheet.write_sheet(
    df,
    spreadsheet_title="Weekly model report",
    sheet_name="Summary",
    share_with="teammate@example.com",
)
```

To share an existing spreadsheet:

```python
gsheet.share_spreadsheet(
    spreadsheet_id="your-spreadsheet-id",
    email_addresses=["teammate@example.com"],
    role="writer",
)
```

## Export To S3

```python
from ml_analytics import S3Connector

s3 = S3Connector(bucket="my-analytics-bucket", s3_root="reports")

gsheet.gsheet_to_s3(
    spreadsheet_id="your-spreadsheet-id",
    sheet_name="Input",
    s3_connector=s3,
    file_name="sheet_data",
    file_format="parquet",
)
```

## Notes

- Share spreadsheets with the service account email before reading or writing.
- Use `sheet_name` for whole-sheet reads, and `range_name` for A1 ranges.
- `write_sheet` cleans null-like values and datetime columns before sending data to Google.
- Keep service account JSON files out of git.
