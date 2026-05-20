# Google Sheets Connector Usage

Comprehensive guide for using the Google Sheets connector in the `ml_analytics` package.

## Table of Contents
- [Setup](#setup)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Sharing Spreadsheets](#sharing-spreadsheets)
- [Complete Examples](#complete-examples)
- [Troubleshooting](#troubleshooting)

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Get Google Service Account Credentials

> **⚠️ Important Note: If you don't have permissions to create a project or need credentials, please contact your team lead or project administrator** to obtain the service account credentials.

<details>
<summary><strong>For administrators only - Creating new credentials</strong></summary>

If you have the necessary permissions and billing account access:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Sheets API** and **Google Drive API**
4. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name and click "Create"
   - Click "Done"
5. Create a key:
   - Click on your service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Select "JSON" and click "Create"
   - Save the downloaded JSON file securely

</details>

### 3. Get Your Service Account Email

```python
from ml_analytics.gsheet_connector import GSheet

gs = GSheet()
print(gs.get_service_account_email())
# Output: your-service-account@your-project.iam.gserviceaccount.com
```

### 4. Share Your Spreadsheet

1. Open your Google Spreadsheet in [Google Sheets](https://sheets.google.com)
2. Click the **Share** button (top right)
3. Add your service account email
4. Set permission to **Editor**
5. Click **Send**

### 5. Get the Spreadsheet ID

From the URL: `https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID/edit`

Copy the `YOUR_SPREADSHEET_ID` part.

## Basic Usage

### Initialize the Connector

```python
from ml_analytics.gsheet_connector import GSheet

# Option 1: Auto-detect credentials (searches for .json file in current and parent directories)
gs = GSheet()

# Option 2: Specify credentials path
gs = GSheet(credentials_path="path/to/credentials.json")

# Option 3: Use environment variable GOOGLE_CREDENTIALS (JSON string)
# Set in .env file:
# GOOGLE_CREDENTIALS='{"type":"service_account","project_id":"...","private_key":"...",...}'
gs = GSheet()  # Automatically picks up GOOGLE_CREDENTIALS env var

# Option 4: Use credentials dictionary (programmatic approach)
import json
import os
creds = json.loads(os.environ['GOOGLE_CREDENTIALS'])
gs = GSheet(credentials_json=creds)
```

### Environment Variable Configuration

The connector supports two approaches for environment-based credentials:

**Option 1: Individual credential components (recommended for Vault/SecretProvider)**
```bash
# Add to .env file or set as environment variables
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_API_PKEY_ID=your-private-key-id
GOOGLE_API_PKEY=-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n
GOOGLE_CLIENT_EMAIL=your-sa@your-project.iam.gserviceaccount.com
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/...
```

**Option 2: Single JSON credential (alternative)**
```bash
# Add to .env file (must be valid JSON as a single line or properly escaped)
GOOGLE_CREDENTIALS='{"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"your-sa@your-project.iam.gserviceaccount.com",...}'
```

**In code:**
```python
# Automatically picks up credentials from environment (individual or JSON)
gs = GSheet()
```

**Credential resolution order:**
```python
# GSheet() will check in this order:
# 1. GOOGLE_CREDENTIALS environment variable (single JSON string)
# 2. Mounted secret at /mnt/ai-data-products/GOOGLE_CREDENTIALS
# 3. Individual Vault secrets (GOOGLE_PROJECT_ID, GOOGLE_API_PKEY_ID, 
#    GOOGLE_API_PKEY, GOOGLE_CLIENT_EMAIL, GOOGLE_CLIENT_ID, GOOGLE_CERT_URL)
# 4. gsheet_credentials.json file (fallback)
# 5. Auto-discovery in project directories (fallback)
gs = GSheet()
```

**Option 1: Use default scope (`ai-data-products`)**
```python
gs = GSheet()  # Uses default scope
```

**Option 2: Use custom scope**
```python
# Will check mounted secret at /mnt/custom-scope/GOOGLE_CREDENTIALS
gs = GSheet(scope="custom-scope")
```

No changes needed to your code - the connector automatically detects and parses JSON credentials from environment variables or production secrets.

### Read from Google Sheets

```python
# Read entire sheet as DataFrame
df = gs.read_sheet(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Sheet1'
)

# Read specific range
df = gs.read_sheet(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    range_name='Sheet1!A1:D10'
)

# Read as list of lists instead of DataFrame
values = gs.read_sheet(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    return_as="list"
)
```

### Write to Google Sheets

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Score': [95, 87, 92],
    'Grade': ['A', 'B', 'A']
})

# Write to spreadsheet (creates sheet if doesn't exist)
gs.write_sheet(
    data=df,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Results'
)

# Sheet names with spaces are automatically handled
gs.write_sheet(
    data=df,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='test raw data'  # Spaces work fine!
)

# Clear before writing
gs.write_sheet(
    data=df,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Results',
    clear_before_write=True
)

# Write to specific range
gs.write_sheet(
    data=df,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    range_name='Sheet1!B2'
)

# Write without headers
gs.write_sheet(
    data=df,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    include_headers=False
)
```

### Append to Google Sheets

```python
# Append new data to existing sheet
new_data = pd.DataFrame({
    'Name': ['David'],
    'Score': [88],
    'Grade': ['B']
})

gs.append_sheet(
    data=new_data,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Results',
    include_headers=False
)
```

### Clear Data

```python
# Clear a range
gs.clear_range(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    range_name='Sheet1!A1:D10'
)
```

## Advanced Usage

### Create New Spreadsheet

```python
# Create a new spreadsheet
spreadsheet_id = gs.create_spreadsheet(
    title="My Analysis Results",
    sheet_names=["Data", "Analysis", "Summary"]
)

print(f"Created: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
```

### Transfer Google Sheet to S3

```python
from ml_analytics.s3_connector import S3Connector

# Initialize connectors
gs = GSheet()
s3 = S3Connector(bucket="my-bucket", s3_root="my-project")

# Transfer as CSV
gs.gsheet_to_s3(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Sheet1',
    s3_connector=s3,
    file_name='data',
    file_format='csv'
)

# Transfer as Parquet
gs.gsheet_to_s3(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Sheet1',
    s3_connector=s3,
    directory='exports',
    file_name='data',
    file_format='parquet'
)
```

### Get Spreadsheet Information

```python
# Get metadata
info = gs.get_spreadsheet_info('1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms')

print(f"Title: {info['properties']['title']}")
print(f"Number of sheets: {len(info['sheets'])}")

# List all sheet names
for sheet in info['sheets']:
    print(f"Sheet: {sheet['properties']['title']}")
```

### Convert Sheet to CSV String

```python
# Get CSV string without saving to file
csv_data = gs.to_csv(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Sheet1'
)

# Save to local file if needed
with open("output.csv", "w") as f:
    f.write(csv_data)
```

### Working with List Data

```python
# Write list of lists
data = [
    ["Product", "Price", "Quantity"],
    ["Apple", 1.5, 100],
    ["Banana", 0.75, 150]
]

gs.write_sheet(
    data=data,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Inventory'
)
```

### Column Number Formatting

Write raw numeric values first, then apply display formatting as a separate step.
Cells keep their numeric type, so sorting and filtering stay numeric while users
see formatted output like `6,302,320.01` or `14.3%`.

Two helpers are available:

- `format_columns_as_number(...)` — arbitrary numeric patterns (thousands separators, decimals, literal suffixes).
- `format_columns_as_percent(...)` — Sheets' native `PERCENT` type (multiplies the stored value by 100 for display).

Both accept either column names (resolved against the header row) or 0-based
integer indices, and can be mixed in the same list.

```python
import pandas as pd

df = pd.DataFrame({
    "country":    ["AR", "BR", "MX"],
    "population": [45195777, 214326223, 128932753],
    "revenue":    [6302320.01, 812445.50, 2450000.00],
    "share":      [0.143, 0.271, 0.098],   # raw ratios
})

gs.write_sheet(df, spreadsheet_id=SID, sheet_name="Summary")

# Integers with thousands separator -> "45,195,777"
gs.format_columns_as_number(
    spreadsheet_id=SID,
    sheet_name="Summary",
    columns=["population"],
    pattern="#,##0",
)

# Two decimals with thousands separator -> "6,302,320.01"
gs.format_columns_as_number(
    spreadsheet_id=SID,
    sheet_name="Summary",
    columns=["revenue"],
    pattern="#,##0.00",
)

# Percentage from a raw ratio -> "14.3%" (Sheets multiplies 0.143 by 100)
gs.format_columns_as_percent(
    spreadsheet_id=SID,
    sheet_name="Summary",
    columns=["share"],
    pattern="0.0%",
)
```

#### When values are already scaled to percentages

If your data stores percentages directly (e.g. `3.11` meaning 3.11%, not
`0.0311`), do **not** use `format_columns_as_percent` — Sheets' `PERCENT` type
would multiply by 100 and show `311.00%`. Instead, use
`format_columns_as_number` with a literal `%` suffix in the pattern:

```python
gs.format_columns_as_number(
    spreadsheet_id=SID,
    sheet_name="Summary",
    columns=["predicted_cvr", "observed_cvr", "search_gap_pct"],
    pattern='0.00"%"',   # displays "3.11%" without multiplying
)
```

#### Pattern cheat-sheet

| Pattern         | Example input    | Displayed as      | Helper                          |
|-----------------|------------------|-------------------|---------------------------------|
| `#,##0`         | `6747815`        | `6,747,815`       | `format_columns_as_number`      |
| `#,##0.0`       | `-1.355`         | `-1.4`            | `format_columns_as_number`      |
| `#,##0.00`      | `6302320.01`     | `6,302,320.01`    | `format_columns_as_number`      |
| `$#,##0.00`     | `6302320.01`     | `$6,302,320.01`   | `format_columns_as_number`      |
| `0.00"%"`       | `3.11`           | `3.11%`           | `format_columns_as_number`      |
| `0.0%`          | `0.143`          | `14.3%`           | `format_columns_as_percent`     |
| `0.00%`         | `0.143`          | `14.30%`          | `format_columns_as_percent`     |

#### Multiple columns, same format

Pass a list to format several columns in one API round-trip:

```python
gs.format_columns_as_number(
    spreadsheet_id=SID,
    sheet_name="Summary",
    columns=[
        "booking_gap", "booking_gap_mean", "booking_gap_sd",
        "booking_gap_p05", "booking_gap_p95", "search_gap", "cvr_gap",
    ],
    pattern="#,##0.00",
)
```

To apply *different* formats to different columns, call the helper once per
format — each call is a single `batchUpdate` request.

#### Options

Both helpers accept:

- `sheet_name` — target tab (defaults to `"Sheet1"`).
- `has_header` — if `True` (default), row 1 is left unformatted and formatting
  starts at row 2. Set to `False` if your data starts at row 1.
- `pattern` — any [Google Sheets number format pattern][sheets-fmt].

[sheets-fmt]: https://developers.google.com/sheets/api/guides/formats

## Sharing Spreadsheets

### Share When Creating New Spreadsheet

```python
# Create and share automatically
result, spreadsheet_id = gs.write_sheet(
    data=df,
    spreadsheet_title="Q1 Financial Report",
    sheet_name="Summary",
    share_with="user@example.com",
    role="writer"  # or "reader"
)

print(f"Created and shared: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
```

### Share When Writing to Existing Spreadsheet

```python
# Write and share in one call
gs.write_sheet(
    data=df,
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Results',
    share_with='user@example.com',
    role='writer'
)
```

### Share with Multiple Users

```python
# Share with multiple emails
gs.write_sheet(
    data=df,
    spreadsheet_title="Team Dashboard",
    sheet_name="Data",
    share_with=[
        "alice@example.com",
        "bob@example.com",
        "charlie@example.com"
    ],
    role="reader"
)
```

### Share Existing Spreadsheet

```python
# Share spreadsheet after creation
gs.share_spreadsheet(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    email_addresses='user@example.com',
    role='writer',
    send_notification=True
)

# Share with multiple users
gs.share_spreadsheet(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    email_addresses=['user1@example.com', 'user2@example.com'],
    role='reader'
)
```

### Permission Roles

- **reader**: Can view but not edit
- **writer**: Can view and edit
- **owner**: Full control (rarely used)

## Complete Examples

### Example 1: Read, Transform, Write Back

```python
from ml_analytics.gsheet_connector import GSheet
import pandas as pd

gs = GSheet()
spreadsheet_id = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'

# Read data
df = gs.read_sheet(spreadsheet_id, sheet_name='Input')

# Transform
df['Total'] = df['Price'] * df['Quantity']
df['Tax'] = df['Total'] * 0.1

# Write results
gs.write_sheet(
    data=df,
    spreadsheet_id=spreadsheet_id,
    sheet_name='Output',
    clear_before_write=True
)
```

### Example 2: Daily Metrics Tracking

```python
from ml_analytics.gsheet_connector import GSheet
from datetime import datetime
import pandas as pd

gs = GSheet()

# Create daily report
report = pd.DataFrame({
    'Date': [datetime.now().strftime('%Y-%m-%d')],
    'Sales': [12500],
    'Users': [450]
})

# Append to tracking sheet
gs.append_sheet(
    data=report,
    spreadsheet_id='YOUR_SPREADSHEET_ID',
    sheet_name='Daily Metrics',
    include_headers=False
)
```

### Example 3: Multiple Sheets in One Spreadsheet

```python
from ml_analytics.gsheet_connector import GSheet
import pandas as pd

gs = GSheet()
spreadsheet_id = 'YOUR_SPREADSHEET_ID'

# Write to different sheets
gs.write_sheet(df_sales, spreadsheet_id, sheet_name='Sales')
gs.write_sheet(df_costs, spreadsheet_id, sheet_name='Costs')
gs.write_sheet(df_profit, spreadsheet_id, sheet_name='Profit')
```

### Example 4: Data Export to S3

```python
from ml_analytics.gsheet_connector import GSheet
from ml_analytics.s3_connector import S3Connector

gs = GSheet()
s3 = S3Connector(s3_root='ml-projects/exports')

# Transfer Google Sheet to S3 as Parquet
gs.gsheet_to_s3(
    spreadsheet_id='1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms',
    sheet_name='Data',
    s3_connector=s3,
    directory='daily-exports',
    file_name=f'data_{datetime.now().strftime("%Y%m%d")}',
    file_format='parquet'
)
```

## Example Spreadsheet

Test with this demo spreadsheet (make a copy and share with your service account):

https://docs.google.com/spreadsheets/d/1GtcR2IEIj5VDrucmmiL3DNn9CiIGBv6ZuWLn4zygOH0
