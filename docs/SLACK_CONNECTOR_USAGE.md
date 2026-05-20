# Slack Connector

Simple interface to send messages, files, and images to Slack channels.

## Setup

### 1. Create a Slack App & Get Bot Token

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app, or use an existing one.
2. Add and check the following **Bot Token Scopes** under OAuth & Permissions:
   - `chat:write` - Send messages
   - `files:write` - Upload files
   - `channels:read` - List public channels
   - `groups:read` - Access private channels (optional)
   - `reactions:write` - Add emoji reactions (optional)
3. Install the app to your workspace and copy the **Bot User OAuth Token** (starts with `xoxb-`)

### 2. Configure Authentication

#### Development Environment

```bash
# Add to .env file
SLACK_BOT_TOKEN=xoxb-your-token-here
```

#### Production Environment (Containerized)

In production containerized environments with SecretProvider, the connector automatically loads credentials from mounted secrets at `/mnt/{scope}/SLACK_BOT_TOKEN` (default scope: `"ai-data-products"`).

**Option 1: Use default scope (`ai-data-products`)**
```python
# Will check for SLACK_BOT_TOKEN in:
# 1. Environment variable
# 2. Mounted secret at /mnt/ai-data-products/SLACK_BOT_TOKEN
# 3. slack_token.txt file (fallback)
slack = SlackConnector()
```

**Option 2: Use custom scope**
```python
# Will check for SLACK_BOT_TOKEN in:
# 1. Environment variable
# 2. Mounted secret at /mnt/custom-scope/SLACK_BOT_TOKEN
# 3. slack_token.txt file (fallback)
slack = SlackConnector(scope="custom-scope")
```

No changes needed to your code - the connector automatically detects production environment secrets.

## Usage

### Initialize

```python
from ml_analytics import SlackConnector
slack = SlackConnector(return_response=True) # Set to False to disable API response return
```

### Send Message

```python
# Simple message
slack.send_message("tech-community", "Hello from Python!")

# With rich formatting
blocks = [
    {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Important Alert*\nSomething happened!"}
    }
]
slack.send_message("#alerts", "Alert", blocks=blocks)

# Reply in thread
response = slack.send_message("tech-community", "Main message")
slack.send_message("tech-community", "Reply", thread_ts=response["ts"])
```

### Upload Files

```python
# Upload a file
slack.send_file(
    channels="#data-team",
    file_path="report.csv",
    title="Daily Report",
    initial_comment="Here's today's report"
)

# Upload from content (e.g., generated data)
import pandas as pd
df = pd.DataFrame({'col1': [1, 2, 3]})
csv_content = df.to_csv(index=False)

slack.send_file(
    channels="#data-team",
    content=csv_content,
    filename="data.csv",
    title="Generated Data"
)
```

### Send Image (Inline)

```python
# Note: Image URL must be publicly accessible
slack.send_message_with_image(
    "tech-community",
    "Check out our results!",
    image_url="https://example.com/chart.png",
    image_alt="Performance chart",
    title="Model Performance"
)
```

### Other Features

```python
# Add emoji reaction
response = slack.send_message("#tech-community", "Great work!")
slack.add_reaction("#tech-community", response["ts"], "thumbsup")


# Get channel ID
channel_id = slack.get_or_resolve_channel_id("tech-community")
```

## Notes

- The bot must be **invited to private channels** before it can post there
- For Slack Connect channels, the connector automatically resolves channel IDs
- Channel names work with or without `#` prefix
