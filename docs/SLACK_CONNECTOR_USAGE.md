# Slack Connector

`SlackConnector` wraps the Slack Web API for the common notebook and pipeline
tasks: post a message, upload a file, add a reaction, or resolve a channel.

## Setup

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps).
2. Add the bot scopes you need:
   - `chat:write` for messages
   - `files:write` for uploads
   - `channels:read` for public channel lookup
   - `groups:read` for private channel lookup
   - `reactions:write` for reactions
3. Install the app to your workspace.
4. Set the bot token:

```bash
SLACK_BOT_TOKEN=xoxb-your-token
```

The connector also checks `/mnt/<scope>/SLACK_BOT_TOKEN`; the default scope is
`ml`. You can pass `scope="custom-scope"` if your mounted secrets use a
different directory.

## Usage

```python
from ml_analytics import SlackConnector

slack = SlackConnector()

slack.send_message("#ml-alerts", "Training finished")
```

## Upload Files

```python
slack.send_file(
    channels="#ml-alerts",
    file_path="reports/metrics.csv",
    title="Model metrics",
    initial_comment="Latest training metrics",
)
```

You can also upload generated content:

```python
csv_content = df.to_csv(index=False)

slack.send_file(
    channels="#ml-alerts",
    content=csv_content,
    filename="metrics.csv",
    title="Model metrics",
)
```

## Threads, Blocks, And Reactions

```python
response = slack.send_message("#ml-alerts", "Training finished")

slack.send_message(
    "#ml-alerts",
    "Full report is attached.",
    thread_ts=response["ts"],
)

slack.add_reaction("#ml-alerts", response["ts"], "white_check_mark")
```

For rich formatting, pass Slack Block Kit payloads through `blocks`:

```python
blocks = [
    {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Training finished*"},
    }
]

slack.send_message("#ml-alerts", "Training finished", blocks=blocks)
```

## Notes

- Invite the bot to private channels before posting.
- Channel names work with or without the `#` prefix.
- For Slack Connect channels, the connector resolves channel IDs when possible.
- Set `return_response=True` when you need raw Slack API responses.
