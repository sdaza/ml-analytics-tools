# Tunnel Manager

A tiny helper to run a persistent VS Code tunnel with automatic restarts and periodic health checks.

- Module: [ml_analytics/tunnel_manager.py](../ml_analytics/tunnel_manager.py)
- Tunnel URL: `https://vscode.dev/tunnel/<tunnel_name>`
- Logs: `~/vscode-tunnel.log`

## Quick Start

**Note:** This is a long-running foreground process. Use one of the background methods below.

### Run in tmux (recommended)

```bash
# Start a new tmux session
tmux new -s vscode-tunnel

# Restart daily at 08:00
ml-analytics-tunnel --name ml --restart-time 08:00

# Or restart every 3 hours
ml-analytics-tunnel --name ml --restart-time 3h

# Detach: Press Ctrl+b then d
# Reattach later: tmux attach -t vscode-tunnel
```

### Run with nohup

```bash
nohup ml-analytics-tunnel --name ml --restart-time 3h > /dev/null 2>&1 &
```

### Run in screen

```bash
screen -S vscode-tunnel
ml-analytics-tunnel --name ml --restart-time 08:00
# Detach: Ctrl+a then d
# Reattach: screen -r vscode-tunnel
```

### Direct module invocation

```bash
python -m ml_analytics.tunnel_manager --name ml --restart-time 3h
```

## Options

- `--name`: Tunnel name (default: `general`)
- `--code-path`: Full path to the VS Code CLI (`code`) executable (default: `~/code`)
- `--restart-time`: Restart schedule — either a daily fixed time in `HH:MM` 24h format (e.g. `08:00`) or a recurring interval in hours (e.g. `3h` for every 3 hours). Default: `04:00`
- `--health-check-interval`: Health check interval in minutes (default: `15`)

## What It Does

- Starts a VS Code tunnel and streams tunnel output to `~/vscode-tunnel.log`.
- Schedules restarts either at a fixed daily time or at a recurring interval (sends the native `r` command to the tunnel).
- Performs a health check at the configured interval (default: 15 minutes) and restarts the tunnel if it dies.
- On Ctrl+C, shuts the tunnel down cleanly (sends `x`).

## Requirements

- VS Code CLI (`code`) installed and accessible at `--code-path`.
- Network access to VS Code tunnel services.

## Troubleshooting

- "Code executable not found": verify `--code-path` and execution permissions.
- No tunnel page: confirm your tunnel name in the URL `https://vscode.dev/tunnel/<name>` and check logs.
