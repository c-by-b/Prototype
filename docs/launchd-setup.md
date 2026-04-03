# Auto-Restart: Safety Socket on Mac Mini

A `launchd` user agent that starts the Flask server on login and restarts it on crash.

## The Plist

Save this as `~/Library/LaunchAgents/com.cbyb.prototype.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cbyb.prototype</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/nathanmeyer/Dev/C-by-B-code/Prototype/venv/bin/python</string>
        <string>-m</string>
        <string>cbyb.app</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/nathanmeyer/Dev/C-by-B-code/Prototype</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/nathanmeyer/Dev/C-by-B-code/Prototype/logs/server-stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/nathanmeyer/Dev/C-by-B-code/Prototype/logs/server-stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
```

## Setup Steps

```bash
# 1. Create logs directory
mkdir -p /Users/nathanmeyer/Dev/C-by-B-code/Prototype/logs

# 2. Copy the plist (or create it from the XML above)
cp docs/com.cbyb.prototype.plist ~/Library/LaunchAgents/

# 3. Load the agent
launchctl load ~/Library/LaunchAgents/com.cbyb.prototype.plist

# 4. Verify it's running
launchctl list | grep cbyb
curl http://localhost:5050/health
```

## Management Commands

```bash
# Stop the server
launchctl unload ~/Library/LaunchAgents/com.cbyb.prototype.plist

# Restart (unload + load)
launchctl unload ~/Library/LaunchAgents/com.cbyb.prototype.plist
launchctl load ~/Library/LaunchAgents/com.cbyb.prototype.plist

# View logs
tail -f /Users/nathanmeyer/Dev/C-by-B-code/Prototype/logs/server-stderr.log
```

## Notes

- **User agent** (LaunchAgents), not system daemon — runs as your user, only when logged in. If the Mini is set to auto-login, this starts on boot.
- **KeepAlive: true** — restarts the process if it crashes.
- **ThrottleInterval: 10** — waits 10 seconds between restart attempts to avoid rapid-fire restarts on persistent errors.
- **Tailscale** manages its own launch agent via its installer — no need to configure that here.
- The `.env` file with API keys is loaded by the app from the working directory.
