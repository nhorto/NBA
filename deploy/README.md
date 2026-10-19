# Deploying the collector

Schedule (US/Eastern, defined in `schedule.sh`):

- `pregame`: 10:00, 13:30, 17:00, 18:30, 20:00, 22:00
- `postgame`: 03:00, 09:00

**Run the scheduler on ONE machine at a time.** The Mac mini is primary; running it on two machines produces duplicate snapshots. Other machines can pull the data instead:

```
rsync -a nicholashorton@mac-mini:"Developer/NBA/var/" var/
```

`run-job.sh <pregame|postgame>` is the wrapper both schedulers call. It cds to the repo (spaces in the path are fine), finds `uv`, runs `uv run --frozen courtside collect <job>`, and exits with the job's exit code.

## macOS (launchd)

```
deploy/macos/install.sh --dry-run   # preview the plists and commands
deploy/macos/install.sh             # write ~/Library/LaunchAgents/com.courtside.{pregame,postgame}.plist and load them
deploy/macos/install.sh --uninstall
```

launchd uses local time. If the Mac is not on `America/New_York`, the installer converts the ET times for today's date and warns; DST transitions can shift them by an hour, so re-run the installer afterwards. The Mac must be awake (or woken) at run time.

Status: `launchctl print gui/$UID/com.courtside.pregame`

## Linux (systemd user timers)

```
deploy/linux/install.sh --dry-run
deploy/linux/install.sh             # units in ~/.config/systemd/user/, then enable --now
deploy/linux/install.sh --uninstall
```

Timers carry an `America/New_York` suffix, so DST is handled by systemd. `Persistent=true` catches up one missed run after downtime. To run while logged out: `loginctl enable-linger $USER` (not done by the script).

Status: `systemctl --user list-timers`

## Logs and manual runs

- Job log: `${COURTSIDE_DATA:-<repo>/var}/logs/<job>.log`, timestamped, rotated once to `<job>.log.1` above ~5 MB.
- launchd's own stdout/stderr (wrapper failures only): `logs/<job>.launchd.log`.
- Linux service output also lands in `journalctl --user -u courtside-pregame`.
- Run manually: `deploy/run-job.sh pregame` (or `postgame`). `--dry-run` as a second argument shows the resolved paths and command without running.
- `COURTSIDE_DATA`, if set when you run an installer, is baked into the units.

## Web app (always-on)

`deploy/run-web.sh` serves the Brief on this machine's Tailscale IP, port 8765, so it's reachable only from your tailnet. Set `COURTSIDE_HOST` or `COURTSIDE_PORT` to override.

- macOS (Mac mini, primary): `bash deploy/macos/install-web.sh`. It installs launchd `com.courtside.web` with KeepAlive. Logs go to `var/logs/web.log`.
- Linux (geekom1): `bash deploy/linux/install-web.sh`. It installs the systemd user unit `courtside-web.service`.

Open `http://mac-mini:8765` (or `http://<tailnet-ip>:8765`) from any device on the tailnet.
