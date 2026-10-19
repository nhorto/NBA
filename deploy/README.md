# Deploying the collector

Schedule (US/Eastern, defined in `schedule.sh`):

- `pregame`: 10:00, 13:30, 17:00, 18:30, 20:00, 22:00
- `postgame`: 03:00, 09:00
- `weekly`: Monday 05:00 (MLOps)
- `dfs-tick`: every 5 minutes, 11:00 to 23:55 (`courtside dfs tick`, the lock-relative DFS watcher, below)

**Run the scheduler on ONE machine at a time.** The Mac mini is primary; running it on two machines produces duplicate snapshots. Other machines can pull the data instead:

```
rsync -a nicholashorton@mac-mini:"Developer/NBA/var/" var/
```

`run-job.sh <pregame|postgame|weekly|dfs-tick>` is the wrapper both schedulers call. It cds to the repo (spaces in the path are fine), finds `uv`, runs `uv run --frozen courtside collect <job>` (`weekly`: the MLOps commands, `dfs-tick`: `courtside dfs tick`), and exits with the job's exit code.

## DFS watcher (`dfs-tick`)

`courtside dfs tick` (docs/PLAYERS.md §5.6) fires the lock-relative DraftKings runs. Every 5 minutes it looks at `dfs_slates` and starts, per slate:

- `lock45` when now is in [lock-47 min, lock-40 min], `lock10` in [lock-12, lock-7];
- `swap` (Classic only, not Showdown) at each later game's tip-10 (same width), and only when late news touches a player of a saved lineup.

Each run refreshes the official injury report, ESPN injuries and the slate's draftables (the draftables draw on the daily DK budget, `dfs_capture`), projects, solves cash + Top 10 and writes `dfs_runs` / `dfs_lineups`. Runs are idempotent on (slate, kind): a second tick in the same window does nothing, a failed run is retried by the next tick.

- **Window:** 11:00 to 23:55 ET, every day (early slates lock around noon). Outside it `run-job.sh dfs-tick` exits 0 immediately (`COURTSIDE_DFS_TICK_ANYTIME=1` overrides it for a manual run). The hours and step live in `schedule.sh` (`DFS_TICK_*`).
- **Idle ticks are free:** with no window hit the tick reads the database and prints nothing: no request, no write, no log line. Only a tick that fires work logs (`logs/dfs-tick.log`).
- **Requests when a window is hit:** 1 ESPN injuries request; the NBA official report finder (newest 15-minute slot first, usually 1-5 probes, at most 40, plus 1 PDF if new); per slate that fires, 1 DK draftables request (2 on a retry). Opening-night worst case, two slates at once (main + Showdown): 4 DK + 1 ESPN + <= 41 NBA.
- **Manual:** `courtside dfs tick --dry-run [--now ISO]` shows what is due; `courtside dfs run --slate ID --kind lock45|lock10|swap|manual [--force]` runs one slate.
- **Overlap:** a tick that finds another tick running skips (`<data>/.dfs-tick.lock`).

## macOS (launchd)

```
deploy/macos/install.sh --dry-run   # preview the plists and commands
deploy/macos/install.sh             # write ~/Library/LaunchAgents/com.courtside.{pregame,postgame,weekly,dfs-tick}.plist and load them
deploy/macos/install.sh --uninstall
```

launchd uses local time. If the Mac is not on `America/New_York`, the installer converts the ET times for today's date and warns; DST transitions can shift them by an hour, so re-run the installer afterwards. The Mac must be awake (or woken) at run time.

`com.courtside.dfs-tick` has 156 `StartCalendarInterval` entries (hours 11-23 x minutes 0, 5, ... 55), which keeps the ticks on the 5-minute grid (a `StartInterval` drifts, and the lock-10 window is only 5 minutes wide) and wakes nothing at night.

Status: `launchctl print gui/$UID/com.courtside.pregame`

## Linux (systemd user timers)

```
deploy/linux/install.sh --dry-run
deploy/linux/install.sh             # units in ~/.config/systemd/user/, then enable --now
deploy/linux/install.sh --uninstall
```

Timers carry an `America/New_York` suffix, so DST is handled by systemd. `Persistent=true` catches up one missed run after downtime (not for `dfs-tick`: `OnCalendar=*-*-* 11..23:00/5:00 America/New_York`, `Persistent=false`, since a stale tick is useless). To run while logged out: `loginctl enable-linger $USER` (not done by the script).

Status: `systemctl --user list-timers`

## Logs and manual runs

- Job log: `${COURTSIDE_DATA:-<repo>/var}/logs/<job>.log`, timestamped, rotated once to `<job>.log.1` above ~5 MB.
- launchd's own stdout/stderr (wrapper failures only): `logs/<job>.launchd.log`.
- Linux service output also lands in `journalctl --user -u courtside-pregame`.
- Run manually: `deploy/run-job.sh pregame` (or `postgame`, `weekly`, `dfs-tick`). `--dry-run` as a second argument shows the resolved paths and command without running.
- `COURTSIDE_DATA`, if set when you run an installer, is baked into the units.

## Web app (always-on)

`deploy/run-web.sh` serves the Brief on this machine's Tailscale IP, port 8765, so it's reachable only from your tailnet. Set `COURTSIDE_HOST` or `COURTSIDE_PORT` to override.

- macOS (Mac mini, primary): `bash deploy/macos/install-web.sh`. It installs launchd `com.courtside.web` with KeepAlive. Logs go to `var/logs/web.log`.
- Linux (geekom1): `bash deploy/linux/install-web.sh`. It installs the systemd user unit `courtside-web.service`.

Open `http://mac-mini:8765` (or `http://<tailnet-ip>:8765`) from any device on the tailnet.
