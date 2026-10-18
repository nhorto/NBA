#!/usr/bin/env bash
# Install (or remove) the Courtside launchd agents on macOS.
#
# Usage: install.sh [--dry-run] [--uninstall]
#
# launchd's StartCalendarInterval uses the machine's LOCAL time. The schedule is
# defined in US/Eastern; if the Mac is not on America/New_York the times are
# converted for today's date, which drifts by an hour across DST changes
# (re-run this script after a DST change).

set -u

dry_run=0
uninstall=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --uninstall) uninstall=1 ;;
    -h | --help)
      echo "usage: ${0##*/} [--dry-run] [--uninstall]"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 64
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
deploy_dir="$(cd "$script_dir/.." && pwd -P)" || exit 1
repo_root="$(cd "$deploy_dir/.." && pwd -P)" || exit 1
run_job="$deploy_dir/run-job.sh"
# shellcheck source=../schedule.sh
. "$deploy_dir/schedule.sh"

data_dir="${COURTSIDE_DATA:-$repo_root/var}"
log_dir="$data_dir/logs"
agents_dir="$HOME/Library/LaunchAgents"
uid="$(id -u)"
jobs="pregame postgame"

xml_escape() {
  local s="$1"
  s="${s//&/&amp;}"
  s="${s//</&lt;}"
  s="${s//>/&gt;}"
  printf '%s' "$s"
}

# Print "H M" lines for a space-separated list of ET "HH:MM" times, converted
# to local time when the system timezone is not America/New_York.
local_times() {
  if [ "$system_tz" = "America/New_York" ]; then
    local t
    for t in $1; do
      printf '%d %d\n' "$((10#${t%%:*}))" "$((10#${t##*:}))"
    done
  else
    python3 - $1 <<'PY'
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

et = ZoneInfo("America/New_York")
today = datetime.now(et).date()
for t in sys.argv[1:]:
    h, m = (int(x) for x in t.split(":"))
    local = datetime(today.year, today.month, today.day, h, m, tzinfo=et).astimezone()
    print(local.hour, local.minute)
PY
  fi
}

times_for() {
  case "$1" in
    pregame) printf '%s' "$PREGAME_TIMES" ;;
    postgame) printf '%s' "$POSTGAME_TIMES" ;;
  esac
}

print_plist() {
  local job="$1" label="com.courtside.$1" hour minute
  cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$(xml_escape "$run_job")</string>
    <string>$job</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
PLIST
  local_times "$(times_for "$job")" | while read -r hour minute; do
    cat <<PLIST
    <dict>
      <key>Hour</key>
      <integer>$hour</integer>
      <key>Minute</key>
      <integer>$minute</integer>
    </dict>
PLIST
  done
  cat <<PLIST
  </array>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$log_dir/$job.launchd.log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$log_dir/$job.launchd.log")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>$(xml_escape "/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")</string>
PLIST
  if [ -n "${COURTSIDE_DATA:-}" ]; then
    cat <<PLIST
    <key>COURTSIDE_DATA</key>
    <string>$(xml_escape "$COURTSIDE_DATA")</string>
PLIST
  fi
  cat <<PLIST
  </dict>
</dict>
</plist>
PLIST
}

# Detect the system timezone name.
system_tz=""
if tzlink="$(readlink /etc/localtime 2>/dev/null)" && [ -n "$tzlink" ]; then
  system_tz="${tzlink#*zoneinfo/}"
fi

if [ "$dry_run" -eq 0 ] && ! command -v launchctl >/dev/null 2>&1; then
  echo "launchctl not found: this installer must run on macOS (use --dry-run to preview)." >&2
  exit 1
fi

if [ "$uninstall" -eq 1 ]; then
  for job in $jobs; do
    label="com.courtside.$job"
    plist="$agents_dir/$label.plist"
    if [ "$dry_run" -eq 1 ]; then
      printf '+ launchctl bootout gui/%s/%s\n' "$uid" "$label"
      printf '+ rm -f %q\n' "$plist"
    else
      printf '+ launchctl bootout gui/%s/%s\n' "$uid" "$label"
      launchctl bootout "gui/$uid/$label" 2>/dev/null || true
      printf '+ rm -f %q\n' "$plist"
      rm -f "$plist"
    fi
  done
  exit 0
fi

if [ "$system_tz" = "America/New_York" ]; then
  echo "# System timezone is America/New_York: using Eastern times as-is."
else
  echo "# WARNING: system timezone is '${system_tz:-unknown}', not America/New_York." >&2
  echo "# WARNING: ET times were converted to local hours for today's date; a DST" >&2
  echo "# WARNING: transition (US or local) can shift them by an hour. Re-run after DST changes." >&2
fi

if [ "$dry_run" -eq 0 ]; then
  mkdir -p "$agents_dir" "$log_dir" || exit 1
else
  printf '+ mkdir -p %q %q\n' "$agents_dir" "$log_dir"
fi

for job in $jobs; do
  label="com.courtside.$job"
  plist="$agents_dir/$label.plist"
  if [ "$dry_run" -eq 1 ]; then
    echo "# --- $plist ---"
    print_plist "$job"
    printf '+ plutil -lint %q\n' "$plist"
    printf '+ launchctl bootout gui/%s/%s   (errors ignored)\n' "$uid" "$label"
    printf '+ launchctl bootstrap gui/%s %q\n' "$uid" "$plist"
  else
    print_plist "$job" >"$plist" || exit 1
    if command -v plutil >/dev/null 2>&1; then
      plutil -lint "$plist" || exit 1
    else
      echo "# plutil not found; skipping lint" >&2
    fi
    printf '+ launchctl bootout gui/%s/%s\n' "$uid" "$label"
    launchctl bootout "gui/$uid/$label" 2>/dev/null || true
    printf '+ launchctl bootstrap gui/%s %q\n' "$uid" "$plist"
    launchctl bootstrap "gui/$uid" "$plist" || exit 1
  fi
done

echo "# Done. Check: launchctl print gui/$uid/com.courtside.pregame"
