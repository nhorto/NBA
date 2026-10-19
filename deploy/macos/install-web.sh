#!/usr/bin/env bash
# Install (or remove) the always-on Courtside web service as a launchd agent.
#
# Usage: install-web.sh [--dry-run] [--uninstall]
#
# KeepAlive restarts the server if it exits (e.g. Tailscale not up yet at login).

set -u

dry_run=0
uninstall=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --uninstall) uninstall=1 ;;
    -h | --help) echo "usage: ${0##*/} [--dry-run] [--uninstall]"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 64 ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
deploy_dir="$(cd "$script_dir/.." && pwd -P)" || exit 1
repo_root="$(cd "$deploy_dir/.." && pwd -P)" || exit 1
run_web="$deploy_dir/run-web.sh"
log_dir="${COURTSIDE_DATA:-$repo_root/var}/logs"
label="com.courtside.web"
plist="$HOME/Library/LaunchAgents/$label.plist"
uid="$(id -u)"

xml_escape() {
  local s="$1"
  s="${s//&/&amp;}"; s="${s//</&lt;}"; s="${s//>/&gt;}"
  printf '%s' "$s"
}

run() {
  echo "+ $*"
  [ "$dry_run" -eq 1 ] || "$@"
}

if [ "$uninstall" -eq 1 ]; then
  command -v launchctl >/dev/null && run launchctl bootout "gui/$uid/$label" 2>/dev/null
  run rm -f "$plist"
  exit 0
fi

content="$(cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$(xml_escape "$run_web")</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>StandardOutPath</key>
  <string>$(xml_escape "$log_dir/web.log")</string>
  <key>StandardErrorPath</key>
  <string>$(xml_escape "$log_dir/web.log")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>$(xml_escape "/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")</string>
  </dict>
</dict>
</plist>
PLIST
)"

if [ "$dry_run" -eq 1 ]; then
  echo "# would write $plist:"
  printf '%s\n' "$content"
else
  mkdir -p "$log_dir" "$(dirname "$plist")"
  printf '%s\n' "$content" >"$plist"
  command -v plutil >/dev/null && plutil -lint "$plist"
fi
if command -v launchctl >/dev/null; then
  run launchctl bootout "gui/$uid/$label" 2>/dev/null
  run launchctl bootstrap "gui/$uid" "$plist"
fi
echo "# Done. Logs: $log_dir/web.log  Status: launchctl print gui/$uid/$label"
