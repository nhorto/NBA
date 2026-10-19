#!/usr/bin/env bash
# Install (or remove) the always-on Courtside web service as a systemd user unit.
#
# Usage: install-web.sh [--dry-run] [--uninstall]

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
run_web="$deploy_dir/run-web.sh"
unit_dir="$HOME/.config/systemd/user"
unit="courtside-web.service"

run() {
  echo "+ $*"
  [ "$dry_run" -eq 1 ] || "$@"
}

if [ "$uninstall" -eq 1 ]; then
  run systemctl --user disable --now "$unit"
  run rm -f "$unit_dir/$unit"
  run systemctl --user daemon-reload
  exit 0
fi

content="[Unit]
Description=Courtside web app (NBA Morning Brief)
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
ExecStart=/bin/bash \"$run_web\"
Restart=always
RestartSec=30

[Install]
WantedBy=default.target"

if [ "$dry_run" -eq 1 ]; then
  echo "# would write $unit_dir/$unit:"
  printf '%s\n' "$content"
else
  mkdir -p "$unit_dir"
  printf '%s\n' "$content" >"$unit_dir/$unit"
fi
run systemctl --user daemon-reload
run systemctl --user enable --now "$unit"
echo "# Done. Logs: journalctl --user -u $unit -f"
