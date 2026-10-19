#!/usr/bin/env bash
# Install (or remove) the Courtside systemd user timers on Linux.
#
# Usage: install.sh [--dry-run] [--uninstall]
#
# Timers use OnCalendar with an explicit America/New_York suffix, so DST is
# handled by systemd regardless of the machine's own timezone.

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
run_job="$deploy_dir/run-job.sh"
# shellcheck source=../schedule.sh
. "$deploy_dir/schedule.sh"

unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
jobs="pregame postgame weekly dfs-tick"

run() {
  printf '+ %s\n' "$*"
  [ "$dry_run" -eq 1 ] && return 0
  "$@"
}

# Quote one argument for a systemd ExecStart= line: double quotes, with
# backslash, double quote, % (specifier) and $ (env expansion) escaped.
systemd_quote() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//%/%%}"
  s="${s//\$/\$\$}"
  printf '"%s"' "$s"
}

times_for() {
  case "$1" in
    pregame) printf '%s' "$PREGAME_TIMES" ;;
    postgame) printf '%s' "$POSTGAME_TIMES" ;;
    weekly) printf '%s' "$WEEKLY_TIMES" ;;
    dfs-tick) dfs_tick_times ;;
  esac
}

print_service() {
  local job="$1"
  cat <<UNIT
[Unit]
Description=Courtside $job job

[Service]
Type=oneshot
UNIT
  if [ -n "${COURTSIDE_DATA:-}" ]; then
    printf 'Environment=%s\n' "$(systemd_quote "COURTSIDE_DATA=$COURTSIDE_DATA")"
  fi
  printf 'ExecStart=/bin/bash %s %s\n' "$(systemd_quote "$run_job")" "$job"
}

print_timer() {
  local job="$1" t persistent
  cat <<UNIT
[Unit]
Description=Courtside $job schedule (US/Eastern)

[Timer]
UNIT
  if [ "$job" = "dfs-tick" ]; then
    # 11:00, 11:05 ... 23:55 ET: hours FIRST..LAST, minutes 00/STEP (systemd range syntax).
    printf 'OnCalendar=*-*-* %s..%s:00/%s:00 America/New_York\n' \
      "$DFS_TICK_FIRST_HOUR" "$DFS_TICK_LAST_HOUR" "$DFS_TICK_STEP"
    persistent=false          # a missed tick is worthless; do not replay it after downtime
  else
    persistent=true
    for t in $(times_for "$job"); do
      if [ "$job" = "weekly" ]; then
        printf 'OnCalendar=%s *-*-* %s:00 America/New_York\n' "$WEEKLY_DAY" "$t"
      else
        printf 'OnCalendar=*-*-* %s:00 America/New_York\n' "$t"
      fi
    done
  fi
  cat <<UNIT
Persistent=$persistent
Unit=courtside-$job.service

[Install]
WantedBy=timers.target
UNIT
}

if [ "$dry_run" -eq 0 ] && ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found: this installer needs systemd (use --dry-run to preview)." >&2
  exit 1
fi

if [ "$uninstall" -eq 1 ]; then
  for job in $jobs; do
    run systemctl --user disable --now "courtside-$job.timer" || true
    run rm -f "$unit_dir/courtside-$job.service" "$unit_dir/courtside-$job.timer"
  done
  run systemctl --user daemon-reload
  exit 0
fi

if [ "$dry_run" -eq 1 ]; then
  printf '+ mkdir -p %q\n' "$unit_dir"
else
  mkdir -p "$unit_dir" || exit 1
fi

for job in $jobs; do
  svc="$unit_dir/courtside-$job.service"
  tmr="$unit_dir/courtside-$job.timer"
  if [ "$dry_run" -eq 1 ]; then
    echo "# --- $svc ---"
    print_service "$job"
    echo "# --- $tmr ---"
    print_timer "$job"
  else
    print_service "$job" >"$svc" || exit 1
    print_timer "$job" >"$tmr" || exit 1
    if command -v systemd-analyze >/dev/null 2>&1; then
      systemd-analyze --user verify "$svc" "$tmr" || echo "# warning: unit verification reported issues" >&2
    fi
  fi
done

run systemctl --user daemon-reload || exit 1
for job in $jobs; do
  run systemctl --user enable --now "courtside-$job.timer" || exit 1
done

echo
echo "# Note: to keep the timers running while you are logged out, run once:"
echo "#   loginctl enable-linger \$USER"
echo "# (not run by this script). Check: systemctl --user list-timers"
