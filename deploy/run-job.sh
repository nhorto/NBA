#!/usr/bin/env bash
# Run one Courtside collection job and append timestamped output to its log.
#
# Usage: run-job.sh <pregame|postgame|weekly|dfs-tick> [--dry-run]
#
# Log: ${COURTSIDE_DATA:-<repo>/var}/logs/<job>.log (rotated once to .log.1
# when it exceeds ~5 MB). Exit code is the job's exit code.
#
# dfs-tick (`courtside dfs tick`, every 5 min): exits 0 at once outside the ET window
# (before DFS_TICK_FIRST_HOUR, from schedule.sh; COURTSIDE_DFS_TICK_ANYTIME=1 overrides) and
# logs nothing when no lock-relative window was hit (the tick prints nothing then).

set -u

usage() {
  echo "usage: ${0##*/} <pregame|postgame|weekly|dfs-tick> [--dry-run]" >&2
  exit 64
}

job="${1:-}"
dry_run=0
[ "${2:-}" = "--dry-run" ] && dry_run=1
case "$job" in
  pregame | postgame | weekly | dfs-tick) ;;
  *) usage ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
repo_root="$(cd "$script_dir/.." && pwd -P)" || exit 1
cd "$repo_root" || exit 1

# shellcheck source=schedule.sh
. "$script_dir/schedule.sh"

# dfs-tick: a no-op outside the ET window (cheap guard, before uv is even looked up).
if [ "$job" = "dfs-tick" ] && [ "$dry_run" -eq 0 ] && [ "${COURTSIDE_DFS_TICK_ANYTIME:-0}" != "1" ]; then
  et_hour="$(TZ=America/New_York date +%H)"
  if [ "$((10#$et_hour))" -lt "$DFS_TICK_FIRST_HOUR" ] || [ "$((10#$et_hour))" -gt "$DFS_TICK_LAST_HOUR" ]; then
    exit 0
  fi
fi

# launchd/systemd start with a minimal PATH.
PATH="/opt/homebrew/bin:$HOME/.local/bin:${PATH:-/usr/bin:/bin}"
export PATH

find_uv() {
  local candidate
  for candidate in "$HOME/.local/bin/uv" "/opt/homebrew/bin/uv"; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if candidate="$(command -v uv 2>/dev/null)" && [ -n "$candidate" ]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if command -v mise >/dev/null 2>&1; then
    candidate="$(mise which uv 2>/dev/null)" || candidate=""
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  return 1
}

[ -n "${COURTSIDE_DATA:-}" ] && export COURTSIDE_DATA
data_dir="${COURTSIDE_DATA:-$repo_root/var}"
log_dir="$data_dir/logs"
log_file="$log_dir/$job.log"
max_bytes=5242880

uv_bin="$(find_uv)" || uv_bin=""

if [ "$dry_run" -eq 1 ]; then
  echo "repo root : $repo_root"
  echo "data dir  : $data_dir"
  echo "log file  : $log_file (rotates to $log_file.1 above $max_bytes bytes)"
  echo "uv        : ${uv_bin:-<not found>}"
  if [ "$job" = "dfs-tick" ]; then
    echo "window    : ET hours $DFS_TICK_FIRST_HOUR-$DFS_TICK_LAST_HOUR (outside it the job exits 0 immediately)"
    echo "command   : cd \"$repo_root\" && \"${uv_bin:-uv}\" run --frozen courtside dfs tick"
  else
    echo "command   : cd \"$repo_root\" && \"${uv_bin:-uv}\" run --frozen courtside collect \"$job\""
  fi
  exit 0
fi

mkdir -p "$log_dir" || exit 1

# Rotate once: keep at most ~5 MB in the live log plus one previous generation.
if [ -f "$log_file" ]; then
  size="$(wc -c <"$log_file" | tr -d ' ')"
  if [ "${size:-0}" -gt "$max_bytes" ]; then
    mv -f "$log_file" "$log_file.1"
  fi
fi

# Prefix each line of stdin with an ISO-8601 timestamp and append to the log.
log_lines() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$line"
  done >>"$log_file"
}

if [ "$job" = "dfs-tick" ]; then
  # Quiet unless something happened: ~150 ticks a day, a handful do any work.
  if [ -z "$uv_bin" ]; then
    echo "[run-job] start $job (repo: $repo_root)" | log_lines
    echo "[run-job] ERROR: uv not found (looked in \$HOME/.local/bin, /opt/homebrew/bin, PATH, mise)" | log_lines
    exit 127
  fi
  out="$("$uv_bin" run --frozen courtside dfs tick 2>&1)"
  rc=$?
  if [ -n "$out" ] || [ "$rc" -ne 0 ]; then
    echo "[run-job] start $job (repo: $repo_root)" | log_lines
    [ -n "$out" ] && printf '%s\n' "$out" | log_lines
    echo "[run-job] end $job exit=$rc" | log_lines
  fi
  exit "$rc"
fi

echo "[run-job] start $job (repo: $repo_root)" | log_lines

if [ -z "$uv_bin" ]; then
  echo "[run-job] ERROR: uv not found (looked in \$HOME/.local/bin, /opt/homebrew/bin, PATH, mise)" | log_lines
  exit 127
fi

if [ "$job" = "weekly" ]; then
  # MLOps: train challengers (offline gate; --if-due = first weekly run of the month or an open
  # calibration/drift alert), evaluate shadows (online gate), refresh the card; then the weekly
  # DFS report.
  rc=0
  for cmd in "retrain --model all --if-due" "shadows" "card"; do
    # shellcheck disable=SC2086
    "$uv_bin" run --frozen courtside mlops $cmd 2>&1 | log_lines
    r="${PIPESTATUS[0]}"
    [ "$r" -ne 0 ] && rc="$r"
  done
  # Weekly DFS report (T3.3): read-only; a failure is logged but does not fail the job.
  "$uv_bin" run --frozen courtside proj weekly-report 2>&1 | log_lines
  r="${PIPESTATUS[0]}"
  [ "$r" -ne 0 ] && echo "[run-job] weekly-report failed (exit $r), not fatal" | log_lines
else
  "$uv_bin" run --frozen courtside collect "$job" 2>&1 | log_lines
  rc="${PIPESTATUS[0]}"
  if [ "$job" = "pregame" ]; then
    # Player-side live shadows (IMPROVEMENTS E1, proj/shadows.py): its own low-priority process
    # after the pregame job, never in the DFS tick; skips itself near a lock build window, 90 s
    # budget, COURTSIDE_SHADOWS=0 turns it off. A failure is logged but never fails the job.
    nice -n 10 "$uv_bin" run --frozen courtside proj shadows run 2>&1 | log_lines
    r="${PIPESTATUS[0]}"
    [ "$r" -ne 0 ] && echo "[run-job] shadows failed (exit $r), not fatal" | log_lines
  fi
fi

echo "[run-job] end $job exit=$rc" | log_lines
exit "$rc"
