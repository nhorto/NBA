# Shared collection schedule, US/Eastern, "HH:MM" entries. Sourced by the
# macOS and Linux installers. Keep in sync with docs/DESIGN.md.
PREGAME_TIMES="10:00 13:30 17:00 18:30 20:00 22:00"
POSTGAME_TIMES="03:00 09:00"
# Weekly MLOps job (retrain challengers, evaluate shadows, refresh model card): Monday.
WEEKLY_TIMES="05:00"
WEEKLY_DAY="Mon"
WEEKLY_WEEKDAY=1   # launchd: 0/7 = Sunday, 1 = Monday
# DFS watcher (`courtside dfs tick`, docs/PLAYERS.md §5.6): every DFS_TICK_STEP minutes from
# DFS_TICK_FIRST_HOUR:00 to DFS_TICK_LAST_HOUR:55 ET, every day. Early slates lock around 12:00
# ET, so lock-47 min is ~11:13. A tick is a no-op (database reads only, no request) unless it
# falls in a lock-45 / lock-10 / late-swap window, so idle ticks are cheap.
DFS_TICK_FIRST_HOUR=11
DFS_TICK_LAST_HOUR=23
DFS_TICK_STEP=5

# "HH:MM" ET entries of the DFS watcher (11:00 11:05 ... 23:55).
dfs_tick_times() {
  local h m
  for ((h = DFS_TICK_FIRST_HOUR; h <= DFS_TICK_LAST_HOUR; h++)); do
    for ((m = 0; m < 60; m += DFS_TICK_STEP)); do
      printf '%02d:%02d ' "$h" "$m"
    done
  done
}
