#!/bin/zsh
# Re-evaluation workstream A (availability and minutes): the Mac scratch runner.
# Docs: docs/research/reeval-A.md (section "Pending backtests"), docs/REEVAL_2026-10.md.
#
#   reeval_a_mac.sh setup     copy code (already rsynced to code/), DB snapshot, bundles, raw tips
#   reeval_a_mac.sh lineup    +lu weights fit < S (2023-24 .. 2026-27) + minutes-level eval
#   reeval_a_mac.sh replay    2026-27 live replay, 5 nights x lock10 / lock45 x off gh rm lu all
#   reeval_a_mac.sh slice     T2.4 slice 2025-26: off / rm (live, live10), lu / rm+lu (live10)
#   reeval_a_mac.sh compare   paired tables (replay report, p8_e3_eval dk on the slice rows)
#   reeval_a_mac.sh all       lineup, replay, slice, compare (one process at a time)
#
# Scratch: ~/Developer/courtside-compute/reeval-a/  (code/ = the model-reeval working tree,
# rsynced from geekom1 with --exclude var --exclude .venv --exclude cache --exclude var-mlops).
# One heavy process at a time, nice 10, 1 thread. Never writes the prod DB or ~/Developer/NBA/var:
# the DB is an online-backup copy opened read-only by every job. Jobs do not START inside the
# lock windows 12:00-12:50 and 15:45-19:45 ET (guard); each job is short enough (< 30 min) to
# finish before the next window if started outside it.
set -u
S=$HOME/Developer/courtside-compute/reeval-a
PY=$HOME/Developer/NBA/.venv/bin/python
PROD=$HOME/Developer/NBA/var
C1R=$HOME/Developer/courtside-compute/c1rship/data/proj
PQ_LOMO=$HOME/Developer/courtside-compute/pq/pq_lomo.json
cd $S || exit 1
mkdir -p logs out data/proj

guard() {          # wait while inside a lock window (ET)
  while true; do
    h=$(TZ=America/New_York date +%H%M)
    if { [ $h -ge 1200 ] && [ $h -lt 1250 ]; } || { [ $h -ge 1545 ] && [ $h -lt 1945 ]; }; then
      echo "guard: waiting ($h ET)"; sleep 120
    else
      break
    fi
  done
}

live_env() {       # the live stack (pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr), new switches OFF
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  export PYTHONPATH=$S/code/src COURTSIDE_NO_WRITER=1 COURTSIDE_DATA=${DATA:-$S/data}
  export COURTSIDE_CONTEXT=1 COURTSIDE_CONTEXT_EXT=0 COURTSIDE_PPLAY=v2 COURTSIDE_COPULA=v1r
  export COURTSIDE_MINRATIO=0 COURTSIDE_PRESEASON=0 COURTSIDE_BLOWOUT=0 COURTSIDE_PPLAYQ=1
  export COURTSIDE_MINFIX=1 COURTSIDE_RECON_RESERVE=0 COURTSIDE_RESERVE_EXTRAS=1 COURTSIDE_INJ_GAMEDAY=0
  export COURTSIDE_LINE_FRESH=0 COURTSIDE_ROLE_BLEND=1 COURTSIDE_TIER_MATCH=0 COURTSIDE_TAIL_REMAP=1
  export COURTSIDE_EARLY_PRIOR=1 COURTSIDE_MIN_RESTRICT=0 COURTSIDE_RECON_WEIGHTED=0
  export COURTSIDE_MIN_SPREAD=1 COURTSIDE_USAGE=0
  export COURTSIDE_GHOST_DROP=0 COURTSIDE_REPORT_MATCH=0 COURTSIDE_LINEUP=0
  export REEVAL_A_OUT=$S/out C1R_OUT=$S/out P8E3_OUT=$S/out
}

run() {            # run LOGNAME SCRIPT ARGS...  (env already set)
  local log=$1; shift
  guard
  echo "start $log $(date)" >> logs/done.txt
  nice -n 10 $PY -u "$@" > logs/$log.log 2>&1
  echo "$? $log $(date)" >> logs/done.txt
}

setup() {
  [ -d code/src ] || { echo "rsync the model-reeval tree to $S/code first"; exit 1; }
  # DB: online backup of prod, source opened read-only (~1-2 min, ~1.2 GB)
  $PY - <<EOF
import sqlite3
src = sqlite3.connect("file:$PROD/courtside.db?mode=ro", uri=True)
dst = sqlite3.connect("$S/data/courtside.db")
src.backup(dst)
dst.close(); src.close()
EOF
  # bundles: live-stack bundles of past seasons (c1rship) + prod 2026-27
  cp $C1R/bundle_20{16-17,17-18,18-19,19-20,20-21,23-24,24-25,25-26}_v1_cx_mf_ep_ms_c1r.json data/proj/
  cp $C1R/corr_20*.json data/proj/
  cp $PROD/proj/bundle_2026-27_v1_cx_mf_ep_ms_c1r.json $PROD/proj/corr_2026-27.json data/proj/
  cp $PROD/proj/minutes_params_2026-27_v1_mf_ms.json data/proj/ 2>/dev/null
  # context feature caches (copies: a cache file is never written through a link into prod)
  mkdir -p data/proj/features
  cp $PROD/proj/features/frame_2026-27_*.parquet data/proj/features/
  cp $C1R/features/*2025-26*.parquet data/proj/features/ 2>/dev/null
  # tip times of 2025-26 (the slice runs each game at its own tip; an empty raw dir = 19:30 ET)
  mkdir -p data/raw/draftkings && rsync -a $PROD/raw/espn_odds data/raw/ \
      && rsync -a $PROD/raw/draftkings/rules data/raw/draftkings/
  cp $PQ_LOMO out/pq_lomo.json
  # one data dir per slice arm (separate per-date caches), sharing DB / bundles / raw / features
  for arm in off rm lu rmlu; do
    mkdir -p data_$arm/proj
    ln -sf $S/data/courtside.db data_$arm/courtside.db
    ln -sfn $S/data/raw data_$arm/raw
    ln -sfn $S/data/proj/features data_$arm/proj/features
    for f in data/proj/*.json; do ln -sf $S/$f data_$arm/proj/; done
  done
  echo "setup done $(date)" >> logs/done.txt
}

lineup() {         # ~45-75 min: 6 season frames (prepare_season) + 64-point grid x 6 seasons
  DATA=$S/data
  live_env
  run lineup code/research/reeval_a_lineup.py all 2023-24 2024-25 2025-26 2026-27 --seasons 3 \
      --eval 2023-24,2024-25,2025-26
  # the fitted weights now live in code/src/courtside/proj/lineup_params.json (package file)
}

replay() {         # ~3-5 min per arm x protocol at 4,000 sims (geekom1: 15 s per night at 1,000)
  DATA=$S/data
  live_env
  for proto in lock10 lock45; do
    for arm in off gh rm lu all; do
      run replay_${proto}_$arm code/research/reeval_a_live_replay.py run $arm --protocol $proto \
          --sims 4000 --db $S/data/courtside.db
    done
  done
}

slice() {          # ~14 min per arm x protocol (c1rship live10 run: 825 s)
  for spec in "off live" "off live10" "rm live" "rm live10" "lu live10" "rmlu live10"; do
    arm=${spec%% *}; proto=${spec##* }
    DATA=$S/data_$arm
    live_env
    [[ $arm == rm || $arm == rmlu ]] && export COURTSIDE_REPORT_MATCH=1
    [[ $arm == lu || $arm == rmlu ]] && export COURTSIDE_LINEUP=1
    run slice_${arm}_$proto code/research/c1r_ship_prep.py harness 2025-26 $proto \
        --tag _$arm --pq-lomo $S/out/pq_lomo.json
  done
}

compare() {        # minutes; reports only
  DATA=$S/data
  live_env
  run replay_report10 code/research/reeval_a_live_replay.py report --protocol lock10
  run replay_report45 code/research/reeval_a_live_replay.py report --protocol lock45
  local B=pathA-v0.3+mf+ep+ms+c1r+p2+pq+rx+rb+tr
  for spec in "rm live" "rm live10" "lu live10" "rmlu live10"; do
    arm=${spec%% *}; proto=${spec##* }
    on=$(ls out/rows_${proto}_2025-26_*_${arm}.parquet | head -1)
    run slicecmp_${arm}_$proto code/research/p8_e3_eval.py dk slice_${arm}_$proto \
        out/absent_${proto}_2025-26_${B}_off.parquet out/rows_${proto}_2025-26_${B}_off.parquet $on
  done
}

case ${1:-} in
  setup) setup ;;
  lineup) lineup ;;
  replay) replay ;;
  slice) slice ;;
  compare) compare ;;
  all) lineup; replay; slice; compare ;;
  *) sed -n '2,12p' $0; exit 2 ;;
esac
