#!/bin/zsh
# Re-evaluation workstream B (docs/research/reeval-B.md): the per-minute ablation of H1-H4 on
# the Mac mini, in scratch ~/Developer/courtside-compute/reeval-b/. One niced, single-threaded
# process; seasons run one after another. Never reads or writes ~/Developer/NBA/var.
#
# 0. From geekom1 (pushes the working tree: nothing of B is committed):
#      rsync -a --delete --exclude .git --exclude .venv --exclude var --exclude __pycache__ \
#        ~/Documents/NBA/ nicholashorton@mac-mini:Developer/courtside-compute/reeval-b/code/
#      ssh nicholashorton@mac-mini 'mkdir -p ~/Developer/courtside-compute/reeval-b/data/proj/features'
#      scp ~/Documents/NBA/var/tmp/reeval-b/data/proj/features/state_params_*.json \
#        nicholashorton@mac-mini:Developer/courtside-compute/reeval-b/data/proj/features/
#      scp ~/Documents/NBA/var/proj/corr_extract_{2023-24,2024-25,2025-26}.pkl \
#        nicholashorton@mac-mini:Developer/courtside-compute/reeval-b/data/proj/
# 1. On the Mac:  cd ~/Developer/courtside-compute/reeval-b && zsh code/research/reeval_b_mac.sh setup
# 2. On the Mac (not 12:00-12:50 or 15:45-19:45 ET):
#      nohup zsh code/research/reeval_b_mac.sh run > logs/run.log 2>&1 &
#    ~45 min on an idle Mac (vc ~5 s, live ~7 min, ext ~7 min per season, 3 seasons), 2-3x that
#    under load. Results: out/reeval_b_{vc,live,ext}_<S>.{txt,json}.
set -e
cd ~/Developer/courtside-compute/reeval-b
CC=~/Developer/courtside-compute
case "$1" in
  setup)
    mkdir -p data/proj/features data/raw/sportsdataverse out logs
    # cached base frames (the cache keys match the state params pushed from geekom1)
    cp -c $CC/xf/data/proj/features/frame_2023-24_fe24174165.parquet data/proj/features/
    cp -c $CC/xf/data/proj/features/frame_2024-25_2e386b7e88.parquet data/proj/features/
    cp -c $CC/pq/data_v1/proj/features/frame_2025-26_8f88c15a91.parquet data/proj/features/
    # officials history (H3) and a DB copy (historical seasons only are read; APFS clone)
    cp -cR ~/Developer/NBA/var/raw/sportsdataverse/nba_stats_officials data/raw/sportsdataverse/
    cp -c $CC/xf/data/courtside.db data/courtside.db
    ls -la data data/proj data/proj/features
    ;;
  run)
    shift
    export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1
    export COURTSIDE_DATA=$PWD/data PYTHONPATH=$PWD/code/src COURTSIDE_NO_WRITER=1
    export COURTSIDE_CONTEXT_B=1
    for S in ${@:-2023-24 2024-25 2025-26}; do
      nice -n 10 ~/Developer/NBA/.venv/bin/python -u code/research/reeval_b_ablation.py $S \
        --stage all --db data/courtside.db --out out > logs/reeval_b_$S.log 2>&1
      echo "done $S $(date)"
    done
    ;;
  *)
    echo "usage: reeval_b_mac.sh setup | run [seasons...]"; exit 1;;
esac
