#!/usr/bin/env bash
# Nightly off-machine backup of the Courtside data on the Mac mini, PULLED by geekom1.
#
# 1. A consistent snapshot of var/courtside.db made on the Mac with SQLite's online backup API
#    from a read-only connection (safe while jobs run), integrity-checked, gzipped, copied here
#    as db/courtside-YYYYMMDD.db.gz; the newest $KEEP snapshots are kept.
# 2. var/ mirrored here (raw archives, sims, proj bundles, models, briefs, reports, logs, ...)
#    WITHOUT --delete, so a file removed on the Mac is still kept here. tmp/ and the live
#    database files are skipped (the snapshot covers the database).
#
# Runs from the systemd user timer deploy/geekom1/courtside-backup.timer (04:30 ET, after the
# 03:00 postgame grading; Persistent=true catches up after the machine was off).
# Status: $DEST/last_backup.txt; log: journalctl --user -u courtside-backup.
set -euo pipefail

MAC="${COURTSIDE_BACKUP_HOST:-nicholashorton@mac-mini}"
REMOTE_REPO="Developer/NBA"                      # relative to the Mac home
REMOTE_TMP="Developer/courtside-backup-tmp"
DEST="${COURTSIDE_BACKUP_DIR:-$HOME/Backups/courtside}"
KEEP="${COURTSIDE_BACKUP_KEEP:-7}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
day="$(date +%Y%m%d)"
started="$(date -Is)"

mkdir -p "$DEST/db" "$DEST/var"
fail() { echo "$(date -Is) FAILED: $*" | tee "$DEST/last_backup.txt" >&2; exit 1; }

# 1. database snapshot on the Mac
"${SSH[@]}" "$MAC" "set -e; mkdir -p ~/$REMOTE_TMP; cd ~/$REMOTE_TMP; rm -f snap.db snap.db.gz
  nice -n 10 ~/$REMOTE_REPO/.venv/bin/python - <<'PY'
import os, sqlite3, sys
src = sqlite3.connect('file:' + os.path.expanduser('~/$REMOTE_REPO/var/courtside.db') + '?mode=ro', uri=True)
dst = sqlite3.connect('snap.db')
src.backup(dst)
ok = dst.execute('pragma quick_check').fetchone()[0]
dst.close(); src.close()
sys.exit(0 if ok == 'ok' else 1)
PY
  nice -n 10 gzip -1 snap.db" || fail "snapshot on the Mac"

part="$DEST/db/courtside-$day.db.gz.part"
rsync -a -e "${SSH[*]}" "$MAC:$REMOTE_TMP/snap.db.gz" "$part" || fail "copy of the snapshot"
gzip -t "$part" || fail "snapshot gzip check"
mv -f "$part" "$DEST/db/courtside-$day.db.gz"
"${SSH[@]}" "$MAC" "rm -f ~/$REMOTE_TMP/snap.db.gz" || true

# 2. mirror var/ (no --delete)
rsync -a -e "${SSH[*]}" --exclude '/courtside.db' --exclude '/courtside.db-*' --exclude '/tmp/' \
  "$MAC:$REMOTE_REPO/var/" "$DEST/var/" || fail "var/ mirror"

# 3. keep the newest $KEEP snapshots
ls -1t "$DEST"/db/courtside-*.db.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -f --

size_db="$(du -h "$DEST/db/courtside-$day.db.gz" | cut -f1)"
size_var="$(du -sh "$DEST/var" | cut -f1)"
echo "$(date -Is) ok: started $started; db snapshot courtside-$day.db.gz ($size_db); var/ mirror $size_var; kept $(ls -1 "$DEST"/db/courtside-*.db.gz | wc -l) snapshots" \
  | tee "$DEST/last_backup.txt"
