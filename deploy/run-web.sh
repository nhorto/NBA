#!/usr/bin/env bash
# Serve the Courtside web app. Used by the launchd / systemd web services.
#
# Binds to COURTSIDE_HOST if set, otherwise this machine's Tailscale IPv4
# (so only tailnet devices can reach it), falling back to 127.0.0.1.
# Port: COURTSIDE_PORT, default 8765.

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 1
repo_root="$(cd "$script_dir/.." && pwd -P)" || exit 1
cd "$repo_root" || exit 1

export PATH="/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:$PATH"

uv_bin=""
for c in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv "$(command -v uv 2>/dev/null)" \
         "$(mise which uv 2>/dev/null)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then uv_bin="$c"; break; fi
done
if [ -z "$uv_bin" ]; then
  echo "run-web: uv not found" >&2
  exit 127
fi

host="${COURTSIDE_HOST:-}"
if [ -z "$host" ]; then
  # On macOS the app-bundled CLI must be called via its wrapper, not a symlink.
  for ts in /usr/local/bin/tailscale "$(command -v tailscale 2>/dev/null)"; do
    if [ -n "$ts" ] && [ -x "$ts" ]; then
      host="$("$ts" ip -4 2>/dev/null | head -n1)"
      [ -n "$host" ] && break
    fi
  done
fi
host="${host:-127.0.0.1}"
port="${COURTSIDE_PORT:-8765}"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [run-web] serving on http://$host:$port"
exec "$uv_bin" run --frozen courtside serve --host "$host" --port "$port"
