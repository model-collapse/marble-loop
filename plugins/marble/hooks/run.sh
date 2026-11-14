#!/bin/sh
# Run a `marble` subcommand for the plugin (MCP server or hook handler).
# Finds the CLI via $MARBLE_BIN, then PATH. If MARBLE is not installed, hooks become no-ops
# (exit 0, empty JSON) instead of erroring in every session; the guard keeps exit code 2 when it blocks.
MARBLE="${MARBLE_BIN:-$(command -v marble 2>/dev/null)}"
if [ -z "$MARBLE" ]; then
  if [ "$1" = "mcp" ]; then
    echo "marble CLI not found: install it (pip install marble-loop) or set MARBLE_BIN" >&2
    exit 1
  fi
  cat >/dev/null
  echo "{}"
  exit 0
fi
exec "$MARBLE" "$@"
