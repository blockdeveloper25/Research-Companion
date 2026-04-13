#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ingest.sh — Add a folder of PDFs to the knowledge base
#
# Usage:
#   ./ingest.sh /path/to/any/folder          — ingest PDFs from any folder
#   ./ingest.sh ~/Desktop/Legal              — home directory paths work too
#   ./ingest.sh /path/to/folder --force      — re-ingest even if already stored
#   ./ingest.sh --remove filename.pdf        — remove a file from knowledge base
#
# The folder can be anywhere on your system — not limited to documents/.
#
# Examples:
#   ./ingest.sh ~/Documents/HR
#   ./ingest.sh /Users/me/Desktop/Policies --force
#   ./ingest.sh --remove "Annual Leave Policy.pdf"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/backend/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
die() { echo -e "${RED}✘  $*${NC}"; exit 1; }
ok()  { echo -e "${GREEN}✔${NC}  $*"; }

[[ -z "${1:-}" ]] && die "Usage: ./ingest.sh /path/to/folder [--force]  or  ./ingest.sh --remove filename.pdf"

[[ -f "$VENV/bin/activate" ]] || die "Backend not set up. Run ./start.sh first."

# ── Remove mode ──────────────────────────────────────────────────────────────
if [[ "$1" == "--remove" ]]; then
  [[ -z "${2:-}" ]] && die "Usage: ./ingest.sh --remove filename.pdf"
  cd "$ROOT/backend"
  "$VENV/bin/python" ingest.py --remove "$2"
  ok "Done."
  exit 0
fi

# ── Ingest mode ──────────────────────────────────────────────────────────────
FOLDER="$1"
shift
EXTRA_ARGS="$*"

# Expand ~ and resolve to absolute path
FOLDER_PATH="$(cd "$FOLDER" 2>/dev/null && pwd)" \
  || die "Folder not found: $FOLDER"

[[ -d "$FOLDER_PATH" ]] || die "Folder not found: $FOLDER_PATH"

echo ""
echo "  Ingesting: $FOLDER_PATH"
echo "  ──────────────────────────────────────"

cd "$ROOT/backend"
"$VENV/bin/python" ingest.py --folder "$FOLDER_PATH" $EXTRA_ARGS

FOLDER_NAME="$(basename "$FOLDER_PATH")"
ok "Done. Open the app and select '$FOLDER_NAME' in the sidebar to search it."
echo ""
