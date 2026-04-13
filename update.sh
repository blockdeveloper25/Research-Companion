#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# update.sh — Pull latest changes and sync all dependencies
#
# Safe to run at any time. Stops the running app, pulls updates,
# reinstalls any changed dependencies, and restarts everything.
#
# Usage:
#   ./update.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✔${NC}  $*"; }
info() { echo -e "${YELLOW}→${NC}  $*"; }
die()  { echo -e "${RED}✘  $*${NC}"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/backend/.venv"

echo ""
echo -e "${BOLD}  Knowledge Companion — Update${NC}"
echo "  ────────────────────────────"

# ── 1. Stop the running app ──────────────────────────────────────────────────
info "Stopping running app (if any)..."
"$ROOT/start.sh" --stop 2>/dev/null || true
ok "App stopped"

# ── 2. Save current version for comparison ───────────────────────────────────
OLD_COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")

# ── 3. Pull latest changes ───────────────────────────────────────────────────
info "Pulling latest changes..."
git -C "$ROOT" pull --ff-only || die "Git pull failed. You may have local changes — commit or stash them first."
NEW_COMMIT=$(git -C "$ROOT" rev-parse HEAD)

if [[ "$OLD_COMMIT" == "$NEW_COMMIT" ]]; then
  ok "Already up to date"
  echo ""
  echo "  No changes to apply. Starting the app..."
  echo ""
  exec "$ROOT/start.sh"
fi

# Show what changed
echo ""
info "Changes pulled:"
git -C "$ROOT" log --oneline "$OLD_COMMIT".."$NEW_COMMIT"
echo ""

# ── 4. Sync Python dependencies (if requirements.txt changed) ────────────────
if git -C "$ROOT" diff --name-only "$OLD_COMMIT".."$NEW_COMMIT" | grep -q "backend/requirements.txt"; then
  info "Python dependencies changed — reinstalling..."
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$ROOT/backend/requirements.txt"
  ok "Python dependencies updated"
else
  ok "Python dependencies unchanged"
fi

# ── 5. Sync Node dependencies (if package.json changed) ─────────────────────
if git -C "$ROOT" diff --name-only "$OLD_COMMIT".."$NEW_COMMIT" | grep -q "frontend/package.json"; then
  info "Node dependencies changed — reinstalling..."
  npm install --prefix "$ROOT/frontend" --silent
  ok "Node dependencies updated"
else
  ok "Node dependencies unchanged"
fi

# ── 6. Run database migrations (schema is auto-applied on startup) ───────────
# init_db() in db/connection.py uses CREATE IF NOT EXISTS — safe on every boot.
# If a migration requires re-ingestion, start.sh will print instructions.
ok "Database schema will auto-update on startup"

# ── 7. Show changelog if it was updated ──────────────────────────────────────
if git -C "$ROOT" diff --name-only "$OLD_COMMIT".."$NEW_COMMIT" | grep -q "CHANGELOG.md"; then
  echo ""
  echo -e "${BOLD}  What's new:${NC}"
  echo "  ──────────"
  # Show the first version block (everything up to the second ## heading)
  sed -n '/^## \[/,/^## \[/{/^## \[/!{/^## \[/!p;};}' "$ROOT/CHANGELOG.md" | head -20
  echo ""
fi

# ── 8. Restart the app ──────────────────────────────────────────────────────
echo ""
ok "Update complete — starting the app..."
echo ""
exec "$ROOT/start.sh"
