#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — One command to set up and run Knowledge Companion
#
# First run:  installs every dependency, sets up the database, pulls AI models
# Every run:  skips what's already done, starts backend + frontend
#
# Usage:
#   ./start.sh           — start the app
#   ./start.sh --stop    — stop background processes
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✔${NC}  $*"; }
info() { echo -e "${YELLOW}→${NC}  $*"; }
die()  { echo -e "${RED}✘  $*${NC}"; exit 1; }
head() { echo -e "\n${BOLD}$*${NC}"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$ROOT/.pids"
LOG_DIR="$ROOT/logs"
VENV="$ROOT/backend/.venv"

# ── --stop flag ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  echo "Stopping Knowledge Companion..."
  for f in "$PID_DIR"/*.pid; do
    [[ -f "$f" ]] || continue
    pid=$(<"$f")
    name=$(basename "$f" .pid)
    if kill "$pid" 2>/dev/null; then
      ok "Stopped $name (pid $pid)"
    fi
    rm -f "$f"
  done
  exit 0
fi

mkdir -p "$PID_DIR" "$LOG_DIR"

echo ""
echo -e "${BOLD}  Knowledge Companion — Setup & Start${NC}"
echo "  ────────────────────────────────────"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Homebrew
# ─────────────────────────────────────────────────────────────────────────────
head "1/6  Homebrew"
if ! command -v brew &>/dev/null; then
  info "Installing Homebrew (this may ask for your password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Apple Silicon homebrew lives at /opt/homebrew
  eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
else
  ok "Homebrew already installed"
fi

# Make sure brew is on PATH for this session
eval "$(brew shellenv)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. PostgreSQL + pgvector
# ─────────────────────────────────────────────────────────────────────────────
head "2/6  PostgreSQL + pgvector"

PG_VERSION="16"

if ! brew list "postgresql@$PG_VERSION" &>/dev/null; then
  info "Installing PostgreSQL $PG_VERSION..."
  brew install "postgresql@$PG_VERSION"
else
  ok "PostgreSQL $PG_VERSION already installed"
fi

# Add pg binaries to PATH for this session
PG_BIN="$(brew --prefix "postgresql@$PG_VERSION")/bin"
export PATH="$PG_BIN:$PATH"

# pgvector must be compiled against the exact PostgreSQL we're using.
# `brew install pgvector` links to the default postgresql, not postgresql@16,
# so we build from source with the correct pg_config instead.
PG_CONFIG="$(brew --prefix "postgresql@$PG_VERSION")/bin/pg_config"
PGVECTOR_VERSION="0.8.0"

if ! "$PG_CONFIG" --pkglibdir &>/dev/null || \
   ! ls "$("$PG_CONFIG" --sharedir)/extension/vector.control" &>/dev/null; then
  info "Building pgvector $PGVECTOR_VERSION for PostgreSQL $PG_VERSION..."
  TMP=$(mktemp -d)
  curl -fsSL "https://github.com/pgvector/pgvector/archive/refs/tags/v${PGVECTOR_VERSION}.tar.gz" \
    | tar -xz -C "$TMP"
  cd "$TMP/pgvector-${PGVECTOR_VERSION}"
  make PG_CONFIG="$PG_CONFIG" >/dev/null
  make install PG_CONFIG="$PG_CONFIG" >/dev/null
  cd "$ROOT"
  rm -rf "$TMP"
  ok "pgvector built and installed"
else
  ok "pgvector already installed for PostgreSQL $PG_VERSION"
fi

# Start PostgreSQL if not running
if ! pg_isready -q 2>/dev/null; then
  info "Starting PostgreSQL service..."
  brew services start "postgresql@$PG_VERSION"
  # Wait up to 10 seconds for it to be ready
  for i in {1..10}; do
    pg_isready -q && break
    sleep 1
  done
  pg_isready -q || die "PostgreSQL failed to start. Check: brew services list"
fi
ok "PostgreSQL is running"

# Homebrew PostgreSQL uses the macOS username as superuser, not "postgres".
# Our app connects as "postgres", so create that role if it doesn't exist.
if ! psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='postgres'" postgres 2>/dev/null | grep -q 1; then
  info "Creating 'postgres' superuser role..."
  createuser -s postgres
fi
ok "postgres role ready"

# Create the database and enable pgvector (safe to run repeatedly)
DB="knowledge_companion"
if ! psql -lqt 2>/dev/null | cut -d'|' -f1 | grep -qw "$DB"; then
  info "Creating database '$DB'..."
  createdb "$DB"
fi
psql "$DB" -c "CREATE EXTENSION IF NOT EXISTS vector;" -q
ok "Database '$DB' ready"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Ollama + AI models
# ─────────────────────────────────────────────────────────────────────────────
head "3/6  Ollama + AI models"

if ! command -v ollama &>/dev/null; then
  info "Installing Ollama..."
  brew install ollama
else
  ok "Ollama already installed"
fi

# Start Ollama server if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  info "Starting Ollama..."
  ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
  echo $! > "$PID_DIR/ollama.pid"
  sleep 2
fi
ok "Ollama is running"

# Pull required models (skips if already present)
for model in "nomic-embed-text" "llama3.2:3b" "llama3.1:8b"; do
  if ollama list 2>/dev/null | grep -q "^$model"; then
    ok "Model $model already downloaded"
  else
    info "Downloading $model (this only happens once)..."
    ollama pull "$model"
    ok "Model $model ready"
  fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 4. Python backend
# ─────────────────────────────────────────────────────────────────────────────
head "4/6  Python backend"

# Pin to Python 3.12 — Pillow ships pre-built wheels for it.
# Python 3.13+ has no pre-built Pillow wheel yet, causing a source build that
# requires system C libraries and frequently fails.
if ! brew list python@3.12 &>/dev/null; then
  info "Installing Python 3.12..."
  brew install python@3.12
else
  ok "Python 3.12 already installed"
fi
PYTHON="$(brew --prefix python@3.12)/bin/python3.12"

# System libraries required by Pillow (image processing) and tesseract (OCR).
# These are C libraries — pip cannot install them, brew must.
for lib in jpeg libtiff openjpeg little-cms2 tesseract; do
  if ! brew list "$lib" &>/dev/null; then
    info "Installing system library: $lib..."
    brew install "$lib"
  fi
done
ok "System image libraries ready"

# Delete venv if it was created with the wrong Python (e.g. 3.14)
if [[ -f "$VENV/bin/python" ]]; then
  VENV_PY=$("$VENV/bin/python" --version 2>&1 | grep -o '3\.[0-9]*')
  if [[ "$VENV_PY" != "3.12" ]]; then
    info "Recreating venv with Python 3.12 (was $VENV_PY)..."
    rm -rf "$VENV"
  fi
fi

if [[ ! -f "$VENV/bin/activate" ]]; then
  info "Creating Python virtual environment..."
  "$PYTHON" -m venv "$VENV"
fi

# Install / upgrade Python dependencies
info "Installing Python dependencies..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$ROOT/backend/requirements.txt"
ok "Python dependencies ready"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Node.js frontend
# ─────────────────────────────────────────────────────────────────────────────
head "5/6  Node.js frontend"

if ! command -v node &>/dev/null; then
  info "Installing Node.js..."
  brew install node
else
  ok "Node.js $(node --version) already installed"
fi

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  info "Installing npm packages..."
  npm install --prefix "$ROOT/frontend" --silent
else
  ok "npm packages already installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Start backend + frontend
# ─────────────────────────────────────────────────────────────────────────────
head "6/6  Starting the app"

# Kill any leftover processes from a previous run — use port-based kill
# so stale processes that lost their PID file are also cleaned up
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :5457 | xargs kill -9 2>/dev/null || true
rm -f "$PID_DIR/backend.pid" "$PID_DIR/frontend.pid"
sleep 1

# Start FastAPI backend
info "Starting backend (port 8000)..."
cd "$ROOT/backend"
"$VENV/bin/uvicorn" main:app --port 8000 --log-level warning \
  >"$LOG_DIR/backend.log" 2>&1 &
echo $! > "$PID_DIR/backend.pid"
cd "$ROOT"

# Wait for backend to be ready
for i in {1..15}; do
  curl -s http://localhost:8000/api/health &>/dev/null && break
  sleep 1
done
curl -s http://localhost:8000/api/health &>/dev/null || die "Backend failed to start. Check logs/backend.log"
ok "Backend running at http://localhost:8000"

# Start React frontend
info "Starting frontend (port 5457)..."
npm run dev --prefix "$ROOT/frontend" \
  >"$LOG_DIR/frontend.log" 2>&1 &
echo $! > "$PID_DIR/frontend.pid"

# Wait for frontend
for i in {1..20}; do
  curl -s http://localhost:5457 &>/dev/null && break
  sleep 1
done

ok "Frontend running at http://localhost:5457"

# Open browser automatically
sleep 1
open "http://localhost:5457" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  Research Companion is running!${NC}"
echo "  ────────────────────────────────────────────────────────────"
echo "  App:      http://localhost:5457"
echo "  API:      http://localhost:8000"
echo "  Logs:     ./logs/backend.log  ./logs/frontend.log"
echo ""
echo "  To ingest documents:"
echo "    ./ingest.sh /path/to/any/folder"
echo ""
echo "  To re-ingest after changing entity/relation types in config.py:"
echo "    ./ingest.sh /path/to/any/folder --force"
echo ""
echo "  To update:  ./update.sh"
echo "  To stop:    ./start.sh --stop"
echo "  ────────────────────────────────────────────────────────────"
echo ""
