#!/bin/bash
# Runs the ASX RTO screener, bakes data into dashboard.html,
# then commits and pushes to GitHub Pages automatically.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/miniforge3/bin/python3"
LOG="$SCRIPT_DIR/screener.log"

echo "" >> "$LOG"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

# 1. Run screener (also bakes dashboard.html)
"$PYTHON" "$SCRIPT_DIR/screener.py" >> "$LOG" 2>&1

# 2. Deploy updated dashboard.html to GitHub Pages
cd "$SCRIPT_DIR"
if git rev-parse --git-dir > /dev/null 2>&1; then
    cp dashboard.html index.html
    git add dashboard.html index.html
    if ! git diff --cached --quiet; then
        git commit -m "Dashboard update $(date '+%Y-%m-%d')"
        git push origin main
        echo "Deployed to GitHub Pages at $(date '+%H:%M:%S')" >> "$LOG"
    else
        echo "No dashboard changes to push" >> "$LOG"
    fi
fi
