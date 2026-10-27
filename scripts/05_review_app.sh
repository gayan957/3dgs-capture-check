#!/usr/bin/env bash
# Step 5: open the human review screen.
set -euo pipefail
cd "$(dirname "$0")/.."
exec streamlit run capturecheck/review/app.py "$@"
