#!/bin/zsh
# macOS: double-click this file to run the app from its own project folder.
set -e
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import flask, sklearn, joblib, certifi' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi
print 'Open http://127.0.0.1:5001 in your browser. Press Ctrl+C to stop.'
exec .venv/bin/python app.py
