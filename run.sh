set -e
cd "$(dirname "$0")"

PY=python3
if ! "$PY" -c "import numpy, scipy, soundfile, librosa, matplotlib, tensorflow" >/dev/null 2>&1; then
  if [ ! -x .venv/bin/python ]; then
    echo "[setup] Creating virtual environment in .venv ..."
    "$PY" -m venv .venv
  fi
  PY=.venv/bin/python
  if [ ! -f .venv/.installed ]; then
    echo "[setup] Installing dependencies - first run only, a few minutes ..."
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r requirements.txt
    echo ok > .venv/.installed
  fi
fi

if [ "$1" = "record" ]; then
  shift
  exec "$PY" record.py "$@"
fi

AUDIO="${1:-audio/street_3min.wav}"
OUT="${2:-output}"
[ -f "$AUDIO" ] || { echo "error: no such file: $AUDIO" >&2; exit 1; }
exec "$PY" what_happened_here.py "$AUDIO" --outdir "$OUT"
