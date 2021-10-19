set -e

ROOT=`dirname "$0"`
flake8 "$ROOT/punctuation_corrector" && python3 -m unittest discover "$ROOT/tests" || echo "Unit tests failed"
