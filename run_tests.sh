set -e

cd `dirname "$0"`
flake8 punctuation_corrector tests && python3 -m unittest discover tests || echo "Unit tests failed"
