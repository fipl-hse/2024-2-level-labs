set -x

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Changed config params"

python config/collect_coverage/coverage_analyzer.py
