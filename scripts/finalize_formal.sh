#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the experiment-artifact directory}"
CONDA_BIN=${CONDA_BIN:-conda}
PY=("$CONDA_BIN" run -n "${CONDA_ENV:-grounding-mechprobe}" python)
ROOT=$EXPERIMENT_ROOT
MODELS=(base-formal instruct-formal random-formal)

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for mode in paper strict; do
  test -s "$ROOT/results/probe-formal-$mode.json"
  "${PY[@]}" -m mechanistic_probe.bootstrap_results \
    --root "$ROOT" --mode "$mode" --result-name "probe-formal-$mode" \
    --iterations 1000 --models "${MODELS[@]}"
done

"${PY[@]}" -m mechanistic_probe.report --root "$ROOT"
