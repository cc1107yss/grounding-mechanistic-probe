#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the experiment-artifact directory}"
: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to a complete Qwen2.5-7B snapshot}"
: "${INSTRUCT_MODEL_PATH:?Set INSTRUCT_MODEL_PATH to a complete Qwen2.5-7B-Instruct snapshot}"
CONDA_BIN=${CONDA_BIN:-conda}
PY=("$CONDA_BIN" run -n "${CONDA_ENV:-grounding-mechprobe}" python)
ROOT=$EXPERIMENT_ROOT
BASE=$BASE_MODEL_PATH
INSTRUCT=$INSTRUCT_MODEL_PATH

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export RANDOM_MODEL_REFERENCE="$BASE"

"${PY[@]}" -m mechanistic_probe.prepare \
  --root "$ROOT" --sample-name formal --limit-per-bucket 1024 --seed 42

if [[ ! -s "$ROOT/artifacts/features-base-formal.jsonl" || ! -f "$ROOT/artifacts/features-base-formal.skipped.json" ]]; then
  "${PY[@]}" -m mechanistic_probe.extract \
    --root "$ROOT" --sample-name formal --model-id "$BASE" --artifact-name base-formal --overwrite
fi

if [[ ! -s "$ROOT/artifacts/features-instruct-formal.jsonl" || ! -f "$ROOT/artifacts/features-instruct-formal.skipped.json" ]]; then
  "${PY[@]}" -m mechanistic_probe.extract \
    --root "$ROOT" --sample-name formal --model-id "$INSTRUCT" --artifact-name instruct-formal --overwrite
fi

if [[ ! -s "$ROOT/artifacts/features-random-formal.jsonl" || ! -f "$ROOT/artifacts/features-random-formal.skipped.json" ]]; then
  "${PY[@]}" -m mechanistic_probe.extract \
    --root "$ROOT" --sample-name formal --model-id __random__ \
    --artifact-name random-formal --random-seed 42 --overwrite
fi

"${PY[@]}" -m mechanistic_probe.probe \
  --root "$ROOT" --mode paper --result-name probe-formal-paper \
  --models base-formal instruct-formal random-formal
"${PY[@]}" -m mechanistic_probe.probe \
  --root "$ROOT" --mode strict --result-name probe-formal-strict \
  --models base-formal instruct-formal random-formal
bash "$REPO_ROOT/scripts/finalize_formal.sh"
