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

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export RANDOM_MODEL_REFERENCE="$BASE"

"${PY[@]}" -m mechanistic_probe.prepare --root "$ROOT" --limit-per-bucket 64
if [[ ! -s "$ROOT/artifacts/features-base.jsonl" ]]; then
  "${PY[@]}" -m mechanistic_probe.extract --root "$ROOT" --model-id "$BASE" --artifact-name base
fi
"${PY[@]}" -m mechanistic_probe.extract --root "$ROOT" --model-id "$INSTRUCT_MODEL_PATH" --artifact-name instruct
if [[ ! -s "$ROOT/artifacts/features-random.jsonl" ]]; then
  "${PY[@]}" -m mechanistic_probe.extract --root "$ROOT" --model-id __random__ --artifact-name random
fi
"${PY[@]}" -m mechanistic_probe.probe --root "$ROOT" --mode paper --models base instruct random
"${PY[@]}" -m mechanistic_probe.probe --root "$ROOT" --mode strict --models base instruct random
