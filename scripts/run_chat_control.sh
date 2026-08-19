#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the existing formal experiment directory}"
: "${INSTRUCT_MODEL_PATH:?Set INSTRUCT_MODEL_PATH to a complete Qwen2.5-7B-Instruct snapshot}"
CONDA_BIN=${CONDA_BIN:-conda}
PY=("$CONDA_BIN" run -n "${CONDA_ENV:-grounding-mechprobe}" python)
ROOT=$EXPERIMENT_ROOT
CHAT_ARTIFACT="$ROOT/artifacts/features-instruct-chat-formal.jsonl"
CHAT_MARKER="$ROOT/artifacts/features-instruct-chat-formal.skipped.json"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

"${PY[@]}" -m mechanistic_probe.validate_chat_control \
  --root "$ROOT" --phase preflight --model-id "$INSTRUCT_MODEL_PATH" --max-length 2048

if [[ ! -s "$CHAT_ARTIFACT" || ! -f "$CHAT_MARKER" ]]; then
  "${PY[@]}" -m mechanistic_probe.extract \
    --root "$ROOT" --sample-name formal \
    --model-id "$INSTRUCT_MODEL_PATH" \
    --artifact-name instruct-chat-formal \
    --prompt-format chat-multiturn \
    --max-length 2048 --overwrite
fi

"${PY[@]}" -m mechanistic_probe.validate_chat_control --root "$ROOT" --phase postflight

for mode in paper strict; do
  "${PY[@]}" -m mechanistic_probe.chat_control_analysis \
    --root "$ROOT" --mode "$mode" --bootstrap-iterations 1000
done

"${PY[@]}" -m mechanistic_probe.chat_control_report --root "$ROOT"
