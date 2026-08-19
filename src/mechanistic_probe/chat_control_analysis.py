from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .probe import (
    _evaluate,
    _layer_curve,
    _load,
    _paired_delta,
    _summarize_by_bucket,
    _task_accuracy,
)


BASE = "base-formal"
RAW_INSTRUCT = "instruct-formal"
CHAT_INSTRUCT = "instruct-chat-formal"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the native Instruct chat-template control")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("paper", "strict"), required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    original_path = args.root / "results" / f"probe-formal-{args.mode}.json"
    original: dict[str, Any] = json.loads(original_path.read_text())
    rows = {
        name: _load(args.root / "artifacts" / f"features-{name}.jsonl")
        for name in (BASE, RAW_INSTRUCT, CHAT_INSTRUCT)
    }
    chat_rows = rows[CHAT_INSTRUCT]
    chat_result = {
        "task_accuracy": _task_accuracy(chat_rows),
        "overall": {
            task: {
                classifier: _evaluate(
                    chat_rows,
                    task,
                    args.mode,
                    classifier,
                    args.bootstrap_iterations,
                )
                for classifier in ("knn", "linear")
            }
            for task in ("sp1", "sp2")
        },
        "buckets": _summarize_by_bucket(chat_rows, args.mode),
        "layer_curves": {
            task: _layer_curve(chat_rows, task, args.mode) for task in ("sp1", "sp2")
        },
    }
    result: dict[str, Any] = {
        "mode": args.mode,
        "protocol": {
            "prompt_control": "Qwen2.5-Instruct official chat template",
            "chat_format": "default system + four user/assistant demonstrations + final user + generation prompt",
            "candidate_strings": ["True", "False"],
            "bootstrap_iterations": args.bootstrap_iterations,
            "reused_model_artifacts": [BASE, RAW_INSTRUCT],
            "new_model_artifact": CHAT_INSTRUCT,
        },
        "models": {
            BASE: original["models"][BASE],
            RAW_INSTRUCT: original["models"][RAW_INSTRUCT],
            CHAT_INSTRUCT: chat_result,
        },
        "paired_differences": {},
    }
    for first, second in ((BASE, CHAT_INSTRUCT), (RAW_INSTRUCT, CHAT_INSTRUCT)):
        name = f"{first}_minus_{second}"
        result["paired_differences"][name] = {
            task: {
                classifier: _paired_delta(
                    rows[first],
                    rows[second],
                    task,
                    args.mode,
                    classifier,
                    args.bootstrap_iterations,
                )
                for classifier in ("knn", "linear")
            }
            for task in ("sp1", "sp2")
        }

    output = args.root / "results" / f"probe-chat-control-{args.mode}.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
