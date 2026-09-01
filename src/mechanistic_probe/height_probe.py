from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .probe import _height_class_curves, _load


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute strict SP2 per-height layer curves")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["paper", "strict"], default="strict")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base-formal", "instruct-formal", "instruct-chat-formal"],
    )
    parser.add_argument("--result-name", default="probe-height-class-strict")
    args = parser.parse_args()

    result: dict[str, Any] = {"mode": args.mode, "models": {}}
    for model_id in args.models:
        rows = _load(args.root / "artifacts" / f"features-{model_id}.jsonl")
        result["models"][model_id] = {
            "height_class_curves": _height_class_curves(rows, args.mode)
        }
        print(model_id, result["models"][model_id]["height_class_curves"]["macro"][-1])

    output = args.root / "results" / f"{args.result_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
