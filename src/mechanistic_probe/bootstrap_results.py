from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .probe import _evaluate, _load, _paired_delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Add cluster-bootstrap intervals to an existing probe result")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["paper", "strict"], required=True)
    parser.add_argument("--result-name", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    result_path = args.root / "results" / f"{args.result_name}.json"
    result: dict[str, Any] = json.loads(result_path.read_text())
    rows_by_model = {
        model: _load(args.root / "artifacts" / f"features-{model}.jsonl")
        for model in args.models
    }

    for model, rows in rows_by_model.items():
        result["models"][model]["overall"] = {
            task: {
                classifier: _evaluate(
                    rows, task, args.mode, classifier, args.iterations
                )
                for classifier in ("knn", "linear")
            }
            for task in ("sp1", "sp2")
        }

    result["paired_differences"] = {}
    for index, model_a in enumerate(args.models):
        for model_b in args.models[index + 1 :]:
            comparison = f"{model_a}_minus_{model_b}"
            result["paired_differences"][comparison] = {
                task: {
                    classifier: _paired_delta(
                        rows_by_model[model_a],
                        rows_by_model[model_b],
                        task,
                        args.mode,
                        classifier,
                        args.iterations,
                    )
                    for classifier in ("knn", "linear")
                }
                for task in ("sp1", "sp2")
            }
    result["bootstrap_iterations"] = args.iterations

    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(result_path)
    print(result_path)


if __name__ == "__main__":
    main()
