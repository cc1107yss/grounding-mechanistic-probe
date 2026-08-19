from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rows_for_task(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    if task == "sp1":
        return rows
    return [row for row in rows if row["proof_depth"] == 1 and row["gold_useful"] == 1 and row["gold_height"] is not None]


def _labels(rows: list[dict[str, Any]], task: str) -> np.ndarray:
    key = "gold_useful" if task == "sp1" else "gold_height"
    return np.asarray([int(row[key]) for row in rows])


def _macro_f1_from_confusion(confusion: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(confusion.shape[0]):
        true_positive = float(confusion[index, index])
        false_positive = float(confusion[:, index].sum() - true_positive)
        false_negative = float(confusion[index, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _cluster_confusions(
    y: np.ndarray, predictions: np.ndarray, cluster_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    class_index = {int(value): index for index, value in enumerate(classes)}
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    matrices = np.zeros((int(inverse.max()) + 1, len(classes), len(classes)), dtype=np.int64)
    y_index = np.asarray([class_index[int(value)] for value in y])
    prediction_index = np.asarray([class_index[int(value)] for value in predictions])
    np.add.at(matrices, (inverse, y_index, prediction_index), 1)
    return classes, matrices


def _bootstrap_ci(
    y: np.ndarray,
    predictions: np.ndarray,
    cluster_ids: np.ndarray,
    iterations: int,
    seed: int,
) -> list[float] | None:
    if iterations <= 0:
        return None
    _, matrices = _cluster_confusions(y, predictions, cluster_ids)
    rng = np.random.default_rng(seed)
    probability = np.full(len(matrices), 1.0 / len(matrices))
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        weights = rng.multinomial(len(matrices), probability)
        values[index] = _macro_f1_from_confusion(np.tensordot(weights, matrices, axes=(0, 0)))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _predict(
    rows: list[dict[str, Any]], task: str, mode: str, classifier: str
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows = _rows_for_task(rows, task)
    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = _labels(rows, task)
    groups = np.asarray([row["theory_group"] for row in rows])
    if mode == "paper":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = splitter.split(x, y)
    else:
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        splits = splitter.split(x, y, groups)
    predictions = np.empty(len(rows), dtype=np.int64)
    for train_index, test_index in splits:
        if classifier == "knn":
            model = KNeighborsClassifier(n_neighbors=8, weights="distance", p=1)
        else:
            model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        model.fit(x[train_index], y[train_index])
        predictions[test_index] = model.predict(x[test_index])
    return rows, y, predictions


def _evaluate(
    rows: list[dict[str, Any]],
    task: str,
    mode: str,
    classifier: str,
    bootstrap_iterations: int = 0,
) -> dict[str, Any]:
    rows = _rows_for_task(rows, task)
    y = _labels(rows, task)
    if len(np.unique(y)) < 2:
        return {"n": len(rows), "f1_macro": None, "reason": "only one class"}
    selected_rows, y, predictions = _predict(rows, task, mode, classifier)
    cluster_key = "theory_group" if mode == "strict" else "example_id"
    result: dict[str, Any] = {
        "n": int(len(rows)),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
        "classes": sorted(int(value) for value in np.unique(y)),
    }
    interval = _bootstrap_ci(
        y,
        predictions,
        np.asarray([row[cluster_key] for row in selected_rows]),
        bootstrap_iterations,
        seed=42,
    )
    if interval is not None:
        result.update(
            {
                "ci95": interval,
                "bootstrap_iterations": bootstrap_iterations,
                "bootstrap_unit": cluster_key,
            }
        )
    return result


def _paired_delta(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    task: str,
    mode: str,
    classifier: str,
    iterations: int,
) -> dict[str, Any]:
    selected_a, y_a, predictions_a = _predict(rows_a, task, mode, classifier)
    selected_b, y_b, predictions_b = _predict(rows_b, task, mode, classifier)
    keys_a = [(row["example_id"], row["statement_index"]) for row in selected_a]
    keys_b = [(row["example_id"], row["statement_index"]) for row in selected_b]
    if keys_a != keys_b or not np.array_equal(y_a, y_b):
        raise ValueError("Paired model artifacts are not aligned")
    cluster_key = "theory_group" if mode == "strict" else "example_id"
    clusters = np.asarray([row[cluster_key] for row in selected_a])
    _, matrices_a = _cluster_confusions(y_a, predictions_a, clusters)
    _, matrices_b = _cluster_confusions(y_b, predictions_b, clusters)
    if matrices_a.shape != matrices_b.shape:
        raise ValueError("Paired bootstrap cluster matrices are not aligned")
    observed = float(
        f1_score(y_a, predictions_a, average="macro")
        - f1_score(y_b, predictions_b, average="macro")
    )
    rng = np.random.default_rng(42)
    probability = np.full(len(matrices_a), 1.0 / len(matrices_a))
    values = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        weights = rng.multinomial(len(matrices_a), probability)
        score_a = _macro_f1_from_confusion(np.tensordot(weights, matrices_a, axes=(0, 0)))
        score_b = _macro_f1_from_confusion(np.tensordot(weights, matrices_b, axes=(0, 0)))
        values[index] = score_a - score_b
    return {
        "delta_f1_macro": observed,
        "ci95": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        "bootstrap_iterations": iterations,
        "bootstrap_unit": cluster_key,
    }


def _task_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        examples.setdefault(row["example_id"], row)
    values = list(examples.values())
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_bucket[int(row["statement_count"])].append(row)
    return {
        "n": len(values),
        "accuracy": float(np.mean([row["correct"] for row in values])),
        "by_bucket": {
            str(bucket): {
                "n": len(bucket_rows),
                "accuracy": float(np.mean([row["correct"] for row in bucket_rows])),
            }
            for bucket, bucket_rows in sorted(by_bucket.items())
        },
    }


def _layer_curve(rows: list[dict[str, Any]], task: str, mode: str) -> list[float | None]:
    rows = _rows_for_task(rows, task)
    number_of_layers = len(rows[0]["features"]) if rows else 0
    curve: list[float | None] = []
    for layer in range(1, number_of_layers + 1):
        copied = [{**row, "features": row["features"][:layer]} for row in rows]
        score = _evaluate(copied, task, mode, "knn")["f1_macro"]
        curve.append(score)
    return curve


def _summarize_by_bucket(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[int(row["statement_count"])].append(row)
    return {
        str(bucket): {
            task: {
                classifier: _evaluate(bucket_rows, task, mode, classifier)
                for classifier in ("knn", "linear")
            }
            for task in ("sp1", "sp2")
        }
        for bucket, bucket_rows in sorted(by_bucket.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit frozen-model MechanisticProbe classifiers")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["paper", "strict"], default="paper")
    parser.add_argument("--models", nargs="+", default=["base", "instruct", "random"], help="Artifact names")
    parser.add_argument("--result-name", help="Output stem; defaults to probe-<mode>")
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    args = parser.parse_args()

    results: dict[str, Any] = {"mode": args.mode, "models": {}, "paired_differences": {}}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_id in args.models:
        path = args.root / "artifacts" / f"features-{model_id}.jsonl"
        rows = _load(path)
        rows_by_model[model_id] = rows
        results["models"][model_id] = {
            "task_accuracy": _task_accuracy(rows),
            "overall": {
                task: {
                    classifier: _evaluate(
                        rows, task, args.mode, classifier, args.bootstrap_iterations
                    )
                    for classifier in ("knn", "linear")
                }
                for task in ("sp1", "sp2")
            },
            "buckets": _summarize_by_bucket(rows, args.mode),
            "layer_curves": {task: _layer_curve(rows, task, args.mode) for task in ("sp1", "sp2")},
        }
    if args.bootstrap_iterations > 0:
        for index, model_a in enumerate(args.models):
            for model_b in args.models[index + 1 :]:
                comparison = f"{model_a}_minus_{model_b}"
                results["paired_differences"][comparison] = {
                    task: {
                        classifier: _paired_delta(
                            rows_by_model[model_a],
                            rows_by_model[model_b],
                            task,
                            args.mode,
                            classifier,
                            args.bootstrap_iterations,
                        )
                        for classifier in ("knn", "linear")
                    }
                    for task in ("sp1", "sp2")
                }
    result_name = args.result_name or f"probe-{args.mode}"
    path = args.root / "results" / f"{result_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(path)


if __name__ == "__main__":
    main()
