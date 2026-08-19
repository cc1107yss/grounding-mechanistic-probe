from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"base": "#2563eb", "instruct": "#dc2626", "random": "#6b7280"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _short_name(name: str) -> str:
    return name.removesuffix("-formal")


def _format_metric(metric: dict[str, Any]) -> str:
    score = metric.get("f1_macro")
    if score is None:
        return "NA"
    interval = metric.get("ci95")
    if interval:
        return f"{score:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"
    return f"{score:.4f}"


def _layer_figure(results_by_mode: dict[str, dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey="row")
    for column, mode in enumerate(("paper", "strict")):
        result = results_by_mode[mode]
        for row_index, task in enumerate(("sp1", "sp2")):
            axis = axes[row_index, column]
            for model_name, model_result in result["models"].items():
                label = _short_name(model_name)
                curve = model_result["layer_curves"][task]
                axis.plot(
                    range(1, len(curve) + 1),
                    curve,
                    label=label,
                    color=COLORS.get(label),
                    linewidth=2,
                )
            axis.set_title(f"{mode.capitalize()} split — {task.upper()}")
            axis.set_ylabel("Macro-F1")
            axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Layers included (prefix)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Formal MechanisticProbe layer-prefix curves")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _bucket_figure(strict: dict[str, Any], output: Path) -> None:
    tasks = ("sp1", "sp2")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    model_names = list(strict["models"])
    buckets = sorted(
        {int(bucket) for model in strict["models"].values() for bucket in model["buckets"]}
    )
    width = 0.24
    for task, axis in zip(tasks, axes):
        for model_index, model_name in enumerate(model_names):
            label = _short_name(model_name)
            values = [
                strict["models"][model_name]["buckets"][str(bucket)][task]["knn"]["f1_macro"]
                for bucket in buckets
            ]
            values = [float("nan") if value is None else value for value in values]
            center = (len(model_names) - 1) / 2
            positions = [index + (model_index - center) * width for index in range(len(buckets))]
            axis.bar(positions, values, width=width, label=label, color=COLORS.get(label))
        axis.set_xticks(range(len(buckets)), [str(bucket) for bucket in buckets])
        axis.set_xlabel("Number of statements")
        axis.set_title(f"Strict split — {task.upper()} kNN")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown(results_by_mode: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Formal reasoning-half probe results",
        "",
        "All language-model weights are frozen. Intervals are 95% cluster-bootstrap intervals; "
        "the paper-style split clusters examples and the strict split clusters complete theories.",
        "",
        "## Overall probe macro-F1",
        "",
        "| Split | Model | Task | kNN | Linear |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for mode in ("paper", "strict"):
        for model_name, model in results_by_mode[mode]["models"].items():
            for task in ("sp1", "sp2"):
                lines.append(
                    f"| {mode} | {_short_name(model_name)} | {task.upper()} | "
                    f"{_format_metric(model['overall'][task]['knn'])} | "
                    f"{_format_metric(model['overall'][task]['linear'])} |"
                )
    lines.extend(
        [
            "",
            "## Base minus Instruct paired differences",
            "",
            "Positive values favor base; negative values favor Instruct.",
            "",
            "| Split | Task | Classifier | Delta macro-F1 | 95% CI |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for mode in ("paper", "strict"):
        comparison = results_by_mode[mode]["paired_differences"]["base-formal_minus_instruct-formal"]
        for task in ("sp1", "sp2"):
            for classifier in ("knn", "linear"):
                metric = comparison[task][classifier]
                low, high = metric["ci95"]
                lines.append(
                    f"| {mode} | {task.upper()} | {classifier} | "
                    f"{metric['delta_f1_macro']:.4f} | [{low:.4f}, {high:.4f}] |"
                )
    lines.extend(
        [
            "",
            "## End-task accuracy",
            "",
            "| Model | Accuracy | N |",
            "| --- | ---: | ---: |",
        ]
    )
    for model_name, model in results_by_mode["strict"]["models"].items():
        accuracy = model["task_accuracy"]
        lines.append(f"| {_short_name(model_name)} | {accuracy['accuracy']:.4f} | {accuracy['n']} |")
    lines.extend(
        [
            "",
            "The probe results characterize the reasoning half only. SP1 is relevance selection, "
            "not entity/predicate grounding; grounding-specific interventions remain a separate experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the formal MechanisticProbe result report")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    results_by_mode = {
        mode: _load(args.root / "results" / f"probe-formal-{mode}.json")
        for mode in ("paper", "strict")
    }
    figure_root = args.root / "results" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    _layer_figure(results_by_mode, figure_root / "formal-layer-curves.png")
    _bucket_figure(results_by_mode["strict"], figure_root / "formal-bucket-knn.png")
    report_path = args.root / "results" / "formal-summary.md"
    report_path.write_text(_markdown(results_by_mode))
    print(report_path)


if __name__ == "__main__":
    main()
