from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ("base-formal", "instruct-formal", "instruct-chat-formal")
LABELS = {
    "base-formal": "Base (raw)",
    "instruct-formal": "Instruct (raw)",
    "instruct-chat-formal": "Instruct (native chat)",
}
COLORS = {
    "base-formal": "#2563eb",
    "instruct-formal": "#dc2626",
    "instruct-chat-formal": "#16a34a",
}
COMPARISONS = {
    "base-formal_minus_instruct-chat-formal": "Base (raw) minus Instruct (native chat)",
    "instruct-formal_minus_instruct-chat-formal": "Instruct (raw) minus Instruct (native chat)",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric(metric: dict[str, Any]) -> str:
    score = metric["f1_macro"]
    interval = metric.get("ci95")
    if interval:
        return f"{score:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"
    return f"{score:.4f}"


def _layer_figure(results: dict[str, dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for column, mode in enumerate(("paper", "strict")):
        for row_index, task in enumerate(("sp1", "sp2")):
            axis = axes[row_index, column]
            for model in MODELS:
                curve = results[mode]["models"][model]["layer_curves"][task]
                axis.plot(
                    range(1, len(curve) + 1),
                    curve,
                    label=LABELS[model],
                    color=COLORS[model],
                    linewidth=2,
                )
            axis.set_title(f"{mode.capitalize()} split — {task.upper()}")
            axis.set_ylabel("Macro-F1")
            axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("Layers included (prefix)")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Native Instruct chat-template control")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _bucket_figure(strict: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    buckets = sorted(int(value) for value in strict["models"][MODELS[0]]["buckets"])
    width = 0.24
    for task, axis in zip(("sp1", "sp2"), axes):
        for model_index, model in enumerate(MODELS):
            values = [
                strict["models"][model]["buckets"][str(bucket)][task]["knn"]["f1_macro"]
                for bucket in buckets
            ]
            values = [float("nan") if value is None else value for value in values]
            center = (len(MODELS) - 1) / 2
            positions = [index + (model_index - center) * width for index in range(len(buckets))]
            axis.bar(
                positions,
                values,
                width=width,
                label=LABELS[model],
                color=COLORS[model],
            )
        axis.set_xticks(range(len(buckets)), [str(bucket) for bucket in buckets])
        axis.set_xlabel("Number of statements")
        axis.set_title(f"Strict split — {task.upper()} kNN")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Native Instruct Chat-Template Control",
        "",
        "The only new language-model forward pass is Qwen2.5-7B-Instruct with its official chat template: the default system message, four fixed user/assistant demonstrations, the final user question, and an assistant generation prompt. Base (raw) and Instruct (raw) features are reused from the frozen formal experiment.",
        "",
        "## Overall probe macro-F1",
        "",
        "| Split | Condition | Task | kNN | Linear |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for mode in ("paper", "strict"):
        for model in MODELS:
            for task in ("sp1", "sp2"):
                overall = results[mode]["models"][model]["overall"][task]
                lines.append(
                    f"| {mode} | {LABELS[model]} | {task.upper()} | "
                    f"{_metric(overall['knn'])} | {_metric(overall['linear'])} |"
                )
    lines.extend(
        [
            "",
            "## Paired differences",
            "",
            "Positive values favor the first condition named in the comparison; negative values favor native-chat Instruct.",
            "",
            "| Split | Comparison | Task | Probe | Delta macro-F1 | 95% CI |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for mode in ("paper", "strict"):
        for comparison, label in COMPARISONS.items():
            for task in ("sp1", "sp2"):
                for classifier in ("knn", "linear"):
                    metric = results[mode]["paired_differences"][comparison][task][classifier]
                    low, high = metric["ci95"]
                    lines.append(
                        f"| {mode} | {label} | {task.upper()} | {classifier} | "
                        f"{metric['delta_f1_macro']:.4f} | [{low:.4f}, {high:.4f}] |"
                    )
    lines.extend(
        [
            "",
            "## End-task accuracy",
            "",
            "| Condition | Accuracy | N |",
            "| --- | ---: | ---: |",
        ]
    )
    for model in MODELS:
        accuracy = results["strict"]["models"][model]["task_accuracy"]
        lines.append(f"| {LABELS[model]} | {accuracy['accuracy']:.4f} | {accuracy['n']} |")
    strict = results["strict"]
    raw_accuracy = strict["models"]["instruct-formal"]["task_accuracy"]["accuracy"]
    chat_accuracy = strict["models"]["instruct-chat-formal"]["task_accuracy"]["accuracy"]
    accuracy_change = chat_accuracy - raw_accuracy
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Base(raw) versus Instruct(native chat) is a native-use comparison that changes both checkpoint and input format. Under the strict split, its SP1 paired intervals include zero for both probe families. Its SP2 result is probe-dependent: kNN favors Base while linear probing favors Instruct(native chat), so it does not support a scalar model ranking.",
            "",
            "Instruct(raw) versus Instruct(native chat) holds the checkpoint fixed and estimates the prompt-format effect. Strict kNN intervals include zero for SP1 and SP2, while the small linear differences favor native chat. This is a modest change in probe geometry, not evidence of a new reasoning stage.",
            "",
            f"End-task accuracy changes from {raw_accuracy:.4f} to {chat_accuracy:.4f} ({accuracy_change:+.4f}) under native chat. The accuracy decrease and the small probe changes must be reported separately because task behavior and frozen-feature decodability need not move together.",
            "",
            "## Interpretation guardrails",
            "",
            "The Base-versus-native-chat comparison is the native model comparison. The raw-versus-chat Instruct comparison isolates the prompt-format effect within the same checkpoint. Probe decodability remains observational: it does not show that the model causally uses the decoded information, and SP1 remains relevance selection rather than entity/predicate grounding.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the native Instruct chat-control report")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    results = {
        mode: _load(args.root / "results" / f"probe-chat-control-{mode}.json")
        for mode in ("paper", "strict")
    }
    figures = args.root / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _layer_figure(results, figures / "chat-control-layer-curves.png")
    _bucket_figure(results["strict"], figures / "chat-control-bucket-knn.png")
    output = args.root / "results" / "chat-control-summary.md"
    output.write_text(_markdown(results))
    print(output)


if __name__ == "__main__":
    main()
