#!/usr/bin/env python3
"""Plot per-height SP2 layer-prefix F1 curves from formal probe results."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    specs = [
        ("base-formal", "Qwen Base (raw)"),
        ("instruct-formal", "Qwen Instruct (raw)"),
        ("instruct-chat-formal", "Qwen Instruct (native chat)"),
    ]
    colors = {"height_0": "#e45756", "height_1": "#4169e1"}
    markers = {"height_0": "s", "height_1": "^"}
    labels = {"height_0": "height = 0", "height_1": "height = 1"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    for axis, (model_id, title) in zip(axes, specs):
        curves = data["models"][model_id]["height_class_curves"]
        for key in ("height_0", "height_1"):
            values = curves[key]
            layers = list(range(1, len(values) + 1))
            axis.plot(
                layers,
                values,
                color=colors[key],
                marker=markers[key],
                markersize=3.8,
                linewidth=1.6,
                label=labels[key],
            )
            axis.scatter(
                [layers[-1]],
                [values[-1]],
                color=colors[key],
                s=54,
                zorder=4,
                edgecolor="white",
                linewidth=0.7,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Layer prefix length")
        axis.set_xlim(1, 28)
        axis.set_xticks([1, 7, 14, 21, 28])
        axis.set_ylim(0.4, 1.0)
        axis.grid(True, alpha=0.25)

    axes[0].set_ylabel("Strict-split kNN class F1")
    axes[0].legend(loc="lower right")
    fig.suptitle("ProofWriter SP2 F1 by proof height (strict split)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
