#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    specs = [('base-formal', 'Qwen Base (raw)'), ('instruct-formal', 'Qwen Instruct (raw)'), ('instruct-chat-formal', 'Qwen Instruct (native chat)')]
    colors, markers = {'sp1':'#e45756','sp2':'#4169e1'}, {'sp1':'o','sp2':'s'}
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    for ax, (model, title) in zip(axes, specs):
        curves = data['models'][model]['layer_curves']
        for task in ('sp1', 'sp2'):
            values = curves[task]; layers = list(range(1, len(values) + 1))
            ax.plot(layers, values, color=colors[task], marker=markers[task], markersize=3.5, linewidth=1.6, label=task.upper())
            ax.scatter([layers[-1]], [values[-1]], color=colors[task], s=52, zorder=4, edgecolor='white', linewidth=0.7)
        ax.set_title(title, fontsize=10); ax.set_xlabel('Layer prefix length')
        ax.set_xlim(1, 28); ax.set_xticks([1, 7, 14, 21, 28]); ax.set_ylim(0.4, 0.95); ax.grid(True, alpha=0.25)
    axes[0].set_ylabel('Strict-split kNN macro-F1'); axes[0].legend(loc='lower right')
    fig.suptitle('ProofWriter SP1/SP2 layer-prefix curves (strict split)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); a.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.output, dpi=220, bbox_inches='tight')

if __name__ == '__main__': main()
