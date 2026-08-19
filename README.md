# Frozen Qwen MechanisticProbe Reproduction

This repository implements the **reasoning-side** milestone of a larger project on separating symbolic grounding from symbolic reasoning in language models. It is a conceptual reproduction of the attention-probe methodology in Hou et al. (EMNLP 2023) on ProofWriter-CWA with frozen `Qwen/Qwen2.5-7B` and `Qwen/Qwen2.5-7B-Instruct` checkpoints.

No language-model weights are fine-tuned. The only learned components are lightweight k-nearest-neighbor and logistic-regression probes over compact attention features.

## Scope

- **SP1 — useful-statement probe:** whether a context statement appears in the gold proof.
- **SP2 — proof-height probe:** for useful statements in depth-1 proofs, whether a statement is the fact or rule step.
- **Paper split:** statement-level stratified five-fold cross-validation, matching the original probe-style protocol.
- **Strict split:** GroupKFold over canonical theory contexts, preventing statements from the same theory from appearing in both training and test folds.

SP1 is relevance selection, **not** entity/predicate binding. This code therefore studies the reasoning side only; it cannot establish a grounding-versus-reasoning separation by itself.

## Formal experiment

The released formal run fixes 6,277 ProofWriter-CWA test questions (85,820 statement-level records) across statement-count buckets `2, 4, 8, 12, 16, 20, 24`. It compares base, Instruct, and an architecture-matched random-weight control. A native Instruct chat-template control is also included. See [RESULTS.md](RESULTS.md) for frozen-model metrics, confidence intervals, and interpretation.

## Setup

Create a CUDA-enabled environment with PyTorch appropriate for your GPU, then install the remaining dependencies:

```bash
conda create -n grounding-mechprobe python=3.10 -y
conda activate grounding-mechprobe
pip install -e .
```

The source dataset is `yyyyifan/MechanisticProbe_ProofWriter_ARC`. `prepare` downloads it automatically when it is not already available below `EXPERIMENT_ROOT/data/source`. If the compute node cannot reach Hugging Face, download the authors' processed `CWA/{train,dev,test}.json` files elsewhere and copy them into that location.

## Run the formal reproduction

Set paths to complete local Hugging Face snapshots. The scripts never download or publish checkpoint weights.

```bash
export EXPERIMENT_ROOT=/absolute/path/to/experiment-artifacts
export BASE_MODEL_PATH=/absolute/path/to/Qwen2.5-7B
export INSTRUCT_MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
export CONDA_ENV=grounding-mechprobe

bash scripts/run_formal.sh
```

The script writes frozen samples, compact feature JSONL files, probe results, bootstrap intervals, and figures under `EXPERIMENT_ROOT`. It is restart-safe: an extraction is reused only when its completion marker exists.

For a small smoke test, use `bash scripts/run_base_pilot.sh` after setting `EXPERIMENT_ROOT` and `BASE_MODEL_PATH`.

## Run the native Instruct chat control

After the formal raw-prompt experiment is complete, the following command reuses its frozen sample and existing Base/Instruct artifacts. The only new language-model forward pass is Instruct with the official Qwen chat template.

```bash
export EXPERIMENT_ROOT=/absolute/path/to/existing-formal-experiment
export INSTRUCT_MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
export CONDA_ENV=grounding-mechprobe

bash scripts/run_chat_control.sh
```

The control uses the template's default system message, four fixed user/assistant demonstration pairs, the final user question, and an assistant generation prompt. It validates that the original artifacts remain byte-identical, then reports both Base(raw) versus Instruct(native chat) and Instruct(raw) versus Instruct(native chat).

## Reproducibility choices

- Four fixed, label-balanced in-context examples are selected from the training split with seed 42.
- Both trained checkpoints receive byte-identical completion prompts in the primary formal run. The additional control changes only Instruct to its native chat template.
- All checkpoint weights are frozen and loaded one at a time in BF16 when available (FP16 fallback).
- Attention pooling follows the paper's order: mean over statement tokens, max over query tokens, then mean over heads, yielding one scalar per layer per statement.
- The random control is initialized with seed 42 from the same local Qwen architecture and tokenizer.
- Formal intervals use 1,000 cluster-bootstrap replicates: examples in the paper split and complete theories in the strict split.

## Important limitations

1. The primary comparison uses a common raw-completion prompt. The native-chat control addresses that prompt-format confound for Instruct, but it remains one fixed multi-turn formatting choice and one candidate-scoring setup.
2. Probe decodability is not causal use. Activation patching or ablation is needed to establish a causal role for SP1/SP2 information.
3. A single random initialization is a sanity-check baseline, not a full random-model variance estimate.
4. Grounding-specific entity/predicate binding interventions are intentionally out of scope and remain the next experimental stage.

## Repository contents

- `src/mechanistic_probe/`: preparation, attention extraction, probes, bootstrap statistics, and report rendering.
- `scripts/`: restart-safe pilot and formal experiment entry points.
- `RESULTS.md`: English summary of the completed formal run.
- `results/chat-control/`: complete native-chat control JSON, validation metadata, report, and figures.

Raw data, checkpoint weights, logs, and extracted feature files are excluded from version control.
