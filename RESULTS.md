# Formal Reasoning-Side Results

## Protocol

The formal run freezes 6,277 ProofWriter-CWA test questions across statement-count buckets `2, 4, 8, 12, 16, 20, 24`, producing 85,820 statement-level records. It compares frozen Qwen2.5-7B base, Qwen2.5-7B-Instruct, and an architecture-matched random-weight control. All intervals are 95% cluster-bootstrap intervals with 1,000 replicates. The strict split holds out complete theories.

## Strict-split kNN results

| Model | SP1 macro-F1 | SP2 macro-F1 | End-task accuracy |
| --- | ---: | ---: | ---: |
| Base | 0.7184 [0.7108, 0.7253] | 0.8889 [0.8786, 0.8991] | 0.7762 |
| Instruct | 0.7122 [0.7053, 0.7190] | 0.8677 [0.8570, 0.8777] | 0.7610 |
| Random | 0.4980 [0.4939, 0.5021] | 0.5992 [0.5850, 0.6131] | 0.4913 |

## Base minus Instruct paired differences

Positive values favor base. Intervals that do not contain zero indicate a reliable difference under the released protocol.

| Task | Probe | Delta macro-F1 | 95% CI |
| --- | --- | ---: | ---: |
| SP1 | kNN | 0.0062 | [0.0019, 0.0105] |
| SP1 | Linear | 0.0049 | [0.0028, 0.0071] |
| SP2 | kNN | 0.0213 | [0.0117, 0.0307] |
| SP2 | Linear | -0.0097 | [-0.0133, -0.0059] |

## Interpretation

Both trained checkpoints are far above the random control, showing that the frozen attention features encode learned proof relevance and proof-step structure. Base and Instruct layer curves are nevertheless close. SP1 slightly favors base under both probe families, while SP2 reverses direction between kNN and linear probes. This pattern is more consistent with a modest rearrangement of representational geometry than with a simple conclusion that instruction tuning improves or degrades symbolic reasoning.

The end-task accuracy comparison is not a pure capability comparison because the released run uses one raw-completion prompt rather than the Instruct model's preferred chat template. The experiment is also about reasoning-side probes; it does not measure symbolic grounding or demonstrate causal use of the decoded information.
