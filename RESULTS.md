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

## Native Instruct chat-template control

The control reruns only Qwen2.5-7B-Instruct with the official chat template: the template's default system message, the same four demonstrations as user/assistant turns, the final question as a user turn, and `add_generation_prompt=True`. The Base(raw) and Instruct(raw) artifacts are reused byte-for-byte. Postflight validation confirms 85,820 aligned rows, zero skipped examples, and unchanged reference-artifact SHA-256 hashes.

### Strict-split results

| Condition | SP1 kNN | SP1 linear | SP2 kNN | SP2 linear | End-task accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base (raw) | 0.7184 | 0.6692 | 0.8889 | 0.6447 | 0.7762 |
| Instruct (raw) | 0.7122 | 0.6643 | 0.8677 | 0.6544 | 0.7610 |
| Instruct (native chat) | 0.7155 | 0.6671 | 0.8685 | 0.6620 | 0.7440 |

### Strict paired differences

Positive values favor the first condition. Each interval is a 1,000-replicate paired cluster-bootstrap 95% interval over complete theories.

| Comparison | Task | Probe | Delta macro-F1 | 95% CI |
| --- | --- | --- | ---: | ---: |
| Base(raw) minus Instruct(chat) | SP1 | kNN | 0.0029 | [-0.0016, 0.0069] |
| Base(raw) minus Instruct(chat) | SP1 | Linear | 0.0022 | [-0.0001, 0.0046] |
| Base(raw) minus Instruct(chat) | SP2 | kNN | 0.0204 | [0.0104, 0.0308] |
| Base(raw) minus Instruct(chat) | SP2 | Linear | -0.0173 | [-0.0222, -0.0122] |
| Instruct(raw) minus Instruct(chat) | SP1 | kNN | -0.0033 | [-0.0071, 0.0004] |
| Instruct(raw) minus Instruct(chat) | SP1 | Linear | -0.0028 | [-0.0046, -0.0009] |
| Instruct(raw) minus Instruct(chat) | SP2 | kNN | -0.0008 | [-0.0103, 0.0080] |
| Instruct(raw) minus Instruct(chat) | SP2 | Linear | -0.0076 | [-0.0116, -0.0036] |

The within-checkpoint prompt-format effect is small for probe decodability: strict kNN differences include zero for both tasks, while linear probes modestly favor native chat. End-task accuracy nevertheless decreases by 1.70 percentage points from Instruct(raw) to Instruct(native chat). Base versus native-chat Instruct still reverses direction across probe families on SP2, so the result does not support a scalar claim that instruction tuning uniformly improves or degrades the reasoning representation.

The complete paper/strict JSON, generated report, validation record, and figures are in [`results/chat-control/`](results/chat-control/). These observational probes do not establish causal use, and SP1 remains relevance selection rather than entity/predicate grounding.
