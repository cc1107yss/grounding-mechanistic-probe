# Native Instruct Chat-Template Control

The only new language-model forward pass is Qwen2.5-7B-Instruct with its official chat template: the default system message, four fixed user/assistant demonstrations, the final user question, and an assistant generation prompt. Base (raw) and Instruct (raw) features are reused from the frozen formal experiment.

## Overall probe macro-F1

| Split | Condition | Task | kNN | Linear |
| --- | --- | --- | ---: | ---: |
| paper | Base (raw) | SP1 | 0.7399 [0.7336, 0.7453] | 0.6692 [0.6654, 0.6726] |
| paper | Base (raw) | SP2 | 0.9038 [0.8951, 0.9124] | 0.6450 [0.6347, 0.6552] |
| paper | Instruct (raw) | SP1 | 0.7332 [0.7272, 0.7390] | 0.6641 [0.6603, 0.6678] |
| paper | Instruct (raw) | SP2 | 0.8913 [0.8817, 0.9000] | 0.6544 [0.6438, 0.6650] |
| paper | Instruct (native chat) | SP1 | 0.7344 [0.7284, 0.7399] | 0.6672 [0.6633, 0.6707] |
| paper | Instruct (native chat) | SP2 | 0.8945 [0.8857, 0.9040] | 0.6621 [0.6507, 0.6724] |
| strict | Base (raw) | SP1 | 0.7184 [0.7108, 0.7253] | 0.6692 [0.6648, 0.6732] |
| strict | Base (raw) | SP2 | 0.8889 [0.8786, 0.8991] | 0.6447 [0.6301, 0.6578] |
| strict | Instruct (raw) | SP1 | 0.7122 [0.7053, 0.7190] | 0.6643 [0.6603, 0.6682] |
| strict | Instruct (raw) | SP2 | 0.8677 [0.8570, 0.8777] | 0.6544 [0.6400, 0.6672] |
| strict | Instruct (native chat) | SP1 | 0.7155 [0.7086, 0.7223] | 0.6671 [0.6629, 0.6708] |
| strict | Instruct (native chat) | SP2 | 0.8685 [0.8580, 0.8793] | 0.6620 [0.6475, 0.6743] |

## Paired differences

Positive values favor the first condition named in the comparison; negative values favor native-chat Instruct.

| Split | Comparison | Task | Probe | Delta macro-F1 | 95% CI |
| --- | --- | --- | --- | ---: | ---: |
| paper | Base (raw) minus Instruct (native chat) | SP1 | knn | 0.0055 | [0.0016, 0.0096] |
| paper | Base (raw) minus Instruct (native chat) | SP1 | linear | 0.0019 | [-0.0004, 0.0040] |
| paper | Base (raw) minus Instruct (native chat) | SP2 | knn | 0.0093 | [0.0005, 0.0175] |
| paper | Base (raw) minus Instruct (native chat) | SP2 | linear | -0.0171 | [-0.0215, -0.0129] |
| paper | Instruct (raw) minus Instruct (native chat) | SP1 | knn | -0.0011 | [-0.0045, 0.0027] |
| paper | Instruct (raw) minus Instruct (native chat) | SP1 | linear | -0.0031 | [-0.0047, -0.0014] |
| paper | Instruct (raw) minus Instruct (native chat) | SP2 | knn | -0.0032 | [-0.0121, 0.0049] |
| paper | Instruct (raw) minus Instruct (native chat) | SP2 | linear | -0.0077 | [-0.0112, -0.0040] |
| strict | Base (raw) minus Instruct (native chat) | SP1 | knn | 0.0029 | [-0.0016, 0.0069] |
| strict | Base (raw) minus Instruct (native chat) | SP1 | linear | 0.0022 | [-0.0001, 0.0046] |
| strict | Base (raw) minus Instruct (native chat) | SP2 | knn | 0.0204 | [0.0104, 0.0308] |
| strict | Base (raw) minus Instruct (native chat) | SP2 | linear | -0.0173 | [-0.0222, -0.0122] |
| strict | Instruct (raw) minus Instruct (native chat) | SP1 | knn | -0.0033 | [-0.0071, 0.0004] |
| strict | Instruct (raw) minus Instruct (native chat) | SP1 | linear | -0.0028 | [-0.0046, -0.0009] |
| strict | Instruct (raw) minus Instruct (native chat) | SP2 | knn | -0.0008 | [-0.0103, 0.0080] |
| strict | Instruct (raw) minus Instruct (native chat) | SP2 | linear | -0.0076 | [-0.0116, -0.0036] |

## End-task accuracy

| Condition | Accuracy | N |
| --- | ---: | ---: |
| Base (raw) | 0.7762 | 6277 |
| Instruct (raw) | 0.7610 | 6277 |
| Instruct (native chat) | 0.7440 | 6277 |

## Interpretation

Base(raw) versus Instruct(native chat) is a native-use comparison that changes both checkpoint and input format. Under the strict split, its SP1 paired intervals include zero for both probe families. Its SP2 result is probe-dependent: kNN favors Base while linear probing favors Instruct(native chat), so it does not support a scalar model ranking.

Instruct(raw) versus Instruct(native chat) holds the checkpoint fixed and estimates the prompt-format effect. Strict kNN intervals include zero for SP1 and SP2, while the small linear differences favor native chat. This is a modest change in probe geometry, not evidence of a new reasoning stage.

End-task accuracy changes from 0.7610 to 0.7440 (-0.0170) under native chat. The accuracy decrease and the small probe changes must be reported separately because task behavior and frozen-feature decodability need not move together.

## Interpretation guardrails

The Base-versus-native-chat comparison is the native model comparison. The raw-versus-chat Instruct comparison isolates the prompt-format effect within the same checkpoint. Probe decodability remains observational: it does not show that the model causally uses the decoded information, and SP1 remains relevance selection rather than entity/predicate grounding.
