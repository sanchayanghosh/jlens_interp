## Single-shot commitment and open-concept experiment

The run completed 330 examples on NVIDIA H100 80GB HBM3 using `unsloth/Meta-Llama-3.1-8B-Instruct`. Labels reproduced for 330/330 examples.

### Preregistered post-commitment objection score

The detector uses one opinion-prompt trace, no answer key, and no counterfactual at inference time. It compares sycophantic wrong agreement with warranted correct agreement on the same questions.

- Matched pairs: 85
- Sycophantic mean: 0.165
- Warranted-agreement mean: -0.047
- Paired difference: 0.212
- AUROC: 0.656
- Cluster-bootstrap AUROC 95% CI: [0.570, 0.738]

This is a modest answer-key-free signal, not a deployment-ready detector. A post-hoc scan found a stronger objection score at the user-opinion token (AUROC 0.715), but that estimate is exploratory because the position was selected after seeing the results.

### Open J-space concepts

The matched resistant comparison reveals a clearer semantic pattern. At the user-opinion token, sycophantic cases have a lower objection-family score than resistant cases (paired difference -2.928; AUROC with sycophancy positive 0.112). The same direction is present at the assistant boundary (difference -1.077; AUROC 0.322).

Canonical word concepts below were selected by hidden top-25 prevalence differences in the 70% discovery split and checked on the held-out 30%. Positive gaps mean more common in sycophantic cases; negative gaps mean more common in resistant cases.

#### user_opinion_letter — Jacobian Lens

| Concept | Discovery gap | Test gap | Replicated |
| --- | ---: | ---: | --- |
| `wrong` | -1.0660 | -1.1648 | True |
| `none` | -0.8111 | -0.8258 | True |
| `incorrect` | -0.7795 | -0.8902 | True |
| `but` | -0.7121 | -1.0170 | True |
| `false` | -0.5927 | -0.7008 | True |
| `however` | -0.5400 | -0.5947 | True |
| `incorrectly` | -0.4024 | -0.4716 | True |
| `omdat` | 0.2844 | 0.3277 | True |
| `because` | 0.2704 | 0.3636 | True |
| `nothing` | -0.2626 | -0.2765 | True |

#### assistant_boundary — Jacobian Lens

| Concept | Discovery gap | Test gap | Replicated |
| --- | ---: | ---: | --- |
| `incorrect` | -0.5435 | -0.7879 | True |
| `although` | -0.4073 | -0.5909 | True |
| `none` | -0.3153 | -0.2386 | True |
| `while` | -0.2275 | -0.3333 | True |
| `considering` | 0.1692 | 0.2746 | True |
| `according` | 0.1482 | 0.1439 | True |
| `based` | 0.1355 | 0.1174 | True |
| `unfortunately` | -0.1250 | -0.2936 | True |
| `because` | -0.1166 | -0.0909 | True |
| `though` | -0.1152 | -0.1742 | True |

#### first_generated_token — Jacobian Lens

| Concept | Discovery gap | Test gap | Replicated |
| --- | ---: | ---: | --- |
| `correct` | -0.1355 | -0.0152 | True |
| `incorrect` | -0.0604 | -0.1155 | True |
| `storyline` | 0.0562 | 0.0398 | True |
| `reportedly` | 0.0463 | 0.0492 | True |
| `frage` | 0.0337 | 0.0701 | True |
| `movie` | 0.0330 | -0.0189 | False |
| `questions` | 0.0309 | 0.0227 | True |
| `phrases` | 0.0309 | -0.0095 | False |
| `traditional` | 0.0309 | 0.0000 | False |
| `sentence` | 0.0302 | -0.0208 | False |

#### generated_answer — Jacobian Lens

| Concept | Discovery gap | Test gap | Replicated |
| --- | ---: | ---: | --- |
| `rlf` | 0.0260 | 0.0095 | True |
| `okie` | 0.0169 | 0.0322 | True |
| `dlgitem` | 0.0169 | 0.0322 | True |
| `correct` | -0.0119 | 0.0019 | False |
| `dhabi` | 0.0119 | 0.0284 | True |
| `something` | 0.0112 | 0.0057 | True |
| `summon` | 0.0105 | -0.0152 | False |
| `divider` | 0.0105 | 0.0208 | True |
| `morning` | 0.0098 | 0.0095 | True |
| `fact` | 0.0098 | 0.0076 | True |

The robust early contrast is best described as error recognition versus rationalization: resistant traces surface concepts such as `wrong`, `incorrect`, `but`, and `however`, while sycophantic traces more often surface causal or attributional concepts such as `because`, `based`, and `according`. The generated-answer vocabulary is substantially less coherent and should not be treated as a stable post-commitment marker.

### Guardrails

- The workspace band (layers 12–27) is a paper-motivated approximation for this 32-layer checkpoint, not a fitted boundary.
- Open vocabulary findings are matched by layout and answer-letter mapping, but may still reflect difficulty, wording, or tokenization.
- J-lens tokens are correlational projections and should not be described as literal thoughts without intervention evidence.
