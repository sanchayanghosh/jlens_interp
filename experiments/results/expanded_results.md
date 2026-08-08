## Expanded results: all historical sycophancy cases

### Run and comparison design

The run completed **936 prompts from 234 cases** on NVIDIA H100 80GB HBM3 in 34.6 minutes using `unsloth/Meta-Llama-3.1-8B-Instruct`. It included all 117 historically sycophantic examples and 117 strict historically non-sycophantic examples matched within each correct-answer → user-answer cell. Each case used four layouts: no opinion, correct opinion, the original wrong opinion, and a second wrong-opinion control. Historical ambiguous examples were excluded from the matched control set.

The primary grouping below is the model's **regenerated behavior**, not the historical label from a different rollout checkpoint.

### Behavioral regeneration

| Regenerated outcome | Wrong-opinion trials |
| --- | ---: |
| Sycophantic | 123 |
| Resistant | 263 |
| Ambiguous | 82 |

Among 386 eligible wrong-opinion trials, 123 were sycophantic (31.9%). The remaining 82 trials were ambiguous, usually because the baseline answer was already wrong or the generated answer was neither the correct answer nor the user's answer.

| Historical label | Cases | Baseline correct | Correct-opinion correct | Eligible wrong trials | Regenerated sycophancy rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| honest | 117 | 115 | 117 | 225 | 12.0% |
| sycophantic | 117 | 86 | 115 | 161 | 59.6% |

The original wrong-opinion result matched its historical strict label for 161/194 eligible cases (83.0%), so the old label remains predictive. But it is not fully question-general: among the 73 cases that were sycophantic for the original distractor, only 38 remained sycophantic for the second distractor (52.1%); 28 became resistant and 7 ambiguous.

The specificity is concentrated in the historical-sycophantic set: its eligible sycophancy rate fell from 74.4% on the original distractor to 44.3% on the second. The matched historical-honest set remained low on both distractors (10.7% and 13.3%). Meanwhile, the correct-opinion control was answered correctly in 232/234 cases, versus 201/234 without an opinion. This shows broad opinion incorporation, while the second-wrong control separates generic suggestibility from distractor-specific behavior.

### New representational patterns

#### 1. User-answer encoding at the assistant boundary is not a sycophancy marker

At layer 30 of the assistant boundary, the correct-minus-user letter margin was negative and nearly identical for both regenerated outcomes:

| Outcome | Jacobian Lens margin | Logit Lens margin |
| --- | ---: | ---: |
| sycophantic | -1.450 ± 0.256 | -1.768 ± 0.318 |
| resistant | -1.307 ± 0.160 | -1.628 ± 0.195 |

Both groups initially represent the user's distractor. The largest sycophantic-versus-resistant gap in boundary opinion shift was only 0.184 Jacobian-Lens units (layer 30) and 0.323 Logit-Lens units (layer 24). The assistant-boundary signal therefore looks like generic prompt uptake or an initial response state, not a behavioral classifier.

#### 2. The outcomes bifurcate late, around layers 20–22

At the pre-answer token, the two behaviors have opposite late-layer margins:

| Outcome | Lens | Layer-30 correct−user margin | Opinion shift vs baseline |
| --- | --- | ---: | ---: |
| sycophantic | jacobian_lens | -2.331 ± 0.146 | -5.160 ± 0.206 |
| sycophantic | logit_lens | -2.808 ± 0.156 | -6.642 ± 0.246 |
| resistant | jacobian_lens | +3.665 ± 0.129 | -2.510 ± 0.100 |
| resistant | logit_lens | +4.916 ± 0.160 | -3.079 ± 0.124 |

The separation becomes visible at layer 20 and grows rapidly: the Jacobian-Lens pre-answer margins are −0.091 versus +0.374 at layer 20, −0.283 versus +0.721 at layer 22, and −1.952 versus +3.810 at layer 29 (sycophantic versus resistant). The largest opinion-shift gap occurs at layer 29: 3.141 Jacobian-Lens units and 5.896 Logit-Lens units. Both groups absorb the user's opinion relative to the no-opinion baseline, but resistant trials recover enough correct-answer evidence to remain positive; sycophantic trials cross over to the user's answer.

#### 3. Correction language, uncertainty, and cross-lens agreement distinguish resistance

At the layer-30 assistant boundary, correction-vocabulary shift from baseline was stronger for resistant trials (Jacobian 1.710; Logit 1.597) than for sycophantic trials (0.595; 0.583). Affirmation shifts were almost identical between the groups (Jacobian 0.491 versus 0.480; Logit 0.295 versus 0.327). This supports a graded correction/conflict signal rather than a binary token cue.

Sycophantic trials also had higher layer-30 entropy than resistant trials at both positions (boundary Jacobian 1.396 versus 1.206 nats; pre-answer Logit 1.428 versus 1.075 nats). Jacobian/Logit margin correlation was similarly low at the boundary (0.267 versus 0.278) but diverged at pre-answer (0.432 versus 0.666). These are descriptive signals of a less concentrated, less cross-lens-aligned sycophantic decision path, not yet causal evidence.

### Interpretation against the research hypotheses

- **Late recovery is supported for resistant generations.** The user opinion is present at the boundary, while correct-answer evidence re-emerges late.
- **Late override is supported for regenerated sycophancy.** Its pre-answer trajectory crosses decisively toward the user's letter in the final third of the network.
- **Conflict/correction magnitude is more informative than mere presence.** Affirmation shifts do not separate behavior, whereas correction shifts do.
- **Generic suggestibility is only partially supported.** Correct opinions strongly help, but only 52.1% of original-distractor sycophancy transfers to a second wrong answer; much of the historical effect is distractor-specific.

### Next steps

1. Fit a case-clustered outcome model using late-layer margin, correction shift, entropy, and historical label, then evaluate question-held-out AUROC.
2. Run causal activation patching across layers 18–30 between matched sycophantic and resistant trials to test whether the late margin is interventionally relevant.
3. Add all alternative wrong letters per question and paraphrased opinion templates to quantify distractor and wording specificity.
4. Refit the Jacobian Lens directly on the exact inference checkpoint and replicate with several seeds/checkpoints before claiming a mechanistic marker.

### Interpretation guardrails

- Historical labels come from a different checkpoint; regenerated behavior is the primary grouping.
- Two wrong-opinion trials share each question, so trial-level standard errors are descriptive rather than independent-sample inference.
- The public lens is a 32-projection third-party fit loaded through Anthropic's Jacobian Lens API; results should be replicated with an exact fit before treating a trajectory as a mechanistic marker.
- Option-text first-token readouts showed weaker and less consistent late separation than letter readouts, so conclusions currently apply most directly to the forced-choice answer representation.

### Expanded-run sources

- [Full Modal run](https://modal.com/apps/aman313/main/ap-al93duEdEbmQkfLPIgXW14)
- [Anthropic Jacobian Lens implementation](https://github.com/anthropics/jacobian-lens)
- [Public Llama-3.1-8B-Instruct lens fit](https://huggingface.co/Kameshr/jspace-lens-Llama8b)
- [Inference checkpoint](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)
