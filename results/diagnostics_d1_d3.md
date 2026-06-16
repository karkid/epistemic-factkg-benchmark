# Gating Diagnostics D1-D3 — Results

_Run 2026-06-11 as pure data + code analysis on the cloned repo (no training required; the evidence is
dispositive at the data/code level). These three diagnostics were the go/no-go fork that decided the
paper framing. All three red flags were confirmed; the paper pivoted to the diagnosis + corrected-benchmark
framing and is now written. Controlled experiment: `benchmark/run_logreg_test.py`._

## TL;DR — all three red flags CONFIRMED
The method-paper framing (epistemic weighting beats baselines on AVeriTeC + working shortcut split) was
foreclosed by the data. The paper is the **diagnosis + corrected-benchmark** path. Details below.

---

## D1 — AVeriTeC: is 0.665 above the majority-class prior? → NO (it ≈ the prior)
Computed on the real AVeriTeC dev set (`model_repo/data/raw/averitec/dev.json`, n=500):

| Class | Count | Share |
|---|---|---|
| Refuted | 305 | 61.0% |
| Supported | 122 | 24.4% |
| Not Enough Evidence | 35 | 7.0% |
| Conflicting Evidence/Cherrypicking | 38 | 7.6% |

- 4-class majority baseline (always "Refuted") = **0.6100**.
- The repo drops Conflicting-Evidence from GNN training (`model_repo/configs/config.yaml:84`, ADR-007), so the
  effective task is 3-class. 3-class majority baseline = 305/462 = **0.6602**.
- The instructor report's best model scores **0.665 on AVeriTeC** → essentially identical to the
  always-Refuted prior (0.6602). The "+7.1pp ladder gain" (0.594 → 0.665) is the baseline starting
  below the prior and v3-nli converging to it, not evidence of epistemic reasoning.
- **Quantified (2026-06-11).** On 3-class AVeriTeC dev: majority "always Refuted" = **acc 0.6602,
  macro-F1 0.2651**. MiniLM+LogReg text classifier (no balancing) collapses to the majority class
  (predicts NEI exactly 1/462) at acc 0.6558, **macro-F1 0.3537**. Accuracy ~0.66 is exactly the
  class-prior collapse; macro-F1 for that behaviour is ~0.27–0.40.
- **In the paper:** AVeriTeC reported as macro-F1 0.265 majority baseline; the system's 0.665 accuracy
  is shown to equal the class prior (Section 3, Table II).

## D2 — Synthetic split: can a trust-blind model solve it WITHOUT source trust? → YES (massive leakage)
Computed on `model_repo/data/raw/synthetic/synthetic_current.jsonl` (n=2500). Single-feature lookup tables predict
the verdict almost perfectly without any trust scalar:

| Single feature | Verdict predictable from it alone |
|---|---|
| `meta.template_type` | **0.99** |
| `source_id` (the `source_type` node feature the trust-blind baseline sees) | **0.82** |
| stance | 0.54 |

Evidence TEXT itself leaks the label via register (first-8-word prefixes, by verdict):
- NEE: "Unconfirmed reports indicate that…", "An anonymous post claimed that…"
- Supported: "The … Authority published its assessment confirming…", "… officially confirmed…"
- Refuted: "… spokesperson clarified…", "… records show no vote on…"

**Mechanism.** `model_repo/src/adapters/synthetic/fictional_generator.py`: low-trust sources always get the
"weak/rumour" register; high-trust get the "official/published" register. Since trust tier → verdict,
text register → verdict. A trust-blind model reaches ~0.99 by learning register, never needing ST.
This fully explains the report's baseline = 0.996 and why the epistemic model (0.889) does worse on
the original split (it is distracted by a trust signal the leaky split does not reward).

**Fix:** the corrected benchmark (v3, `benchmark-repo/data/expanded_split.jsonl`) holds evidence text byte-identical
across trust tiers and constant source category, removing all non-trust paths to the label.

## D3 — AI2THOR: is the 1.000 accuracy real capability or triviality? → triviality
Computed on `model_repo/data/raw/ai2thor/claims_all.jsonl` (n=1800): perfectly balanced 900
supported / 900 refuted, no NEI (majority baseline = 0.500). Every evidence item is
`source_id=sensor_perception`, IS=1.0, evidence_type perception → ST and EW are constants, so the EC
formula contributes nothing discriminative. The label is recoverable from closed-world triple/number
matching generated from simulator ground-truth.

The GNN model scores 1.000 accuracy on AI2THOR (instructor report). This reflects a clean,
constant-trust matching task, not epistemic reasoning — any model that memorises the triple→label
mapping achieves perfect accuracy, but that does not generalise to graded source trust.

**In the paper:** AI2THOR described as "closed-world trivial" (Section 3); the GNN's 1.000 accuracy
is listed as a diagnostic finding, not a benchmark result.

---

## Outcome
All three diagnostics confirmed the pivot to the diagnosis + corrected-benchmark paper:

> **"Beyond Binary Trust: A Source-Trust-Isolating Benchmark for Continuous Epistemic Reasoning
> in Fact Verification"**

Contributions delivered:
1. Diagnosis of three evaluation failures in the original system (D1–D3 above)
2. Corrected trust-isolating benchmark (v3, 2200 records, expanded vocabulary, mid-tier probes)
3. Reproducible evaluation: 3 GNN variants × 3 independent runs + LogReg probe
