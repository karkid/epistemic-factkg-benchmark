# Gating Diagnostics D1-D3 — Results

_Run 2026-06-11 as pure data + code analysis on the cloned repo (no training required; the evidence is
dispositive at the data/code level). These three diagnostics were defined in `plan/experiment_plan.md`
as the go/no-go fork that decides which paper we write._

## TL;DR — all three red flags from the novelty gate are CONFIRMED
The green path (method paper showing epistemic weighting beats baselines on AVeriTeC + a working
shortcut split) is **foreclosed by the data as built**. The honest, defensible paper is the **amber
path: a diagnosis + corrected-benchmark paper**. Details below.

---

## D1 — AVeriTeC: is 0.665 above the majority-class prior? → NO (it ≈ the prior)
Computed on the real AVeriTeC dev set (`data/raw/averitec/dev.json`, n=500):

| Class | Count | Share |
|---|---|---|
| Refuted | 305 | 61.0% |
| Supported | 122 | 24.4% |
| Not Enough Evidence | 35 | 7.0% |
| Conflicting Evidence/Cherry-picking | 38 | 7.6% |

- 4-class majority baseline (always "Refuted") = **0.6100**.
- The repo **drops Conflicting-Evidence** from GNN training (`configs/config.yaml:84`, ADR-007), so the
  effective task is 3-class. 3-class majority baseline = 305/462 = **0.6602**.
- The instructor report's best model scores **0.665 on AVeriTeC** → essentially **identical to the
  always-Refuted prior (0.6602)**. The "+7.1pp ladder gain" (0.594 → 0.665) is the baseline starting
  *below* the prior and v3-nli converging *to* it, not evidence of epistemic reasoning.
- **Required for the paper:** report **macro-F1** and the majority/always-Refuted baseline, never bare
  accuracy. Confirmation step: re-run v3-nli and inspect per-class predictions (a model collapsing onto
  Refuted would post ~0.66 accuracy with poor Supported/NEI recall). Until then, treat the AVeriTeC
  "win" as **very likely the class prior**.
- **QUANTIFIED (2026-06-11).** On 3-class AVeriTeC dev: majority "always Refuted" = **acc 0.6602,
  macro-F1 0.2651**. A realistic MiniLM+LogReg text classifier (no balancing) **collapses to the
  majority class** (predicts NEI exactly 1/462) at acc 0.6558, **macro-F1 0.3537**. So accuracy ~0.66
  is exactly the class-prior collapse, and macro-F1 for that behavior is ~0.27-0.40. The report's 0.665
  with NO per-AVeriTeC macro-F1 reported is this collapse; the "+7.1pp gain" is converging to the prior.
  The report's overall macro-F1 0.8155 is inflated by AI2THOR (1.000) and the leaked synthetic split
  (0.889); the AVeriTeC-specific macro-F1 was never reported and is ~0.3-0.4.

## D2 — Synthetic split: can a trust-blind model solve it WITHOUT source trust? → YES (massive leakage)
Computed on `data/raw/synthetic/synthetic_current.jsonl` (n=2500). The split is meant to *require*
source trust (identical text, varied source → low-trust must yield NEE). It does not. Single-feature
lookup tables predict the verdict almost perfectly **without any trust scalar**:

| Single feature | Verdict predictable from it alone |
|---|---|
| `meta.template_type` | **0.99** |
| `source_id` (the `source_type` node feature the "trust-blind" baseline SEES) | **0.82** |
| stance | 0.54 |

And the **evidence TEXT itself leaks the label** via style (first-8-word prefixes, by verdict):
- NEE: "Unconfirmed reports indicate that…", "An anonymous post claimed that…", "Sources allegedly suggest…"
- Supported: "The … Authority published its assessment confirming…", "… officially confirmed…"
- Refuted: "… spokesperson clarified…", "… records show no vote on…"

**Mechanism (confirmed in code).** `src/adapters/synthetic/fictional_generator.py` docstring: *"Shortcut-
breaking is **guaranteed by construction**: the template type determines source trust and inference
strength, which determines the verdict… regardless of evidence stance text."* But
`src/adapters/synthetic/client/local_client.py` draws evidence text from **reliability-stratified pools**
`_TEXT_POOLS[type]{strong|weak|hedged}` and `_WEAK_PREFIXES` (e.g. "An anonymous post claimed that ",
"Unconfirmed reports indicate that "). Low-trust sources always get the "weak/rumour" register; high-trust
get the "official/published" register. Since trust tier → verdict, **text register → verdict**. A model
reading only the 384-d text embedding (the "trust-blind" baseline) reaches ~0.99+ by learning the
register, never needing the ST feature. This fully explains the report's baseline = 0.996 and why the
epistemic model (0.889) does *worse* (it is distracted by a trust signal the leaky split doesn't reward).

**Verdict:** the synthetic split is a **template/text-style artifact**, not a test of source trust. It
**cannot** validate or refute H3 as built. It must be rebuilt so the SAME text appears under high- and
low-trust sources (a true counterfactual), with a trust-blind baseline driven to chance (~0.5) before
the epistemic model is reported.

## D3 — AI2THOR: is the 1.000 accuracy real capability or triviality? → triviality
Computed on `data/raw/ai2thor/claims_all.jsonl` (n=1800): perfectly balanced 900 supported / 900
refuted, **no NEI**. Every evidence item is `source_id=sensor_perception`, IS=1.0, evidence_type
perception → **ST and EW are constants**, so the epistemic formula contributes nothing discriminative
here. The label is recoverable from closed-world triple/number matching (claim "stool weighs 3.18 kg"
↔ evidence states the matching or a mismatching number), generated from the same simulator ground-truth.
Perfect accuracy reflects a clean, noise-free, constant-trust matching task, **not perception-based
epistemic reasoning**.

**Verdict:** AI2THOR is a **sanity check / proof-of-concept**, not a headline result, and must not be
allowed to inflate the overall number (it is 180/767 ≈ 23% of the val set at 1.000).

---

## Decision: take the AMBER path (diagnosis + corrected-benchmark paper)
The data foreclose the original method-paper framing: the AVeriTeC gain is ~the class prior (D1), the
shortcut split is a text-style artifact (D2), and AI2THOR is trivial (D3). Per the go/no-go fork in
`critique/novelty_review.md`, this is the amber path. The honest, novel, workshop-viable paper is:

**"Source-Trust Shortcuts in Fact Verification Are Easy to Build Badly: A Diagnosis and a Corrected
Benchmark."** Contributions:
1. **Diagnosis** — characterize how a natural-looking trust-contrastive synthetic split leaks the verdict
   through text register (with the quantified leakage above and the generator mechanism), and show the
   field's habit of reporting AVeriTeC *accuracy* hides the majority-class prior.
2. **Corrected artifact** — a rebuilt, trust-isolating split where text is held fixed/counterbalanced
   across trust tiers and a trust-blind baseline is provably at chance.
3. **Honest method evaluation** — the EC formula + Pramana taxonomy + two-layer auditable head as the
   proposed *mitigation*, evaluated on the corrected split and on AVeriTeC with macro-F1 + majority
   baselines + the EW/trust ablation (does the taxonomy beat flat/learned trust?).

This is more defensible than the original framing and uses the existing system as the mitigation. It does
require the corrected-split rebuild + re-runs (plan item R1, now promoted to central), but no green-path
fabrication. Next: confirm D1 with a macro-F1 re-run, build the corrected split, and re-scope the paper
outline to this framing — pending the user's go-ahead on the pivot.
