# Corrected Trust-Isolating Split — Design, Iteration & Verification

_Built 2026-06-11. Generator: `experiments/build_corrected_split.py` (seed 20260611). Data:
`experiments/data/corrected_split.jsonl` (2080 records). Controlled test:
`experiments/run_trust_isolation_test.py`. This is the paper's central methodological artifact: a
fact-verification split that ONLY graded source trust can solve._

## Motivation
The original synthetic split leaks the verdict through text register (diagnostics D2): low-trust sources
always get rumour-style phrasings, high-trust get official phrasings, so a trust-blind text model scores
~0.996 without using source trust. The corrected split must remove every non-trust path to the label.

## Design iteration (this is itself a finding)
**v1 (text isolation only) — STILL LEAKED.** First attempt held the evidence text byte-identical across a
high-trust and a low-trust source, neutral register, with disjoint train/test source *sets*. Data-level
single-feature checks looked clean (text predicted the label at chance 0.50). But the controlled
experiment exposed a residual leak: the trust-blind baseline scored **macro-F1 0.82 on held-out test**.
Why: the baseline sees a 6-d `source_type` (category) one-hot, category correlates with trust in the
registry, so it transferred to most held-out sources; combined with a "supporting-text -> supported"
stance default, it recovered the label without ever using ST. The single-feature data check missed this
`source_type × stance` interaction. **Lesson (a paper point): text-isolation alone does not isolate
trust; source category leaks it.**

**v2 (text isolation + constant source category) — CLEAN.** Hold `source_type` CONSTANT across all
records (all `news_media`) and vary ONLY the continuous ST scalar (synthetic source ids `nm_*`). Now the
trust-blind input (MiniLM text emb + constant source_type one-hot) is byte-identical within each
high/low pair, so the model is provably at chance; ST is the single differing variable. Train and test
hold out distinct ST VALUES so a trust-aware model must use ST as a continuous monotonic signal, not
memorize values. `source_id` carries ST but is NOT a model feature (the repo featurizer only one-hots
`source_type`).

Labels use the repo's own EC formula + thresholds (testimony EW=0.80, IS=1.0): single item ->
support/refute score = EC = 1-(1-ST)^0.8; ST>=0.85 -> EC>=0.781 -> SUPPORTED/REFUTED; ST<=0.45 ->
EC<=0.380 -> NEE. ST tiers: high train {0.85, 0.90}, high test {0.86, 0.88}; low train {0.30, 0.40},
low test {0.35, 0.45}. Within each (claim, stance) the ONLY thing that flips the label is ST.

## Statistics
| Split | n | supported | NEE | refuted |
|-------|---|-----------|-----|---------|
| train | 1600 | 400 | 800 | 400 |
| test (held-out ST values) | 480 | 120 | 240 | 120 |

## Verification 1 — data-level (no training)
- 747/747 (100%) distinct evidence texts appear with >1 gold label -> text alone cannot determine the
  label.
- Distinct `source_type` values in the split: **1** (`news_media`) -> the trust-blind baseline's only
  non-text feature is constant -> it cannot distinguish high from low at all.

## Verification 2 — controlled feature-access experiment (the headline)
Repo-faithful features (all-MiniLM-L6-v2 text emb + source_type one-hot = exactly the trust-blind GNN's
node features), plain LogisticRegression so the result reflects FEATURES not capacity. Train on corrected
train, evaluate on held-out-ST test:

| Condition | features | test accuracy | test macro-F1 |
|---|---|---|---|
| Trust-blind | text emb + source_type (NO ST) | 0.500 | **0.447** (chance; NEE F1=0.008) |
| Trust-aware | + ST scalar (+ EC) | 1.000 | **1.000** |
| **Δ macro-F1 (trust effect)** | | | **+0.553** |

Reading: the corrected split is solvable iff the model uses the ST scalar. A trust-blind model is at
chance (it can only default supporting-text->supported); adding the single ST feature makes the
held-out test perfectly separable. The 1.000 is NOT a capability claim — it proves ST is necessary AND
sufficient here, which is exactly what "trust-isolating" means. (Contrast: on the ORIGINAL split a
trust-blind model scores ~0.996 because text register leaks the label.)

## What still needs the full repo GNN (Tier 2, next)
The controlled experiment proves the benchmark is valid. To show "the actual system" behaves the same and
to satisfy the experiment-review gate, still to run:
1. Repo `baseline` (trust-blind GNN) vs `v3-nli` (trust-aware) on the corrected split, >=3 seeds.
2. D1 confirmation: macro-F1 + majority baseline on AVeriTeC (does 0.665 collapse under macro-F1?).
3. EW ablation (Pramana vs random vs learned vs uniform) + trust ablation (ST vs flat learned trust).
   Requires adding the `nm_*` synthetic sources to the registry so the repo's ST lookup resolves them.

## Limitations to disclose
- Controlled, single-evidence, testimony-only, fictional-claim, constant-category by construction
  (deliberately, to isolate trust). It tests trust-sensitivity in a clean setting, not open-web realism;
  AVeriTeC (macro-F1 + majority baseline) carries the realism axis.
- The trust-aware 1.000 reflects a deterministic label = f(ST, stance); the value of the split is the
  CONTRAST with the trust-blind 0.447, not the absolute number. Report both.
- NEE is 50% by construction; report macro-F1 and per-class, with always-NEE as the floor.
