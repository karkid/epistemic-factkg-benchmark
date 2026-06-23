# Expanded Trust-Isolating Split (v3) — Design, Iteration & Verification

_Generator: `benchmark/build_expanded_split.py` (seed 20260611, date the data was generated). Data: `data/expanded_split.jsonl`
(2200 records). Controlled test: `benchmark/run_logreg_test.py`. GNN evaluation:
`benchmark/run_gnn_eval.py`. This is the paper's central methodological artifact: a fact-verification
split that only graded source trust can solve._

## Motivation
The original synthetic split leaks the verdict through text register (diagnostic D2): low-trust sources
always get rumour-style phrasings, high-trust get official phrasings, so a trust-blind text model scores
~0.996 without using source trust. The corrected split removes every non-trust path to the label.

---

## Design iteration (this is itself a finding)

**v1 (text isolation only) — STILL LEAKED.**
First attempt held evidence text byte-identical across a high-trust and a low-trust source, neutral
register, with disjoint train/test source sets. Data-level single-feature checks looked clean (text
predicted the label at chance 0.50). But the controlled experiment exposed a residual leak: trust-blind
baseline scored **macro-F1 0.824 on held-out test**. Cause: the baseline sees a 6-d `source_type` one-hot;
category correlates with trust in the registry, so it transferred to most held-out sources combined with a
supporting-text→supported stance default. Text-isolation alone does not isolate trust; source category
leaks it. This is a paper point (Section 3).

**v2 (text isolation + constant source category) — CLEAN.**
Hold `source_type` constant across all records (all `news_media`), vary only the continuous ST scalar
(synthetic source ids `nm_*`). The trust-blind input (MiniLM text emb + constant source_type one-hot) is
byte-identical within each high/low pair; ST is the single differing variable. Train and test hold out
distinct ST values so a trust-aware model must use ST as a continuous monotonic signal, not memorise
values.

Labels use the EC formula (testimony EW=0.80, IS=1.0): EC = 1−(1−ST)^0.8.
- ST ≥ 0.85 → EC ≥ 0.781 → SUPPORTED / REFUTED
- ST ≤ 0.45 → EC ≤ 0.380 → NOT_ENOUGH_EVIDENCE

**v3 (expanded vocabulary + mid-tier boundary probes) — CURRENT.**
Expands claim vocabulary (24 things × 16 attrs × 24 values) to reduce repetition, and adds mid-tier
boundary probes (ST ∈ {0.50, 0.62, 0.72}, EC ∈ {0.426, 0.539, 0.639}) that fall below the EC decision
threshold (0.75) and are labelled NOT_ENOUGH_EVIDENCE. These probes test whether a model reasons over
the ST scalar continuously rather than learning a binary high/low threshold. Trust tiers:

| ST values | Tier | EC range | Label |
|-----------|------|----------|-------|
| 0.85, 0.90 (train) / 0.86, 0.88 (test) | high | ≥ 0.781 | SUPPORTED / REFUTED |
| 0.30, 0.40 (train) / 0.35, 0.45 (test) | low | ≤ 0.380 | NOT_ENOUGH_EVIDENCE |
| 0.50, 0.62, 0.72 (test only) | mid | 0.426–0.639 | NOT_ENOUGH_EVIDENCE |

---

## Statistics

| Split | n | supported | refuted | NEE |
|-------|---|-----------|---------|-----|
| train | 1600 | 400 | 400 | 800 |
| test — standard (high/low) | 480 | 120 | 120 | 240 |
| test — mid-tier probes | 120 | 0 | 0 | 120 |
| **test total** | **600** | **120** | **120** | **360** |

---

## Design note — trust tags and registry
Each record carries `meta.source_trust` and `meta.trust_tier` directly because
`benchmark/run_logreg_test.py` reads these fields from the record — it is a self-contained
evaluation script that does not load the registry. The GNN evaluation (`run_gnn_eval.py`) takes
the opposite path: it ignores `meta` entirely and derives source trust from `registry_with_nm.jsonl`
via the `--registry` flag passed to model training.

All three GNN variants (baseline, v2-HGNN, v3-NLI) read `source_trust` from the registry in the
same way — the distinction between trust-blind and trust-aware is not where they read the value
but what they do with it. The baseline computes EC internally but routes verdict through a plain MLP
with `has_ec: False`, discarding the EC score. v2-HGNN and v3-NLI activate the EC decision path
(`has_ec: True`), making the symbolic verdict directly from the EC score when EC ≥ 0.75.

The `nm_*` source IDs were created specifically for this benchmark; their ST values in the registry
and in each record's `meta` are the same values by construction. Consistency between the two
representations is verified by `just validate` (split_design check: `registry_mismatches: 0`).

## Verification 1 — data-level (no training)
- All 1028 standard-tier texts (high/low pairs) appear with >1 gold label → same text yields
  SUPPORTED/REFUTED (high-trust) and NEE (low-trust); text alone cannot determine the verdict.
- 40 mid-tier test texts are unique to mid-tier probes (test-only, all NEE) — they share vocabulary
  and source category with standard records; only the ST value distinguishes them.
- Distinct `source_type` values: **1** (`news_media`) → trust-blind baseline's only non-text feature
  is constant → cannot distinguish high from low at all.

## Verification 2 — controlled feature-access experiment
Repo-faithful features (all-MiniLM-L6-v2 text emb + source_type one-hot = exactly the trust-blind GNN's
node features), plain LogisticRegression so the result reflects features not model capacity:

| Condition | Features | All test macro-F1 | Standard macro-F1 | Mid-tier macro-F1 |
|-----------|----------|-------------------|-------------------|-------------------|
| Trust-blind | text + source_type (no ST) | **0.389** | 0.447 | 0.016 |
| Trust-aware | + ST scalar + EC | **0.882** | 0.997 | 0.204 |
| **Δ macro-F1 (trust effect)** | | **+0.493** | +0.551 | +0.188 |

Key readings:
- Trust-blind collapses (0.389) — cannot use ST, text is identical within pairs.
- Trust-aware near-perfect on standard pairs (0.997) — ST is necessary and sufficient for binary tiers.
- Trust-aware collapses on mid-tier (0.204) — LogReg learns a binary threshold from training; mid-tier
  ST values (0.50–0.72) are not seen during training, so the model cannot generalise. This is the
  paper's central new finding: continuous trust reasoning is harder than binary thresholding.

## Verification 3 — GNN evaluation (3 models × 3 runs)
Results from `benchmark/run_gnn_eval.py` (see `results/gnn_results.jsonl`):

| Model | Macro-F1 (mean ± std) | NEE F1 |
|-------|----------------------|--------|
| Baseline GNN (trust-blind) | 0.385 ± 0.007 | 0.014 |
| v2-HGNN (EC symbolic layer) | 0.918 ± 0.017 | 0.930 |
| v3-NLI (EC + DeBERTa NLI) | 0.907 ± 0.019 | 0.920 |

v3-NLI underperforms v2-HGNN because DeBERTa NLI assigns near-perfect entailment/contradiction to all
template text regardless of source trust, creating a competing signal the EC layer must overcome.
This is itself a benchmark finding: NLI-enriched models cannot bypass the trust-isolation constraint.

---

## Limitations
- Controlled, single-evidence, testimony-only, fictional-claim, constant-category by construction
  (deliberately, to isolate trust). Tests trust-sensitivity in a clean setting, not open-web realism;
  AVeriTeC (macro-F1 + majority baseline) carries the realism axis.
- Mid-tier probes are all NOT_ENOUGH_EVIDENCE by the EC formula; models do not see mid-tier ST during
  training, so the 0.204 measures generalisation to unseen continuous ST, not classification accuracy.
- Report macro-F1 and per-class F1; always-NEE floor is macro-F1 0.250.
