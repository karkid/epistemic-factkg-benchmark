# Benchmark Results Summary — Expanded Trust-Isolating Split (v3)

**Dataset:** `expanded_split.jsonl`  | train=1280  val=320  test=600 (480 standard + 120 mid-tier boundary probes)

---

## 1. Baselines

| Baseline | Accuracy | Macro-F1 | NEE F1 |
|----------|----------|----------|--------|
| Always-NEE | 0.600 | 0.250 | 1.000 |
| Always-majority (not_enough_evidence) | 0.600 | 0.250 | — |

---

## 2. LogReg Probe (feature-access test)

| Model | Accuracy | Macro-F1 | NEE F1 | mid-tier F1 |
|-------|----------|----------|--------|-------------|
| LogReg trust-blind (no ST) | 0.405 | 0.389 | 0.022 | 0.016 |
| LogReg trust-aware (+ ST + EC) | 0.887 | 0.882 | 0.897 | 0.204 |
| Δ trust effect | — | +0.493 | — | — |

**Key finding:** Trust-aware LogReg achieves high macro-F1 on standard pairs but scores 0.204 on mid-tier boundary probes — LogReg learns a binary threshold, not continuous ST reasoning.

---

## 3. GNN Models (3 independent runs each)

| Model | Runs | Accuracy | Macro-F1 | SUP F1 | REF F1 | NEE F1 |
|-------|------|----------|----------|--------|--------|--------|
| Baseline (no EC) | 3 | 0.402 ± 0.003 | 0.385 ± 0.007 | 0.569 ± 0.005 | 0.572 ± 0.001 | 0.014 ± 0.025 |
| v2-HGNN (EC layer) | 3 | 0.922 ± 0.018 | 0.918 ± 0.017 | 0.901 ± 0.035 | 0.923 ± 0.000 | 0.930 ± 0.017 |
| v3-NLI (EC + NLI) | 3 | 0.911 ± 0.019 | 0.907 ± 0.019 | 0.879 ± 0.038 | 0.923 ± 0.000 | 0.920 ± 0.019 |

---

## 4. Analysis

### What the benchmark tests
Each test record has byte-identical evidence text attached to either a high-trust or low-trust source.
A model that reasons over source trust correctly should predict SUPPORTED/REFUTED for high-trust
sources and NEE for low-trust sources — same text, different verdict purely from ST.

### Findings

**F1. Baseline collapses on NEE** — NEE F1 = 0.014. Without an EC layer, the model
predicts SUPPORTED/REFUTED based on text semantics alone. Since all evidence text is
semantically supporting or refuting (by construction), it never predicts NEE.

**F2. v2-HGNN partially solves the benchmark** — macro-F1 = 0.918 ± 0.017, NEE F1 = 0.930.
The EC layer correctly defers low-trust evidence to NEE. The HybridVerdictHead's access
to claim embeddings lets it resolve ambiguous cases better than a pure EC threshold.

**F3. v3-NLI underperforms v2-HGNN** — macro-F1 = 0.907 ± 0.019 vs 0.918 ± 0.017.
v3-NLI still far outperforms the trust-blind baseline (0.907 vs 0.385), confirming
the EC layer remains effective. However, on template text DeBERTa assigns near-perfect
entailment/contradiction regardless of source trust, creating a competing signal against the
EC layer. This instability raises variance (σ = 0.019 vs σ = 0.017 for v2-HGNN)
and reduces accuracy compared to v2-HGNN alone. The benchmark penalises text-semantic
shortcuts (including NLI) — models relying on them underperform v2-HGNN but do not fail outright.

**F4. Mid-tier boundary zone** — LogReg trust-aware scores only 0.204 on mid-tier probes (ST=0.50, 0.62, 0.72).
This shows that access to the ST scalar is not sufficient — a model must use it
continuously, not just learn a binary high/low threshold.

---

## 5. Decision-Path Analysis (v2-HGNN vs v3-NLI)

40% of test decisions are resolved symbolically (EC ≥ 0.75, high-trust records); 60% are deferred to the HybridVerdictHead (EC < 0.75, all NEE labels).

| Model | Symbolic (EC≥0.75) | VerdictHead acc | Mean errors/run | Error type |
|-------|-------------------|-----------------|-----------------|------------|
| v2-HGNN | 240/600, 100% | 87.0% | 47 | SUP/REF prediction on NEE records |
| v3-NLI  | 240/600, 100% | 85.2% | 53 | SUP/REF prediction on NEE records |

**Key finding:** The EC symbolic path is identical and perfect for both models. All errors come from the VerdictHead fallback path. v3-NLI makes ~53 errors/run vs ~47 for v2-HGNN — NLI features push VerdictHead toward SUPPORTED/REFUTED on records that should be NEE, amplifying the text-trust conflict in the fallback path.

---

## 6. Mid-tier GNN analysis

Mid-tier probes (ST ∈ {0.50, 0.62, 0.72}, all NEE) test whether models reason over ST continuously
rather than applying a binary learned threshold. On the 120 mid-tier records in isolation, the
**always-NEE floor is macro-F1 0.333** (one of three classes, none of the other two present).

| Model | Mid-tier macro-F1 (mean ± std) | vs always-NEE floor (0.333) |
|-------|-------------------------------|------------------------------|
| Baseline (trust-blind) | 0.005 ± 0.009 | −0.328 (predicts SUP/REF on all) |
| v2-HGNN (EC layer) | 0.252 ± 0.024 | −0.081 (below floor) |
| v3-NLI (EC + NLI)  | 0.237 ± 0.026 | −0.096 (below floor) |
| LogReg trust-aware | 0.204 | −0.129 (below floor) |

**F5. All models score below the always-NEE floor on mid-tier.** The trust-aware GNNs (v2-HGNN 0.252,
v3-NLI 0.237) and the trust-aware LogReg (0.204) all fall short of 0.333 — meaning they actively
mislabel some mid-tier records as SUPPORTED/REFUTED rather than NEE. The EC layer was trained on
binary high/low ST values (0.30–0.45 for low, 0.85–0.90 for high); mid-tier ST values (0.50–0.72)
fall in an untrained gap. The VerdictHead, seeing ST values closer to the high-trust training range
than to low-trust, partially fires SUPPORTED/REFUTED where the answer is NEE. This is the benchmark's
central open problem: continuous trust reasoning cannot be learned from binary-tier training data alone.

---

## 7. Full run log

| exp_id | acc | macro_f1 | NEE_f1 |
|--------|-----|----------|--------|
| GNN-baseline-run1 | 0.4000 | 0.3810 | 0.0000 |
| GNN-baseline-run2 | 0.4000 | 0.3810 | 0.0000 |
| GNN-baseline-run3 | 0.4050 | 0.3932 | 0.0429 |
| GNN-v2-hgnn-run1 | 0.9317 | 0.9274 | 0.9396 |
| GNN-v2-hgnn-run2 | 0.9333 | 0.9291 | 0.9412 |
| GNN-v2-hgnn-run3 | 0.9017 | 0.8980 | 0.9107 |
| GNN-v3-nli-run1 | 0.9000 | 0.8964 | 0.9091 |
| GNN-v3-nli-run2 | 0.9000 | 0.8964 | 0.9091 |
| GNN-v3-nli-run3 | 0.9333 | 0.9291 | 0.9412 |

_Generated by `analysis/aggregate.py`_
