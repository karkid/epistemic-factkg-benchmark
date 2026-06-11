# When "Shortcut-Breaking" Splits Leak: A Source-Trust-Isolating Benchmark for Fact Verification

**Dheeraj Karki, Aiswarya Konavoor — Togo AI Labs** (contact: aiswarya@togolabs.ai)

A diagnosis-and-benchmark paper on measuring **source-trust reasoning** in fact verification.

📄 **[Read the paper (PDF)](paper/paper.pdf)** &nbsp;·&nbsp; 🌐 **[Project website](https://sreedath.github.io/epistemic-factkg-benchmark/)**

---

## The one-paragraph version
To claim a fact-verification model uses *source trust*, you need a test only a trust-using model can pass.
The natural construction, pairing the same claim with a high-trust and a low-trust source, **secretly
leaks the verdict**: the low-trust evidence is written in a different register, so a model reads the style
instead of the source. We quantify the leak, show that AVeriTeC *accuracy* of 0.665 is just the
majority-class prior (macro-F1 0.265), build a **trust-isolating benchmark** that admits no detectable
non-trust path to the label, and report an **inversion**: a trust-blind model that *wins* on the leaky
split (0.996 accuracy) collapses to chance on the corrected one (macro-F1 0.444), while a trust-aware
model wins (0.772 with the original graph network, 1.000 with a controlled probe).

## The inversion (headline result)
| Setting | Metric | Trust-blind | Trust-aware |
|---|---|---|---|
| Original split (leaky) | accuracy | **0.996** | 0.889 |
| Corrected, controlled probe | macro-F1 | 0.447 | **1.000** |
| Corrected, graph network | macro-F1 | 0.444 | **0.772** |

Source trust helps **only once the split stops leaking**.

## Repository layout
```
paper/        paper.pdf, paper.tex, figures/        # the paper
benchmark/    build_corrected_split.py              # seeded generator (seed 20260611)
              corrected_split.jsonl                 # the 2080-record benchmark (train 1600 / test 480)
              run_trust_isolation_test.py           # controlled trust-blind vs trust-aware probe
results/      results.jsonl                         # every number in the paper, traceable
              diagnostics_d1_d3.md                  # the leakage diagnosis (D1-D3)
              corrected_split_design.md             # benchmark design + verification + v1->v2 lesson
docs/         index.html                            # the project website (GitHub Pages)
```

## Reproduce
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install scikit-learn sentence-transformers numpy
python benchmark/build_corrected_split.py          # regenerate the corrected split (seeded)
python benchmark/run_trust_isolation_test.py        # trust-blind ~chance vs trust-aware solves it
```
The audited system (Epistemic FactKG) lives at https://github.com/karkid/epistemic-factkg ; the faithful
graph-network numbers (baseline vs v2-hgnn) were produced by running that system on this benchmark.

## How the benchmark isolates source trust
1. **Text isolation** — the same evidence text is attached to a high-trust and a low-trust source.
2. **Category isolation** — the source category is held constant, so a category feature cannot proxy trust
   (a first version leaked at macro-F1 0.82 through exactly this proxy).
3. **Value generalization** — trust values are held out at test, so a model must use trust as a continuous
   signal, not memorize values.

Verification: 747/747 distinct texts appear with more than one gold label; every non-trust feature
predicts the verdict at chance (0.50).

## Limitations
Controlled, single-evidence, testimony-only, fictional-claim, single-category by design. It measures trust
sensitivity, not open-web realism. Single training run (multi-seed is future work). The orthogonal
evidence-type axis is deliberately scoped out.

## License
MIT (see [LICENSE](LICENSE)).
