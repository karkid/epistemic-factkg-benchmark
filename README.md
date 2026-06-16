# Beyond Binary Trust: A Source-Trust-Isolating Benchmark for Continuous Epistemic Reasoning in Fact Verification

**Dheeraj Karki** — Independent Researcher (dheerajkarki1790@gmail.com)  
**Aiswarya Konavoor** — Togo AI Labs (aiswarya@togolabs.ai)

A diagnosis-and-benchmark paper on measuring **source-trust reasoning** in fact verification.

📄 **[Read the paper (PDF)](paper/paper.pdf)** &nbsp;·&nbsp; 🌐 **[Project website](docs/index.html)** &nbsp;·&nbsp; ⚙️ **[System repo](https://github.com/karkid/epistemic-factkg)** &nbsp;·&nbsp; 🧪 **[Benchmark repo](https://github.com/AISWARYA-NANDAKUMAR/epistemic-factkg-benchmark)**

---

## The one-paragraph version
To claim a fact-verification model uses *source trust*, you need a test only a trust-using model can pass.
The natural construction, pairing the same claim with a high-trust and a low-trust source, **secretly
leaks the verdict**: the low-trust evidence is written in a different register, so a model reads the style
instead of the source. We quantify the leak, show that AVeriTeC *accuracy* of 0.665 is just the
majority-class prior (macro-F1 0.265), build a **trust-isolating benchmark** that admits no detectable
non-trust path to the label, and report an **inversion**: a trust-blind model that *wins* on the leaky
split (0.996 accuracy) collapses to chance on the corrected one (macro-F1 0.385 ± 0.007), while a
trust-aware model wins (v2-HGNN macro-F1 0.918 ± 0.017). Mid-tier boundary probes further show that
access to the trust scalar alone is insufficient — trust must be reasoned over continuously.

## The inversion (headline result)
| Setting | Metric | Trust-blind | Trust-aware (v2-HGNN) | v3-NLI |
|---|---|---|---|---|
| Original split (leaky) | accuracy | **0.996** | 0.889 | — |
| Corrected, LogReg probe | macro-F1 | 0.389 | **0.882** | — |
| Corrected, GNN (3 runs) | macro-F1 | 0.385 ± 0.007 | **0.918 ± 0.017** | 0.907 ± 0.019 |
| Always-NEE floor | macro-F1 | 0.250 | — | — |
| Mid-tier probes (LogReg) | macro-F1 | 0.016 | — | 0.204 |

Source trust helps **only once the split stops leaking**.

## Repository layout
```
Justfile                       # just probe / eval / results / figures / pdf / all
INSTRUCTIONS.md                # step-by-step setup guide
paper/
  paper.pdf                    # camera-ready PDF
  paper.tex                    # LaTeX source
  numbers.tex                  # auto-generated LaTeX macros (from analysis/aggregate.py)
  generate_figures.py          # generates all 5 paper figures from results/
  figures/                     # pb_construction.png  pb_leakage.png  pb_ec_curve.png
                               #   pb_inversion.png  pb_architecture.png
benchmark/
  build_expanded_split.py      # seeded v3 generator (vocabulary expanded, mid-tier probes added)
  run_logreg_test.py           # controlled trust-blind vs trust-aware LogReg probe
  run_gnn_eval.py              # GNN training + evaluation harness (requires system repo)
analysis/
  aggregate.py                 # aggregate multi-seed GNN runs → numbers.tex + summary.md
  midtier_eval.py              # standard vs mid-tier breakdown (reads per_record_predictions.jsonl)
data/
  expanded_split.jsonl         # 2200-record benchmark (train 1600 / test 600)
  registry_with_nm.jsonl       # source trust registry used for generation
  splits/                      # train/val/test index files
results/
  gnn_results.jsonl            # per-seed GNN results (3 seeds × 3 models)
  logreg_results.json          # LogReg probe results incl. mid-tier breakdown
  midtier_results.json         # standard vs mid-tier macro-F1 breakdown
  summary.md                   # human-readable results summary
docs/
  index.html                   # project website (GitHub Pages)
  figures/                     # same figures as paper/figures/ (auto-synced)
```

## Reproduce

### Quick start (requires [`just`](https://just.systems) and [`uv`](https://docs.astral.sh/uv/))
```bash
# install tools (once)
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv
brew install just                                   # macOS; or: cargo install just

# install Python dependencies
just setup

# run what you need
just probe      # LogReg probe — ~2 min, no GPU
just figures    # regenerate paper figures from committed results
just report     # aggregate results + regenerate figures (shortcut)
just pdf        # compile paper PDF (requires LaTeX)
just all        # full pipeline: data → probe → eval → results → figures → pdf
just clean-runs # clear runs/ and gnn_results.jsonl before a fresh just all
```

### Step by step

**1 — Install dependencies**
```bash
uv sync   # installs from uv.lock — deterministic, no version surprises
```

**2 — LogReg probe** (no GPU, no system repo, ~2 min)
```bash
uv run python benchmark/run_logreg_test.py    # → results/logreg_results.json
```

**3 — GNN evaluation** (GPU recommended, ~10 min/run × 9 runs)
```bash
# auto-clones https://github.com/karkid/epistemic-factkg on first use
for model in baseline v2-hgnn v3-nli; do
  for run in 1 2 3; do
    uv run python benchmark/run_gnn_eval.py --model $model --run $run
  done
done
uv run python analysis/aggregate.py           # → results/summary.md + paper/numbers.tex
```

**4 — Regenerate paper figures**
```bash
uv run python paper/generate_figures.py       # → paper/figures/*.png + docs/figures/*.png
```

## How the benchmark isolates source trust
1. **Text isolation** — the same evidence text is attached to a high-trust and a low-trust source.
2. **Category isolation** — the source category is held constant, so a category feature cannot proxy
   trust (a first version leaked at macro-F1 0.82 through exactly this proxy).
3. **Value generalization** — trust values are held out at test, so a model must use trust as a
   continuous signal, not memorize values.
4. **Mid-tier probes** — records with ST ∈ {0.50, 0.62, 0.72} test whether the model generalises
   beyond binary thresholding; a LogReg probe with the trust scalar collapses to macro-F1 0.204.

Verification: 747/747 distinct texts appear with more than one gold label; every non-trust feature
predicts the verdict at chance (0.50).

## Limitations
Controlled, single-evidence, testimony-only, fictional-claim, single-category by design. It measures
trust sensitivity, not open-web realism. GNN results reported as mean ± std across three independent
training runs (seeds 43, 44, 45); v2-HGNN σ = 0.017, v3-NLI σ = 0.019. Results are fully
reproducible — run `just clean-runs && just all` to verify. The orthogonal evidence-type axis is
deliberately scoped out.

## License
MIT (see [LICENSE](LICENSE)).
