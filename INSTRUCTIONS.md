# Running the Expanded Trust-Isolating Benchmark (v3)

This benchmark includes:
- Larger claim vocabulary (24 things × 16 attrs × 24 values, reduces repetition)
- Mid-tier boundary probes (ST=0.50, 0.62, 0.72) that test continuous trust reasoning
- 3 independent GNN runs per model for variance estimation
- Always-NEE baseline reported alongside majority baseline

---

## Quick start (requires [`just`](https://just.systems))

```bash
just probe      # LogReg probe only — ~2 min, no GPU needed
just eval       # all 9 GNN runs — ~90 min, GPU recommended
just results    # aggregate results → summary.md + numbers.tex
just figures    # regenerate paper figures
just report     # results + figures in one step
just pdf        # compile paper PDF (requires LaTeX)
just all        # full rerun: data → probe → eval → results (stops here)
just paper      # regenerate figures + compile PDF (run after just all once results look correct)
just clean-checkpoints  # clear runs/ so next eval retrains from scratch
just clean-runs         # clear runs/ AND wipe gnn_results.jsonl (full wipe)
just validate   # verify paper numbers match committed references (no reruns needed)
just generate-refs  # regenerate reference snapshots after intentional result changes
```

> **`just validate` vs `just all` — two different questions:**
>
> - `just validate` answers: *"Are the paper numbers consistent with the committed result files?"*
>   Run this on the unmodified repo. It requires no GPU and always passes on the committed state.
>
> - `just all` answers: *"Can I reproduce similar results on my hardware?"*
>   This reruns all experiments from scratch. Results will be numerically close to the paper
>   but not bit-exact across platforms due to GPU floating-point non-determinism (e.g. CUDA
>   vs CPU, cuDNN version differences). After `just all`, `just validate` will fail against
>   the committed reference — this is expected, not a bug.
>
> **Do not run `just all` before `just validate`** if your goal is to verify the paper.
> If you rerun experiments and want validation to pass on your platform, run
> `just generate-refs` after `just all` to update the references to your results — but note
> this updates the reference, not the paper numbers.

---

## Prerequisites

**Install tools (once):**
```bash
# uv — fast Python package manager (handles all dependencies)
curl -LsSf https://astral.sh/uv/install.sh | sh

# just — task runner for the Justfile
brew install just          # macOS
# or: cargo install just   # cross-platform via Rust
```

**Install Python dependencies (from lock file — deterministic):**
```bash
just setup
# equivalent to: uv sync
```

The GNN evaluation additionally requires `git` on PATH. The GNN script auto-clones the
[system repo](https://github.com/karkid/epistemic-factkg) from GitHub on first use —
no manual setup required.

---

## Step 1 — Generate expanded benchmark data (optional)

The data is already committed at `data/expanded_split.jsonl` (2200 records).
Only re-run if you want to verify generation or change parameters:

```bash
just data
# or: uv run python benchmark/build_expanded_split.py
# → data/expanded_split.jsonl  (2200 records: train=1600, test=600)
```

---

## Step 2 — LogReg probe

Tests the paper's central claim with a linear model — no GNN needed.

```bash
just probe
# or: uv run python benchmark/run_logreg_test.py
# → results/logreg_results.json
```

**Expected output:**
```
always-NEE baseline: acc=0.600  macroF1=0.250
trust-blind        : acc=0.405  macroF1=0.389
trust-aware        : acc=0.887  macroF1=0.882
Δ macro-F1 (trust effect) = +0.493
mid-tier probe (trust-aware): macro-F1=0.204
```

---

## Step 3 — GNN evaluation (9 runs: 3 models × 3 runs)

Run from the repo root. Each run replaces its existing entry in `results/gnn_results.jsonl` — safe to interrupt and resume without creating duplicates.

```bash
just eval
# or a single run: just eval-one v2-hgnn 1
```

**Note:** Each run takes ~10 minutes. Completed checkpoints are cached under `runs/` —
re-running a finished run skips training and loads existing metrics. To force full
retraining from scratch, clear checkpoints first:
```bash
just clean-checkpoints   # removes runs/ only, leaves result files intact
just eval                # retrains all 9 runs fresh
```

---

## Step 4 — Aggregate results

```bash
just results
# or: uv run python analysis/aggregate.py
# → results/summary.md + paper/numbers.tex
```

---

## Step 5 — Verify numbers (optional but recommended)

Checks that all numbers in the results MDs and `paper/numbers.tex` match the committed
reference snapshots. No reruns needed — pure read-and-compare.

```bash
just validate
```

Three areas are checked:
- **split design** — record counts, EC values, LogReg and GNN aggregates vs `data/` and `results/`
- **reproducibility** — `paper/numbers.tex` macros and result file consistency
- **diagnostics** — D1/D2/D3 recomputed from `model_repo/data/raw/` source files
  (diagnostics are skipped with a SKIP notice if `model_repo/` is not present)

If any check fails, the output shows which value mismatched and what was expected.
After intentional result changes (e.g. re-running `just eval`), regenerate references with:

```bash
just generate-refs   # updates verification/*/reference.json — commit after running
```

---

## Expected results

| Model | Macro-F1 | NEE F1 |
|-------|----------|--------|
| Always-NEE baseline | 0.250 | 1.000 |
| LogReg trust-blind | 0.389 | — |
| LogReg trust-aware | 0.882 | — |
| Baseline GNN (no EC) | 0.385 ± 0.007 | 0.000 |
| v2-HGNN (EC layer) | 0.918 ± 0.017 | — |
| v3-NLI (EC + NLI) | 0.907 ± 0.019 | — |

Results are seeded (seeds 43, 44, 45 per run). Values are numerically close but not
bit-exact across platforms (CPU vs CUDA, different BLAS libraries). The committed results
above were obtained on macOS/CPU. Run `just all` to reproduce on your hardware.

**Key finding:** Trust-blind models (baseline GNN, LogReg without ST) collapse to NEE F1 ≈ 0.
Models with EC layer and correct threshold (0.75) correctly classify NEE for low-trust sources.
v2-HGNN outperforms v3-NLI because NLI features correlate perfectly with stance on template text,
creating a competing signal that the EC layer must overcome.

---

## File layout

```
├── Justfile                    ← just probe / eval / results / validate / all / paper / pdf
├── INSTRUCTIONS.md             ← this file
├── benchmark/
│   ├── build_expanded_split.py ← data generator (seed 20260611)
│   ├── run_logreg_test.py      ← LogReg probe (trust-blind vs trust-aware)
│   └── run_gnn_eval.py         ← GNN training + evaluation harness
├── analysis/
│   ├── aggregate.py            ← mean±std table + numbers.tex generator
│   └── midtier_eval.py         ← standard vs mid-tier breakdown
├── data/
│   ├── expanded_split.jsonl    ← 2200-record benchmark (committed)
│   ├── registry_with_nm.jsonl  ← source trust registry + nm_* entries
│   └── splits/                 ← train/val/test index files
├── results/
│   ├── logreg_results.json     ← LogReg probe results
│   ├── gnn_results.jsonl       ← 9 GNN run entries
│   ├── midtier_results.json    ← standard vs mid-tier breakdown
│   ├── summary.md              ← aggregated results + analysis findings
│   ├── corrected_split_design.md ← benchmark design, iteration history, verification
│   └── diagnostics_d1_d3.md   ← D1/D2/D3 diagnostic findings (motivation)
├── verification/
│   ├── compare.py              ← shared comparator: stdin JSON vs reference → ✓/✗
│   ├── split_design/           ← verifies corrected_split_design.md numbers
│   ├── reproducibility/        ← verifies numbers.tex + result file consistency
│   └── diagnostics/            ← recomputes D1/D2/D3 from model_repo/ source data
│       (requires model_repo/ clone — see just validate output if absent)
├── paper/
│   ├── paper.tex / paper.pdf   ← LaTeX source and compiled PDF
│   ├── numbers.tex             ← auto-generated macros (\input{numbers} in paper.tex)
│   ├── generate_figures.py     ← generates all figures + syncs to docs/figures/
│   └── figures/                ← pb_*.png
└── docs/
    ├── index.html              ← project website
    └── figures/                ← auto-synced from paper/figures/
```
