# epistemic-factkg-benchmark
# Run `just` to list all commands.

[default]
[doc("List available commands")]
default:
    @just --list

# ── Variables ────────────────────────────────────────────────────────────────
PYTHON        := "uv run python"
LATEXMK       := "latexmk"
PAPER_DIR     := "paper"
PAPER_TEX     := PAPER_DIR / "paper.tex"
PAPER_PDF     := PAPER_DIR / "paper.pdf"

# ── Setup ─────────────────────────────────────────────────────────────────────

[doc("Install all Python dependencies from uv.lock (deterministic, reproducible).")]
setup:
    uv sync

# ── Data ─────────────────────────────────────────────────────────────────────

[doc("Generate expanded benchmark dataset (2200 records). Data already committed — only re-run to verify or regenerate.")]
data:
    {{PYTHON}} benchmark/build_expanded_split.py

# ── Evaluation ───────────────────────────────────────────────────────────────

[doc("Run LogReg probe (trust-blind vs trust-aware). Fast (~2 min, no GPU needed).")]
probe:
    {{PYTHON}} benchmark/run_logreg_test.py

[doc("Run all 9 GNN evaluations (3 models × 3 runs). Slow (~10 min/run). Completed runs are cached and skipped.")]
eval:
    for model in baseline v2-hgnn v3-nli; do \
        for run in 1 2 3; do \
            {{PYTHON}} benchmark/run_gnn_eval.py --model $model --run $run; \
        done; \
    done

[doc("Run a single GNN evaluation. Usage: just eval-one baseline 1")]
eval-one model run:
    {{PYTHON}} benchmark/run_gnn_eval.py --model {{model}} --run {{run}}

# ── Results & figures ─────────────────────────────────────────────────────────

[doc("Aggregate GNN + LogReg results into summary.md and numbers.tex. Auto-triggers midtier_eval.py if stale.")]
results:
    {{PYTHON}} analysis/aggregate.py

[doc("Regenerate all paper figures and sync to docs/figures/. Reads from results/*.json.")]
figures:
    {{PYTHON}} paper/generate_figures.py

[doc("Aggregate results + regenerate figures in one step (analytics shortcut).")]
report: results figures

# ── Paper ─────────────────────────────────────────────────────────────────────

[doc("Compile paper PDF with latexmk. Requires a LaTeX installation.")]
pdf:
    {{LATEXMK}} -pdf -cd {{PAPER_TEX}}

[doc("Clean LaTeX build artifacts.")]
pdf-clean:
    {{LATEXMK}} -C -cd {{PAPER_TEX}}

# ── Compound targets ──────────────────────────────────────────────────────────

[doc("Regenerate figures + compile PDF (does not re-run benchmark).")]
paper: figures pdf

[doc("Full reproducibility run: data → probe → eval → results → figures → PDF.")]
all: data probe eval results figures pdf

[doc("Clear GNN results and cached runs so the next eval re-trains from scratch.")]
clean-runs:
    rm -rf runs/
    @> results/gnn_results.jsonl
    @echo "Cleared runs/ and gnn_results.jsonl — next just eval will retrain from scratch."

# ── Verification ──────────────────────────────────────────────────────────────

[doc("Validate all four areas (split design, reproducibility, diagnostics, figures) against committed references. No reruns — pure read-and-compare.")]
validate:
    @echo "── split design ──────────────────────────────"
    {{PYTHON}} verification/split_design/generate.py | \
      {{PYTHON}} verification/compare.py verification/split_design/reference.json
    @echo "── reproducibility ───────────────────────────"
    {{PYTHON}} verification/reproducibility/generate.py | \
      {{PYTHON}} verification/compare.py verification/reproducibility/reference.json
    @echo "── diagnostics ───────────────────────────────"
    {{PYTHON}} verification/diagnostics/generate.py | \
      {{PYTHON}} verification/compare.py verification/diagnostics/reference.json
    @echo "── figures ───────────────────────────────────"
    {{PYTHON}} verification/figures/generate.py | \
      {{PYTHON}} verification/compare.py verification/figures/reference.json

[doc("Regenerate all reference.json snapshots after intentional result changes. Commit after running.")]
generate-refs:
    {{PYTHON}} verification/split_design/generate.py --save
    {{PYTHON}} verification/reproducibility/generate.py --save
    {{PYTHON}} verification/diagnostics/generate.py --save
    {{PYTHON}} verification/figures/generate.py --save
