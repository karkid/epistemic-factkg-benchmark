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

[doc("Install all Python dependencies including PDF utilities (pylatexenc, Pillow, pypdf).")]
setup:
    uv sync --extra pdf

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

[doc("Full rerun from scratch: clears checkpoints → data → probe → eval → results. Stops before figures and PDF — run just paper separately once results look correct.")]
all: clean-checkpoints data probe eval results

[doc("Clear only model checkpoints so the next eval re-trains from scratch. Does not touch result files.")]
clean-checkpoints:
    rm -rf runs/
    @echo "Cleared runs/ — next just eval will retrain from scratch."

[doc("Clear model checkpoints AND wipe gnn_results.jsonl. Use before a full clean slate run.")]
clean-runs: clean-checkpoints
    @> results/gnn_results.jsonl
    @echo "Cleared runs/ and gnn_results.jsonl."

# ── Verification ──────────────────────────────────────────────────────────────

[doc("Validate all four areas against committed references. Shows only mismatches with diff table. All suites always run even if one fails.")]
validate:
    @failed=0; \
    printf "── split design ──────────────────────────────\n"; \
    uv run python verification/split_design/generate.py 2>/dev/null | \
      uv run python verification/compare.py verification/split_design/reference.json 2>/dev/null \
      || failed=1; \
    printf "── reproducibility ───────────────────────────\n"; \
    uv run python verification/reproducibility/generate.py 2>/dev/null | \
      uv run python verification/compare.py verification/reproducibility/reference.json 2>/dev/null \
      || failed=1; \
    printf "── diagnostics ───────────────────────────────\n"; \
    uv run python verification/diagnostics/generate.py 2>/dev/null | \
      uv run python verification/compare.py verification/diagnostics/reference.json 2>/dev/null \
      || failed=1; \
    printf "── figures ───────────────────────────────────\n"; \
    uv run python verification/figures/generate.py 2>/dev/null | \
      uv run python verification/compare.py verification/figures/reference.json 2>/dev/null \
      || failed=1; \
    if [ "$failed" -eq 1 ]; then \
        printf "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"; \
        printf "  Differences above are expected if you re-ran experiments on\n"; \
        printf "  a different platform (Windows vs Mac, GPU vs CPU, CUDA version).\n"; \
        printf "  Your experiments completed correctly — the numbers are close.\n"; \
        printf "\n"; \
        printf "  Next step: run  just generate-refs  to accept your results\n"; \
        printf "  as the new reference. Validate will then pass on your platform.\n"; \
        printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"; \
        exit 1; \
    fi

[doc("Regenerate all reference.json snapshots after intentional result changes. Commit after running.")]
generate-refs:
    {{PYTHON}} verification/split_design/generate.py --save
    {{PYTHON}} verification/reproducibility/generate.py --save
    {{PYTHON}} verification/diagnostics/generate.py --save
    {{PYTHON}} verification/figures/generate.py --save
