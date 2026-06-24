"""GNN training + evaluation harness for the expanded trust-isolating split.

Trains a specified model on the expanded_split.jsonl (1600 train records) and evaluates
on the held-out test set (600 records: 480 standard + 120 mid-tier boundary probes).

Model repo resolution (in order):
  1. If model_repo/ already contains a clone, reuse it (cached).
  2. Otherwise clone https://github.com/karkid/epistemic-factkg into model_repo/ and
     run `uv sync` once.

No file in the original benchmark/ or in the main model repo is modified.

Usage (from any directory):
    python run_gnn_eval.py --model baseline --run 1
    python run_gnn_eval.py --model v2-hgnn  --run 2
    python run_gnn_eval.py --model v3-nli   --run 3

Run matrix: {baseline, v2-hgnn, v3-nli} × {run 1, 2, 3} = 9 total runs.
Results appended to review/12062026/results/gnn_results.jsonl.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

GITHUB_URL   = "https://github.com/karkid/epistemic-factkg"
ROOT   = Path(__file__).resolve().parents[1]   # review/12062026/
MODEL_REPO   = ROOT / "model_repo"             # cloned here if local not found
SPLIT_JSONL  = ROOT / "data" / "expanded_split.jsonl"
REGISTRY_OUT = ROOT / "data" / "registry_with_nm.jsonl"
SPLITS_DIR   = ROOT / "data" / "splits"
EMBED_CACHE  = ROOT / "data" / "embed_cache.pkl"
RESULTS_OUT  = ROOT / "results" / "gnn_results.jsonl"
RUNS_DIR     = ROOT / "runs"

NM_ENTRIES = [
    {"source_id": "nm_h85a", "source_trust": 0.85, "prior_trust": 0.85},
    {"source_id": "nm_h90a", "source_trust": 0.90, "prior_trust": 0.90},
    {"source_id": "nm_l30a", "source_trust": 0.30, "prior_trust": 0.30},
    {"source_id": "nm_l40a", "source_trust": 0.40, "prior_trust": 0.40},
    {"source_id": "nm_h88x", "source_trust": 0.88, "prior_trust": 0.88},
    {"source_id": "nm_h86x", "source_trust": 0.86, "prior_trust": 0.86},
    {"source_id": "nm_l35x", "source_trust": 0.35, "prior_trust": 0.35},
    {"source_id": "nm_l45x", "source_trust": 0.45, "prior_trust": 0.45},
    {"source_id": "nm_m50x", "source_trust": 0.50, "prior_trust": 0.50},
    {"source_id": "nm_m62x", "source_trust": 0.62, "prior_trust": 0.62},
    {"source_id": "nm_m72x", "source_trust": 0.72, "prior_trust": 0.72},
]
_NM_TEMPLATE = {
    "source_type": "news_media", "modality": "web_text", "category": "news_media",
    "default_inference_strength": 0.80,
    "trust_metadata": {"methodology_ref": "tier1-corrected_benchmark", "version": "expanded_v3"},
}


# ============================================================
# Step 0 — Locate or clone the model repo
# ============================================================
def _find_or_clone_repo() -> Path:
    """Return path to epistemic-factkg. Use cached clone if present, else clone."""
    if (MODEL_REPO / "src" / "pipeline" / "model" / "train.py").exists():
        print(f"  model repo: using cached clone at {MODEL_REPO}")
        return MODEL_REPO

    print(f"  model repo not found — cloning {GITHUB_URL}")
    MODEL_REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", GITHUB_URL, str(MODEL_REPO)], check=True)
    print("  running uv sync to install dependencies...")
    subprocess.run(["uv", "sync"], cwd=MODEL_REPO, check=True)
    print("  clone ready.")
    return MODEL_REPO


# ============================================================
# Step 1 — Infrastructure setup (registry + split index files)
# ============================================================
def setup_registry(main_repo: Path):
    """Write registry_with_nm.jsonl: original registry + nm_* synthetic entries."""
    if REGISTRY_OUT.exists():
        print(f"  registry: already exists ({REGISTRY_OUT.name})")
        return
    REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    registry_src = main_repo / "data" / "registry" / "source_trust_registry.jsonl"
    existing = {json.loads(l)["source_id"] for l in open(registry_src)}
    REGISTRY_OUT.write_text(registry_src.read_text())
    with open(REGISTRY_OUT, "a") as out:
        for entry in NM_ENTRIES:
            if entry["source_id"] not in existing:
                out.write(json.dumps({**_NM_TEMPLATE, **entry}) + "\n")
    print(f"  registry: written with +{len(NM_ENTRIES)} nm_* entries")


def setup_splits():
    """Create train/val/test_indices.json in the format train.py expects: {indices, meta}."""
    if (SPLITS_DIR / "test_indices.json").exists():
        # Verify format is correct (not a plain list from an earlier broken run)
        raw = json.loads((SPLITS_DIR / "test_indices.json").read_text())
        if isinstance(raw, dict) and "indices" in raw:
            print(f"  splits: already exist ({SPLITS_DIR.name}/)")
            return
        print("  splits: found old plain-list format — regenerating with correct format")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(SPLIT_JSONL)]
    train_all = [i for i, r in enumerate(recs) if r["provenance"]["split"] == "train"]
    test_idx  = [i for i, r in enumerate(recs) if r["provenance"]["split"] == "test"]
    split_pt  = int(len(train_all) * 0.80)
    train_idx = train_all[:split_pt]
    val_idx   = train_all[split_pt:]

    for name, indices, frac in [
        ("train", train_idx, 0.80),
        ("val",   val_idx,   0.20),
        ("test",  test_idx,  1.00),
    ]:
        payload = {"indices": indices, "meta": {"seed": 0, "train_frac": frac}}
        (SPLITS_DIR / f"{name}_indices.json").write_text(json.dumps(payload))

    print(f"  splits: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")


# ============================================================
# Step 2 — Train + evaluate one run
# ============================================================
def run_model(model: str, run_id: int, main_repo: Path) -> dict:
    run_name  = f"{model}_run{run_id}"
    ckpt_dir  = RUNS_DIR / run_name / "checkpoints"
    eval_dir  = RUNS_DIR / run_name / "eval"
    metrics_file = eval_dir / "eval" / "verdict_metrics.json"

    if metrics_file.exists():
        print(f"  [{run_name}] already done — loading existing metrics")
        return json.loads(metrics_file.read_text())

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / "best_model.pt"
    if not best_ckpt.exists():
        print(f"\n{'='*60}\n  TRAINING  model={model}  run={run_id}\n{'='*60}")
        subprocess.run([
            "uv", "run", "python", "-m", "src.pipeline.model.train",
            "--jsonl",          str(SPLIT_JSONL),
            "--model",          model,
            "--model-name",     run_name,
            "--registry",       str(REGISTRY_OUT),
            "--splits-dir",     str(SPLITS_DIR),
            "--checkpoint-dir", str(ckpt_dir),
            "--report-dir",     str(RUNS_DIR / run_name / "train_report"),
            "--embed-cache",    str(EMBED_CACHE),
            "--ec-threshold",   "0.75",
            "--epochs",         "30",
            "--seed",           str(42 + run_id),
            "--verbose",
        ], cwd=main_repo, check=True)
    else:
        print(f"  [{run_name}] checkpoint exists — skipping training")

    print(f"\n{'='*60}\n  EVALUATING  model={model}  run={run_id}\n{'='*60}")
    # evaluate.py writes to {output}/{run-id}/verdict_metrics.json; --run-id makes path deterministic
    subprocess.run([
        "uv", "run", "python", "-m", "src.pipeline.model.evaluate",
        "--checkpoint",  str(best_ckpt),
        "--jsonl",       str(SPLIT_JSONL),
        "--model",       model,
        "--model-name",  run_name,
        "--registry",    str(REGISTRY_OUT),
        "--splits-dir",  str(SPLITS_DIR),
        "--output",      str(eval_dir),
        "--run-id",      "eval",
        "--embed-cache", str(EMBED_CACHE),
    ], cwd=main_repo, check=True)

    return json.loads(metrics_file.read_text())


# ============================================================
# Step 3 — Append result to gnn_results.jsonl
# ============================================================
def save_result(model: str, run_id: int, metrics: dict) -> dict:
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "exp_id":    f"GNN-{model}-run{run_id}",
        "model":     model,
        "run_id":    run_id,
        "dataset":   "expanded_split_v3_test",
        "n_claims":  metrics.get("n_claims"),
        "accuracy":  metrics.get("accuracy"),
        "macro_f1":  metrics.get("macro_f1"),
        "per_class": {
            cls: v.get("f1") for cls, v in metrics.get("per_class", {}).items()
        },
    }
    existing = []
    if RESULTS_OUT.exists():
        existing = [json.loads(l) for l in open(RESULTS_OUT) if l.strip()]
    existing = [e for e in existing if e.get("exp_id") != entry["exp_id"]]
    existing.append(entry)
    with open(RESULTS_OUT, "w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")
    print(f"\n  → saved: acc={entry['accuracy']:.4f}  macro_f1={entry['macro_f1']:.4f}")
    return entry


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train + evaluate a GNN model on the expanded trust-isolating split.")
    ap.add_argument("--model", required=True, choices=["baseline", "v2-hgnn", "v3-nli"])
    ap.add_argument("--run",   type=int, required=True, choices=[1, 2, 3],
                    help="Run number 1-3 (independent runs for variance estimation)")
    args = ap.parse_args()

    print("\n[setup] locating model repo...")
    main_repo = _find_or_clone_repo()

    print("\n[setup] preparing data infrastructure...")
    setup_registry(main_repo)
    setup_splits()

    print(f"\n[run] model={args.model}  run={args.run}")
    metrics = run_model(args.model, args.run, main_repo)
    entry   = save_result(args.model, args.run, metrics)

    print(f"\n[done] {args.model} run={args.run}: "
          f"acc={entry['accuracy']:.4f}  macro-F1={entry['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
