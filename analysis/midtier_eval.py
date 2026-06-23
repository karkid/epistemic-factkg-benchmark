"""
Mid-tier vs standard accuracy breakdown — pure analytics.

Reads per_record_predictions.jsonl produced by evaluate.py for each run,
cross-references with ST tiers from expanded_split.jsonl, and computes
separate macro-F1 for mid-tier (ST ∈ {0.50, 0.62, 0.72}) and standard records.

No model loading. Run after evaluate.py has produced per_record_predictions.jsonl.

Run from the benchmark repo root:
  uv run python analysis/midtier_eval.py
"""
from __future__ import annotations
import json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR   = ROOT / "runs"
DATA_JSONL = ROOT / "data" / "expanded_split.jsonl"
OUT_FILE   = ROOT / "results" / "midtier_results.json"

MID_ST = {0.50, 0.62, 0.72}
VERDICT_TO_INT = {"supported": 0, "refuted": 1, "not_enough_evidence": 2}


def _macro_f1(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"accuracy": None, "macro_f1": None, "n": 0}
    correct = sum(p == t for p, t in pairs)
    f1s = []
    for c in range(3):
        tp = sum(p == c and t == c for p, t in pairs)
        fp = sum(p == c and t != c for p, t in pairs)
        fn = sum(p != c and t == c for p, t in pairs)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return {"accuracy": round(correct / len(pairs), 4),
            "macro_f1": round(sum(f1s) / 3, 4), "n": len(pairs)}


def main():
    # Build ST tier map from the test split
    st_map: dict[str, str] = {}
    for line in DATA_JSONL.open():
        r = json.loads(line)
        if r["provenance"]["split"] == "test":
            st = round(r["meta"]["source_trust"], 2)
            st_map[r["id"]] = "mid" if st in MID_ST else "standard"
    print(f"ST map: {sum(1 for v in st_map.values() if v=='mid')} mid-tier, "
          f"{sum(1 for v in st_map.values() if v=='standard')} standard")

    if not RUNS_DIR.exists():
        print(f"runs/ not found at {RUNS_DIR} — writing empty results")
        OUT_FILE.write_text(json.dumps({"per_run": [], "summary": {}}, indent=2))
        return

    results = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        # evaluate.py writes to run_dir/eval/eval/per_record_predictions.jsonl
        # 'latest' symlink points to 'eval' inside run_dir/eval/
        pred_file = run_dir / "eval" / "latest" / "per_record_predictions.jsonl"
        if not pred_file.exists():
            pred_file = run_dir / "eval" / "eval" / "per_record_predictions.jsonl"
        if not pred_file.exists():
            print(f"  skip {run_dir.name}: per_record_predictions.jsonl not found")
            continue

        run_name  = run_dir.name
        model_key = run_name.rsplit("_run", 1)[0]

        records_by_tier: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for line in pred_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec  = json.loads(line)
            p    = VERDICT_TO_INT.get(rec["predicted"],  2)
            t    = VERDICT_TO_INT.get(rec["true_label"], 2)
            tier = st_map.get(rec["claim_id"], "standard")
            records_by_tier[tier].append((p, t))

        row = {
            "run":      run_name,
            "model":    model_key,
            "mid":      _macro_f1(records_by_tier["mid"]),
            "standard": _macro_f1(records_by_tier["standard"]),
        }
        results.append(row)
        print(f"  {run_name}: mid={row['mid']}  standard={row['standard']}")

    # Aggregate by model
    by_model: dict[str, list] = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)

    summary = {}
    for model, runs in sorted(by_model.items()):
        mid_f1s = [r["mid"]["macro_f1"] for r in runs if r["mid"]["macro_f1"] is not None]
        std_f1s = [r["standard"]["macro_f1"] for r in runs if r["standard"]["macro_f1"] is not None]
        summary[model] = {
            "mid_macro_f1_mean": round(statistics.mean(mid_f1s), 4)  if mid_f1s else None,
            "mid_macro_f1_std":  round(statistics.stdev(mid_f1s), 4) if len(mid_f1s) > 1 else 0.0,
            "std_macro_f1_mean": round(statistics.mean(std_f1s), 4)  if std_f1s else None,
            "std_macro_f1_std":  round(statistics.stdev(std_f1s), 4) if len(std_f1s) > 1 else 0.0,
            "n_runs": len(runs),
        }
        print(f"\n{model}: mid={summary[model]['mid_macro_f1_mean']} "
              f"± {summary[model]['mid_macro_f1_std']}  "
              f"std={summary[model]['std_macro_f1_mean']} "
              f"± {summary[model]['std_macro_f1_std']}")

    OUT_FILE.write_text(json.dumps({"per_run": results, "summary": summary}, indent=2))
    print(f"\nResults → {OUT_FILE}")


if __name__ == "__main__":
    main()
