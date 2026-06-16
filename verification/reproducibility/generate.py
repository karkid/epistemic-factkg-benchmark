"""Generate reproducibility snapshot from result and derived files.

Reads gnn_results.jsonl, logreg_results.json, midtier_results.json,
paper/numbers.tex, and results/summary.md — then prints a JSON snapshot to
stdout (or writes to reference.json with --save).

Usage:
    uv run python verification/reproducibility/generate.py               # stdout
    uv run python verification/reproducibility/generate.py --save        # update reference
    uv run python verification/reproducibility/generate.py | \\
        uv run python verification/compare.py verification/reproducibility/reference.json
"""
from __future__ import annotations
import argparse, json, re, statistics
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).resolve().parent / "reference.json"

GNN_JSONL   = ROOT / "results" / "gnn_results.jsonl"
LOGREG_JSON = ROOT / "results" / "logreg_results.json"
MIDTIER_JSON= ROOT / "results" / "midtier_results.json"
NUMBERS_TEX = ROOT / "paper"   / "numbers.tex"
SUMMARY_MD  = ROOT / "results" / "summary.md"


def _parse_tex(path: Path) -> dict:
    """Parse \\newcommand{\\Name}{value} lines → {Name: value_str}."""
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _gnn_snapshot() -> dict:
    runs = [json.loads(l) for l in open(GNN_JSONL)]
    out  = {}
    for model in ("baseline", "v2-hgnn", "v3-nli"):
        entries = [r for r in runs if r["model"] == model]
        mf1s    = [e["macro_f1"] for e in entries]
        nees    = [e["per_class"].get("not_enough_evidence") or 0.0 for e in entries]
        out[model] = {
            "n_runs":   len(entries),
            "mf1_mean": round(statistics.mean(mf1s), 3),
            "mf1_std":  round(statistics.stdev(mf1s), 3),
            "nee_mean": round(statistics.mean(nees), 3),
        }
    return out


def _logreg_snapshot() -> dict:
    lr = json.loads(LOGREG_JSON.read_text())
    return {
        "trust_blind_all_mf1":      lr["trust_blind"]["all"]["macro_f1"],
        "trust_blind_standard_mf1": lr["trust_blind"]["standard"]["macro_f1"],
        "trust_blind_mid_mf1":      lr["trust_blind"]["mid_tier"]["macro_f1"],
        "trust_aware_all_mf1":      lr["trust_aware"]["all"]["macro_f1"],
        "trust_aware_standard_mf1": lr["trust_aware"]["standard"]["macro_f1"],
        "trust_aware_mid_mf1":      lr["trust_aware"]["mid_tier"]["macro_f1"],
        "delta_macro_f1":           lr["delta_macro_f1"],
    }


def _midtier_snapshot() -> dict:
    mt = json.loads(MIDTIER_JSON.read_text())
    return {
        model: {
            "mid_macro_f1_mean": s["mid_macro_f1_mean"],
            "mid_macro_f1_std":  s["mid_macro_f1_std"],
            "n_runs":            s["n_runs"],
        }
        for model, s in mt["summary"].items()
    }


def build_snapshot() -> dict:
    return {
        "numbers_tex": _parse_tex(NUMBERS_TEX),
        "gnn":         _gnn_snapshot(),
        "logreg":      _logreg_snapshot(),
        "midtier":     _midtier_snapshot(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="Write snapshot to reference.json instead of stdout")
    args = ap.parse_args()

    snapshot = build_snapshot()
    payload  = json.dumps(snapshot, indent=2)

    if args.save:
        REFERENCE.write_text(payload)
        print(f"reference written → {REFERENCE}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
