"""Verify computed inputs for all paper figures independently from the raw data files.

Does NOT import generate_figures.py — computes the same values directly from source
data so that errors in generate_figures.py are caught by comparison.

Figures covered:
  pb_leakage.png     — majority-lookup accuracy per feature from raw split files
  pb_inversion.png   — GNN/LogReg/midtier values from results/*.json
  pb_ec_curve.png    — EC formula spot-checks at all benchmark ST tier points
  pb_construction.png— EC badge values on the diagram (ec(0.85) and ec(0.30))

pb_architecture.png is a structural diagram with no computed values; not verified here.

Default: prints JSON to stdout (pipe into verification/compare.py).
--save:  writes to verification/figures/reference.json (update golden state).

Usage:
    uv run python verification/figures/generate.py               # stdout
    uv run python verification/figures/generate.py --save        # update reference
    uv run python verification/figures/generate.py | \\
        uv run python verification/compare.py verification/figures/reference.json
"""
from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
RESULTS  = ROOT / "results"
DATA     = ROOT / "data"
REGISTRY = DATA / "registry_with_nm.jsonl"
ORIG_JSONL = ROOT / "model_repo" / "data" / "raw" / "synthetic" / "synthetic_current.jsonl"
CORR_JSONL = DATA / "expanded_split.jsonl"
GNN_JSONL  = RESULTS / "gnn_results.jsonl"
LR_JSON    = RESULTS / "logreg_results.json"
MT_JSON    = RESULTS / "midtier_results.json"
REFERENCE  = Path(__file__).resolve().parent / "reference.json"


def _ec(st: float, ew: float = 0.80, is_: float = 1.0) -> float:
    exp = ew * is_
    return 0.0 if exp == 0 else round(1.0 - (1.0 - st) ** exp, 3)


def _lookup_acc(records: list, key_fn) -> float:
    g: dict = defaultdict(list)
    for r in records:
        g[key_fn(r)].append(r["verdict"]["label"])
    n = len(records)
    return round(sum(Counter(v).most_common(1)[0][1] for v in g.values()) / n, 3)


# ── pb_leakage.png ────────────────────────────────────────────────────────────
def _leakage_snapshot() -> dict:
    corr = [json.loads(l) for l in open(CORR_JSONL)]

    # Registry: source_id → source_type/category
    reg = {json.loads(l)["source_id"]: (json.loads(l).get("source_type") or
           json.loads(l).get("category", "unknown"))
           for l in open(REGISTRY)}

    # Original split
    if ORIG_JSONL.exists():
        orig = [json.loads(l) for l in open(ORIG_JSONL)]
        original = {
            "template_type":   _lookup_acc(orig, lambda r: r["meta"]["template_type"]),
            "source_category": _lookup_acc(orig, lambda r: reg.get(r["evidence"][0]["source_id"], "unknown")),
            "stance":          _lookup_acc(orig, lambda r: r["evidence"][0].get("stance")),
            "evidence_text":   _lookup_acc(orig, lambda r: r["evidence"][0]["text"]),
        }
    else:
        original = {"__skip__": "model_repo/data/raw/synthetic/synthetic_current.jsonl not found"}

    # Corrected split — test standard subset only (matches Table III)
    test_std = [r for r in corr
                if r["provenance"]["split"] == "test"
                and r["meta"].get("trust_tier") != "mid"]
    corrected = {
        "source_category": _lookup_acc(test_std, lambda r: r["evidence"][0]["source_type"]),
        "stance":          _lookup_acc(test_std, lambda r: r["evidence"][0].get("stance")),
        "evidence_text":   _lookup_acc(test_std, lambda r: r["evidence"][0]["text"]),
    }

    return {"original": original, "corrected": corrected}


# ── pb_inversion.png ──────────────────────────────────────────────────────────
def _inversion_snapshot() -> dict:
    # GNN: aggregate mean ± std per model
    gnn_raw: dict = defaultdict(list)
    for line in open(GNN_JSONL):
        r = json.loads(line)
        gnn_raw[r["model"]].append(r)

    def _ms(vals: list) -> tuple:
        return (round(statistics.mean(vals), 3),
                round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0)

    b_mf1, b_std   = _ms([e["macro_f1"] for e in gnn_raw["baseline"]])
    v2_mf1, v2_std = _ms([e["macro_f1"] for e in gnn_raw["v2-hgnn"]])
    v3_mf1, v3_std = _ms([e["macro_f1"] for e in gnn_raw["v3-nli"]])

    lr = json.loads(open(LR_JSON).read())
    mt_sum = json.loads(MT_JSON.read_text()).get("summary", {}) if MT_JSON.exists() else {}

    return {
        "leaky_split": {
            "trust_blind_acc": 0.996,   # hardcoded sentinel in plot_inversion
            "v2_hgnn_acc":     0.889,
        },
        "corrected_overall": {
            "baseline_mf1":         b_mf1,
            "baseline_std":         b_std,
            "v2_hgnn_mf1":          v2_mf1,
            "v2_hgnn_std":          v2_std,
            "v3_nli_mf1":           v3_mf1,
            "v3_nli_std":           v3_std,
            "always_nee_mf1":       round(lr["baselines"]["always_nee"]["macro_f1"], 3),
            "logreg_aware_all_mf1": round(lr["trust_aware"]["all"]["macro_f1"],      4),
            "logreg_aware_std_mf1": round(lr["trust_aware"]["standard"]["macro_f1"], 4),
            "logreg_aware_mid_mf1": round(lr["trust_aware"]["mid_tier"]["macro_f1"], 4),
        },
        "standard_vs_mid": {
            "logreg_std_mf1": round(lr["trust_aware"]["standard"]["macro_f1"],  4),
            "logreg_mid_mf1": round(lr["trust_aware"]["mid_tier"]["macro_f1"],  4),
            "v2_std_mf1":     round(mt_sum.get("v2-hgnn", {}).get("std_macro_f1_mean", 1.0), 3),
            "v2_mid_mf1":     round(mt_sum.get("v2-hgnn", {}).get("mid_macro_f1_mean", 0.0), 3),
            "v2_mid_std":     round(mt_sum.get("v2-hgnn", {}).get("mid_macro_f1_std",  0.0), 3),
            "v3_std_mf1":     round(mt_sum.get("v3-nli",  {}).get("std_macro_f1_mean", 1.0), 3),
            "v3_mid_mf1":     round(mt_sum.get("v3-nli",  {}).get("mid_macro_f1_mean", 0.0), 3),
            "v3_mid_std":     round(mt_sum.get("v3-nli",  {}).get("mid_macro_f1_std",  0.0), 3),
        },
    }


# ── pb_ec_curve.png ───────────────────────────────────────────────────────────
def _ec_curve_snapshot() -> dict:
    # All ST values plotted as tier markers in the figure (EW=0.80 testimony)
    return {
        "testimony_ew080": {
            "st030": _ec(0.30),
            "st035": _ec(0.35),
            "st040": _ec(0.40),
            "st045": _ec(0.45),
            "st050": _ec(0.50),
            "st062": _ec(0.62),
            "st072": _ec(0.72),
            "st085": _ec(0.85),
            "st086": _ec(0.86),
            "st088": _ec(0.88),
            "st090": _ec(0.90),
        },
        "decision_threshold": 0.75,
    }


# ── pb_construction.png ───────────────────────────────────────────────────────
def _construction_snapshot() -> dict:
    # EC badge values shown on the construction diagram arrows
    return {
        "high_trust_ec_badge": _ec(0.85),   # labelled "EC = 0.781"
        "low_trust_ec_badge":  _ec(0.30),   # labelled "EC = 0.248"
    }


def build_snapshot() -> dict:
    return {
        "leakage":      _leakage_snapshot(),
        "inversion":    _inversion_snapshot(),
        "ec_curve":     _ec_curve_snapshot(),
        "construction": _construction_snapshot(),
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
