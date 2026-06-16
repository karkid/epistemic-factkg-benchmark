"""Generate split design snapshot from source files.

Default: prints JSON to stdout (pipe into verification/compare.py).
--save:  writes to verification/split_design/reference.json (update golden state).

Usage:
    uv run python verification/split_design/generate.py               # stdout
    uv run python verification/split_design/generate.py --save        # update reference
    uv run python verification/split_design/generate.py | \\
        uv run python verification/compare.py verification/split_design/reference.json
"""
from __future__ import annotations
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
SPLIT_JSONL = ROOT / "data" / "expanded_split.jsonl"
REGISTRY    = ROOT / "data" / "registry_with_nm.jsonl"
ORIG_JSONL  = ROOT / "model_repo" / "data" / "raw" / "synthetic" / "synthetic_current.jsonl"
LOGREG_JSON = ROOT / "results" / "logreg_results.json"
GNN_JSONL   = ROOT / "results" / "gnn_results.jsonl"
REFERENCE   = Path(__file__).resolve().parent / "reference.json"


def ec(st: float, ew: float = 0.80) -> float:
    return 1 - (1 - st) ** ew


def build_snapshot() -> dict:
    recs = [json.loads(l) for l in open(SPLIT_JSONL)]

    # ── Statistics ──────────────────────────────────────────────────────────
    stats: dict = {"total": len(recs)}
    for sp in ("train", "test"):
        for tier in ("high", "low", "mid"):
            subset = [r for r in recs
                      if r["provenance"]["split"] == sp
                      and r["meta"].get("trust_tier") == tier]
            if not subset:
                continue
            c = Counter(r["verdict"]["label"] for r in subset)
            stats[f"{sp}_{tier}"] = {
                "n":         len(subset),
                "supported": c.get("supported", 0),
                "refuted":   c.get("refuted", 0),
                "nee":       c.get("not_enough_evidence", 0),
            }

    # ── Text isolation ───────────────────────────────────────────────────────
    text_info: dict = {}
    for r in recs:
        t    = r["evidence"][0]["text"]
        tier = r["meta"].get("trust_tier", "")
        text_info.setdefault(t, {"labels": set(), "tiers": set()})
        text_info[t]["labels"].add(r["verdict"]["label"])
        text_info[t]["tiers"].add(tier)

    standard_texts  = [t for t, v in text_info.items() if "mid" not in v["tiers"]]
    multilabel_std  = sum(1 for t in standard_texts if len(text_info[t]["labels"]) > 1)
    mid_only        = sum(1 for t, v in text_info.items() if v["tiers"] == {"mid"})
    src_types       = sorted({r["evidence"][0]["source_type"] for r in recs})

    isolation = {
        "standard_texts_total":      len(standard_texts),
        "standard_texts_multilabel": multilabel_std,
        "mid_only_texts":            mid_only,
        "distinct_source_types":     src_types,
    }

    # ── EC formula ───────────────────────────────────────────────────────────
    ec_values = {
        "high_st085": round(ec(0.85), 3),
        "low_st045":  round(ec(0.45), 3),
        "mid_st050":  round(ec(0.50), 3),
        "mid_st062":  round(ec(0.62), 3),
        "mid_st072":  round(ec(0.72), 3),
    }

    # ── LogReg ───────────────────────────────────────────────────────────────
    lr = json.loads(Path(LOGREG_JSON).read_text())
    delta_std = round(
        lr["trust_aware"]["standard"]["macro_f1"]
        - lr["trust_blind"]["standard"]["macro_f1"], 3
    )
    logreg = {
        "trust_blind_all_mf1":      lr["trust_blind"]["all"]["macro_f1"],
        "trust_blind_standard_mf1": lr["trust_blind"]["standard"]["macro_f1"],
        "trust_blind_mid_mf1":      lr["trust_blind"]["mid_tier"]["macro_f1"],
        "trust_aware_all_mf1":      lr["trust_aware"]["all"]["macro_f1"],
        "trust_aware_standard_mf1": lr["trust_aware"]["standard"]["macro_f1"],
        "trust_aware_mid_mf1":      lr["trust_aware"]["mid_tier"]["macro_f1"],
        "delta_all":                lr["delta_macro_f1"],
        "delta_standard":           delta_std,
    }

    # ── Registry consistency ─────────────────────────────────────────────────
    # meta.source_trust in each record must match registry_with_nm.jsonl.
    # These are two separate representations of the same value; drift would
    # mean the tag in the record no longer reflects what the model actually sees.
    reg = {json.loads(l)["source_id"]: float(json.loads(l).get("source_trust") or
                                              json.loads(l).get("trust_score", 0))
           for l in open(REGISTRY)}
    mismatches = sum(
        1 for r in recs
        if abs(r["meta"]["source_trust"] - reg.get(r["evidence"][0]["source_id"], -1)) > 0.001
    )
    registry_consistency = {
        "records_checked":      len(recs),
        "registry_mismatches":  mismatches,
    }

    # ── GNN ──────────────────────────────────────────────────────────────────
    runs = [json.loads(l) for l in open(GNN_JSONL)]
    gnn: dict = {}
    for model in ("baseline", "v2-hgnn", "v3-nli"):
        entries = [r for r in runs if r["model"] == model]
        mf1s    = [e["macro_f1"] for e in entries]
        nees    = [e["per_class"].get("not_enough_evidence") or 0.0 for e in entries]
        gnn[model] = {
            "n_runs":   len(entries),
            "mf1_mean": round(statistics.mean(mf1s), 3),
            "mf1_std":  round(statistics.stdev(mf1s), 3),
            "nee_mean": round(statistics.mean(nees), 3),
        }

    # ── Leakage figure values ────────────────────────────────────────────────
    # Verifies the computed inputs to pb_leakage.png so figure values stay
    # consistent with the source data files.
    def _lookup_acc(records, key_fn):
        g: dict = defaultdict(list)
        for r in records:
            g[key_fn(r)].append(r["verdict"]["label"])
        n = len(records)
        return round(sum(Counter(v).most_common(1)[0][1] for v in g.values()) / n, 3)

    reg_cat = {json.loads(l)["source_id"]: (json.loads(l).get("source_type") or
               json.loads(l).get("category", "unknown"))
               for l in open(REGISTRY)}

    if ORIG_JSONL.exists():
        orig = [json.loads(l) for l in open(ORIG_JSONL)]
        leakage_original = {
            "template_type": _lookup_acc(orig, lambda r: r["meta"]["template_type"]),
            "source_category": _lookup_acc(orig, lambda r: reg_cat.get(r["evidence"][0]["source_id"], "unknown")),
            "stance":          _lookup_acc(orig, lambda r: r["evidence"][0].get("stance")),
            "evidence_text":   _lookup_acc(orig, lambda r: r["evidence"][0]["text"]),
        }
    else:
        leakage_original = {"__skip__": "model_repo/data/raw/synthetic/synthetic_current.jsonl not found"}

    test_std = [r for r in recs
                if r["provenance"]["split"] == "test"
                and r["meta"].get("trust_tier") != "mid"]
    leakage_corrected = {
        "source_category": _lookup_acc(test_std, lambda r: r["evidence"][0]["source_type"]),
        "stance":          _lookup_acc(test_std, lambda r: r["evidence"][0].get("stance")),
        "evidence_text":   _lookup_acc(test_std, lambda r: r["evidence"][0]["text"]),
    }

    leakage = {"original": leakage_original, "corrected": leakage_corrected}

    return {"statistics": stats, "isolation": isolation, "ec_values": ec_values,
            "registry_consistency": registry_consistency, "leakage": leakage,
            "logreg": logreg, "gnn": gnn}


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
