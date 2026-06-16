"""Generate diagnostics snapshot by recomputing D1/D2/D3 from source data.

Reads raw data files from model_repo/ and computes diagnostic numbers
from scratch — does NOT read diagnostics_d1_d3.md (that is an output, not
a source of truth).

D1 — AVeriTeC majority baseline (dev.json, n=500, 3-class after dropping
     Conflicting Evidence rows per ADR-007 / configs/config.yaml:84)
D2 — Synthetic shortcut leakage: per-feature majority accuracy
     (synthetic_current.jsonl, n=2500)
D3 — AI2THOR class balance and source-id homogeneity
     (claims_all.jsonl, n=1800)

Note: the textclf_mf1=0.3537 (MiniLM+LogReg) requires embedding inference and
is intentionally excluded — it is documented in the MD as historical context,
not as a recomputable verification target. The model accuracy 1.000 for AI2THOR
comes from the instructor report, also excluded for the same reason.

Usage:
    uv run python verification/diagnostics/generate.py               # stdout
    uv run python verification/diagnostics/generate.py --save        # update reference
    uv run python verification/diagnostics/generate.py | \\
        uv run python verification/compare.py verification/diagnostics/reference.json
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from sklearn.metrics import f1_score
except ImportError:
    raise SystemExit("sklearn required: uv add scikit-learn")

ROOT      = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).resolve().parent / "reference.json"

MODEL_REPO   = ROOT / "model_repo"
AVERITEC_DEV = MODEL_REPO / "data" / "raw" / "averitec" / "dev.json"
SYNTHETIC    = MODEL_REPO / "data" / "raw" / "synthetic" / "synthetic_current.jsonl"
AI2THOR      = MODEL_REPO / "data" / "raw" / "ai2thor" / "claims_all.jsonl"

SKIP_MSG = (
    f"model_repo/ not found at {MODEL_REPO} — "
    "diagnostics require the original model repository. "
    "Clone it into temp/ with: git clone <model-repo-url> model_repo/"
)


def _skip_if_no_model_repo() -> None:
    """Emit a __skip__ JSON sentinel and exit 0 when model_repo is absent.

    compare.py recognises the sentinel and prints a SKIP notice without
    treating the missing data as a mismatch failure.
    """
    if not MODEL_REPO.exists():
        print(json.dumps({"__skip__": SKIP_MSG}))
        raise SystemExit(0)

CONFLICTING_KEYWORD = "Conflict"


def _d1_averitec() -> dict:
    data   = json.load(open(AVERITEC_DEV))
    # Drop Conflicting Evidence (ADR-007) — same filter as GNN training
    keep   = [r for r in data if CONFLICTING_KEYWORD not in r["label"]]
    labs3  = [r["label"] for r in keep]
    c3     = Counter(labs3)
    # Majority = always "Refuted"
    majority = c3.most_common(1)[0][0]
    pred3    = [majority] * len(labs3)
    majority_acc = c3[majority] / len(labs3)
    majority_mf1 = f1_score(labs3, pred3, average="macro", zero_division=0)
    return {
        "n_total":      len(data),
        "n_3class":     len(keep),
        "class_counts": dict(c3),
        "majority_class": majority,
        "majority_acc": round(majority_acc, 4),
        "majority_mf1": round(majority_mf1, 4),
    }


def _d2_synthetic() -> dict:
    recs   = [json.loads(l) for l in open(SYNTHETIC)]
    labels = [r["verdict"]["label"] for r in recs]

    def majority_lookup_acc(groups: dict) -> float:
        correct = sum(
            Counter(lbls).most_common(1)[0][1]
            for lbls in groups.values()
        )
        return round(correct / len(labels), 2)

    tt_groups:     dict = defaultdict(list)
    si_groups:     dict = defaultdict(list)
    stance_groups: dict = defaultdict(list)
    for r in recs:
        lbl = r["verdict"]["label"]
        tt_groups[r["meta"]["template_type"]].append(lbl)
        si_groups[r["evidence"][0]["source_id"]].append(lbl)
        stance_groups[r["evidence"][0].get("stance", "unknown")].append(lbl)

    return {
        "n":                    len(recs),
        "n_template_types":     len(tt_groups),
        "n_source_ids":         len(si_groups),
        "template_type_acc":    majority_lookup_acc(tt_groups),
        "source_id_acc":        majority_lookup_acc(si_groups),
        "stance_acc":           majority_lookup_acc(stance_groups),
    }


def _d3_ai2thor() -> dict:
    recs        = [json.loads(l) for l in open(AI2THOR)]
    labels      = [r["verdict"]["label"] for r in recs]
    source_ids  = sorted({r["evidence"][0]["source_id"] for r in recs})
    c           = Counter(labels)
    majority    = c.most_common(1)[0][0]
    majority_acc = c[majority] / len(recs)
    return {
        "n":              len(recs),
        "class_counts":   dict(c),
        "source_ids":     source_ids,
        "majority_acc":   round(majority_acc, 3),
        "nee_count":      c.get("not_enough_evidence", 0),
    }


def build_snapshot() -> dict:
    _skip_if_no_model_repo()
    return {
        "d1_averitec": _d1_averitec(),
        "d2_synthetic": _d2_synthetic(),
        "d3_ai2thor":  _d3_ai2thor(),
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
