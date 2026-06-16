"""Controlled LogReg feature-access test on the expanded split.

Reports:
  - Always-NEE baseline (floor)
  - Majority baseline
  - Trust-blind: text emb + source_type (no ST)
  - Trust-aware: + ST scalar + EC
  - Breakdown by tier (standard high/low pairs vs mid-tier boundary probes)

Writes results to results/logreg_results.json.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "expanded_split.jsonl"
OUT   = ROOT / "results" / "logreg_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

def ec(st, ew=0.80, is_=1.0):
    return 1.0 - (1.0 - st) ** (ew * is_)

recs  = [json.loads(l) for l in open(SPLIT)]
SRC_TYPES = sorted({r["evidence"][0]["source_type"] for r in recs})
ST_IDX    = {t: i for i, t in enumerate(SRC_TYPES)}
print(f"{len(recs)} records | source_types: {SRC_TYPES}")

print("loading all-MiniLM-L6-v2 ...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
texts = [r["evidence"][0]["text"] for r in recs]
emb   = model.encode(texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
print(f"embedded: {emb.shape}")

def feats(i, r, with_st):
    ev = r["evidence"][0]
    stype = ev["source_type"]
    st    = float(r["meta"]["source_trust"])
    oh    = np.zeros(len(SRC_TYPES)); oh[ST_IDX.get(stype, 0)] = 1.0
    base  = np.concatenate([emb[i], oh])
    return np.concatenate([base, [st, ec(st)]]) if with_st else base

labels = np.array([r["verdict"]["label"] for r in recs])
splits = np.array([r["provenance"]["split"] for r in recs])
tiers  = np.array([r["meta"]["trust_tier"] for r in recs])
tr = splits == "train"; te = splits == "test"
te_std = te & (tiers != "mid")   # standard high/low test records
te_mid = te & (tiers == "mid")   # mid-tier boundary probes

print(f"\nTest split: {te.sum()} total  |  standard={te_std.sum()}  mid-tier={te_mid.sum()}")

# ---- Baselines ----
yte_all = labels[te]
yte_std = labels[te_std]
yte_mid = labels[te_mid]

maj_label = Counter(yte_all).most_common(1)[0][0]
maj_f1    = f1_score(yte_all, [maj_label] * len(yte_all), average="macro")
maj_acc   = accuracy_score(yte_all, [maj_label] * len(yte_all))

nei_f1  = f1_score(yte_all, ["not_enough_evidence"] * len(yte_all), average="macro")
nei_acc = accuracy_score(yte_all, ["not_enough_evidence"] * len(yte_all))

print(f"\nTEST majority baseline (always '{maj_label}'): acc={maj_acc:.4f}  macro-F1={maj_f1:.4f}")
print(f"TEST always-NEE baseline:                      acc={nei_acc:.4f}  macro-F1={nei_f1:.4f}")

# ---- LogReg experiments ----
def run(with_st, name):
    Xtr = np.vstack([feats(i, r, with_st) for i, r in enumerate(recs) if tr[i]])
    Xte = np.vstack([feats(i, r, with_st) for i, r in enumerate(recs) if te[i]])
    Xte_std = np.vstack([feats(i, r, with_st) for i, r in enumerate(recs) if te_std[i]])
    Xte_mid = np.vstack([feats(i, r, with_st) for i, r in enumerate(recs) if te_mid[i]])
    ytr = labels[tr]
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)

    def metrics(X, y, label):
        pred = clf.predict(X)
        acc  = accuracy_score(y, pred)
        mf1  = f1_score(y, pred, average="macro")
        pcf1 = {k: round(v, 3) for k, v in
                zip(sorted(set(y)), f1_score(y, pred, average=None, labels=sorted(set(y))))}
        print(f"  {label}: acc={acc:.4f}  macro-F1={mf1:.4f}  per-class={pcf1}")
        return {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "per_class_f1": pcf1}

    print(f"\n### {name}  (dim={Xtr.shape[1]})")
    r_all = metrics(Xte,     yte_all, "ALL test       ")
    r_std = metrics(Xte_std, yte_std, "standard pairs ")
    r_mid = metrics(Xte_mid, yte_mid, "mid-tier probes")
    return {"all": r_all, "standard": r_std, "mid_tier": r_mid}

res_blind = run(False, "TRUST-BLIND  (text + source_type, NO ST)")
res_aware = run(True,  "TRUST-AWARE  (+ ST scalar + EC)")

print("\n==================== HEADLINE ====================")
print(f"  always-NEE baseline: acc={nei_acc:.3f}  macroF1={nei_f1:.3f}")
print(f"  trust-blind        : acc={res_blind['all']['accuracy']:.3f}  macroF1={res_blind['all']['macro_f1']:.3f}")
print(f"  trust-aware        : acc={res_aware['all']['accuracy']:.3f}  macroF1={res_aware['all']['macro_f1']:.3f}")
print(f"  Δ macro-F1 (trust effect) = {res_aware['all']['macro_f1'] - res_blind['all']['macro_f1']:+.3f}")
print(f"  mid-tier probe (trust-aware): macro-F1={res_aware['mid_tier']['macro_f1']:.3f} "
      f"(tests continuous-ST reasoning; 1.000=model uses continuous ST)")

output = {
    "benchmark_version": "expanded_v3",
    "split_file": str(SPLIT),
    "n_train": int(tr.sum()),
    "n_test_total": int(te.sum()),
    "n_test_standard": int(te_std.sum()),
    "n_test_mid_tier": int(te_mid.sum()),
    "baselines": {
        "majority": {"label": maj_label, "accuracy": round(maj_acc, 4), "macro_f1": round(maj_f1, 4)},
        "always_nee": {"accuracy": round(nei_acc, 4), "macro_f1": round(nei_f1, 4)},
    },
    "trust_blind": res_blind,
    "trust_aware": res_aware,
    "delta_macro_f1": round(res_aware["all"]["macro_f1"] - res_blind["all"]["macro_f1"], 4),
}
with open(OUT, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nresults written → {OUT}")
