"""Controlled feature-access experiment on the corrected trust-isolating split.

Tests the paper's central causal claim with the repo's real feature set (all-MiniLM-L6-v2 text
embedding + source_type one-hot, exactly what the trust-blind `baseline` GNN sees), isolating the
single variable of interest: access to the source-trust scalar ST.

  trust-blind  = MiniLM text emb (384d) + source_type one-hot (6d)            [no ST]
  trust-aware  = trust-blind + ST scalar (+ EC of the single evidence item)   [+ ST]

Train on the corrected train split; evaluate on the HELD-OUT-SOURCE test split. If the corrected split
is truly trust-isolating, trust-blind must collapse to ~the majority baseline on test (it cannot use
text — identical within pairs — and source_type does not transfer), while trust-aware generalizes.
Classifier is a plain LogisticRegression so the result reflects the FEATURES, not model capacity.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "experiments" / "data" / "corrected_split.jsonl"

def ec(st, ew=0.80, is_=1.0):
    return 1.0 - (1.0 - st) ** (ew * is_)

recs = [json.loads(l) for l in open(SPLIT)]
# v2: source_type + ST are stored on the record. source_type is CONSTANT (news_media) by design, so it
# is uninformative; ST is the single trust variable. Build the source_type vocab from the data.
SRC_TYPES = sorted({r["evidence"][0].get("source_type", "unknown") for r in recs})
ST_IDX = {t: i for i, t in enumerate(SRC_TYPES)}
print(f"{len(recs)} records | distinct source_types in split: {len(SRC_TYPES)} ({SRC_TYPES})")

# embed evidence texts once (MiniLM, same as the repo)
print("loading all-MiniLM-L6-v2 ...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
texts = [r["evidence"][0]["text"] for r in recs]
emb = model.encode(texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False)
print("embedded:", emb.shape)

def feats(i, r, with_st):
    ev = r["evidence"][0]
    stype = ev.get("source_type", "unknown"); st = float(r["meta"]["source_trust"])
    oh = np.zeros(len(SRC_TYPES)); oh[ST_IDX.get(stype, 0)] = 1.0
    base = np.concatenate([emb[i], oh])
    return np.concatenate([base, [st, ec(st)]]) if with_st else base

labels = np.array([r["verdict"]["label"] for r in recs])
split = np.array([r["provenance"]["split"] for r in recs])
tr = split == "train"; te = split == "test"

def run(with_st, name):
    Xtr = np.vstack([feats(i, r, with_st) for i, r in enumerate(recs) if tr[i]])
    Xte = np.vstack([feats(i, r, with_st) for i, r in enumerate(recs) if te[i]])
    ytr, yte = labels[tr], labels[te]
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred); mf1 = f1_score(yte, pred, average="macro")
    print(f"\n### {name}  (dim={Xtr.shape[1]})")
    print(f"  TEST (held-out sources): accuracy={acc:.4f}  macro-F1={mf1:.4f}")
    print("  per-class F1:", {k: round(v, 3) for k, v in
          zip(sorted(set(yte)), f1_score(yte, pred, average=None, labels=sorted(set(yte))))})
    return acc, mf1

# majority baseline on test
from collections import Counter
maj = Counter(labels[te]).most_common(1)[0]
maj_acc = maj[1] / te.sum()
print(f"\nTEST majority baseline: always '{maj[0]}' -> acc={maj_acc:.4f} "
      f"(macro-F1 of a constant predictor = {1/ (3) * (2*maj_acc/(1+maj_acc)):.3f} approx; report real below)")

a0, f0 = run(False, "TRUST-BLIND  (text + source_type, NO ST)")
a1, f1v = run(True,  "TRUST-AWARE  (+ ST scalar + EC)")
print("\n==================== HEADLINE ====================")
print(f"  trust-blind  : acc={a0:.3f}  macroF1={f0:.3f}")
print(f"  trust-aware  : acc={a1:.3f}  macroF1={f1v:.3f}")
print(f"  Δ macro-F1 (trust effect) = {f1v - f0:+.3f}")
print("  Expectation if the split is trust-isolating: trust-blind ~ majority/chance, trust-aware high.")
