"""Build a CORRECTED, trust-isolating synthetic split for Epistemic FactKG.

Why: the original synthetic split (data/raw/synthetic/synthetic_current.jsonl) leaks the verdict
through TEXT REGISTER — low-trust sources always get rumour-style phrasings ("An anonymous post
claimed...") and high-trust get official phrasings, so a trust-blind text model scores ~0.99 without
ever using source trust (see experiments/diagnostics_d1_d3.md, D2). That split cannot test the
source-trust hypothesis.

This corrected split enforces two properties so that ONLY graded source trust can solve it:
  (1) TEXT ISOLATION — the SAME evidence text string is attached to a high-trust and a low-trust
      source. Within each (claim, stance) the text is byte-identical across tiers, and the register is
      uniformly neutral (no "weak"/"strong" pools). So any text/stance/evidence-type/modality feature is
      uninformative about the label; a text-only model is forced to chance on the high-vs-low decision.
  (2) SOURCE GENERALIZATION — train and test use DISJOINT source sets. A model that memorizes source
      categories (the 6-d source_type one-hot the trust-blind baseline sees) cannot transfer to the
      held-out test sources; only a model using the continuous ST scalar generalizes.

Labels are derived with the repo's own EC formula + thresholds (src/epistemic/formula.py), replicated
inline here for zero-dependency reproducibility:
  EC_i = 1 - (1-ST)^(EW*IS);  single item -> support/refute score = EC_i;
  derive_verdict: supported if support>=0.75 & refute<0.40; refuted if refute>=0.75 & support<0.40; else NEE.
With testimony EW=0.80 and IS=1.0: ST>=0.85 -> EC>=0.781 -> SUPPORTED/REFUTED; ST<=0.45 -> EC<=0.380 -> NEE.

Output: experiments/data/corrected_split.jsonl  (schema v3.0, drop-in for the existing pipeline).
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path

SEED = 20260611
random.seed(SEED)
OUT = Path(__file__).parent / "data" / "corrected_split.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- EC formula (replicated from src/epistemic/formula.py, verified) ----
EW_TESTIMONY = 0.80
SUPPORT_THRESHOLD = REFUTE_THRESHOLD = 0.75
CONFLICT_FLOOR = 0.40

def ec(st: float, ew: float, is_: float) -> float:
    exp = ew * is_
    return 0.0 if exp == 0 else round(1.0 - (1.0 - st) ** exp, 4)

def derive_verdict(support: float, refute: float) -> str:
    if support >= SUPPORT_THRESHOLD and refute < CONFLICT_FLOOR: return "supported"
    if refute >= REFUTE_THRESHOLD and support < CONFLICT_FLOOR:  return "refuted"
    if support >= CONFLICT_FLOOR and refute >= CONFLICT_FLOOR:   return "conflicting_evidence"
    return "not_enough_evidence"

# ---- v2 design: CONSTANT source_type, vary only the continuous ST scalar ----
# v1 leaked: the trust-blind baseline sees source_type (6-d category), which correlates with trust in
# the registry, so it transferred to held-out sources (plus a "supporting-text->supported" stance
# default). To TRULY isolate trust we hold source_type constant (all "news_media") and vary ONLY ST.
# Then the trust-blind input is byte-identical within each high/low pair -> provably at chance; the ST
# scalar is the single differing variable. Train/test hold out distinct ST VALUES so the trust-aware
# model must use ST as a continuous monotonic signal, not memorize exact values.
# source_id carries the ST but is NOT a model feature (the featurizer only one-hots source_TYPE).
SRC_TYPE = "news_media"
# (source_id, ST). High ST>=0.85 -> EC>=0.781 -> SUPPORTED/REFUTED; Low ST<=0.45 -> EC<=0.380 -> NEE.
HIGH_TRAIN = [("nm_h85a",0.85),("nm_h90a",0.90)]
LOW_TRAIN  = [("nm_l30a",0.30),("nm_l40a",0.40)]
HIGH_TEST  = [("nm_h88x",0.88),("nm_h86x",0.86)]   # held-out ST values
LOW_TEST   = [("nm_l35x",0.35),("nm_l45x",0.45)]   # held-out ST values

# ---- Neutral fictional claim material (NO trust-register words anywhere) ----
THINGS = ["Velora battery cell","Тrentwood reservoir","Kaplen turbine","Mistral alloy plate",
          "Orenco water sample","Brexford soil core","Nuvance display panel","Halcyon coolant tank",
          "Pendle bridge cable","Sarn district aquifer","Lumen-7 sensor array","Caldera ceramic tile"]
ATTRS  = [("mass","kg"),("output","megawatts"),("concentration","ppm"),("thickness","mm"),
          ("flow rate","litres per second"),("temperature","degrees C"),("efficiency","percent"),
          ("tensile strength","megapascals")]
VALUES = ["12.4","318","27","6.5","145","58","91","204","3.18","76","430","19"]

# Single neutral evidence register, identical for high & low tiers.
def support_text(thing, attr, unit, val):
    return f"Recorded measurements state that the {thing} {attr} is {val} {unit}."
def refute_text(thing, attr, unit, val, other):
    return f"Recorded measurements state that the {thing} {attr} is {other} {unit}, not {val} {unit}."

def claim_text(thing, attr, unit, val):
    return f"The {thing} {attr} is {val} {unit}."

def make_records(base_id, claim, ev_text, stance, high_src, low_src, split):
    """Emit the high-trust and low-trust instantiation of ONE identical evidence text."""
    recs = []
    for (sid, st), tier in ((high_src,"high"), (low_src,"low")):
        is_ = 1.0
        e = ec(st, EW_TESTIMONY, is_)
        support = e if stance == "supports" else 0.0
        refute  = e if stance == "refutes"  else 0.0
        label = derive_verdict(support, refute)
        rid = f"corrected-{base_id}-{stance[:3]}-{tier}"
        recs.append({
            "schema_version": "3.0",
            "id": rid,
            "claim": claim,
            "verdict": {"label": label, "justification": None,
                        "derivation_method": "ec_formula_trust_isolated"},
            "epistemic": {"evidence_types_all": ["testimony"], "assignment_method": "corrected_split"},
            "claim_triples": None,
            "reasoning": {"structural": "one_hop", "strategy": "testimonial_lookup"},
            "evidence": [{
                "evidence_id": f"{rid}-e0", "text": ev_text, "triples": [], "triple_source": None,
                "modality": "web_text", "stance": stance, "evidence_types": ["testimony"],
                "source_id": sid, "source_type": SRC_TYPE, "inference_strength": is_, "source_url": None,
            }],
            "provenance": {"dataset": "corrected_synthetic", "split": split, "context_id": base_id},
            "meta": {"trust_tier": tier, "source_trust": st, "source_type": SRC_TYPE, "ec": e,
                     "text_isolated": True, "is_shortcut_breaking": True},
        })
    return recs

def build(n_train=400, n_test=120):
    out = []
    def gen(n, high_pool, low_pool, split, off):
        for i in range(n):
            thing = random.choice(THINGS); attr, unit = random.choice(ATTRS)
            val = random.choice(VALUES); other = random.choice([v for v in VALUES if v != val])
            claim = claim_text(thing, attr, unit, val)
            base = hashlib.md5(f"{split}{i}{claim}".encode()).hexdigest()[:10]
            sup = support_text(thing, attr, unit, val)
            ref = refute_text(thing, attr, unit, val, other)
            hi = random.choice(high_pool); lo = random.choice(low_pool)
            # SAME support text -> high (SUPPORTED) & low (NEE); SAME refute text -> high (REFUTED) & low (NEE)
            out.extend(make_records(base, claim, sup, "supports", hi, lo, split))
            out.extend(make_records(base, claim, ref, "refutes",  hi, lo, split))
    gen(n_train, HIGH_TRAIN, LOW_TRAIN, "train", 0)
    gen(n_test,  HIGH_TEST,  LOW_TEST,  "test", n_train)
    return out

if __name__ == "__main__":
    recs = build()
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records -> {OUT}")
