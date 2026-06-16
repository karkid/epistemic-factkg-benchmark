"""Build an EXPANDED trust-isolating split (v3).

Improvements over the original build_corrected_split.py (v2):
  1. Larger vocabulary (THINGS×24, ATTRS×16, VALUES×24) to eliminate ~30% claim repetition.
  2. Mid-trust boundary tier in test only (ST=0.50, 0.62, 0.72 → always NEE by formula).
     These span the untested 0.45–0.85 zone and probe continuous-ST reasoning.

Nothing in benchmark/ is modified. Output goes to data/expanded_split.jsonl.
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path

SEED = 20260611
random.seed(SEED)
OUT = Path(__file__).resolve().parents[1] / "data" / "expanded_split.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- EC formula (inline, same as original) ----
EW_TESTIMONY = 0.80
SUPPORT_THRESHOLD = REFUTE_THRESHOLD = 0.75
CONFLICT_FLOOR = 0.40

def ec(st: float, ew: float = EW_TESTIMONY, is_: float = 1.0) -> float:
    exp = ew * is_
    return 0.0 if exp == 0 else round(1.0 - (1.0 - st) ** exp, 4)

def derive_verdict(support: float, refute: float) -> str:
    if support >= SUPPORT_THRESHOLD and refute < CONFLICT_FLOOR: return "supported"
    if refute >= REFUTE_THRESHOLD and support < CONFLICT_FLOOR:  return "refuted"
    if support >= CONFLICT_FLOOR and refute >= CONFLICT_FLOOR:   return "conflicting_evidence"
    return "not_enough_evidence"

# ---- Source tiers ----
SRC_TYPE = "news_media"
HIGH_TRAIN = [("nm_h85a", 0.85), ("nm_h90a", 0.90)]
LOW_TRAIN  = [("nm_l30a", 0.30), ("nm_l40a", 0.40)]
HIGH_TEST  = [("nm_h88x", 0.88), ("nm_h86x", 0.86)]
LOW_TEST   = [("nm_l35x", 0.35), ("nm_l45x", 0.45)]
# Mid-tier: EC(0.50)=0.426, EC(0.62)=0.539, EC(0.72)=0.639 → all NEE regardless of stance.
# TEST ONLY — these probe whether the model uses continuous ST, not a binary high/low threshold.
MID_TEST   = [("nm_m50x", 0.50), ("nm_m62x", 0.62), ("nm_m72x", 0.72)]

# ---- Expanded vocabulary (original 12+8+12 → 24+16+24) ----
THINGS = [
    # original
    "Velora battery cell", "Trentwood reservoir", "Kaplen turbine", "Mistral alloy plate",
    "Orenco water sample", "Brexford soil core", "Nuvance display panel", "Halcyon coolant tank",
    "Pendle bridge cable", "Sarn district aquifer", "Lumen-7 sensor array", "Caldera ceramic tile",
    # new
    "Ardent pressure vessel", "Celdric flow meter", "Durnham heat exchanger", "Elvast pipe joint",
    "Falwick rotor blade", "Groven insulation panel", "Hydrel valve housing", "Iverstone beam section",
    "Jarrow catalyst bed", "Kelwyn gear assembly", "Lorvik compressor stage", "Mendal filter cartridge",
]
ATTRS = [
    # original
    ("mass", "kg"), ("output", "megawatts"), ("concentration", "ppm"), ("thickness", "mm"),
    ("flow rate", "litres per second"), ("temperature", "degrees C"), ("efficiency", "percent"),
    ("tensile strength", "megapascals"),
    # new
    ("pressure", "bar"), ("voltage", "volts"), ("diameter", "centimetres"), ("volume", "cubic metres"),
    ("density", "kilograms per cubic metre"), ("frequency", "hertz"), ("torque", "newton-metres"),
    ("resistance", "ohms"),
]
VALUES = [
    # original
    "12.4", "318", "27", "6.5", "145", "58", "91", "204", "3.18", "76", "430", "19",
    # new
    "0.42", "87", "512", "1.7", "2300", "44", "0.08", "660", "33", "7.9", "1050", "255",
]

# ---- Text templates (neutral register, identical across trust tiers) ----
def support_text(thing, attr, unit, val):
    return f"Recorded measurements state that the {thing} {attr} is {val} {unit}."

def refute_text(thing, attr, unit, val, other):
    return f"Recorded measurements state that the {thing} {attr} is {other} {unit}, not {val} {unit}."

def claim_text(thing, attr, unit, val):
    return f"The {thing} {attr} is {val} {unit}."

# ---- Record construction ----
def make_paired_records(base_id, claim, ev_text, stance, high_src, low_src, split):
    """High-trust + low-trust pair for one evidence text. Core isolation design."""
    recs = []
    for (sid, st), tier in ((high_src, "high"), (low_src, "low")):
        e = ec(st)
        support = e if stance == "supports" else 0.0
        refute  = e if stance == "refutes"  else 0.0
        label = derive_verdict(support, refute)
        rid = f"expanded-{base_id}-{stance[:3]}-{tier}"
        recs.append(_record(rid, claim, ev_text, stance, sid, st, split, label, tier, e))
    return recs

def make_mid_record(base_id, claim, ev_text, stance, mid_src, split):
    """Standalone mid-tier record. Always NEE (EC < 0.75); no high-tier pairing."""
    sid, st = mid_src
    e = ec(st)
    label = "not_enough_evidence"  # guaranteed by EC < 0.75 at all mid-tier ST values
    rid = f"expanded-mid-{base_id}-{stance[:3]}-{sid}"
    return _record(rid, claim, ev_text, stance, sid, st, split, label, "mid", e)

def _record(rid, claim, ev_text, stance, sid, st, split, label, tier, ec_val):
    return {
        "schema_version": "3.0",
        "id": rid,
        "claim": claim,
        "verdict": {"label": label, "justification": None,
                    "derivation_method": "ec_formula_trust_isolated"},
        "epistemic": {"evidence_types_all": ["testimony"], "assignment_method": "expanded_split"},
        "claim_triples": None,
        "reasoning": {"structural": "one_hop", "strategy": "testimonial_lookup"},
        "evidence": [{
            "evidence_id": f"{rid}-e0", "text": ev_text, "triples": [], "triple_source": None,
            "modality": "web_text", "stance": stance, "evidence_types": ["testimony"],
            "source_id": sid, "source_type": SRC_TYPE, "inference_strength": 1.0, "source_url": None,
        }],
        "provenance": {"dataset": "expanded_synthetic", "split": split, "context_id": rid[:20]},
        "meta": {
            "trust_tier": tier, "source_trust": st, "source_type": SRC_TYPE, "ec": ec_val,
            "text_isolated": True, "is_shortcut_breaking": True,
            "benchmark_version": "expanded_v3",
        },
    }

# ---- Build ----
def build(n_train: int = 400, n_test: int = 120, n_mid: int = 20) -> list:
    out = []

    def gen_paired(n, high_pool, low_pool, split):
        for _ in range(n):
            thing = random.choice(THINGS)
            attr, unit = random.choice(ATTRS)
            val = random.choice(VALUES)
            other = random.choice([v for v in VALUES if v != val])
            claim = claim_text(thing, attr, unit, val)
            base = hashlib.md5(f"{split}{_}{claim}".encode()).hexdigest()[:10]
            hi = random.choice(high_pool)
            lo = random.choice(low_pool)
            out.extend(make_paired_records(base, claim, support_text(thing, attr, unit, val),
                                           "supports", hi, lo, split))
            out.extend(make_paired_records(base, claim, refute_text(thing, attr, unit, val, other),
                                           "refutes", hi, lo, split))

    def gen_mid(n, mid_pool, split):
        for _ in range(n):
            thing = random.choice(THINGS)
            attr, unit = random.choice(ATTRS)
            val = random.choice(VALUES)
            other = random.choice([v for v in VALUES if v != val])
            claim = claim_text(thing, attr, unit, val)
            base = hashlib.md5(f"mid{split}{_}{claim}".encode()).hexdigest()[:10]
            for mid_src in mid_pool:
                out.append(make_mid_record(base, claim,
                                           support_text(thing, attr, unit, val),
                                           "supports", mid_src, split))
                out.append(make_mid_record(base, claim,
                                           refute_text(thing, attr, unit, val, other),
                                           "refutes", mid_src, split))

    gen_paired(n_train, HIGH_TRAIN, LOW_TRAIN, "train")
    gen_paired(n_test,  HIGH_TEST,  LOW_TEST,  "test")
    gen_mid(n_mid, MID_TEST, "test")  # 20 iters × 3 sources × 2 stances = 120 extra records
    return out

if __name__ == "__main__":
    recs = build()
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    splits  = Counter(r["provenance"]["split"] for r in recs)
    verdicts = Counter(r["verdict"]["label"] for r in recs)
    tiers   = Counter(r["meta"]["trust_tier"] for r in recs)
    st_vals = sorted(set(r["meta"]["source_trust"] for r in recs))
    print(f"wrote {len(recs)} records → {OUT}")
    print(f"  splits:   {dict(splits)}")
    print(f"  verdicts: {dict(verdicts)}")
    print(f"  tiers:    {dict(tiers)}")
    print(f"  ST values: {st_vals}")
    # Verify EC values for mid-tier
    print("\nMid-tier EC check (should all be < 0.75 → NEE):")
    for sid, st in [("nm_m50x", 0.50), ("nm_m62x", 0.62), ("nm_m72x", 0.72)]:
        print(f"  {sid} ST={st}: EC={ec(st):.4f}  → {'NEE ok' if ec(st) < 0.75 else 'FAIL'}")
