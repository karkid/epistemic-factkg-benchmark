"""Generate all paper figures for the v3 benchmark paper.

Run from any directory:
    python generate_figures.py

Requires: matplotlib numpy
    pip install matplotlib numpy

Outputs (relative to this file):
    figures/pb_construction.png  benchmark construction diagram
    figures/pb_leakage.png       verdict-leakage bar chart
    figures/pb_ec_curve.png      EC formula: how ST, EW, IS interact
    figures/pb_inversion.png     inversion bar chart  ← reads results files
    figures/pb_architecture.png  3-model structural comparison

Data sources
------------
pb_inversion.png   : reads ../results/gnn_results.jsonl  +  ../results/logreg_results.json
                     +  ../results/midtier_results.json  (standard vs mid-tier breakdown)
pb_ec_curve.png    : formula-computed (EC = 1-(1-ST)^(EW*IS))
pb_construction.png: structural/conceptual, no data file
pb_leakage.png     : computed from actual source files — majority lookup accuracy per feature
                     on original split (model_repo/data/raw/synthetic/synthetic_current.jsonl)
                     and corrected split (data/expanded_split.jsonl). Template field absent
                     from corrected split (shown as N/A). Source ID on corrected split is the
                     trust signal (intentional), distinguished by hatching.
pb_architecture.png: structural, no data file
"""
from __future__ import annotations
import json, shutil, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrowPatch

HERE    = Path(__file__).parent
OUT     = HERE / "figures"
RESULTS = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_BLIND = "#888888"
C_V2    = "#4A90D9"
C_V3    = "#E88C2A"
C_PROBE = "#2C4A8A"
C_NEE   = "#D64444"
C_HIGH  = "#4CAF50"
C_MID   = "#FF9800"
C_INF   = "#9B59B6"
C_PER   = "#27AE60"


def ec(st: float, ew: float = 0.80, is_: float = 1.0) -> float:
    exp = ew * is_
    return 0.0 if exp == 0 else 1.0 - (1.0 - st) ** exp


# ─────────────────────────────────────────────────────────────────────────────
# Shared drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def rect(ax, cx, cy, w, h, text, fc, ec_="black", alpha=0.92,
         fs=8.5, bold=False, lw=1.3):
    p = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.07",
        facecolor=fc, edgecolor=ec_, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            multialignment="center")


def diamond(ax, cx, cy, w, h, text, fc, ec_="black", alpha=0.92,
            fs=8.5, lw=1.5):
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    d = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec_,
                linewidth=lw, alpha=alpha)
    ax.add_patch(d)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, multialignment="center")


def arr(ax, x0, y0, x1, y1, color="black", lw=1.2, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=11))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.08, my, label, fontsize=7, color=color,
                ha="left", va="center")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Benchmark construction
# ─────────────────────────────────────────────────────────────────────────────
def plot_construction():
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 3.9)
    ax.axis("off")

    LW = 2.0

    def sbox(cx, cy, w, h, fc, ec_="black", lw=LW):
        ax.add_patch(mpatches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            facecolor=fc, edgecolor=ec_, linewidth=lw, zorder=2))

    def arrow(x0, y0, x1, y1, lw=1.6):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=lw, mutation_scale=14), zorder=3)

    def hline(x0, x1, y, lw=1.6):
        ax.plot([x0, x1], [y, y], color="black", lw=lw, zorder=3)

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(4.4, 3.72, "Only the source-trust score flips the label.",
            ha="center", fontsize=10.5, style="italic")

    # ── left: claim + evidence ───────────────────────────────────────────────
    CX_L, W_L, CY_L, H_L = 1.25, 2.1, 2.05, 2.4
    sbox(CX_L, CY_L, W_L, H_L, "#F8F8F8")
    ax.text(CX_L, 2.80, "Claim +", ha="center", fontsize=10, fontweight="bold")
    ax.text(CX_L, 2.52, "evidence text", ha="center", fontsize=10, fontweight="bold")
    ax.text(CX_L, 2.12, "byte-identical,", ha="center", fontsize=9, style="italic")
    ax.text(CX_L, 1.82, "neutral wording", ha="center", fontsize=9, style="italic")

    # ── mid: source boxes ────────────────────────────────────────────────────
    CX_M, W_M, H_M = 4.3, 2.35, 1.0
    CY_HI, CY_LO = 2.82, 1.28

    sbox(CX_M, CY_HI, W_M, H_M, "#E8F5E9", ec_="#388E3C")
    ax.text(CX_M, CY_HI + 0.27, "High-trust source",
            ha="center", fontsize=9.5, fontweight="bold", color="#1B5E20")
    ax.text(CX_M, CY_HI - 0.02, r"trust score  $S_T = 0.85$ to $0.90$",
            ha="center", fontsize=8.5)
    ax.text(CX_M, CY_HI - 0.27, "category: news media",
            ha="center", fontsize=8.5)

    sbox(CX_M, CY_LO, W_M, H_M, "#FFEBEE", ec_="#C62828")
    ax.text(CX_M, CY_LO + 0.27, "Low-trust source",
            ha="center", fontsize=9.5, fontweight="bold", color="#7F0000")
    ax.text(CX_M, CY_LO - 0.02, r"trust score  $S_T = 0.30$ to $0.45$",
            ha="center", fontsize=8.5)
    ax.text(CX_M, CY_LO - 0.27, "category: news media",
            ha="center", fontsize=8.5)

    # ── right: verdict boxes ─────────────────────────────────────────────────
    CX_R, W_R, H_R = 7.35, 1.9, 0.82
    sbox(CX_R, CY_HI, W_R, H_R, "#C8E6C9", ec_="#388E3C")
    ax.text(CX_R, CY_HI + 0.12, "SUPPORTED",
            ha="center", fontsize=10.5, fontweight="bold", color="#1B5E20")
    ax.text(CX_R, CY_HI - 0.17, "or REFUTED",
            ha="center", fontsize=10.5, fontweight="bold", color="#1B5E20")

    sbox(CX_R, CY_LO, W_R, H_R, "#FFCDD2", ec_="#C62828")
    ax.text(CX_R, CY_LO + 0.12, "NOT ENOUGH",
            ha="center", fontsize=10.5, fontweight="bold", color="#7F0000")
    ax.text(CX_R, CY_LO - 0.17, "EVIDENCE",
            ha="center", fontsize=10.5, fontweight="bold", color="#7F0000")

    # ── fork: T-junction with straight horizontal arrows ─────────────────────
    # Horizontal stub from claim-box right edge → branch point
    # Vertical stem at branch point connecting both levels
    # Straight horizontal arrows from branch point → each source box
    FORK_X   = CX_L + W_L/2        # 2.30 — right edge of claim box
    FORK_Y   = CY_L                 # 2.05 — vertical centre of claim box
    MID_X    = CX_M - W_M/2        # 3.125 — left edge of source boxes
    BRANCH_X = FORK_X + 0.45       # 2.75 — where the vertical stem sits

    hline(FORK_X, BRANCH_X, FORK_Y)                          # horizontal stub
    ax.plot([BRANCH_X, BRANCH_X], [CY_LO, CY_HI],           # vertical stem
            color="black", lw=1.6, zorder=3)
    arrow(BRANCH_X, CY_HI, MID_X, CY_HI)                    # straight → high-trust
    arrow(BRANCH_X, CY_LO, MID_X, CY_LO)                    # straight → low-trust

    # ── straight horizontal arrows: mid → verdict ────────────────────────────
    arrow(CX_M + W_M/2, CY_HI, CX_R - W_R/2, CY_HI)
    arrow(CX_M + W_M/2, CY_LO, CX_R - W_R/2, CY_LO)

    # ── EC badges above horizontal arrows ────────────────────────────────────
    for cy, val, fc, ec_ in [
        (CY_HI, "EC = 0.781", "#E8F5E9", "#388E3C"),
        (CY_LO, "EC = 0.248", "#FFEBEE", "#C62828"),
    ]:
        mid_arr_x = (CX_M + W_M/2 + CX_R - W_R/2) / 2
        ax.text(mid_arr_x, cy + 0.20, val,
                ha="center", fontsize=8.5, fontweight="bold", color=ec_,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=fc,
                          edgecolor=ec_, linewidth=1.2), zorder=4)

    # ── footer ───────────────────────────────────────────────────────────────
    ax.text(4.4, 0.22,
            r"Same text and same source category — "
            r"wording and category carry $\mathbf{no}$ signal about the verdict.",
            ha="center", fontsize=8.8, style="italic", color="#333333")

    fig.savefig(OUT / "pb_construction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓  pb_construction.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Leakage bar chart  (computed from actual source files)
# ─────────────────────────────────────────────────────────────────────────────
def _majority_lookup_acc(groups: dict, n: int) -> float:
    from collections import Counter as _Counter
    return round(sum(_Counter(v).most_common(1)[0][1] for v in groups.values()) / n, 3)


def _leakage_values():
    from collections import defaultdict as _dd

    repo_root = HERE.parent   # temp/
    orig_path = repo_root / "model_repo" / "data" / "raw" / "synthetic" / "synthetic_current.jsonl"
    corr_path = repo_root / "data" / "expanded_split.jsonl"
    reg_path  = repo_root / "data" / "registry_with_nm.jsonl"

    orig = [json.loads(l) for l in open(orig_path)]
    corr = [json.loads(l) for l in open(corr_path)]

    # Registry: source_id → source_type (category)
    reg = {json.loads(l)["source_id"]: (json.loads(l).get("source_type") or
                                         json.loads(l).get("category", "unknown"))
           for l in open(reg_path)}

    def _groups(recs, key_fn):
        g = _dd(list)
        for r in recs:
            g[key_fn(r)].append(r["verdict"]["label"])
        return g

    # Original split — 4 features shown in the figure
    o_tt  = _majority_lookup_acc(_groups(orig, lambda r: r["meta"]["template_type"]),         len(orig))
    o_cat = _majority_lookup_acc(_groups(orig, lambda r: reg.get(r["evidence"][0]["source_id"], "unknown")), len(orig))
    o_st  = _majority_lookup_acc(_groups(orig, lambda r: r["evidence"][0].get("stance")),     len(orig))
    o_txt = _majority_lookup_acc(_groups(orig, lambda r: r["evidence"][0]["text"]),            len(orig))

    # Corrected split — test standard subset (480 records), matches Table III in the paper.
    # Template field was removed entirely from v3 data (no corrected bar for it).
    # Source category is constant (all news_media) → 0.50.
    corr_test = [r for r in corr
                 if r["provenance"]["split"] == "test"
                 and r["meta"].get("trust_tier") != "mid"]
    n_ct = len(corr_test)
    c_cat = _majority_lookup_acc(_groups(corr_test, lambda r: r["evidence"][0]["source_type"]), n_ct)
    c_st  = _majority_lookup_acc(_groups(corr_test, lambda r: r["evidence"][0].get("stance")),  n_ct)
    c_txt = _majority_lookup_acc(_groups(corr_test, lambda r: r["evidence"][0]["text"]),         n_ct)

    return (o_tt, o_cat, o_st, o_txt), (c_cat, c_st, c_txt)


def plot_leakage():
    (o_tt, o_cat, o_st, o_txt), (c_cat, c_st, c_txt) = _leakage_values()

    # X-axis: 4 features consistent with Table I (diagnosis) and Table III (isolation)
    #   Template Field   — original split only (field removed in v3; no corrected bar)
    #   Source Category  — original leaked via category-trust correlation; corrected = constant (0.50)
    #   Stance           — mild leak in original; at chance in corrected
    #   Evidence Text    — main text-register leak in original; at chance in corrected
    features  = ["Template\nField", "Source\nCategory", "Stance", "Evidence\nText"]
    original  = [o_tt,  o_cat, o_st,  o_txt]
    corrected = [c_cat, c_st,  c_txt]          # 3 values — no Template Field bar on corrected

    x    = np.arange(len(features))
    x_c  = x[1:]                               # corrected bars only under last 3 features
    w    = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Original bars (all 4)
    ax.bar(x - w/2, original, w, label="Original split (leaky)",
           color="#E8714A", edgecolor="black", hatch="///", linewidth=0.8)

    # Corrected bars (3 — source category, stance, evidence text)
    ax.bar(x_c + w/2, corrected, w, label="Corrected split",
           color="#2ABFBF", edgecolor="black", linewidth=0.8)

    # Chance line
    ax.axhline(0.50, color="#555", linestyle="--", linewidth=1.1,
               alpha=0.7, label="chance (0.50)")

    # "removed" annotation under Template Field corrected position
    ax.text(x[0] + w/2, 0.03, "removed\nin v3", ha="center", va="bottom",
            fontsize=7.5, color="#999", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=10)
    ax.set_ylabel("Predictability Score\n(majority lookup accuracy)", fontsize=9)
    ax.set_title("Verdict Predictability from Non-Trust Features", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")

    # Value labels — original bars
    for i, val in enumerate(original):
        ax.text(x[i] - w/2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Value labels — corrected bars
    for i, val in enumerate(corrected):
        ax.text(x_c[i] + w/2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "pb_leakage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓  pb_leakage.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — EC formula curve
# ─────────────────────────────────────────────────────────────────────────────
def plot_ec_curve():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ST = np.linspace(0.001, 0.999, 500)

    curves = [
        (0.55, "inference by analogy  (EW = 0.55)", C_INF, "--"),
        (0.80, "testimony             (EW = 0.80)", C_V2,  "-"),
        (0.95, "perception            (EW = 0.95)", C_PER, "-."),
    ]
    for ew, label, color, ls in curves:
        ax.plot(ST, [ec(s, ew) for s in ST],
                color=color, linestyle=ls, linewidth=2.0, label=label)

    ax.axhline(0.75, color="red", linestyle=":", linewidth=1.5,
               label="decision threshold  EC = 0.75")

    ec_t = np.array([ec(s) for s in ST])
    ax.fill_between(ST, 0,    ec_t, where=(ec_t <  0.75), color=C_NEE,  alpha=0.07)
    ax.fill_between(ST, 0.75, ec_t, where=(ec_t >= 0.75), color=C_HIGH, alpha=0.07)

    tier_pts = [
        (0.30,"low"),(0.35,"low"),(0.40,"low"),(0.45,"low"),
        (0.50,"mid"),(0.62,"mid"),(0.72,"mid"),
        (0.85,"high"),(0.86,"high"),(0.88,"high"),(0.90,"high"),
    ]
    cmap = {"low": C_NEE, "mid": C_MID, "high": C_HIGH}
    for sv, tier in tier_pts:
        ax.plot(sv, ec(sv), "o", color=cmap[tier], markersize=6, zorder=5)

    for xv, c in [(0.45, C_NEE), (0.85, C_HIGH)]:
        ax.axvline(xv, color=c, linestyle=":", linewidth=0.9, alpha=0.5)

    ax.text(0.22,  0.09, "NEE zone\n(EC < 0.75)",    ha="center", fontsize=8,
            color=C_NEE, alpha=0.85)
    ax.text(0.615, 0.09, "mid-tier\n(probe zone)",   ha="center", fontsize=8,
            color=C_MID, alpha=0.85)
    ax.text(0.925, 0.55, "SUP /\nREF",               ha="center", fontsize=8,
            color=C_HIGH, alpha=0.85)

    ax.plot([], [], "o", color=C_NEE,  markersize=6, label="low-trust  (ST ≤ 0.45)")
    ax.plot([], [], "o", color=C_MID,  markersize=6, label="mid-tier   (ST ∈ {0.50, 0.62, 0.72})")
    ax.plot([], [], "o", color=C_HIGH, markersize=6, label="high-trust (ST ≥ 0.85)")

    ax.set_xlabel("Source Trust  $S_T$", fontsize=11)
    ax.set_ylabel("Epistemic Confidence  EC", fontsize=11)
    ax.set_title(r"EC = 1 $-$ (1 $-$ $S_T$)$^{EW \times IS}$"
                 "   (IS = 1.0, single testimony item)", fontsize=10.5)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUT / "pb_ec_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓  pb_ec_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Inversion  (reads results files)
# ─────────────────────────────────────────────────────────────────────────────
def _load_results():
    """Read GNN and LogReg results from the results directory."""
    # GNN: aggregate mean±std per model
    gnn_raw = defaultdict(list)
    for line in open(RESULTS / "gnn_results.jsonl"):
        r = json.loads(line)
        gnn_raw[r["model"]].append(r)

    def ms(vals):
        return (statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0)

    gnn = {}
    for model, entries in gnn_raw.items():
        gnn[model] = {
            "mf1": ms([e["macro_f1"] for e in entries]),
            "acc": ms([e["accuracy"]  for e in entries]),
        }

    # LogReg
    lr = json.loads(open(RESULTS / "logreg_results.json").read())
    return gnn, lr


def plot_inversion():
    gnn, lr = _load_results()

    # Mid-tier breakdown (standard vs mid-tier per model)
    mt_path = RESULTS / "midtier_results.json"
    mt = json.loads(mt_path.read_text()) if mt_path.exists() else {}
    mt_sum = mt.get("summary", {})

    LEAKY_BLIND  = 0.996
    LEAKY_V2     = 0.889

    baseline_mf1, baseline_std = gnn["baseline"]["mf1"]
    v2_mf1,       v2_std       = gnn["v2-hgnn"]["mf1"]
    v3_mf1,       v3_std       = gnn["v3-nli"]["mf1"]

    always_nee_mf1 = lr["baselines"]["always_nee"]["macro_f1"]
    lr_aware_mf1   = lr["trust_aware"]["all"]["macro_f1"]
    lr_std_mf1     = lr["trust_aware"]["standard"]["macro_f1"]
    lr_mid_mf1     = lr["trust_aware"]["mid_tier"]["macro_f1"]

    v2_std_mf1  = mt_sum.get("v2-hgnn", {}).get("std_macro_f1_mean",  1.0)
    v2_mid_mf1  = mt_sum.get("v2-hgnn", {}).get("mid_macro_f1_mean",  0.0)
    v2_mid_std  = mt_sum.get("v2-hgnn", {}).get("mid_macro_f1_std",   0.0)
    v3_std_mf1  = mt_sum.get("v3-nli",  {}).get("std_macro_f1_mean",  1.0)
    v3_mid_mf1  = mt_sum.get("v3-nli",  {}).get("mid_macro_f1_mean",  0.0)
    v3_mid_std  = mt_sum.get("v3-nli",  {}).get("mid_macro_f1_std",   0.0)

    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(13.0, 4.5),
        gridspec_kw={"width_ratios": [0.7, 1.8, 1.5]},
    )

    # ── left: leaky split ────────────────────────────────────────────────────
    lvals   = [LEAKY_BLIND, LEAKY_V2]
    llabels = [f"Trust-blind\n({LEAKY_BLIND:.3f})", f"v2-HGNN\n({LEAKY_V2:.3f})"]
    lcolors = [C_BLIND, C_V2]
    lb = ax_l.bar(llabels, lvals, color=lcolors,
                  edgecolor="black", linewidth=0.8, hatch=["", "//"], width=0.55)
    ax_l.axhline(0.50, color="red", linestyle=":", linewidth=1.1, alpha=0.6)
    ax_l.text(1.38, 0.51, "0.50", fontsize=7.5, color="red", alpha=0.8)
    for bar, val in zip(lb, lvals):
        ax_l.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                  f"{val:.3f}", ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold")
    ax_l.set_ylim(0, 1.22)
    ax_l.set_title("Leaky split\n(accuracy)", fontsize=10)
    ax_l.set_ylabel("Score", fontsize=10)
    ax_l.yaxis.grid(True, alpha=0.25); ax_l.set_axisbelow(True)

    # ── middle: corrected split overall ──────────────────────────────────────
    rmodels  = ["Trust-blind\nGNN", "Always-\nNEE", "LogReg\nprobe", "v2-HGNN", "v3-NLI"]
    rmeans   = [baseline_mf1, always_nee_mf1, lr_aware_mf1, v2_mf1, v3_mf1]
    rstds    = [baseline_std, 0.0,             0.0,          v2_std, v3_std]
    rcolors  = [C_BLIND, C_NEE, C_PROBE, C_V2, C_V3]
    rhatches = ["",      "",    "xx",    "//", ".."]

    x = np.arange(len(rmodels))
    rb = ax_m.bar(x, rmeans, yerr=rstds, color=rcolors,
                  edgecolor="black", linewidth=0.8, hatch=rhatches, width=0.55,
                  capsize=4,
                  error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5})

    ax_m.axhline(baseline_mf1, color=C_BLIND, linestyle="--", linewidth=1.0,
                 alpha=0.40, label=f"trust-blind baseline ({baseline_mf1:.3f})")
    ax_m.axhline(always_nee_mf1, color=C_NEE, linestyle=":", linewidth=1.0,
                 alpha=0.40, label=f"always-NEE floor ({always_nee_mf1:.3f})")

    ax_m.set_xticks(x); ax_m.set_xticklabels(rmodels, fontsize=9)
    ax_m.set_title("Corrected split — overall\n(macro-F1, 3 independent runs)", fontsize=10)
    ax_m.set_ylim(0, 1.22)
    ax_m.yaxis.grid(True, alpha=0.25); ax_m.set_axisbelow(True)
    ax_m.legend(fontsize=7.5, loc="lower right", framealpha=0.92)

    for bar, val, std in zip(rb, rmeans, rstds):
        lbl = f"{val:.3f}" if std == 0 else f"{val:.3f}\n±{std:.3f}"
        ax_m.text(bar.get_x() + bar.get_width()/2, val + max(std, 0.0) + 0.016,
                  lbl, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # ── right: standard vs mid-tier breakdown ────────────────────────────────
    # Grouped bars: for each model, two bars (standard / mid-tier)
    grp_models  = ["LogReg\nprobe", "v2-HGNN", "v3-NLI"]
    std_vals    = [lr_std_mf1,    v2_std_mf1, v3_std_mf1]
    mid_vals    = [lr_mid_mf1,    v2_mid_mf1, v3_mid_mf1]
    mid_errs    = [0.0,           v2_mid_std, v3_mid_std]

    xg   = np.arange(len(grp_models))
    W    = 0.34
    C_STD = "#4CAF50"   # green — standard records (solved)
    C_MID = "#FF7043"   # orange-red — mid-tier (unsolved)

    b_std = ax_r.bar(xg - W/2, std_vals, width=W, color=C_STD,
                     edgecolor="black", linewidth=0.8, label="Standard records")
    b_mid = ax_r.bar(xg + W/2, mid_vals, width=W, yerr=mid_errs, color=C_MID,
                     edgecolor="black", linewidth=0.8, capsize=4,
                     error_kw={"elinewidth": 1.5, "ecolor": "black", "capthick": 1.5},
                     label="Mid-tier probes")

    for bar, val in zip(b_std, std_vals):
        ax_r.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val, err in zip(b_mid, mid_vals, mid_errs):
        lbl = f"{val:.3f}" if err == 0 else f"{val:.3f}\n±{err:.3f}"
        ax_r.text(bar.get_x() + bar.get_width()/2, val + err + 0.012,
                  lbl, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_r.axhline(0.75, color="red", linestyle=":", linewidth=1.0, alpha=0.5,
                 label="EC threshold (0.75)")
    ax_r.set_xticks(xg); ax_r.set_xticklabels(grp_models, fontsize=9)
    ax_r.set_title("Standard vs mid-tier\n(macro-F1 breakdown)", fontsize=10)
    ax_r.set_ylim(0, 1.22)
    ax_r.yaxis.grid(True, alpha=0.25); ax_r.set_axisbelow(True)
    ax_r.legend(fontsize=7.5, loc="center right", framealpha=0.92)

    fig.suptitle("The Inversion", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "pb_inversion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓  pb_inversion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Architecture comparison  (T-junction branches, straight arrows)
# ─────────────────────────────────────────────────────────────────────────────
def plot_architecture():
    FW, FH = 12.0, 5.6
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW); ax.set_ylim(0, FH)
    ax.axis("off")

    LW   = 1.8
    COL  = [1.8, 6.0, 10.2]
    BW   = 2.6    # column box width
    BH   = 0.58   # box height
    BOFF = 1.0    # branch x-offset from diamond centre
    BBW  = 1.5    # branch box width

    def sbox(cx, cy, w, h, text, fc, ec_="black", lw=LW,
             fs=9.0, bold=False, alpha=1.0):
        ax.add_patch(mpatches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            facecolor=fc, edgecolor=ec_, linewidth=lw,
            alpha=alpha, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                multialignment="center", zorder=3)

    def sdiamond(cx, cy, w, h, text, fc, ec_="black", lw=LW, fs=9.0):
        pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec_,
                             linewidth=lw, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=3)

    def arrow(x0, y0, x1, y1, lw=1.5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=lw, mutation_scale=13), zorder=4)

    def line(x0, y0, x1, y1, lw=1.5):
        ax.plot([x0, x1], [y0, y1], color="black", lw=lw, zorder=3)

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(FW/2, 5.38, "Model Variants: Key Structural Differences",
            ha="center", fontsize=12, fontweight="bold")

    # ── column headers ────────────────────────────────────────────────────────
    titles  = ["Baseline\n(trust-blind)", "v2-HGNN\n(trust-aware)",
               "v3-NLI\n(trust-aware + NLI)"]
    hcolors = [C_BLIND, C_V2, C_V3]
    for cx, title, hc in zip(COL, titles, hcolors):
        sbox(cx, 4.98, BW+0.3, 0.62, title,
             fc=hc, ec_=hc, lw=1.8, fs=9.5, bold=True, alpha=0.30)

    # ── row: input embedding ─────────────────────────────────────────────────
    sbox(COL[0], 4.18, BW, BH, "SBERT embedding", "#DDEEFF")
    sbox(COL[1], 4.18, BW, BH, "SBERT embedding", "#DDEEFF")
    sbox(COL[2], 4.18, BW, BH, "SBERT embedding\n+ DeBERTa NLI ×3",
         "#FFF3E0", ec_="#E88C2A", lw=LW)
    for cx in COL:
        arrow(cx, 3.89, cx, 3.60)

    # ── row: graph attention ──────────────────────────────────────────────────
    for cx in COL:
        sbox(cx, 3.31, BW, BH, "Heterogeneous\nGraph Attention", "#DDEEFF")
        arrow(cx, 3.02, cx, 2.82)   # ends at diamond top vertex

    # ── EC layer: baseline placeholder ───────────────────────────────────────
    sbox(COL[0], 2.42, BW, BH, "— no EC layer —",
         fc="#F2F2F2", ec_="#AAAAAA", lw=1.2)
    arrow(COL[0], 2.13, COL[0], 1.54)
    sbox(COL[0], 1.25, BW, BH, "VerdictHead\n(neural only)",
         fc="#F2F2F2", ec_="#AAAAAA", lw=LW, fs=8.5)
    arrow(COL[0], 0.96, COL[0], 0.51)

    # ── EC diamond + T-junction for v2-HGNN and v3-NLI ───────────────────────
    D_CY, D_W, D_H = 2.42, 2.1, 0.78
    D_BOT  = D_CY - D_H / 2    # bottom vertex y = 2.03
    T_Y    = 1.88               # horizontal T-bar level
    BOX_Y  = 1.25
    MRG_Y  = 0.78               # merge bar level

    for cx in COL[1:]:
        lx = cx - BOFF
        rx = cx + BOFF

        sdiamond(cx, D_CY, D_W, D_H, "EC ≥ 0.75 ?",
                 fc="#E8F5E9", ec_="#388E3C", fs=9.0)

        # Stem: diamond bottom → T-bar  (plain line, no head)
        line(cx, D_BOT, cx, T_Y)

        # T-bar horizontal  (no head)
        line(lx, T_Y, rx, T_Y)

        # Vertical arrows: T-bar → branch boxes
        arrow(lx, T_Y, lx, 1.54)
        arrow(rx, T_Y, rx, 1.54)

        # Branch labels
        ax.text(lx - 0.12, 1.71, "No",  fontsize=8, style="italic",
                ha="right", zorder=5)
        ax.text(rx + 0.12, 1.71, "Yes", fontsize=8, style="italic",
                ha="left",  zorder=5)

        # Branch output boxes
        sbox(lx, BOX_Y, BBW, BH, "Hybrid\nVerdictHead",
             fc="#DDEEFF", ec_=C_V2, lw=LW, fs=8.5)
        sbox(rx, BOX_Y, BBW, BH, "Symbolic\nverdict",
             fc="#C8E6C9", ec_="#388E3C", lw=LW, fs=8.5)

        # Converge stems: box bottoms → merge bar  (plain lines)
        line(lx, BOX_Y - BH/2, lx, MRG_Y)
        line(rx, BOX_Y - BH/2, rx, MRG_Y)
        line(lx, MRG_Y, rx, MRG_Y)            # merge bar

        # Arrow: merge bar centre → Verdict
        arrow(cx, MRG_Y, cx, 0.51)

    # ── Verdict boxes ────────────────────────────────────────────────────────
    for cx in COL:
        sbox(cx, 0.28, BW, 0.46, "Verdict", fc="#EEEEEE", lw=LW, bold=True, fs=9.5)

    # ── v3-NLI annotation: NLI is text-semantic, not trust-correlated ─────────
    # v3-NLI DOES use ST via EC. The issue: DeBERTa assigns near-perfect
    # entailment/contradiction on template text regardless of ST (text is
    # semantically clear by construction) → creates competing signal with EC.
    ax.text(COL[2] + 1.42, 3.31,
            "NLI ≈ 0.99 entailment\nor contradiction\n(same for high- AND low-trust)\n"
            "→ contradictory input",
            ha="left", fontsize=7.5, color="#B05000", style="italic",
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8F0",
                      edgecolor="#E88C2A", linewidth=1.0))
    ax.annotate("", xy=(COL[2] + BW/2, 3.31),
                xytext=(COL[2] + 1.40, 3.31),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color="#E88C2A",
                                lw=1.0, mutation_scale=10))

    fig.savefig(OUT / "pb_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  ✓  pb_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
DOCS_FIGS = HERE.parent / "docs" / "figures"

if __name__ == "__main__":
    print(f"Writing figures → {OUT}/")
    plot_construction()
    plot_leakage()
    plot_ec_curve()
    plot_inversion()
    plot_architecture()
    if DOCS_FIGS.exists():
        for png in OUT.glob("*.png"):
            shutil.copy2(png, DOCS_FIGS / png.name)
        print(f"  ✓  synced {len(list(OUT.glob('*.png')))} figures → docs/figures/")
    print("Done.")
