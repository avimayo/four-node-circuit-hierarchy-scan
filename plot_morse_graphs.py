#!/usr/bin/env python3
"""
plot_morse_graphs.py  —  Morse complex / heteroclinic network examples.

For a given phase portrait type (stable_pat, semi_pat, unstable_pat), draws the
full 4-bit hypercube with nodes coloured by stability class.  Edges shown only
between patterns differing by exactly 1 bit (natural gradient-flow channels).
Arrows indicate the inferred direction of heteroclinic flow (lower stability →
higher stability attractor through the saddle).

Generates one PNG with 4 panels: circuit 114, 66, 8, and 2.
"""

import csv, itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Bit-hypercube helpers ─────────────────────────────────────────────────────
ALL_PATS = [f"{i:04b}" for i in range(16)]      # 0000..1111
LABELS   = {
    "0000": "∅",   "1000": "F",   "0100": "M",   "0010": "T",   "0001": "B",
    "1100": "FM",  "1010": "FT",  "1001": "FB",  "0110": "MT",  "0101": "MB",
    "0011": "TB",  "1110": "FMT", "1101": "FMB", "1011": "FTB", "0111": "MTB",
    "1111": "FMTB",
}

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# Positions on the 4-bit hypercube: arrange by level (popcount), spread by index
def hypercube_layout():
    level_x = {0: [0], 1: [-1.5, -0.5, 0.5, 1.5],
                2: [-2, -1, 0, 1, 2, 3], 3: [-1.5, -0.5, 0.5, 1.5], 4: [0]}
    pos = {}
    by_level = {k: [] for k in range(5)}
    for p in ALL_PATS:
        by_level[p.count("1")].append(p)
    for lv, pats in by_level.items():
        xs = level_x[lv]
        for i, p in enumerate(sorted(pats)):
            x_idx = i - len(pats) // 2
            pos[p] = (xs[i] if i < len(xs) else i, lv * 1.5)
    return pos

POS = hypercube_layout()

# ── Stability class parser ────────────────────────────────────────────────────
def parse_pat(s):
    if not s or s == "none":
        return set()
    return set(s.split("|"))

def classify(stable, semi1, semi2, semi3, unstable):
    """Return dict pattern→class for all patterns present."""
    cls = {}
    for p in parse_pat(stable):   cls[p] = "stable"
    for p in parse_pat(semi1):
        if p not in cls:          cls[p] = "semi1"
    for p in parse_pat(semi2):
        if p not in cls:          cls[p] = "semi2"
    for p in parse_pat(semi3):
        if p not in cls:          cls[p] = "semi3"
    for p in parse_pat(unstable):
        if p not in cls:          cls[p] = "unstable"
    return cls

# ── Arrow direction heuristic ─────────────────────────────────────────────────
RANK = {"stable": 0, "semi1": 1, "semi2": 2, "semi3": 3, "unstable": 4}

def infer_flow(cls_dict):
    """Return list of (source, target) for heteroclinic arrows."""
    arrows = []
    for a, b in itertools.combinations(ALL_PATS, 2):
        if hamming(a, b) != 1:
            continue
        ca = cls_dict.get(a)
        cb = cls_dict.get(b)
        if ca is None or cb is None:
            continue
        if RANK.get(ca, 3) > RANK.get(cb, 3):   # a is less stable → b
            arrows.append((a, b))
        elif RANK.get(cb, 3) > RANK.get(ca, 3): # b is less stable → a
            arrows.append((b, a))
    return arrows

# ── Colour scheme ─────────────────────────────────────────────────────────────
NODE_STYLE = {
    "stable":   dict(facecolor="#2ecc71", edgecolor="#145a32", zorder=5, s=600, linewidths=2.5),
    "semi1":    dict(facecolor="#f1c40f", edgecolor="#7d6608", zorder=4, s=360, linewidths=1.8),
    "semi2":    dict(facecolor="#e67e22", edgecolor="#784212", zorder=4, s=250, linewidths=1.5),
    "semi3":    dict(facecolor="#e74c3c", edgecolor="#7b241c", zorder=4, s=160, linewidths=1.2),
    "unstable": dict(facecolor="#7f8c8d", edgecolor="#2c3e50", zorder=3, s=100, linewidths=1.0),
    "absent":   dict(facecolor="#ecf0f1", edgecolor="#bdc3c7", zorder=1, s=80,  linewidths=0.5),
}
# Shrink = node radius ≈ sqrt(s/π) + half linewidth  (in display points)
NODE_SHRINK_A = {"stable": 15, "semi1": 12, "semi2": 10, "semi3": 8, "unstable": 6, "absent": 5}
NODE_SHRINK_B = {"stable": 15, "semi1": 12, "semi2": 10, "semi3": 8, "unstable": 6, "absent": 5}

# ── Single panel ─────────────────────────────────────────────────────────────
def draw_morse_panel(ax, title, stable_str, semi1_str, semi2_str, semi3_str,
                     unstable_str, freq_pct=None):
    cls_dict = classify(stable_str, semi1_str, semi2_str, semi3_str, unstable_str)
    all_semis = parse_pat(semi1_str) | parse_pat(semi2_str) | parse_pat(semi3_str)

    # Draw hypercube edges (background)
    for a, b in itertools.combinations(ALL_PATS, 2):
        if hamming(a, b) == 1:
            xa, ya = POS[a]; xb, yb = POS[b]
            ax.plot([xa, xb], [ya, yb], color="#cccccc", lw=0.7, zorder=0)

    # Draw putative semi→semi connections (thin gray arrows)
    for a, b in itertools.combinations(all_semis, 2):
        if hamming(a, b) != 1:
            continue
        pa, pb = a.count("1"), b.count("1")
        pairs = [(a,b)] if pa < pb else ([(b,a)] if pb < pa else [(a,b),(b,a)])
        for src, tgt in pairs:
            xs, ys = POS[src]; xt, yt = POS[tgt]
            sc, tc = cls_dict.get(src,"absent"), cls_dict.get(tgt,"absent")
            ax.annotate("", xy=(xt, yt), xytext=(xs, ys),
                        arrowprops=dict(arrowstyle="-|>", color="#999999",
                                        lw=0.7, mutation_scale=8,
                                        shrinkA=NODE_SHRINK_A.get(sc,5),
                                        shrinkB=NODE_SHRINK_B.get(tc,5)),
                        zorder=1)

    # Draw main heteroclinic flow arrows (stable ↔ semi ↔ unstable)
    for src, tgt in infer_flow(cls_dict):
        xs, ys = POS[src]; xt, yt = POS[tgt]
        shrink_a = NODE_SHRINK_A.get(cls_dict.get(src, "absent"), 3)
        shrink_b = NODE_SHRINK_B.get(cls_dict.get(tgt, "absent"), 5)
        ax.annotate("", xy=(xt, yt), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle="-|>", color="#333333",
                                    lw=1.4, mutation_scale=12,
                                    shrinkA=shrink_a, shrinkB=shrink_b),
                    zorder=2)

    # Draw nodes
    for pat in ALL_PATS:
        x, y = POS[pat]
        cls  = cls_dict.get(pat, "absent")
        style = NODE_STYLE[cls].copy()
        ax.scatter(x, y, **style)
        lbl = LABELS[pat]
        if cls in ("stable", "semi1", "semi2", "semi3"):
            fs = 7 if cls == "stable" else 6
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=fs, fontweight="normal", color="black", zorder=6)
        else:
            fs  = 5
            col = "#555555" if cls == "unstable" else "#aaaaaa"
            ax.text(x, y + 0.18, lbl, ha="center", va="bottom",
                    fontsize=fs, color=col, zorder=6)

    # Counts
    n_sm = len(parse_pat(semi1_str))+len(parse_pat(semi2_str))+len(parse_pat(semi3_str))
    info = (f"stable={len(parse_pat(stable_str))}  "
            f"semi1={len(parse_pat(semi1_str))} semi2={len(parse_pat(semi2_str))} "
            f"semi3={len(parse_pat(semi3_str))}  "
            f"unstable={len(parse_pat(unstable_str))}")
    if freq_pct:
        info = f"{freq_pct:.1f}% of samples\n{info}"
    ax.text(0.5, -0.04, info, transform=ax.transAxes, ha="center",
            va="top", fontsize=6.5, color="#555555")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(-3, 4); ax.set_ylim(-0.5, 6.5)
    ax.axis("off")

# ── Load top phase types from atlas ──────────────────────────────────────────
ATLAS = "phase_atlas_v2.csv"

def _row_to_tuple(row):
    return (float(row["freq_pct"]),
            row["stable_pat"], row["semi1_pat"], row["semi2_pat"],
            row["semi3_pat"],  row["unstable_pat"])

def top_type(circ, canon_only=False):
    with open(ATLAS) as f:
        for row in csv.DictReader(f):
            if int(row["circuit_idx"]) != circ: continue
            if int(row["rank"]) == 1:
                if canon_only and not int(row["is_canonical_hier"]): continue
                return _row_to_tuple(row)
    return None

def get_type_by_rank(circ, rank):
    with open(ATLAS) as f:
        for row in csv.DictReader(f):
            if int(row["circuit_idx"]) == circ and int(row["rank"]) == rank:
                return _row_to_tuple(row)

# ── Main ─────────────────────────────────────────────────────────────────────
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.patch.set_facecolor("#fafafa")

# Row 1: dominant types for the four illustrative circuits
panels_r1 = [
    (114, "Circuit 114 — dominant type\n(all 4 backward edges, ~72%)"),
    (27,  "Circuit 27 — dominant type\n(T→F + B→F, ~99%)"),
    (8,   "Circuit 8 — dominant type\n(T→F only, ~90%)"),
    (2,   "Circuit 2 — dominant type\n(B→M only, ~47%)"),
]
for ax, (circ, title) in zip(axes[0], panels_r1):
    res = top_type(circ)
    if res:
        freq, sp, sm1, sm2, sm3, usp = res
        draw_morse_panel(ax, title, sp, sm1, sm2, sm3, usp, freq)

# Row 2: circuit 114 rare variant (0011 becomes stable) and circuit 2 non-canonical type
panels_r2 = [
    (114, 2, "Circuit 114 — type 2\n(semi variant)"),
    (114, 4, "Circuit 114 — rare type\n(0011 stabilises → bifurcation)"),
    (2,   2, "Circuit 2 — type 2\n(0011 stable, different semi set)"),
    (6,   1, "Circuit 6 — dominant type\n(T→M only, ~67%)"),
]
for ax, (circ, rank, title) in zip(axes[1], panels_r2):
    res = get_type_by_rank(circ, rank)
    if res:
        freq, sp, sm1, sm2, sm3, usp = res
        draw_morse_panel(ax, title, sp, sm1, sm2, sm3, usp, freq)

# Shared legend
legend_handles = [
    mpatches.Patch(facecolor="#2ecc71", edgecolor="#145a32", label="Stable attractor (n=0)"),
    mpatches.Patch(facecolor="#f1c40f", edgecolor="#7d6608", label="Saddle n=1  (codim-3)"),
    mpatches.Patch(facecolor="#e67e22", edgecolor="#784212", label="Saddle n=2  (codim-2)"),
    mpatches.Patch(facecolor="#e74c3c", edgecolor="#7b241c", label="Saddle n=3  (codim-1)"),
    mpatches.Patch(facecolor="#7f8c8d", edgecolor="#2c3e50", label="Unstable / repeller (n=4)"),
    mpatches.Patch(facecolor="#ecf0f1", edgecolor="#bdc3c7", label="Not detected"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           frameon=True, fontsize=9, bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Morse Complex / Phase Portrait Atlas  ·  TME Circuit Attractor Scan",
             fontsize=13, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0.06, 1, 0.98])

out = os.path.expanduser("~/circuit_hpc/morse_graphs.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
plt.show()
