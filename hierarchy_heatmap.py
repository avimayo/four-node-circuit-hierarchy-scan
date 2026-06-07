"""
Hierarchy heatmap: fraction of parameter samples where the stable-state set
contains {1000 (F), 1100 (F+M), 1111 (F+M+T+B)} — the F → F+M → all hierarchy.

X-axis: 16 FM→TB forward-edge combinations  (F→T, M→T, F→B, M→B)
Y-axis: 16 TB→FM backward-edge combinations (T→F, T→M, B→F, B→M)
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product
from collections import defaultdict

N_SAMPLES = 10000
HIER_STATES   = {"1000", "1100", "1111"}                    # must contain these
HIER_ALLOWED1 = {"0000", "1000", "1100", "1111"}            # pattern A: pure
HIER_ALLOWED2 = {"0000", "0011", "1000", "1100", "1111"}    # pattern B: + T+B state

# ── Circuit index → edge vector ──────────────────────────────────────────────
# Edge vector order: (pft, ptf, pmt, ptm, pfb, pbf, pmb, pbm)
# Forward  FM→TB: indices 0,2,4,6  (pft, pmt, pfb, pmb)
# Backward TB→FM: indices 1,3,5,7  (ptf, ptm, pbf, pbm)
all_vecs = sorted(product([0, 1], repeat=8), key=sum)   # same as Mathematica SortBy[...,Total]
idx_to_vec = {i + 1: v for i, v in enumerate(all_vecs)}  # 1-based

def fwd_bits(vec):   return (vec[0], vec[2], vec[4], vec[6])   # F→T, M→T, F→B, M→B
def bwd_bits(vec):   return (vec[1], vec[3], vec[5], vec[7])   # T→F, T→M, B→F, B→M

# ── Read final_results.csv ────────────────────────────────────────────────────
# For each circuit, compute fraction of samples that contain all HIER_STATES
hier_freq   = defaultdict(float)   # circuit_index → hierarchy frequency
has_results = set()                # circuits that actually have result data

with open("final_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        c    = int(row["circuit_index"])
        pat  = row["phenotype_pattern"]
        cnt  = int(row["count"])
        has_results.add(c)
        states_in_pat = set(pat.split("|"))
        if HIER_STATES.issubset(states_in_pat) and (
                states_in_pat == HIER_ALLOWED1 or states_in_pat == HIER_ALLOWED2):
            hier_freq[c] += cnt / N_SAMPLES

# ── Build 16×16 matrix ───────────────────────────────────────────────────────
# Row index = backward bits (TB→FM), col index = forward bits (FM→TB)
# Sorted by number of active edges so groups 0,1,2,3,4 are contiguous
fwd_combos = sorted(product([0, 1], repeat=4), key=sum)
bwd_combos = sorted(product([0, 1], repeat=4), key=sum)
# Group boundary positions (after n=0,1,2,3 edge groups): 0.5, 4.5, 10.5, 14.5
GROUP_SEPS = [0.5, 4.5, 10.5, 14.5]

mat   = np.full((16, 16), np.nan)
mask  = np.zeros((16, 16), dtype=bool)          # True = no free params (circuit 1)

for c, vec in idx_to_vec.items():
    fb = fwd_bits(vec)
    bb = bwd_bits(vec)
    ri = bwd_combos.index(bb)
    ci = fwd_combos.index(fb)
    if c == 1 or c not in has_results:   # no edges or not yet computed → grey
        mask[ri, ci] = True
    else:
        mat[ri, ci] = hier_freq.get(c, 0.0)

masked_mat = np.ma.array(mat, mask=mask)

# ── Edge labels for tick marks ────────────────────────────────────────────────
FWD_LABELS = ["F→T", "M→T", "F→B", "M→B"]
BWD_LABELS = ["T→F", "T→M", "B→F", "B→M"]

def combo_label(bits, names):
    active = [n for n, b in zip(names, bits) if b]
    return ", ".join(active) if active else "∅"

fwd_tick_labels = [combo_label(b, FWD_LABELS) for b in fwd_combos]
bwd_tick_labels = [combo_label(b, BWD_LABELS) for b in bwd_combos]

# ── Mini circuit diagram at each tick ────────────────────────────────────────
NODE_POS = {"F": (0, 1), "M": (1, 1), "T": (1, 0), "B": (0, 0)}
EDGE_MAP  = {
    "F→T": ("F", "T"), "M→T": ("M", "T"),
    "F→B": ("F", "B"), "M→B": ("M", "B"),
    "T→F": ("T", "F"), "T→M": ("T", "M"),
    "B→F": ("B", "F"), "B→M": ("B", "M"),
}

def draw_mini_circuit(ax, center_x, center_y, active_edges, size=0.22):
    s = size
    cols = {"F": "#4C72B0", "M": "#DD8452", "T": "#55A868", "B": "#C44E52"}
    for node, (nx, ny) in NODE_POS.items():
        ax.plot(center_x + (nx - 0.5) * s, center_y + (ny - 0.5) * s,
                "o", ms=3, color=cols[node], transform=ax.transData, zorder=5)
    for ename in active_edges:
        src, dst = EDGE_MAP[ename]
        sx, sy = NODE_POS[src]; dx, dy = NODE_POS[dst]
        ax.annotate("",
            xy  =(center_x + (dx-0.5)*s, center_y + (dy-0.5)*s),
            xytext=(center_x + (sx-0.5)*s, center_y + (sy-0.5)*s),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                            mutation_scale=4),
            zorder=4)

def bits_to_edge_names(bits, names_list, edge_names):
    return [edge_names[i] for i, b in enumerate(names_list) if b]

FWD_ENAMES = ["F→T", "M→T", "F→B", "M→B"]
BWD_ENAMES = ["T→F", "T→M", "B→F", "B→M"]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

cmap = plt.cm.YlOrRd.copy()
cmap.set_bad("lightgrey")

try:
    from matplotlib.colors import AsinhNorm
    norm = AsinhNorm(linear_width=0.05, vmin=0, vmax=1)
except ImportError:
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)

im = ax.imshow(masked_mat, cmap=cmap, norm=norm,
               aspect="auto", origin="lower")

# Grid lines
ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Group separator lines
for sep in GROUP_SEPS:
    ax.axvline(sep, color="black", linewidth=1.5, zorder=3)
    ax.axhline(sep, color="black", linewidth=1.5, zorder=3)

# Annotate cells with values
for ri in range(16):
    for ci in range(16):
        if not mask[ri, ci] and not np.isnan(mat[ri, ci]):
            v = mat[ri, ci]
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if v < 0.6 else "white")

# Tick positions and labels (mini diagrams replace text labels)
ax.set_xticks(range(16))
ax.set_yticks(range(16))
ax.set_xticklabels([""] * 16)
ax.set_yticklabels([""] * 16)

# Draw mini diagrams outside the axes
DIAG_X_OFF = -1.5   # columns to the left of row 0
DIAG_Y_OFF = -1.5   # rows below col 0

# We draw in data coordinates by expanding the axes limits after drawing
ax.set_xlim(-2.5, 15.5)
ax.set_ylim(-2.5, 15.5)

for ci, fb in enumerate(fwd_combos):
    active = [FWD_ENAMES[i] for i, b in enumerate(fb) if b]
    draw_mini_circuit(ax, ci, -1.5, active, size=0.32)

for ri, bb in enumerate(bwd_combos):
    active = [BWD_ENAMES[i] for i, b in enumerate(bb) if b]
    draw_mini_circuit(ax, -1.5, ri, active, size=0.32)

# Axis labels
ax.set_xlabel("FM → TB forward edges  (F→T, M→T, F→B, M→B)", fontsize=11, labelpad=50)
ax.set_ylabel("TB → FM backward edges  (T→F, T→M, B→F, B→M)", fontsize=11, labelpad=50)

cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_ticks([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
cb.set_ticklabels(["0", "0.05", "0.1", "0.2", "0.4", "0.6", "0.8", "1.0"])
cb.set_label("Fraction of samples with {0000,1000,1100,1111} or {0000,0011,1000,1100,1111} (arcsinh scale)", fontsize=9)

n_done = len(has_results)
ax.set_title(
    f"Hierarchy frequency by circuit topology  "
    f"({'PARTIAL: ' if n_done < 256 else ''}{n_done}/256 circuits)",
    fontsize=13, pad=12
)

plt.tight_layout()
plt.savefig("hierarchy_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved hierarchy_heatmap.png  ({n_done}/256 circuits)")
