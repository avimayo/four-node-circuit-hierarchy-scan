"""
Three complementary analyses of the 256-circuit dataset:

1. fig_state_heatmaps.png  — 16x16 heatmap per individual stable state
                             (fraction of samples where that state is a stable attractor)
2. fig_diversity_heatmap.png — 16x16 heatmap of attractor diversity
                               (number of distinct solution types per circuit)
3. fig_bwd_collapse.png    — stacked bar: backward-edge combinations (x) vs
                             fraction of samples in each major solution type (y),
                             averaged over all 16 forward-edge combinations
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from itertools import product
from collections import defaultdict

# ── MatrixPlot-style icon helpers ─────────────────────────────────────────────
ICON_YELLOW = np.array([0.976, 0.878, 0.157])   # active node
ICON_TEAL   = np.array([0.122, 0.627, 0.518])   # inactive node
NODES       = ["F", "M", "T", "B"]

def pattern_icon(pat):
    """Return (n_states x 4 x 3) RGB array for a phenotype pattern string."""
    if pat == "other":
        return np.full((1, 4, 3), 0.72)
    states = sorted(pat.split("|"), key=lambda s: s.count("1"))
    img = np.zeros((len(states), 4, 3), dtype=float)
    for i, s in enumerate(states):
        for j, bit in enumerate(s):
            img[i, j] = ICON_YELLOW if bit == "1" else ICON_TEAL
    return img

# ── Mini circuit diagram helpers (same style as hierarchy_heatmap.py) ─────────
_NP = {"F": (0.20, 0.80), "M": (0.80, 0.80),   # node positions in [0,1]^2
       "T": (0.80, 0.20), "B": (0.20, 0.20)}
_NC = {"F": "#4C72B0", "M": "#DD8452", "T": "#55A868", "B": "#C44E52"}
# Backward-edge order: T->F, T->M, B->F, B->M
_BWD_SD = [
    ((0.80, 0.20), (0.20, 0.80)),   # T -> F
    ((0.80, 0.20), (0.80, 0.80)),   # T -> M
    ((0.20, 0.20), (0.20, 0.80)),   # B -> F
    ((0.20, 0.20), (0.80, 0.80)),   # B -> M
]

def draw_circuit_axes(ax, bb, ms=2.5, lw=0.6, ms_scale=3):
    """Draw 4-node mini circuit in ax; bb = (T->F, T->M, B->F, B->M) bits."""
    for node, (nx, ny) in _NP.items():
        ax.plot(nx, ny, "o", ms=ms, color=_NC[node], zorder=5)
    for b, (src, dst) in zip(bb, _BWD_SD):
        if b:
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=lw, mutation_scale=ms_scale),
                        zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

N_SAMPLES = 10000

EDGE_NAMES = ["F->T", "T->F", "M->T", "T->M", "F->B", "B->F", "M->B", "B->M"]
FWD_ENAMES = ["F->T", "M->T", "F->B", "M->B"]
BWD_ENAMES = ["T->F", "T->M", "B->F", "B->M"]

all_vecs    = sorted(product([0, 1], repeat=8), key=sum)
idx_to_vec  = {i + 1: v for i, v in enumerate(all_vecs)}
fwd_combos  = sorted(product([0, 1], repeat=4), key=sum)
bwd_combos  = sorted(product([0, 1], repeat=4), key=sum)
GROUP_SEPS  = [0.5, 4.5, 10.5, 14.5]

def fwd_bits(vec): return (vec[0], vec[2], vec[4], vec[6])
def bwd_bits(vec): return (vec[1], vec[3], vec[5], vec[7])

def combo_label(bits, names):
    active = [n for n, b in zip(names, bits) if b]
    return "{" + ",".join(active) + "}" if active else "{}"

fwd_labels = [combo_label(b, FWD_ENAMES) for b in fwd_combos]
bwd_labels = [combo_label(b, BWD_ENAMES) for b in bwd_combos]

# ── Read data ─────────────────────────────────────────────────────────────────
ALL_STATES = [format(i, "04b") for i in range(16)]

state_freq = defaultdict(lambda: defaultdict(float))
pat_freq   = defaultdict(lambda: defaultdict(float))
n_distinct = defaultdict(int)

with open("final_results.csv") as f:
    for row in csv.DictReader(f):
        c   = int(row["circuit_index"])
        pat = row["phenotype_pattern"]
        cnt = int(row["count"])
        frac = cnt / N_SAMPLES
        pat_freq[c][pat] += frac
        n_distinct[c] += 1
        for s in pat.split("|"):
            state_freq[c][s] += frac

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: per-state heatmaps
# ─────────────────────────────────────────────────────────────────────────────
state_means = {}
for s in ALL_STATES:
    if s == "0000":
        continue
    vals = [state_freq[c].get(s, 0.0) for c in range(1, 257)]
    state_means[s] = np.mean(vals)

top_states = sorted(state_means, key=lambda s: -state_means[s])[:12]

STATE_LABELS = {
    "0000": "none",  "0001": "B",     "0010": "T",     "0011": "T+B",
    "0100": "M",     "0101": "M+B",   "0110": "M+T",   "0111": "M+T+B",
    "1000": "F",     "1001": "F+B",   "1010": "F+T",   "1011": "F+T+B",
    "1100": "F+M",   "1101": "F+M+B", "1110": "F+M+T", "1111": "F+M+T+B"
}

ncols, nrows = 4, 3
fig1, axes = plt.subplots(nrows, ncols, figsize=(16, 11))

try:
    from matplotlib.colors import AsinhNorm
    use_asinh = True
except ImportError:
    use_asinh = False

for ax_idx, s in enumerate(top_states):
    ax = axes[ax_idx // ncols, ax_idx % ncols]
    mat  = np.full((16, 16), np.nan)
    mask = np.zeros((16, 16), dtype=bool)
    for c, vec in idx_to_vec.items():
        ri = bwd_combos.index(bwd_bits(vec))
        ci = fwd_combos.index(fwd_bits(vec))
        if c not in n_distinct:
            mask[ri, ci] = True
        else:
            mat[ri, ci] = state_freq[c].get(s, 0.0)
    mmat = np.ma.array(mat, mask=mask)
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("lightgrey")
    norm = AsinhNorm(linear_width=0.05, vmin=0, vmax=1) if use_asinh else mcolors.Normalize(0, 1)
    im = ax.imshow(mmat, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    for sep in GROUP_SEPS:
        ax.axvline(sep, color="black", linewidth=1.0, zorder=3)
        ax.axhline(sep, color="black", linewidth=1.0, zorder=3)
    ax.set_title(f"{s}  ({STATE_LABELS[s]})  mean={state_means[s]:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig1.suptitle("Fraction of samples where each state is a stable attractor\n"
              "X: FM->TB forward edges  |  Y: TB->FM backward edges  (arcsinh scale)",
              fontsize=11)
plt.tight_layout()
fig1.savefig("fig_state_heatmaps.png", dpi=150, bbox_inches="tight")
print("Saved fig_state_heatmaps.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: attractor diversity heatmap
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(9, 8))
mat  = np.full((16, 16), np.nan)
mask = np.zeros((16, 16), dtype=bool)
for c, vec in idx_to_vec.items():
    ri = bwd_combos.index(bwd_bits(vec))
    ci = fwd_combos.index(fwd_bits(vec))
    if c not in n_distinct:
        mask[ri, ci] = True
    else:
        mat[ri, ci] = n_distinct[c]

mmat = np.ma.array(mat, mask=mask)
cmap2 = plt.cm.viridis.copy(); cmap2.set_bad("lightgrey")
im2 = ax.imshow(mmat, cmap=cmap2, aspect="auto", origin="lower",
                vmin=1, vmax=np.nanmax(mat))
for ri in range(16):
    for ci in range(16):
        if not mask[ri, ci] and not np.isnan(mat[ri, ci]):
            ax.text(ci, ri, int(mat[ri, ci]), ha="center", va="center",
                    fontsize=5, color="white" if mat[ri, ci] > 15 else "black")

ax.set_xticks(range(16)); ax.set_xticklabels(fwd_labels, rotation=90, fontsize=5)
ax.set_yticks(range(16)); ax.set_yticklabels(bwd_labels, fontsize=5)
ax.set_xlabel("FM->TB forward edges", fontsize=10, labelpad=8)
ax.set_ylabel("TB->FM backward edges", fontsize=10, labelpad=8)
ax.set_title("Number of distinct attractor types per circuit", fontsize=12)
ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)
for sep in GROUP_SEPS:
    ax.axvline(sep, color="black", linewidth=1.5, zorder=3)
    ax.axhline(sep, color="black", linewidth=1.5, zorder=3)
fig2.colorbar(im2, ax=ax, fraction=0.03, pad=0.02, label="n distinct solution types")
plt.tight_layout()
fig2.savefig("fig_diversity_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved fig_diversity_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: backward-edge collapse — stacked bar
# ─────────────────────────────────────────────────────────────────────────────

pat_totals = defaultdict(float)
for c in range(1, 257):
    for pat, frac in pat_freq[c].items():
        pat_totals[pat] += frac

top_pats = sorted(pat_totals, key=lambda p: -pat_totals[p])[:8]
top_pats = sorted(top_pats, key=lambda p: len(p.split("|")))

bwd_pat_mean = np.zeros((16, len(top_pats) + 1))  # +1 for "other"

for bi, bb in enumerate(bwd_combos):
    circuits_in_row = [c for c, vec in idx_to_vec.items()
                       if bwd_bits(vec) == bb and c in n_distinct]
    if not circuits_in_row:
        continue
    for pi, pat in enumerate(top_pats):
        bwd_pat_mean[bi, pi] = np.mean([pat_freq[c].get(pat, 0.0) for c in circuits_in_row])
    bwd_pat_mean[bi, -1] = 1.0 - bwd_pat_mean[bi, :-1].sum()

def pat_to_label(pat):
    states = pat.split("|")
    return "{" + ",".join(STATE_LABELS.get(s, s) for s in states) + "}"

pat_labels = [pat_to_label(p) for p in top_pats] + ["other"]
colors = plt.cm.tab10(np.linspace(0, 0.9, len(pat_labels)))

# ── Layout constants ───────────────────────────────────────────────────────────
FIG_W, FIG_H = 16.0, 12.0
fig3 = plt.figure(figsize=(FIG_W, FIG_H))

BAR_LEFT    = 0.07
BAR_WIDTH   = 0.90
BAR_BOTTOM  = 0.32
BAR_HEIGHT  = 0.64

# ── Stacked bar ────────────────────────────────────────────────────────────────
ax = fig3.add_axes([BAR_LEFT, BAR_BOTTOM, BAR_WIDTH, BAR_HEIGHT])

bottoms = np.zeros(16)
for pi, color in enumerate(colors):
    ax.bar(range(16), bwd_pat_mean[:, pi], bottom=bottoms,
           color=color, width=0.8, edgecolor="white", linewidth=0.3)
    bottoms += bwd_pat_mean[:, pi]

ax.set_xticks([])
ax.set_ylabel("Fraction of samples\n(averaged over all forward-edge combos)", fontsize=10)
ax.set_title("Solution-type distribution by backward-edge combination\n"
             "(each bar = average over 16 forward-edge variants)", fontsize=11)
ax.set_ylim(0, 1)
ax.set_xlim(-0.5, 15.5)
for sep in GROUP_SEPS:
    ax.axvline(sep, color="black", linewidth=1.2, zorder=3)

# ── Mini circuit tick icons for x-axis ────────────────────────────────────────
# One square axes per bar: 4-node diagram showing which backward edges are active
TICK_SQ_IN = 0.50                          # square side in inches
tick_hf    = TICK_SQ_IN / FIG_H           # figure-y fraction
tick_wf    = TICK_SQ_IN / FIG_W           # figure-x fraction
tick_y0    = BAR_BOTTOM - 0.005 - tick_hf

for bi, bb in enumerate(bwd_combos):
    x_c = BAR_LEFT + BAR_WIDTH * (bi + 0.5) / 16.0
    at  = fig3.add_axes([x_c - tick_wf / 2.0, tick_y0, tick_wf, tick_hf])
    draw_circuit_axes(at, bb, ms=2.5, lw=0.7, ms_scale=4)

fig3.text(BAR_LEFT + BAR_WIDTH * 0.5, tick_y0 - 0.010,
          "TB→FM backward edges", ha="center", va="top", fontsize=11)

# ── Icon legend (bottom-aligned, smaller MatrixPlot icons) ────────────────────
ICON_CELL_IN = 0.14
icon_cell_hf = ICON_CELL_IN / FIG_H
icon_wf      = 4 * ICON_CELL_IN / FIG_W

swatch_hf   = icon_cell_hf * 0.80
tick_lbl_hf = icon_cell_hf * 1.10
gap_hf      = icon_cell_hf * 0.10

SWATCH_BOTTOM = 0.04
ICON_AXES_Y0  = SWATCH_BOTTOM + swatch_hf + tick_lbl_hf + gap_hf

n_ent    = len(pat_labels)
entry_wf = (0.97 - 0.03) / n_ent

for pi, (pat, color) in enumerate(zip(top_pats + ["other"], colors)):
    xi = 0.03 + (pi + 0.5) * entry_wf - icon_wf / 2.0

    ax_sw = fig3.add_axes([xi, SWATCH_BOTTOM, icon_wf, swatch_hf])
    ax_sw.set_facecolor(color)
    ax_sw.set_xticks([]); ax_sw.set_yticks([])
    for sp in ax_sw.spines.values():
        sp.set_color("#555555"); sp.set_linewidth(0.5)

    if pat == "other":
        ax_sw.text(0.5, 0.5, "other", ha="center", va="center",
                   fontsize=7, fontweight="bold", color="black",
                   transform=ax_sw.transAxes)
        continue

    icon    = pattern_icon(pat)
    n_rows  = icon.shape[0]
    icon_hf = n_rows * icon_cell_hf

    ax_ic = fig3.add_axes([xi, ICON_AXES_Y0, icon_wf, icon_hf])
    ax_ic.imshow(icon, aspect="auto", interpolation="nearest", origin="upper")

    for x in [0.5, 1.5, 2.5]:
        ax_ic.axvline(x, color="black", linewidth=0.8, zorder=5)
    for row in range(n_rows - 1):
        ax_ic.axhline(row + 0.5, color="black", linewidth=0.8, zorder=5)

    ax_ic.set_xticks([0, 1, 2, 3])
    ax_ic.set_xticklabels(NODES, fontsize=7, fontweight="bold")
    ax_ic.tick_params(axis="x", bottom=False, labelbottom=True, pad=1, length=0)
    ax_ic.set_yticks([])
    for sp in ax_ic.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.8)

plt.savefig("fig_bwd_collapse.png", dpi=150, bbox_inches="tight")
print("Saved fig_bwd_collapse.png")
