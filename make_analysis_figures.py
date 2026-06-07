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
from itertools import product
from collections import defaultdict

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

# per-circuit per-state frequency (state appears as a stable attractor)
state_freq = defaultdict(lambda: defaultdict(float))
# per-circuit per-pattern frequency (exact joint pattern)
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
# Select states with mean freq > 0.01 (excluding 0000 which is always ~1)
# ─────────────────────────────────────────────────────────────────────────────
state_means = {}
for s in ALL_STATES:
    if s == "0000":
        continue
    vals = [state_freq[c].get(s, 0.0) for c in range(1, 257)]
    state_means[s] = np.mean(vals)

# Pick top 12 states by mean frequency
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
# For each of 16 backward-edge combos, average the pattern frequencies
# across all 16 forward-edge combos, then show top solution types stacked
# ─────────────────────────────────────────────────────────────────────────────

# Identify top patterns by total count across all circuits
pat_totals = defaultdict(float)
for c in range(1, 257):
    for pat, frac in pat_freq[c].items():
        pat_totals[pat] += frac

top_pats = sorted(pat_totals, key=lambda p: -pat_totals[p])[:8]

# For each backward-edge combo, average over forward-edge combos
bwd_pat_mean = np.zeros((16, len(top_pats) + 1))  # +1 for "other"

for bi, bb in enumerate(bwd_combos):  # already sorted by edge count
    circuits_in_row = [c for c, vec in idx_to_vec.items()
                       if bwd_bits(vec) == bb and c in n_distinct]
    if not circuits_in_row:
        continue
    for pi, pat in enumerate(top_pats):
        bwd_pat_mean[bi, pi] = np.mean([pat_freq[c].get(pat, 0.0) for c in circuits_in_row])
    # "other" = remainder
    bwd_pat_mean[bi, -1] = 1.0 - bwd_pat_mean[bi, :-1].sum()

# Clean labels for patterns
def pat_to_label(pat):
    states = pat.split("|")
    return "{" + ",".join(STATE_LABELS.get(s, s) for s in states) + "}"

pat_labels = [pat_to_label(p) for p in top_pats] + ["other"]
colors = plt.cm.tab10(np.linspace(0, 0.9, len(pat_labels)))

fig3, ax = plt.subplots(figsize=(14, 6))
bottoms = np.zeros(16)
for pi, (label, color) in enumerate(zip(pat_labels, colors)):
    ax.bar(range(16), bwd_pat_mean[:, pi], bottom=bottoms,
           label=label, color=color, width=0.8, edgecolor="white", linewidth=0.3)
    bottoms += bwd_pat_mean[:, pi]

ax.set_xticks(range(16))
ax.set_xticklabels(bwd_labels, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("TB->FM backward edges", fontsize=11)
ax.set_ylabel("Fraction of samples (averaged over all forward-edge combos)", fontsize=10)
ax.set_title("Solution-type distribution by backward-edge combination\n"
             "(each bar = average over 16 forward-edge variants)", fontsize=11)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, framealpha=0.9)
ax.set_ylim(0, 1)
ax.set_xlim(-0.5, 15.5)
plt.tight_layout()
fig3.savefig("fig_bwd_collapse.png", dpi=150, bbox_inches="tight")
print("Saved fig_bwd_collapse.png")
