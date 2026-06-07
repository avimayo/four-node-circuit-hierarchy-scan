"""
Combined figure: all analyses in one canvas.
Canonical hierarchy = {0000,1000,1100,1111} only.

Layout (GridSpec):
  Row 0: [A] canonical hierarchy heatmap  |  [B] attractor diversity heatmap
  Row 1: [C] top-12 per-state heatmaps (3 rows × 4 cols, nested)
  Row 2: [D] backward-edge collapse stacked bar
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from itertools import product
from collections import defaultdict

N_SAMPLES  = 10000
CANONICAL  = "0000|1000|1100|1111"   # canonical hierarchy only

EDGE_NAMES = ["F->T", "T->F", "M->T", "T->M", "F->B", "B->F", "M->B", "B->M"]
FWD_ENAMES = ["F->T", "M->T", "F->B", "M->B"]
BWD_ENAMES = ["T->F", "T->M", "B->F", "B->M"]

all_vecs   = sorted(product([0, 1], repeat=8), key=sum)
idx_to_vec = {i + 1: v for i, v in enumerate(all_vecs)}
fwd_combos = sorted(product([0, 1], repeat=4), key=sum)
bwd_combos = sorted(product([0, 1], repeat=4), key=sum)
GROUP_SEPS = [0.5, 4.5, 10.5, 14.5]

def fwd_bits(vec): return (vec[0], vec[2], vec[4], vec[6])
def bwd_bits(vec): return (vec[1], vec[3], vec[5], vec[7])

def combo_label(bits, names):
    active = [n for n, b in zip(names, bits) if b]
    return "{" + ",".join(active) + "}" if active else "{}"

fwd_labels = [combo_label(b, FWD_ENAMES) for b in fwd_combos]
bwd_labels = [combo_label(b, BWD_ENAMES) for b in bwd_combos]

ALL_STATES = [format(i, "04b") for i in range(16)]
STATE_LABELS = {
    "0000": "none",    "0001": "B",       "0010": "T",       "0011": "T+B",
    "0100": "M",       "0101": "M+B",     "0110": "M+T",     "0111": "M+T+B",
    "1000": "F",       "1001": "F+B",     "1010": "F+T",     "1011": "F+T+B",
    "1100": "F+M",     "1101": "F+M+B",   "1110": "F+M+T",   "1111": "F+M+T+B"
}

# ── Read data ─────────────────────────────────────────────────────────────────
hier_freq  = defaultdict(float)
state_freq = defaultdict(lambda: defaultdict(float))
pat_freq   = defaultdict(lambda: defaultdict(float))
n_distinct = defaultdict(int)
has_results = set()

with open("final_results.csv") as f:
    for row in csv.DictReader(f):
        c   = int(row["circuit_index"])
        pat = row["phenotype_pattern"]
        cnt = int(row["count"])
        frac = cnt / N_SAMPLES
        has_results.add(c)
        pat_freq[c][pat] += frac
        n_distinct[c] += 1
        for s in pat.split("|"):
            state_freq[c][s] += frac
        if pat == CANONICAL:
            hier_freq[c] += frac

# ── Colour norm ───────────────────────────────────────────────────────────────
try:
    from matplotlib.colors import AsinhNorm
    def make_norm(): return AsinhNorm(linear_width=0.05, vmin=0, vmax=1)
except ImportError:
    def make_norm(): return mcolors.Normalize(vmin=0, vmax=1)

def add_seps(ax, lw=1.5):
    for sep in GROUP_SEPS:
        ax.axvline(sep, color="black", linewidth=lw, zorder=3)
        ax.axhline(sep, color="black", linewidth=lw, zorder=3)

def build_mat(value_dict):
    mat  = np.full((16, 16), np.nan)
    mask = np.zeros((16, 16), dtype=bool)
    for c, vec in idx_to_vec.items():
        ri = bwd_combos.index(bwd_bits(vec))
        ci = fwd_combos.index(fwd_bits(vec))
        if c not in has_results:
            mask[ri, ci] = True
        else:
            mat[ri, ci] = value_dict.get(c, 0.0)
    return mat, mask

# ── Top 12 states by mean frequency (excluding 0000) ─────────────────────────
state_means = {s: np.mean([state_freq[c].get(s, 0.0) for c in range(1, 257)])
               for s in ALL_STATES if s != "0000"}
top_states = sorted(state_means, key=lambda s: -state_means[s])[:12]

# ── Top 8 patterns for stacked bar ───────────────────────────────────────────
pat_totals = defaultdict(float)
for c in range(1, 257):
    for pat, frac in pat_freq[c].items():
        pat_totals[pat] += frac
top_pats = sorted(pat_totals, key=lambda p: -pat_totals[p])[:8]

def pat_to_label(pat):
    return "{" + ",".join(STATE_LABELS.get(s, s) for s in pat.split("|")) + "}"

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 24))
outer = gridspec.GridSpec(3, 1, figure=fig,
                          height_ratios=[5, 3.5, 3.5],
                          hspace=0.45)

# ── Row 0: hierarchy heatmap (A) + diversity heatmap (B) ─────────────────────
row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.35)
axA = fig.add_subplot(row0[0])
axB = fig.add_subplot(row0[1])

# Panel A — canonical hierarchy
mat_h, mask_h = build_mat(hier_freq)
mmat_h = np.ma.array(mat_h, mask=mask_h)
cmap_r = plt.cm.YlOrRd.copy(); cmap_r.set_bad("lightgrey")
imA = axA.imshow(mmat_h, cmap=cmap_r, norm=make_norm(), aspect="auto", origin="lower")
for ri in range(16):
    for ci in range(16):
        if not mask_h[ri, ci] and not np.isnan(mat_h[ri, ci]):
            v = mat_h[ri, ci]
            axA.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                     fontsize=4, color="white" if v > 0.6 else "black")
axA.set_xticks(range(16)); axA.set_xticklabels(fwd_labels, rotation=90, fontsize=5)
axA.set_yticks(range(16)); axA.set_yticklabels(bwd_labels, fontsize=5)
axA.set_xticks(np.arange(-0.5,16,1), minor=True)
axA.set_yticks(np.arange(-0.5,16,1), minor=True)
axA.grid(which="minor", color="white", linewidth=0.4); axA.tick_params(which="minor", bottom=False, left=False)
add_seps(axA)
axA.set_xlabel("FM->TB forward edges", fontsize=9, labelpad=6)
axA.set_ylabel("TB->FM backward edges", fontsize=9, labelpad=6)
axA.set_title("A  |  Canonical hierarchy {0000,1000,1100,1111}\n(arcsinh scale)", fontsize=10, loc="left")
cb = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
cb.set_ticks([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
cb.set_ticklabels(["0","0.05","0.1","0.2","0.4","0.6","0.8","1.0"], fontsize=7)

# Panel B — diversity
mat_d = np.full((16, 16), np.nan); mask_d = np.zeros((16,16), dtype=bool)
for c, vec in idx_to_vec.items():
    ri = bwd_combos.index(bwd_bits(vec)); ci = fwd_combos.index(fwd_bits(vec))
    if c not in has_results: mask_d[ri,ci] = True
    else: mat_d[ri,ci] = n_distinct[c]
mmat_d = np.ma.array(mat_d, mask=mask_d)
cmap_v = plt.cm.viridis.copy(); cmap_v.set_bad("lightgrey")
imB = axB.imshow(mmat_d, cmap=cmap_v, aspect="auto", origin="lower",
                 vmin=1, vmax=np.nanmax(mat_d))
for ri in range(16):
    for ci in range(16):
        if not mask_d[ri,ci] and not np.isnan(mat_d[ri,ci]):
            v = mat_d[ri,ci]
            axB.text(ci, ri, int(v), ha="center", va="center",
                     fontsize=4, color="white" if v > 15 else "black")
axB.set_xticks(range(16)); axB.set_xticklabels(fwd_labels, rotation=90, fontsize=5)
axB.set_yticks(range(16)); axB.set_yticklabels(bwd_labels, fontsize=5)
axB.set_xticks(np.arange(-0.5,16,1), minor=True)
axB.set_yticks(np.arange(-0.5,16,1), minor=True)
axB.grid(which="minor", color="white", linewidth=0.4); axB.tick_params(which="minor", bottom=False, left=False)
add_seps(axB)
axB.set_xlabel("FM->TB forward edges", fontsize=9, labelpad=6)
axB.set_ylabel("TB->FM backward edges", fontsize=9, labelpad=6)
axB.set_title("B  |  Number of distinct attractor types", fontsize=10, loc="left")
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04, label="n solution types")

# ── Row 1: backward-edge collapse bar (C) ────────────────────────────────────
axD = fig.add_subplot(outer[1])

bwd_pat_mean = np.zeros((16, len(top_pats) + 1))
for bi, bb in enumerate(bwd_combos):
    cs = [c for c, vec in idx_to_vec.items() if bwd_bits(vec) == bb and c in n_distinct]
    if not cs: continue
    for pi, pat in enumerate(top_pats):
        bwd_pat_mean[bi, pi] = np.mean([pat_freq[c].get(pat, 0.0) for c in cs])
    bwd_pat_mean[bi, -1] = max(0.0, 1.0 - bwd_pat_mean[bi, :-1].sum())

pat_labels = [pat_to_label(p) for p in top_pats] + ["other"]
colors = plt.cm.tab10(np.linspace(0, 0.9, len(pat_labels)))
bottoms = np.zeros(16)
for pi, (label, color) in enumerate(zip(pat_labels, colors)):
    axD.bar(range(16), bwd_pat_mean[:, pi], bottom=bottoms,
            label=label, color=color, width=0.8, edgecolor="white", linewidth=0.3)
    bottoms += bwd_pat_mean[:, pi]

for sep in GROUP_SEPS:
    axD.axvline(sep, color="black", linewidth=1.5, zorder=3)

axD.set_xticks(range(16))
axD.set_xticklabels(bwd_labels, rotation=45, ha="right", fontsize=7)
axD.set_xlabel("TB->FM backward edges  (sorted by edge count)", fontsize=10)
axD.set_ylabel("Fraction of samples", fontsize=10)
axD.set_title("C  |  Solution-type distribution by backward-edge combination\n"
              "(each bar = average over all 16 forward-edge variants)", fontsize=10, loc="left")
axD.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, framealpha=0.9)
axD.set_ylim(0, 1); axD.set_xlim(-0.5, 15.5)

# ── Row 2: hierarchy freq vs feedback fraction (D) ───────────────────────────
axE = fig.add_subplot(outer[2])

frac_data = []   # (feedback_fraction, hier_freq, n_bwd, n_fwd, n_total)
for c, vec in idx_to_vec.items():
    if c not in has_results:
        continue
    n_bwd = sum(bwd_bits(vec))
    n_fwd = sum(fwd_bits(vec))
    n_total = n_bwd + n_fwd
    if n_total == 0:
        continue
    frac = n_bwd / n_total
    frac_data.append((frac, hier_freq.get(c, 0.0), n_bwd, n_fwd, n_total))

fracs   = np.array([d[0] for d in frac_data])
hfreqs  = np.array([d[1] for d in frac_data])
ntotals = np.array([d[4] for d in frac_data])

# Scatter — colour by total edge count
cmap_sc = plt.cm.plasma
norm_sc = mcolors.Normalize(vmin=1, vmax=8)
sc = axE.scatter(fracs, hfreqs, c=ntotals, cmap=cmap_sc, norm=norm_sc,
                 s=60, alpha=0.7, edgecolors="grey", linewidths=0.4, zorder=3)

# Mean at each unique fraction
from collections import defaultdict as dd
bins = dd(list)
for f, h in zip(fracs, hfreqs):
    bins[round(f, 6)].append(h)
ux = sorted(bins)
uy = [np.mean(bins[x]) for x in ux]
axE.plot(ux, uy, "k-o", lw=2, ms=6, zorder=4, label="mean")

axE.set_xlabel("Feedback fraction  =  n(TB→FM) / [n(TB→FM) + n(FM→TB)]", fontsize=11)
axE.set_ylabel("Canonical hierarchy frequency\n{0000, 1000, 1100, 1111}", fontsize=10)
axE.set_title("D  |  Hierarchy frequency vs. feedback fraction\n"
              "(each point = one circuit; line = mean at each fraction value)", fontsize=10, loc="left")
axE.set_xlim(-0.05, 1.05)
axE.set_ylim(-0.02, 1.05)
axE.axvline(0.5, color="grey", lw=1, ls="--", alpha=0.5)
axE.legend(fontsize=9)
cb2 = fig.colorbar(sc, ax=axE, orientation="horizontal", fraction=0.04, pad=0.18, aspect=40)
cb2.set_label("Total cross-edges", fontsize=9)
cb2.set_ticks(range(1, 9))

plt.savefig("combined_figure.png", dpi=150, bbox_inches="tight")
print("Saved combined_figure.png")
