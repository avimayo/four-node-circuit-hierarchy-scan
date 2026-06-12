#!/usr/bin/env python3
"""
Forward-edge analysis across 256 circuits.

Panel A: Heatmap — mean stable-state count over the 5×5 (n_fwd × n_bwd) grid.
Panel B: Scatter — n_stable vs forward fraction, one point per circuit,
         colored by total edge count.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import product
from pathlib import Path

BASE = Path("/Users/Avimayo/circuit_hpc")
OUT  = BASE / "fig_3d_forward_analysis.png"

FWD_EDGES = ["F->T", "M->T", "F->B", "M->B"]   # FM→TB
BWD_EDGES = ["T->F", "T->M", "B->F", "B->M"]   # TB→FM

# sorted (by sum, then lex) combos for each group — same ordering as heatmap
fwd_combos = sorted(product([0, 1], repeat=4), key=sum)
bwd_combos = sorted(product([0, 1], repeat=4), key=sum)
fwd_xi = {c: i for i, c in enumerate(fwd_combos)}
bwd_yi = {c: i for i, c in enumerate(bwd_combos)}

GROUP_SEPS = [0.5, 4.5, 10.5, 14.5]


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_edges(label: str):
    """Return set of edge strings from a label like '{F->T,B->M}'."""
    inner = label.strip("{}").strip()
    if not inner:
        return set()
    return set(re.split(r",\s*", inner))


def circuit_combos(edges: set):
    fwd = tuple(1 if e in edges else 0 for e in FWD_EDGES)
    bwd = tuple(1 if e in edges else 0 for e in BWD_EDGES)
    return fwd, bwd


# ── load & annotate ──────────────────────────────────────────────────────────

df = pd.read_csv(BASE / "phenotype_table.csv")
sol_cols = [c for c in df.columns if c.startswith("solution_")]

records = []
for _, row in df.iterrows():
    edges = parse_edges(str(row["added_edges"]))
    fwd_c, bwd_c = circuit_combos(edges)
    n_fwd = sum(fwd_c)
    n_bwd = sum(bwd_c)
    n_total = n_fwd + n_bwd
    fwd_frac = n_fwd / n_total if n_total > 0 else np.nan

    n_stable = int(df[sol_cols].loc[row.name].notna().sum())

    records.append(dict(
        circuit_index=row["circuit_index"],
        n_fwd=n_fwd, n_bwd=n_bwd, n_total=n_total,
        fwd_frac=fwd_frac, n_stable=n_stable,
        xi=fwd_xi[fwd_c], yi=bwd_yi[bwd_c],
    ))

data = pd.DataFrame(records)

# ── colormap helpers ─────────────────────────────────────────────────────────

cmap_total = cm.viridis
norm_total = mcolors.Normalize(vmin=data.n_total.min(), vmax=data.n_total.max())

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Stable-state diversity vs edge-type composition across 256 circuits",
             fontsize=13, fontweight="bold")

# ── Panel A: heatmap — mean n_stable over (n_fwd, n_bwd) grid ────────────────

grouped = (data.groupby(["n_fwd", "n_bwd"])
               .agg(mean_stable=("n_stable", "mean"),
                    sem_stable=("n_stable", "sem"),
                    count=("n_stable", "size"))
               .reset_index())

# Build 5×5 matrix (row = n_bwd 0..4, col = n_fwd 0..4)
heat   = np.full((5, 5), np.nan)
counts = np.zeros((5, 5), dtype=int)
for _, r in grouped.iterrows():
    nf, nb = int(r.n_fwd), int(r.n_bwd)
    heat[nb, nf]   = r.mean_stable
    counts[nb, nf] = int(r["count"])

im = ax2.imshow(heat, aspect="auto", origin="lower",
                cmap="YlOrRd", vmin=0, vmax=np.nanmax(heat))
ax2.set_xticks(range(5))
ax2.set_xticklabels([str(i) for i in range(5)], fontsize=11)
ax2.set_yticks(range(5))
ax2.set_yticklabels([str(i) for i in range(5)], fontsize=11)
ax2.set_xlabel("# Forward edges  (FM→TB)", fontsize=12, labelpad=8)
ax2.set_ylabel("# Backward edges  (TB→FM)", fontsize=12, labelpad=8)
ax2.set_title("A.  Mean stable states\nvs edge-type count", fontsize=12, pad=10)

# Annotate each cell: mean value + circuit count
vmax = np.nanmax(heat)
for nb in range(5):
    for nf in range(5):
        if not np.isnan(heat[nb, nf]):
            txt_color = "white" if heat[nb, nf] > 0.65 * vmax else "black"
            ax2.text(nf, nb, f"{heat[nb, nf]:.2f}\n(n={counts[nb, nf]})",
                     ha="center", va="center", fontsize=9,
                     color=txt_color, fontweight="bold")

# draw grid lines
for i in range(6):
    ax2.axhline(i - 0.5, color="white", lw=0.8)
    ax2.axvline(i - 0.5, color="white", lw=0.8)

fig.colorbar(im, ax=ax2, label="Mean # stable states", fraction=0.046, pad=0.04)

# ── Panel B: scatter n_stable vs fwd_fraction ────────────────────────────────

ax3 = ax3

# add jitter on x to reveal overplotted points
rng = np.random.default_rng(42)
mask = data.fwd_frac.notna()
jx   = rng.uniform(-0.012, 0.012, mask.sum())

sc = ax3.scatter(
    data.loc[mask, "fwd_frac"] + jx,
    data.loc[mask, "n_stable"],
    c=data.loc[mask, "n_total"],
    cmap=cmap_total,
    norm=norm_total,
    alpha=0.65, s=28, edgecolors="none",
)
cb_c = fig.colorbar(sc, ax=ax3, label="Total edges")

# overlay means ± SEM per fwd_fraction bucket
grp = data[mask].groupby("fwd_frac")["n_stable"]
means = grp.mean()
sems  = grp.sem().fillna(0)
ax3.errorbar(means.index, means.values, yerr=sems.values,
             fmt="o", color="black", ms=6, capsize=4, lw=1.5,
             zorder=5, label="mean ± SEM")

ax3.set_xlabel("Forward fraction  (n_fwd / n_total)")
ax3.set_ylabel("# Distinct stable states")
ax3.set_title("B.  Stable-state diversity\nvs forward fraction")
ax3.legend(loc="upper left", fontsize=9)
ax3.set_xlim(-0.05, 1.05)

plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()
