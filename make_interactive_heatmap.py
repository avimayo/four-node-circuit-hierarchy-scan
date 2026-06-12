#!/usr/bin/env python3
"""
Interactive heatmap — fig_interactive_heatmap.html

Dropdown selects what is displayed:
  • Hierarchy frequency (4 strictness levels)
  • Number of distinct attractors
  • Attractor Shannon entropy
  • All-active (F+M+T+B) state frequency
  • Dominant-pattern share
Hovering any cell shows the full attractor-pattern breakdown.
"""

import csv
import numpy as np
from itertools import product
from collections import defaultdict
from pathlib import Path

try:
    import plotly.graph_objects as go
except ImportError:
    raise SystemExit("plotly not found — install with:  pip install plotly")

BASE = Path(__file__).parent
N_SAMPLES = 10000

FWD_ENAMES = ["F→T", "M→T", "F→B", "M→B"]
BWD_ENAMES = ["T→F", "T→M", "B→F", "B→M"]

all_vecs   = sorted(product([0, 1], repeat=8), key=sum)
idx_to_vec = {i + 1: v for i, v in enumerate(all_vecs)}
fwd_combos = sorted(product([0, 1], repeat=4), key=sum)
bwd_combos = sorted(product([0, 1], repeat=4), key=sum)
ALL_STATES = [format(i, "04b") for i in range(16)]

def fwd_bits(vec): return (vec[0], vec[2], vec[4], vec[6])
def bwd_bits(vec): return (vec[1], vec[3], vec[5], vec[7])

def combo_label(bits, names):
    active = [n for n, b in zip(names, bits) if b]
    return ", ".join(active) if active else "∅"

fwd_labels = [combo_label(b, FWD_ENAMES) for b in fwd_combos]
bwd_labels = [combo_label(b, BWD_ENAMES) for b in bwd_combos]

STATE_LABELS = {
    "0000": "∅",       "0001": "B",       "0010": "T",       "0011": "T+B",
    "0100": "M",       "0101": "M+B",     "0110": "M+T",     "0111": "M+T+B",
    "1000": "F",       "1001": "F+B",     "1010": "F+T",     "1011": "F+T+B",
    "1100": "F+M",     "1101": "F+M+B",   "1110": "F+M+T",   "1111": "F+M+T+B",
}

def pat_label(pat):
    return " + ".join(STATE_LABELS.get(s, s) for s in sorted(pat.split("|")))

def freq_bar(frac, width=10):
    n = round(frac * width)
    return "█" * n + "░" * (width - n)

# ── Read data ──────────────────────────────────────────────────────────────────
pat_freq   = defaultdict(lambda: defaultdict(float))
n_distinct = defaultdict(int)

with open(BASE / "final_results.csv") as f:
    for row in csv.DictReader(f):
        c   = int(row["circuit_index"])
        pat = row["phenotype_pattern"]
        cnt = int(row["count"])
        pat_freq[c][pat] += cnt / N_SAMPLES
        n_distinct[c] += 1

# ── Matrix builders ────────────────────────────────────────────────────────────
def _cell_iter():
    for c, vec in idx_to_vec.items():
        if c in pat_freq:
            yield c, bwd_combos.index(bwd_bits(vec)), fwd_combos.index(fwd_bits(vec))

def hier_matrix(check_fn):
    m = np.full((16, 16), np.nan)
    for c, ri, ci in _cell_iter():
        m[ri, ci] = sum(frac for pat, frac in pat_freq[c].items()
                        if check_fn(frozenset(pat.split("|"))))
    return m

def ndistinct_matrix():
    m = np.full((16, 16), np.nan)
    for c, ri, ci in _cell_iter():
        m[ri, ci] = n_distinct[c]
    return m

def entropy_matrix():
    m = np.full((16, 16), np.nan)
    for c, ri, ci in _cell_iter():
        probs = np.array([v for v in pat_freq[c].values() if v > 0])
        m[ri, ci] = float(-np.sum(probs * np.log2(probs)))
    return m

def state_freq_matrix(state):
    """Fraction of samples where `state` appears in the attractor set."""
    m = np.full((16, 16), np.nan)
    for c, ri, ci in _cell_iter():
        m[ri, ci] = sum(frac for pat, frac in pat_freq[c].items()
                        if state in pat.split("|"))
    return m

def dominant_share_matrix():
    m = np.full((16, 16), np.nan)
    for c, ri, ci in _cell_iter():
        m[ri, ci] = max(pat_freq[c].values())
    return m

# ── Hierarchy definitions ──────────────────────────────────────────────────────
_HIER_CANONICAL = frozenset({"1000", "1100", "1111"})
_HIER_EXACT_A   = frozenset({"0000", "1000", "1100", "1111"})
_HIER_EXACT_B   = frozenset({"0000", "0011", "1000", "1100", "1111"})
_ONE_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 1)
_TWO_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 2)

HIERARCHIES = [
    ("Canonical  F→F+M→all  (strict)",
     "Attractor set is exactly {∅,F,F+M,all} or {∅,T+B,F,F+M,all}",
     lambda s: _HIER_CANONICAL <= s and (s == _HIER_EXACT_A or s == _HIER_EXACT_B)),
    ("Contains  F→F+M→all  (relaxed)",
     "Attractor set includes {F, F+M, all} — extra states allowed",
     lambda s: _HIER_CANONICAL <= s),
    ("Any  1-node → 2-node → all",
     "Attractor set has ≥1 single-node state, ≥1 two-node state, and the all-active state",
     lambda s: "1111" in s and bool(_ONE_NODE & s) and bool(_TWO_NODE & s)),
    ("Any  single-node → all-active",
     "Attractor set has at least one single-node state and the all-active state",
     lambda s: "1111" in s and bool(_ONE_NODE & s)),
]

# ── Build all views ────────────────────────────────────────────────────────────
# Each view: label, description, matrix, colorscale, zmin, zmax, ann_fmt, colorbar_title
VIEWS = []

for name, desc, fn in HIERARCHIES:
    VIEWS.append(dict(
        label=f"Hierarchy: {name}",
        desc=desc,
        matrix=hier_matrix(fn),
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        ann_fmt=".0%",
        colorbar_title="Hierarchy<br>frequency",
    ))

nd_mat = ndistinct_matrix()
VIEWS.append(dict(
    label="# distinct attractors",
    desc="Number of different attractor-pattern types found across 10,000 parameter samples",
    matrix=nd_mat,
    colorscale="Viridis",
    zmin=1, zmax=int(np.nanmax(nd_mat)),
    ann_fmt=".0f",
    colorbar_title="# distinct<br>attractors",
))

ent_mat = entropy_matrix()
VIEWS.append(dict(
    label="Attractor entropy (bits)",
    desc="Shannon entropy of the attractor-pattern distribution — 0 = monostable, high = highly diverse",
    matrix=ent_mat,
    colorscale="Plasma",
    zmin=0, zmax=float(np.nanmax(ent_mat)),
    ann_fmt=".1f",
    colorbar_title="Entropy<br>(bits)",
))

aa_mat = state_freq_matrix("1111")
VIEWS.append(dict(
    label="All-active (F+M+T+B) frequency",
    desc="Fraction of samples where the fully-active state is a stable attractor",
    matrix=aa_mat,
    colorscale="YlOrRd",
    zmin=0, zmax=1,
    ann_fmt=".0%",
    colorbar_title="F+M+T+B<br>frequency",
))

f_only_mat = state_freq_matrix("1000")
VIEWS.append(dict(
    label="F-only state frequency",
    desc="Fraction of samples where F alone (1000) is a stable attractor",
    matrix=f_only_mat,
    colorscale="YlOrRd",
    zmin=0, zmax=1,
    ann_fmt=".0%",
    colorbar_title="F-only<br>frequency",
))

dom_mat = dominant_share_matrix()
VIEWS.append(dict(
    label="Dominant attractor share",
    desc="Fraction of samples in the single most-frequent pattern — near 1 = near-monostable",
    matrix=dom_mat,
    colorscale="YlGnBu",
    zmin=0, zmax=1,
    ann_fmt=".0%",
    colorbar_title="Dominant<br>share",
))

# ── Hover text (shared across all views) ───────────────────────────────────────
def freq_bar(frac, width=10):
    n = round(frac * width)
    return "█" * n + "░" * (width - n)

hover_text = np.full((16, 16), "", dtype=object)

for c, ri, ci in _cell_iter():
    fwd_str = combo_label(fwd_combos[ci], FWD_ENAMES)
    bwd_str = combo_label(bwd_combos[ri], BWD_ENAMES)
    pats    = sorted(pat_freq[c].items(), key=lambda x: -x[1])

    hier_lines = []
    for name, _, fn in HIERARCHIES:
        freq  = sum(frac for pat, frac in pat_freq[c].items()
                    if fn(frozenset(pat.split("|"))))
        short = name.split("(")[0].strip()
        hier_lines.append(f"  {freq_bar(freq, 6)}  {freq:4.0%}  {short}")

    pat_lines = [
        f"  {frac:5.1%}  {freq_bar(frac)}  {pat_label(pat)}"
        for pat, frac in pats
    ]

    probs = np.array(list(pat_freq[c].values()))
    entropy = float(-np.sum(probs * np.log2(np.where(probs > 0, probs, 1))))

    lines = [
        f"<b>Circuit {c}</b>  —  {n_distinct[c]} attractor type{'s' if n_distinct[c]>1 else ''}  |  "
        f"H = {entropy:.2f} bits",
        f"<span style='color:#666'>Fwd: {fwd_str}   Bwd: {bwd_str}</span>",
        "─" * 38,
        "<b>Hierarchy frequencies</b>",
        *hier_lines,
        "─" * 38,
        "<b>All attractor patterns</b>",
        *pat_lines,
    ]
    hover_text[ri, ci] = "<br>".join(lines)

# ── Annotation helper ──────────────────────────────────────────────────────────
def make_annotations(view):
    z, zmin, zmax, fmt = view["matrix"], view["zmin"], view["zmax"], view["ann_fmt"]
    span = zmax - zmin if zmax > zmin else 1
    anns = []
    for ri in range(16):
        for ci in range(16):
            v = z[ri, ci]
            if not np.isnan(v):
                norm = (v - zmin) / span
                anns.append(dict(
                    x=ci, y=ri,
                    text=format(v, fmt),
                    showarrow=False, xref="x", yref="y",
                    font=dict(size=7, color="white" if norm > 0.55 else "black"),
                ))
    return anns

# ── Build Plotly figure ────────────────────────────────────────────────────────
fig = go.Figure()

for i, view in enumerate(VIEWS):
    fig.add_trace(go.Heatmap(
        z=view["matrix"],
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=view["colorscale"],
        zmin=view["zmin"], zmax=view["zmax"],
        colorbar=dict(
            title=dict(text=view["colorbar_title"], side="right"),
            tickformat=view["ann_fmt"] if "%" in view["ann_fmt"] else "",
            thickness=16, len=0.85,
        ),
        visible=(i == 0),
        name=view["label"],
    ))

# Group separator lines
SEPS = [0.5, 4.5, 10.5, 14.5]
for sep in SEPS:
    fig.add_shape(type="line",
                  x0=sep, x1=sep, y0=-0.5, y1=15.5,
                  line=dict(color="black", width=3.0))
    fig.add_shape(type="line",
                  x0=-0.5, x1=15.5, y0=sep, y1=sep,
                  line=dict(color="black", width=3.0))

# Dropdown buttons
buttons = []
for i, view in enumerate(VIEWS):
    visibility = [j == i for j in range(len(VIEWS))]
    buttons.append(dict(
        label=view["label"],
        method="update",
        args=[
            {"visible": visibility},
            {
                "title.text": (
                    f"{view['label']}"
                    f"<br><sup>{view['desc']}"
                    f"  —  hover for full breakdown</sup>"
                ),
                "annotations": make_annotations(view),
            },
        ],
    ))

fig.update_layout(
    annotations=make_annotations(VIEWS[0]),
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.0, xanchor="left",
        y=1.18, yanchor="top",
        bgcolor="white",
        bordercolor="#aaa",
        font=dict(size=11),
    )],
    title=dict(
        text=(
            f"{VIEWS[0]['label']}"
            f"<br><sup>{VIEWS[0]['desc']}"
            f"  —  hover for full breakdown</sup>"
        ),
        font=dict(size=13),
        x=0.5, xanchor="center",
    ),
    width=980, height=920,
    plot_bgcolor="black",
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#aaa",
        font=dict(size=11, family="monospace"),
    ),
    margin=dict(l=160, r=80, t=130, b=160),
)

fig.update_xaxes(
    tickmode="array", tickvals=list(range(16)), ticktext=fwd_labels,
    tickangle=90, tickfont=dict(size=8),
    title=dict(text="FM→TB forward edges", font=dict(size=11)),
)
fig.update_yaxes(
    tickmode="array", tickvals=list(range(16)), ticktext=bwd_labels,
    tickfont=dict(size=8),
    title=dict(text="TB→FM backward edges", font=dict(size=11)),
)

out = BASE / "fig_interactive_heatmap.html"
fig.write_html(str(out), include_plotlyjs="cdn")
print(f"Saved {out}  —  open in any browser")
