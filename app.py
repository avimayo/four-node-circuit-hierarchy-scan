"""
Streamlit app: Four-node circuit attractor landscape
Repo: avimayo/four-node-circuit-hierarchy-scan
"""

import csv
import io
import re
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Circuit Attractor Landscape",
    page_icon="🔬",
    layout="wide",
)

BASE = Path(__file__).parent
N_SAMPLES = 10000

# ── Static lookup tables ───────────────────────────────────────────────────────
FWD_ENAMES = ["F→T", "M→T", "F→B", "M→B"]
BWD_ENAMES = ["T→F", "T→M", "B→F", "B→M"]
FWD_ENAMES_ASCII = ["F->T", "M->T", "F->B", "M->B"]
BWD_ENAMES_ASCII = ["T->F", "T->M", "B->F", "B->M"]

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

SEPS  = [0.5, 4.5, 10.5, 14.5]
TAB10 = px.colors.qualitative.Plotly

def pat_label(pat):
    return " + ".join(STATE_LABELS.get(s, s) for s in sorted(pat.split("|")))

# ── State-pattern icon helpers (bar legend) ────────────────────────────────────
# Two-colour scheme: yellow = node ON, teal = node OFF
_COL_ON  = "#FDD835"   # Material Yellow 600
_COL_OFF = "#00897B"   # Material Teal 600
_NODES   = ["F", "M", "T", "B"]

def _state_row_svg(state, cell=14):
    parts = []
    for ni in range(4):
        fill = _COL_ON if state[ni] == "1" else _COL_OFF
        parts.append(f'<rect x="{ni*cell}" y="0" width="{cell}" height="{cell}" '
                     f'fill="{fill}" stroke="white" stroke-width="0.5"/>')
    return "".join(parts)

def pat_icon_html(pat_str, swatch_color, cell=14):
    """State grid (F/M/T/B columns) + column letter labels + swatch bar at bottom."""
    LBL_H  = 12   # pixels for F M T B row
    SW_H   = 7    # pixels for the colour swatch
    GAP    = 2
    w = 4 * cell

    # column letter labels (F / M / T / B in node colours)
    col_labels = "".join(
        f'<text x="{ni*cell + cell//2}" y="{LBL_H - 1}" '
        f'text-anchor="middle" font-size="9" fill="{_NC_HEX[n]}" '
        f'font-weight="bold" font-family="sans-serif">{n}</text>'
        for ni, n in enumerate(_NODES)
    )

    if pat_str == "other":
        total_h = cell + GAP + LBL_H + GAP + SW_H
        return (f'<svg width="{w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect x="0" y="0" width="{w}" height="{cell}" fill="#555" rx="2"/>'
                f'<text x="{w//2}" y="{cell//2+4}" text-anchor="middle" '
                f'font-size="10" fill="#eee" font-family="monospace">other</text>'
                f'<g transform="translate(0,{cell+GAP})">{col_labels}</g>'
                f'<rect x="0" y="{cell+GAP+LBL_H+GAP}" width="{w}" height="{SW_H}" '
                f'fill="{swatch_color}" rx="1"/></svg>')

    states = sorted(pat_str.split("|"))
    N = len(states)
    grid_h = N * cell
    total_h = grid_h + GAP + LBL_H + GAP + SW_H
    rows = "".join(
        f'<g transform="translate(0,{si*cell})">{_state_row_svg(s, cell)}</g>'
        for si, s in enumerate(states)
    )
    swatch = (f'<rect x="0" y="{grid_h+GAP+LBL_H+GAP}" '
              f'width="{w}" height="{SW_H}" fill="{swatch_color}" rx="1"/>')
    return (f'<svg width="{w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">'
            f'{rows}'
            f'<g transform="translate(0,{grid_h+GAP})">{col_labels}</g>'
            f'{swatch}</svg>')

def build_bar_legend_html(top_pats, colors):
    """Returns icons table with active/off key below — no outer wrapper."""
    key_svg = (
        f'<svg width="140" height="18" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="1" width="15" height="15" fill="{_COL_ON}" rx="2"/>'
        f'<text x="19" y="13" font-size="12" fill="currentColor" font-family="sans-serif">active</text>'
        f'<rect x="76" y="1" width="15" height="15" fill="{_COL_OFF}" rx="2"/>'
        f'<text x="95" y="13" font-size="12" fill="currentColor" font-family="sans-serif">off</text>'
        f'</svg>'
    )
    all_pats   = list(top_pats) + ["other"]
    all_colors = list(colors) + ["#AAAAAA"]
    items = []
    for pat, col in zip(all_pats, all_colors):
        svg = pat_icon_html(pat, col)
        lbl = pat_label(pat) if pat != "other" else "other"
        items.append(
            f'<td style="text-align:center;vertical-align:top;padding:4px 10px;">'
            f'{svg}'
            f'<div style="font-size:11px;color:inherit;max-width:80px;'
            f'word-wrap:break-word;margin-top:4px;">{lbl}</div></td>'
        )
    return (
        f'<table style="border-collapse:collapse;">'
        f'<tr>' + "".join(items) + '</tr></table>'
        f'<div style="margin-top:8px;">{key_svg}'
        f'  <span style="font-size:12px;color:inherit;vertical-align:middle;">'
        f'  Columns: F · M · T · B</span></div>'
    )

# ── Mini circuit PNG helpers (heatmap + bar tick labels) ───────────────────────
_NP_MPL = {"F": (0.20, 0.80), "M": (0.80, 0.80),
           "T": (0.80, 0.20), "B": (0.20, 0.20)}
_NC_MPL = {"F": "#4C72B0", "M": "#DD8452", "T": "#55A868", "B": "#C44E52"}
_NC_HEX = _NC_MPL   # alias used in SVG icon helpers
_FWD_SD = [
    ((0.20, 0.80), (0.80, 0.20)),   # F→T
    ((0.80, 0.80), (0.80, 0.20)),   # M→T
    ((0.20, 0.80), (0.20, 0.20)),   # F→B
    ((0.80, 0.80), (0.20, 0.20)),   # M→B
]
_BWD_SD = [
    ((0.80, 0.20), (0.20, 0.80)),   # T→F
    ((0.80, 0.20), (0.80, 0.80)),   # T→M
    ((0.20, 0.20), (0.20, 0.80)),   # B→F
    ((0.20, 0.20), (0.80, 0.80)),   # B→M
]

def _circuit_png_b64(bits, edge_sds, size_in=0.42, dpi=120, arrow_color="white"):
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0))
    for b, (src, dst) in zip(bits, edge_sds):
        if b:
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                        lw=0.7, mutation_scale=5))
    for node, (nx, ny) in _NP_MPL.items():
        ax.plot(nx, ny, "o", ms=3.2, color=_NC_MPL[node], zorder=5, markeredgewidth=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                transparent=True, dpi=dpi, pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

@st.cache_data
def build_logo_bytes(size_in=4.8, dpi=130):
    """Simplex logo: 4 cell-type nodes (letters only) with directed forward/backward arrows."""
    from matplotlib.patches import Circle

    BG = "#12122a"
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    # generous padding so nodes + arc arrows never clip
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Diamond layout: F=top, M=left, T=right, B=bottom
    pos = {"F": (0.50, 0.92), "M": (0.05, 0.42), "T": (0.95, 0.42), "B": (0.50, 0.02)}
    nc  = {"F": "#4C72B0", "M": "#DD8452", "T": "#55A868", "B": "#C44E52"}

    fwd = [("F","T"),("F","B"),("M","T"),("M","B")]
    bwd = [("T","F"),("T","M"),("B","F"),("B","M")]
    for src, dst in fwd + bwd:
        is_fwd = (src, dst) in fwd
        rad = 0.22 if is_fwd else -0.22
        col = "#6BAED6" if is_fwd else "#FD8D3C"
        ax.annotate("", xy=pos[dst], xytext=pos[src],
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                    mutation_scale=14,
                                    connectionstyle=f"arc3,rad={rad}", alpha=0.80),
                    zorder=2)

    R = 0.12
    for node, (nx, ny) in pos.items():
        ax.add_patch(Circle((nx, ny), R*1.6, color=nc[node], alpha=0.12, zorder=3))
        ax.add_patch(Circle((nx, ny), R,     color=nc[node], zorder=4))
        ax.add_patch(Circle((nx, ny), R, fill=False, edgecolor="white", lw=1.4,
                             alpha=0.80, zorder=5))
        ax.text(nx, ny+0.008, node, ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
                zorder=6, family="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data
def build_node_legend_bytes(dpi=110):
    """2×2 grid: node letter + colour + cell-type name, matching tick-icon layout."""
    # Grid: F(top-left) M(top-right) / B(bottom-left) T(bottom-right)
    grid = [["F","M"],["B","T"]]
    cnames = {"F":"Fibro-\nblasts","M":"Macro-\nphages","T":"T-cells","B":"B-cells"}
    BG = "#12122a"
    fig, axes = plt.subplots(2, 2, figsize=(1.9, 1.9), dpi=dpi)
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    for ri, row in enumerate(grid):
        for ci, node in enumerate(row):
            ax = axes[ri][ci]
            ax.set_facecolor(BG)
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.axis("off")
            from matplotlib.patches import Circle
            ax.add_patch(Circle((0.5, 0.62), 0.32, color=_NC_MPL[node], zorder=2))
            ax.text(0.5, 0.62, node, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white", zorder=3)
            ax.text(0.5, 0.16, cnames[node], ha="center", va="center",
                    fontsize=5.5, color=_NC_MPL[node], fontweight="bold",
                    zorder=3, multialignment="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data
def build_tick_images():
    fwd_imgs      = [_circuit_png_b64(bb, _FWD_SD, arrow_color="white")   for bb in fwd_combos]
    bwd_imgs      = [_circuit_png_b64(bb, _BWD_SD, arrow_color="white")   for bb in bwd_combos]
    bwd_imgs_dark = [_circuit_png_b64(bb, _BWD_SD, arrow_color="#333333") for bb in bwd_combos]
    return fwd_imgs, bwd_imgs, bwd_imgs_dark

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_attractor_data():
    pat_freq   = defaultdict(lambda: defaultdict(float))
    n_distinct = defaultdict(int)
    with open(BASE / "final_results.csv") as f:
        for row in csv.DictReader(f):
            c   = int(row["circuit_index"])
            pat = row["phenotype_pattern"]
            cnt = int(row["count"])
            pat_freq[c][pat] += cnt / N_SAMPLES
            n_distinct[c] += 1
    return dict(pat_freq), dict(n_distinct)

@st.cache_data
def load_phenotype_table():
    df = pd.read_csv(BASE / "phenotype_table.csv")
    sol_cols = [c for c in df.columns if c.startswith("solution_")]

    def parse_edges(label):
        inner = label.strip("{}").strip()
        return set(re.split(r",\s*", inner)) if inner else set()

    def circuit_combos(edges):
        fwd = tuple(1 if e in edges else 0 for e in FWD_ENAMES_ASCII)
        bwd = tuple(1 if e in edges else 0 for e in BWD_ENAMES_ASCII)
        return fwd, bwd

    records = []
    for _, row in df.iterrows():
        edges = parse_edges(str(row["added_edges"]))
        fwd_c, bwd_c = circuit_combos(edges)
        n_fwd   = sum(fwd_c); n_bwd = sum(bwd_c); n_total = n_fwd + n_bwd
        fwd_frac = n_fwd / n_total if n_total > 0 else float("nan")
        n_stable = int(df[sol_cols].loc[row.name].notna().sum())
        records.append(dict(
            circuit_index=row["circuit_index"],
            n_fwd=n_fwd, n_bwd=n_bwd, n_total=n_total,
            fwd_frac=fwd_frac, n_stable=n_stable,
        ))
    return pd.DataFrame(records)

pat_freq_raw, n_distinct = load_attractor_data()
pat_freq = defaultdict(lambda: defaultdict(float),
                       {k: defaultdict(float, v) for k, v in pat_freq_raw.items()})

def _cells():
    for c, vec in idx_to_vec.items():
        if c in pat_freq:
            yield c, bwd_combos.index(bwd_bits(vec)), fwd_combos.index(fwd_bits(vec))

# ── Matrix builders ────────────────────────────────────────────────────────────
@st.cache_data
def build_all_matrices():
    _HIER_CANONICAL = frozenset({"1000", "1100", "1111"})
    _HIER_EXACT_A   = frozenset({"0000", "1000", "1100", "1111"})
    _HIER_EXACT_B   = frozenset({"0000", "0011", "1000", "1100", "1111"})
    _HIER_4STEP     = frozenset({"0000", "1000", "1100", "1110", "1111"})
    _ONE_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 1)
    _TWO_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 2)

    def hier_mat(check):
        m = np.full((16, 16), np.nan)
        for c, ri, ci in _cells():
            m[ri, ci] = sum(frac for pat, frac in pat_freq[c].items()
                            if check(frozenset(pat.split("|"))))
        return m

    def state_mat(state):
        m = np.full((16, 16), np.nan)
        for c, ri, ci in _cells():
            m[ri, ci] = sum(frac for pat, frac in pat_freq[c].items()
                            if state in pat.split("|"))
        return m

    nd = np.full((16, 16), np.nan)
    ent = np.full((16, 16), np.nan)
    dom = np.full((16, 16), np.nan)
    for c, ri, ci in _cells():
        nd[ri, ci]  = n_distinct[c]
        probs = np.array([v for v in pat_freq[c].values() if v > 0])
        ent[ri, ci] = float(-np.sum(probs * np.log2(probs)))
        dom[ri, ci] = max(pat_freq[c].values())

    return {
        "hier_strict":  hier_mat(lambda s: _HIER_CANONICAL <= s and (
                            s == _HIER_EXACT_A or s == frozenset({"0000","0011","1000","1100","1111"}))),
        "hier_relaxed": hier_mat(lambda s: _HIER_CANONICAL <= s),
        "hier_4step":   hier_mat(lambda s: _HIER_4STEP <= s),
        "hier_1_2_all": hier_mat(lambda s: "1111" in s and bool(_ONE_NODE & s) and bool(_TWO_NODE & s)),
        "hier_1_all":   hier_mat(lambda s: "1111" in s and bool(_ONE_NODE & s)),
        "ndistinct":    nd,
        "entropy":      ent,
        "all_active":   state_mat("1111"),
        "f_only":       state_mat("1000"),
        "dominant":     dom,
    }

@st.cache_data
def build_hover():
    def bar(frac, w=10):
        return "█" * round(frac * w) + "░" * (w - round(frac * w))

    _HIER_CANONICAL = frozenset({"1000", "1100", "1111"})
    _HIER_EXACT_A   = frozenset({"0000", "1000", "1100", "1111"})
    _HIER_EXACT_B   = frozenset({"0000", "0011", "1000", "1100", "1111"})
    _HIER_4STEP     = frozenset({"0000", "1000", "1100", "1110", "1111"})
    _ONE_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 1)
    _TWO_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 2)

    fns = [
        ("Canonical F→F+M→all (strict)", lambda s: _HIER_CANONICAL <= s and (s == _HIER_EXACT_A or s == _HIER_EXACT_B)),
        ("Contains F→F+M→all",           lambda s: _HIER_CANONICAL <= s),
        ("∅→F→F+M→F+M+T→all",           lambda s: _HIER_4STEP <= s),
        ("Any 1→2→all",                  lambda s: "1111" in s and bool(_ONE_NODE & s) and bool(_TWO_NODE & s)),
        ("Any 1→all",                    lambda s: "1111" in s and bool(_ONE_NODE & s)),
    ]

    hover = np.full((16, 16), "", dtype=object)
    for c, ri, ci in _cells():
        fwd_str = combo_label(fwd_combos[ci], FWD_ENAMES)
        bwd_str = combo_label(bwd_combos[ri], BWD_ENAMES)
        pats    = sorted(pat_freq[c].items(), key=lambda x: -x[1])
        probs   = np.array(list(pat_freq[c].values()))
        entropy = float(-np.sum(probs * np.log2(np.where(probs > 0, probs, 1))))

        hier_freqs = [sum(frac for p, frac in pat_freq[c].items()
                         if fn(frozenset(p.split("|")))) for _, fn in fns]
        hier_lines = [f"  {bar(f, 6)}  {f:4.0%}  {name}"
                      for (name, _), f in zip(fns, hier_freqs)]
        if all(f >= 0.999 for f in hier_freqs):
            hier_lines.append(
                f"  100% because all {n_distinct[c]} attractor pattern"
                f"{'s' if n_distinct[c]>1 else ''} independently satisfy every criterion"
            )

        pat_lines = [f"  {frac:5.1%}  {bar(frac)}  {pat_label(pat)}" for pat, frac in pats]
        hover[ri, ci] = "<br>".join([
            f"<b>Circuit {c}</b>  —  {n_distinct[c]} attractor type{'s' if n_distinct[c]>1 else ''}"
            f"  |  H = {entropy:.2f} bits",
            f"<span style='color:#333'>Fwd: {fwd_str}   Bwd: {bwd_str}</span>",
            "─" * 38,
            "<b>Hierarchy frequencies</b>  "
            "<span style='color:#555'>(fraction of 100 k param samples)</span>",
            *hier_lines,
            "─" * 38,
            "<b>All attractor patterns</b>",
            *pat_lines,
        ])
    return hover

@st.cache_data
def build_tick_hover():
    def _stats(circuits):
        ent_vals, dom_vals, nd_vals = [], [], []
        for c in circuits:
            probs = np.array([v for v in pat_freq[c].values() if v > 0])
            ent_vals.append(-np.sum(probs * np.log2(probs)))
            dom_vals.append(max(pat_freq[c].values()))
            nd_vals.append(n_distinct[c])
        return np.mean(ent_vals), np.mean(dom_vals), np.mean(nd_vals)

    col_hover = []
    for ci, bb in enumerate(fwd_combos):
        circuits = [c for c, vec in idx_to_vec.items()
                    if fwd_bits(vec) == bb and c in pat_freq]
        lbl = combo_label(bb, FWD_ENAMES)
        if not circuits:
            col_hover.append(f"<b>Forward: {lbl}</b><br>No data")
            continue
        me, md, mn = _stats(circuits)
        col_hover.append("<br>".join([
            f"<b>Forward edges: {lbl}</b>",
            f"16 backward-edge variants",
            "─" * 26,
            f"Mean entropy:        {me:.2f} bits",
            f"Mean dominant share: {md:.0%}",
            f"Mean # attractors:   {mn:.1f}",
        ]))

    row_hover = []
    for ri, bb in enumerate(bwd_combos):
        circuits = [c for c, vec in idx_to_vec.items()
                    if bwd_bits(vec) == bb and c in pat_freq]
        lbl = combo_label(bb, BWD_ENAMES)
        if not circuits:
            row_hover.append(f"<b>Backward: {lbl}</b><br>No data")
            continue
        me, md, mn = _stats(circuits)
        row_hover.append("<br>".join([
            f"<b>Backward edges: {lbl}</b>",
            f"16 forward-edge variants",
            "─" * 26,
            f"Mean entropy:        {me:.2f} bits",
            f"Mean dominant share: {md:.0%}",
            f"Mean # attractors:   {mn:.1f}",
        ]))

    return col_hover, row_hover

mats       = build_all_matrices()
hover_text = build_hover()

# ── Heatmap view definitions ───────────────────────────────────────────────────
VIEWS = [
    dict(label="Hierarchy: Canonical F→F+M→all (strict)",
         desc="Attractor set is exactly {∅, F, F+M, all} or {∅, T+B, F, F+M, all}",
         key="hier_strict",   colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="Hierarchy<br>frequency"),
    dict(label="Hierarchy: Contains F→F+M→all (relaxed)",
         desc="Attractor set includes {F, F+M, all} — extra states allowed",
         key="hier_relaxed",  colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="Hierarchy<br>frequency"),
    dict(label="Hierarchy: ∅→F→F+M→F+M+T→all",
         desc="Attractor set includes {∅, F, F+M, F+M+T, all} — full 5-step cascade",
         key="hier_4step",    colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="Hierarchy<br>frequency"),
    dict(label="Hierarchy: Any 1-node → 2-node → all",
         desc="Attractor set has ≥1 single-node, ≥1 two-node, and the all-active state",
         key="hier_1_2_all",  colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="Hierarchy<br>frequency"),
    dict(label="Hierarchy: Any single-node → all-active",
         desc="Attractor set has at least one single-node state and the all-active state",
         key="hier_1_all",    colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="Hierarchy<br>frequency"),
    dict(label="# distinct attractors",
         desc="Number of different attractor-pattern types found across 10,000 parameter samples",
         key="ndistinct",     colorscale="Viridis",
         zmin=1, zmax=int(np.nanmax(mats["ndistinct"])),
         ann_fmt=".0f", colorbar_title="# distinct<br>attractors"),
    dict(label="Attractor entropy (bits)",
         desc="Shannon entropy of the attractor distribution — 0 = monostable, high = highly diverse",
         key="entropy",       colorscale="Plasma",
         zmin=0, zmax=float(np.nanmax(mats["entropy"])),
         ann_fmt=".1f", colorbar_title="Entropy<br>(bits)"),
    dict(label="All-active (F+M+T+B) frequency",
         desc="Fraction of samples where the fully-active state is a stable attractor",
         key="all_active",    colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="F+M+T+B<br>frequency"),
    dict(label="F-only state frequency",
         desc="Fraction of samples where F alone (1000) is a stable attractor",
         key="f_only",        colorscale="YlOrRd", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="F-only<br>frequency"),
    dict(label="Dominant attractor share",
         desc="Fraction of samples in the single most-frequent pattern — near 100% = near-monostable",
         key="dominant",      colorscale="YlGnBu", zmin=0, zmax=1,
         ann_fmt=".0%", colorbar_title="Dominant<br>share"),
]

# ── Heatmap figure ─────────────────────────────────────────────────────────────
def build_heatmap_figure(view):
    z, zmin, zmax = mats[view["key"]], view["zmin"], view["zmax"]
    span = zmax - zmin if zmax > zmin else 1
    fmt  = view["ann_fmt"]
    annotations = [
        dict(x=ci, y=ri, text=format(z[ri, ci], fmt),
             showarrow=False, xref="x", yref="y",
             font=dict(size=9, color="white" if (z[ri,ci]-zmin)/span > 0.55 else "black"))
        for ri in range(16) for ci in range(16) if not np.isnan(z[ri, ci])
    ]

    fwd_imgs, bwd_imgs, _ = build_tick_images()
    col_hover, row_hover = build_tick_hover()

    fig = go.Figure(go.Heatmap(
        z=z, text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=view["colorscale"], zmin=zmin, zmax=zmax,
        colorbar=dict(title=dict(text=view["colorbar_title"], side="right"),
                      tickformat=fmt if "%" in fmt else "",
                      thickness=16, len=0.85),
    ))
    for sep in SEPS:
        fig.add_shape(type="line", x0=sep, x1=sep, y0=-0.5, y1=15.5,
                      line=dict(color="black", width=3))
        fig.add_shape(type="line", x0=-0.5, x1=15.5, y0=sep, y1=sep,
                      line=dict(color="black", width=3))

    IMG_SZ = 0.82
    for ci, img in enumerate(fwd_imgs):
        fig.add_layout_image(dict(source=img, layer="above",
                                  xref="x", x=ci, yref="y", y=-1.0,
                                  xanchor="center", yanchor="middle",
                                  sizex=IMG_SZ, sizey=IMG_SZ))
    for ri, img in enumerate(bwd_imgs):
        fig.add_layout_image(dict(source=img, layer="above",
                                  xref="x", x=-1.0, yref="y", y=ri,
                                  xanchor="center", yanchor="middle",
                                  sizex=IMG_SZ, sizey=IMG_SZ))

    fig.add_trace(go.Scatter(
        x=list(range(16)), y=[-1.0] * 16, mode="markers",
        marker=dict(size=32, opacity=0, color="rgba(0,0,0,0)"),
        text=col_hover, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[-1.0] * 16, y=list(range(16)), mode="markers",
        marker=dict(size=32, opacity=0, color="rgba(0,0,0,0)"),
        text=row_hover, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))

    # Embed node key to the right of the colorbar
    _nk_b64 = "data:image/png;base64," + base64.b64encode(build_node_legend_bytes()).decode()
    fig.add_layout_image(dict(
        source=_nk_b64, layer="above",
        xref="paper", x=1.14, yref="paper", y=0.42,
        xanchor="left", yanchor="top",
        sizex=0.17, sizey=0.17,
    ))
    nk_label = dict(
        text="Node key",
        xref="paper", x=1.225, yref="paper", y=0.43,
        xanchor="center", yanchor="bottom",
        showarrow=False, font=dict(size=12, color="white"),
    )

    fig.update_layout(
        annotations=annotations + [nk_label],
        width=1050, height=920,
        plot_bgcolor="black",
        margin=dict(l=60, r=190, t=20, b=60),
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa",
                        font=dict(size=13, family="monospace", color="black")),
    )
    fig.update_xaxes(tickmode="array", tickvals=list(range(16)), ticktext=[""] * 16,
                     ticklen=0, range=[-1.5, 15.5],
                     title=dict(text="FM→TB forward edges", font=dict(size=14)))
    fig.update_yaxes(tickmode="array", tickvals=list(range(16)), ticktext=[""] * 16,
                     ticklen=0, range=[-1.5, 15.5],
                     title=dict(text="TB→FM backward edges", font=dict(size=14)),
                     scaleanchor="x", scaleratio=1)
    return fig

# ── Bar-plot figure ─────────────────────────────────────────────────────────────
# Figure geometry constants (fixed width so layout_image coords are predictable)
_BAR_W, _BAR_H = 1160, 700
_BAR_ML, _BAR_MR, _BAR_MT, _BAR_MB = 70, 40, 60, 220

@st.cache_data
def build_bar_figure():
    pat_totals = defaultdict(float)
    for c in range(1, 257):
        for pat, frac in pat_freq[c].items():
            pat_totals[pat] += frac

    top_pats = sorted(pat_totals, key=lambda p: -pat_totals[p])[:8]
    top_pats = sorted(top_pats, key=lambda p: len(p.split("|")))

    bwd_pat_mean = np.zeros((16, len(top_pats) + 1))
    for bi, bb in enumerate(bwd_combos):
        circuits = [c for c, vec in idx_to_vec.items()
                    if bwd_bits(vec) == bb and c in pat_freq]
        if not circuits:
            continue
        for pi, pat in enumerate(top_pats):
            bwd_pat_mean[bi, pi] = np.mean([pat_freq[c].get(pat, 0.0) for c in circuits])
        bwd_pat_mean[bi, -1] = max(0, 1.0 - bwd_pat_mean[bi, :-1].sum())

    all_pats    = top_pats + ["other"]
    colors_used = [TAB10[pi % len(TAB10)] for pi in range(len(all_pats))]

    fig = go.Figure()
    for pi, (pat, col) in enumerate(zip(all_pats, colors_used)):
        label = pat_label(pat) if pat != "other" else "other"
        fig.add_trace(go.Bar(
            name=label, x=list(range(16)), y=bwd_pat_mean[:, pi],
            marker_color=col, showlegend=False,
            customdata=bwd_labels,
            hovertemplate=(
                f"<b>%{{customdata}}</b><br>"
                f"{label}: %{{y:.1%}}<extra></extra>"
            ),
        ))

    for sep in SEPS:
        fig.add_vline(x=sep, line_width=2, line_color="black")

    # Mini circuit tick images: use data coordinates (same strategy as heatmap)
    # Extend y below 0 so images sit in visible space beneath the x-axis line.
    _, _, bwd_imgs_dark = build_tick_images()
    bwd_imgs = bwd_imgs_dark
    for bi, img in enumerate(bwd_imgs):
        fig.add_layout_image(dict(
            source=img, layer="above",
            xref="x", x=bi, yref="y", y=-0.02,
            xanchor="center", yanchor="top",
            sizex=0.82, sizey=0.14,
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Solution-type distribution by backward-edge combination<br>"
                        "<sup>Each bar = average over all 16 forward-edge variants</sup>",
                   font=dict(size=16)),
        xaxis=dict(tickmode="array", tickvals=list(range(16)), ticktext=[""] * 16,
                   ticklen=0, title=dict(text="TB→FM backward edges", font=dict(size=14))),
        yaxis=dict(title=dict(text="Fraction of samples", font=dict(size=14)),
                   tickfont=dict(size=13), tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                   range=[-0.18, 1]),
        height=_BAR_H,
        margin=dict(l=_BAR_ML, r=_BAR_MR, t=_BAR_MT, b=_BAR_MB),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="#1e293b", font=dict(color="white", size=14)),
    )
    return fig, top_pats, colors_used[:len(top_pats)]

# ── Forward-analysis figure ────────────────────────────────────────────────────
@st.cache_data
def build_forward_figure():
    data = load_phenotype_table()

    grouped = (data.groupby(["n_fwd", "n_bwd"])
                   .agg(mean_stable=("n_stable", "mean"),
                        sem_stable=("n_stable", "sem"),
                        count=("n_stable", "size"))
                   .reset_index())

    heat   = np.full((5, 5), np.nan)
    counts = np.zeros((5, 5), dtype=int)
    for _, r in grouped.iterrows():
        nf, nb = int(r.n_fwd), int(r.n_bwd)
        heat[nb, nf]   = r.mean_stable
        counts[nb, nf] = int(r["count"])

    vmax = float(np.nanmax(heat))
    heat_anns = [
        dict(x=nf, y=nb,
             text=f"{heat[nb,nf]:.2f}<br>(n={counts[nb,nf]})",
             showarrow=False, xref="x", yref="y",
             font=dict(size=12, color="white" if heat[nb,nf] > 0.65*vmax else "black",
                       family="Arial"))
        for nb in range(5) for nf in range(5) if not np.isnan(heat[nb, nf])
    ]

    fig_heat = go.Figure(go.Heatmap(
        z=heat, colorscale="YlOrRd", zmin=0, zmax=vmax,
        colorbar=dict(title="Mean # stable<br>states", thickness=14),
        hovertemplate="n_fwd=%{x}  n_bwd=%{y}<br>mean=%{z:.2f}<extra></extra>",
    ))
    for i in range(6):
        fig_heat.add_shape(type="line", x0=i-0.5, x1=i-0.5, y0=-0.5, y1=4.5,
                           line=dict(color="white", width=0.8))
        fig_heat.add_shape(type="line", x0=-0.5, x1=4.5, y0=i-0.5, y1=i-0.5,
                           line=dict(color="white", width=0.8))
    fig_heat.update_layout(
        annotations=heat_anns,
        title="Mean stable states vs edge-type count",
        xaxis=dict(tickvals=list(range(5)), title=dict(text="# Forward edges (FM→TB)", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis=dict(tickvals=list(range(5)), title=dict(text="# Backward edges (TB→FM)", font=dict(size=13)), tickfont=dict(size=12),
                   scaleanchor="x", scaleratio=1),
        width=560, height=560,
        margin=dict(l=60, r=80, t=50, b=60),
    )

    # Per-circuit hierarchy frequency (relaxed: attractor set contains {F, F+M, all-active})
    _HIER_CANON = frozenset({"1000", "1100", "1111"})
    hier_freq_map = {
        c: sum(frac for pat, frac in pat_freq[c].items()
               if _HIER_CANON <= frozenset(pat.split("|")))
        for c in pat_freq
    }
    data["hier_freq"] = data["circuit_index"].map(hier_freq_map)
    data["bwd_frac"]  = data["n_bwd"] / data["n_total"]

    _ax = dict(color="black", linecolor="black", linewidth=1,
               tickcolor="black", tickfont=dict(color="black", size=12),
               title_font=dict(color="black", size=13),
               showgrid=True, gridcolor="#EEEEEE", zeroline=False)
    _scatter_layout = dict(height=340, width=540,
                           margin=dict(l=70, r=80, t=50, b=50),
                           legend=dict(x=0.02, y=0.98, font=dict(color="black")),
                           plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(color="black"))

    # ── helper to build one scatter ───────────────────────────────────────────
    def _scatter(x_col, y_col, x_title, y_title, color_col, cscale,
                  cbar_title, ytickfmt=None, color_range=None):
        mask = data[x_col].notna() & data[y_col].notna()
        jx   = rng.uniform(-0.012, 0.012, mask.sum())
        grp  = data[mask].groupby(x_col)[y_col]
        means, sems = grp.mean(), grp.sem().fillna(0)
        cmin, cmax = (color_range or (data.loc[mask, color_col].min(),
                                       data.loc[mask, color_col].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.loc[mask, x_col] + jx,
            y=data.loc[mask, y_col],
            mode="markers",
            marker=dict(color=data.loc[mask, color_col], colorscale=cscale,
                        cmin=cmin, cmax=cmax, size=7, opacity=0.55,
                        colorbar=dict(title=cbar_title, thickness=12, x=1.02)),
            hovertemplate=f"{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=means.index, y=means.values,
            error_y=dict(type="data", array=sems.values, visible=True),
            mode="markers", marker=dict(color="black", size=9, symbol="diamond"),
            name="mean ± SEM",
        ))
        yaxis_kw = dict(title=y_title, **_ax)
        if ytickfmt:
            yaxis_kw["tickformat"] = ytickfmt
            yaxis_kw["range"] = [-0.02, 1.02]
        fig.update_layout(
            title=dict(text=f"{y_title}<br><sup>vs {x_title}</sup>",
                       font=dict(color="black", size=13)),
            xaxis=dict(title=x_title, range=[-0.05, 1.05], **_ax),
            yaxis=yaxis_kw,
            height=320, width=500,
            margin=dict(l=70, r=80, t=55, b=50),
            legend=dict(x=0.02, y=0.98, font=dict(color="black")),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
        )
        return fig

    rng = np.random.default_rng(42)

    fig_fwd_stable = _scatter("fwd_frac", "n_stable",
        "Forward fraction  (n_fwd / n_total)", "# Distinct stable states",
        "n_total", "Viridis", "Total<br>edges")
    fig_bwd_stable = _scatter("bwd_frac", "n_stable",
        "Backward fraction  (n_bwd / n_total)", "# Distinct stable states",
        "n_total", "Viridis", "Total<br>edges")
    fig_fwd_hier = _scatter("fwd_frac", "hier_freq",
        "Forward fraction  (n_fwd / n_total)", "Hierarchy freq (relaxed)",
        "n_bwd", "Oranges", "# Bwd<br>edges",
        ytickfmt=".0%", color_range=(0, 4))
    fig_bwd_hier = _scatter("bwd_frac", "hier_freq",
        "Backward fraction  (n_bwd / n_total)", "Hierarchy freq (relaxed)",
        "n_fwd", "Blues", "# Fwd<br>edges",
        ytickfmt=".0%", color_range=(0, 4))

    return fig_heat, fig_fwd_stable, fig_bwd_stable, fig_fwd_hier, fig_bwd_hier

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("Four-node circuit attractor landscape")

tab_home, tab_heat, tab_bar, tab_fwd, tab_takeaway = st.tabs([
    "📖 About",
    "🗺️ Topology heatmap",
    "📊 Solution types",
    "🔍 Edge analysis",
    "💡 Take-home",
])

# ── Tab 0: Landing page ────────────────────────────────────────────────────────
with tab_home:
    col_l, col_m, col_r = st.columns([1, 3, 1])
    with col_l:
        st.image(build_logo_bytes(), width=320)
    with col_m:
        st.markdown("""
## What are we studying?

The **tumor microenvironment (TME)** is shaped by complex communication between immune
and stromal cells. We model a four-cell circuit of key TME players and ask: which
patterns of cell–cell signalling can give rise to a **hierarchical activation cascade** —
a stepwise progression from a single active cell type all the way to a fully co-activated
community?

The four cell types are:

| Symbol | Cell type | Role in the TME |
|--------|-----------|-----------------|
| **F** | Fibroblasts | Stromal scaffold; remodel the ECM and secrete pro- and anti-tumour factors |
| **M** | Macrophages | Innate immune sentinels; switch between pro-inflammatory (M1) and immunosuppressive (M2) phenotypes |
| **T** | T-cells | Adaptive cytotoxic effectors; key mediators of anti-tumour immunity |
| **B** | B-cells | Humoral immunity; can form tertiary lymphoid structures that correlate with better prognosis |

The **canonical hierarchy F → F+M → F+M+T+B** represents a cascade in which fibroblast
activation precedes macrophage recruitment, followed by full lymphocyte co-activation —
a pattern associated with productive anti-tumour immune responses.

---

## The circuit model

Each directed edge represents a regulatory interaction from one cell type to another.
Up to **8 directed edges** are possible:

| Direction | Edges | Interpretation |
|-----------|-------|----------------|
| Forward (FM → TB) | F→T, M→T, F→B, M→B | Fibroblasts / macrophages signal to lymphocytes |
| Backward (TB → FM) | T→F, T→M, B→F, B→M | Lymphocytes feed back onto stromal / innate cells |

Each edge is either present or absent, giving **2⁸ = 256 distinct circuit topologies**.
For every topology we sample **100,000 random parameter sets** (interaction strengths drawn
uniformly from [0, 5]) and integrate the ODE system to its stable steady state, recording
which cell types are active at the attractor.

### ODE system
        """)

        st.latex(r"""
\dot{F} = F \bigl(P_{BF}\,B + p_{FF}\,F + p_{MF}\,M + P_{TF}\,T - r_F\bigr)
""")
        st.latex(r"""
\dot{M} = M \bigl(P_{BM}\,B + p_{FM}\,F + p_{MM}\,M + P_{TM}\,T - r_M\bigr)
""")
        st.latex(r"""
\dot{T} = T \bigl(P_{FT}\,F + P_{MT}\,M + p_{TT}\,T - p_{BT}\,B - r_T\bigr)
""")
        st.latex(r"""
\dot{B} = B \bigl(P_{FB}\,F + P_{MB}\,M + p_{TB}\,T - p_{BB}\,B - r_B\bigr)
""")
        st.markdown(r"""
**Uppercase** $P_{XY} \sim \mathcal{U}[0,5]$ — free edge-strength parameters
(set to 0 when edge X→Y is absent).
**Lowercase** $p_{xy}$ — fixed kinetic constants; $r_x$ — basal decay rates.
Each equation is multiplicative-linear: the $x \cdot (\cdots)$ structure ensures
$\dot{x}=0$ at $x=0$, so the all-inactive state is always a trivial fixed point.

---

## How the data were collected

Simulations ran on a **high-performance computing cluster** using
**Wolfram Mathematica 14.3** to solve the ODE system.
Jobs were submitted via the LSF scheduler — 2,560 array jobs in total
(256 circuits × 10 independent chunks of 10,000 samples each = **100,000 samples per circuit**).
Results were aggregated per circuit and stored in `final_results.csv`.
        """)

        st.markdown("---")
        st.markdown("### Run statistics")

        # Compute summary numbers from loaded data
        n_circuits  = len(pat_freq)
        nd_vals     = list(n_distinct.values())
        hier_m      = mats["hier_strict"]
        n_any_hier  = int(np.sum((~np.isnan(hier_m)) & (hier_m > 0)))
        n_maj_hier  = int(np.sum((~np.isnan(hier_m)) & (hier_m >= 0.5)))
        max_hier    = float(np.nanmax(hier_m))
        ent_m       = mats["entropy"]
        mean_ent    = float(np.nanmean(ent_m))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Circuits analysed", f"{n_circuits} / 256")
        c2.metric("Samples per circuit", "100,000")
        c3.metric("Total ODE integrations", f"{n_circuits * 100_000:,}")
        c4.metric("HPC jobs", "2,560")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean distinct attractors", f"{np.mean(nd_vals):.1f}")
        c2.metric("Max distinct attractors", f"{max(nd_vals)}")
        c3.metric("Circuits with any\ncanonical hierarchy", f"{n_any_hier}")
        c4.metric("Circuits >50 % canonical", f"{n_maj_hier}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max canonical frequency", f"{max_hier:.0%}")
        c2.metric("Mean attractor entropy", f"{mean_ent:.2f} bits")
        c3.metric("Parameter range", "[0, 5]")
        c4.metric("Scheduler", "LSF")

        st.markdown("---")
        st.markdown("""
### What each tab shows

| Tab | Contents |
|-----|----------|
| **Topology heatmap** | 16 × 16 grid of all circuit topologies, coloured by a chosen metric (hierarchy frequency, entropy, attractor diversity, …). Rows = backward-edge combinations; columns = forward-edge combinations. Hover any cell for the full attractor-pattern breakdown, hover a tick icon for row/column averages. |
| **Solution types** | Stacked bar chart showing how often each attractor-pattern type appears, averaged over the 16 forward-edge variants for each backward-edge combination. Hover bars for exact fractions. The icon legend below the chart uses **yellow = cell type active** and **teal = cell type inactive** (columns: F · M · T · B). |
| **Edge analysis** | Left: heat map of mean number of distinct stable states as a function of forward- vs backward-edge count. Right: hierarchy frequency (relaxed) vs number of backward (TB→FM) edges — tests whether backward signaling is necessary and sufficient for a hierarchical cascade. |

---
*Data collected June 2026*
        """)

# ── Tab 1: heatmap ─────────────────────────────────────────────────────────────
with tab_heat:
    if "view_key" not in st.session_state:
        st.session_state["view_key"] = VIEWS[0]["key"]

    _ctrl_col, _map_col = st.columns([1, 4])
    with _ctrl_col:
        st.markdown("**Display metric**")
        view_groups = {
            "🔗 Hierarchy cascades": VIEWS[:5],
            "📊 Attractor statistics": VIEWS[5:8],
            "🔵 State frequencies": VIEWS[8:],
        }
        for group_name, group_views in view_groups.items():
            st.markdown(f"**{group_name}**")
            for v in group_views:
                short = (v["label"]
                         .replace("Hierarchy: ", "")
                         .replace(" (strict)", " ★")
                         .replace(" (relaxed)", " ○"))
                if st.button(short, key=v["key"], use_container_width=True):
                    st.session_state["view_key"] = v["key"]
            st.markdown("---")
        st.caption("Hover cells for full breakdown · hover tick icons for row/column averages.")

    with _map_col:
        view = next(v for v in VIEWS if v["key"] == st.session_state["view_key"])
        st.markdown(f"**{view['label']}** — {view['desc']}")
        st.plotly_chart(build_heatmap_figure(view), use_container_width=False)

# ── Tab 2: stacked bar ─────────────────────────────────────────────────────────
with tab_bar:
    st.markdown(
        "Each bar shows the fraction of parameter samples in each attractor-pattern type, "
        "averaged over all 16 forward-edge combinations for that backward-edge combo."
    )
    _bar_fig, _bar_top_pats, _bar_colors = build_bar_figure()
    st.plotly_chart(_bar_fig, use_container_width=True)
    _node_b64 = base64.b64encode(build_node_legend_bytes()).decode()
    _pat_html  = build_bar_legend_html(_bar_top_pats, _bar_colors)
    st.markdown(
        f'<div style="display:flex;justify-content:center;align-items:flex-start;'
        f'gap:30px;padding:0;margin-top:-20px;">'
        f'<div>{_pat_html}</div>'
        f'<div style="text-align:center;padding-top:8px;flex-shrink:0;">'
        f'<div style="font-size:11px;color:#aaa;margin-bottom:4px;">Node key</div>'
        f'<img src="data:image/png;base64,{_node_b64}" width="130">'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Tab 3: forward-edge analysis ───────────────────────────────────────────────
with tab_fwd:
    st.markdown(
        "**Left:** mean # distinct stable states by (n_fwd, n_bwd) edge counts.  \n"
        "**Right (2 × 2):** each outcome metric (rows) × each edge direction (columns). "
        "Colour encodes the complementary edge count."
    )
    fig_heat, fig_fwd_stable, fig_bwd_stable, fig_fwd_hier, fig_bwd_hier = build_forward_figure()
    _heat_col, _scatter_col = st.columns([1, 2], gap="large")
    with _heat_col:
        st.plotly_chart(fig_heat, use_container_width=False)
    with _scatter_col:
        _r1c1, _r1c2 = st.columns(2)
        with _r1c1:
            st.plotly_chart(fig_fwd_stable, use_container_width=False)
        with _r1c2:
            st.plotly_chart(fig_bwd_stable, use_container_width=False)
        _r2c1, _r2c2 = st.columns(2)
        with _r2c1:
            st.plotly_chart(fig_fwd_hier, use_container_width=False)
        with _r2c2:
            st.plotly_chart(fig_bwd_hier, use_container_width=False)

# ── Tab 4: take-home message ───────────────────────────────────────────────────
with tab_takeaway:
    _, _tm, _ = st.columns([1, 5, 1])
    with _tm:
        st.markdown("""
## Take-home message

### ✅ Obvious / expected results

- **More edges → more attractor diversity.** Circuits with more active edges have higher Shannon entropy and more distinct attractor types. This reflects a larger effective parameter space.
- **All-active state (F+M+T+B) becomes more accessible.** In denser circuits the all-active attractor appears as a stable state across a larger fraction of parameter samples.
- **Forward-edge count alone does not predict hierarchy.** Increasing the proportion of FM→TB forward edges does not raise hierarchy frequency — the forward-fraction scatter is essentially flat (see Edge analysis tab, bottom-left).

---

### 🔑 Key / surprising results

- **Backward signaling (TB→FM) is necessary *and* sufficient for the canonical hierarchy.**
  The hierarchy-frequency vs backward-fraction scatter shows a sharp jump from ~0 % at zero backward edges to substantial frequency as soon as any backward edge is added.
  The forward-fraction scatter shows no corresponding trend — ruling out forward edges as drivers.

- **Which specific backward edges matter is visible in the topology heatmap.**
  The row structure of the heatmap (each row = one backward-edge combination) reveals which TB→FM feedback patterns are most potent. Inspect the "Hierarchy: Canonical F→F+M→all (strict)" view for the clearest signal.

- **Hierarchy is largely insensitive to forward-edge count once feedback is present.**
  Within the high-hierarchy region of the heatmap, colour (forward combination) varies broadly without strongly tracking hierarchy frequency — suggesting forward edges modulate *how* the hierarchy is expressed, not *whether* it exists.

---

### 🔬 In progress

- **Semistable (ghost) attractors.** The current analysis covers only stable fixed points.
  Semistable states — stable within their face but with a positive invasion eigenvalue — are available for ~1,670 circuits from the first HPC run and represent quasi-stable transient phenotypes.

---

### ❓ Open questions

- **Mechanistic explanation.** *Why* does backward signaling enable hierarchy?
  A candidate story: TB→FM feedback creates a positive loop that stabilises the
  F-only and F+M intermediate states, making them genuine attractors rather than
  transients. This could be tested analytically in reduced two-node sub-circuits.

- **Robustness to kinetic constants.** The fixed parameters ($p_{FF}$, $p_{MF}$, …, $K_x$, $r_x$)
  were set from a single biological parameterisation. Do the topology-level results
  (especially the backward-signaling threshold) survive if those constants are varied?
""")
