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

# ── Mini state-pattern icon helpers (bar legend) ───────────────────────────────
_NC_HEX = {"F": "#4C72B0", "M": "#DD8452", "T": "#55A868", "B": "#C44E52"}
_NODES  = ["F", "M", "T", "B"]

def _state_row_svg(state, cell=11):
    parts = []
    for ni, node in enumerate(_NODES):
        fill = _NC_HEX[node] if state[ni] == "1" else "#DDDDDD"
        parts.append(f'<rect x="{ni*cell}" y="0" width="{cell}" height="{cell}" '
                     f'fill="{fill}" stroke="white" stroke-width="0.5"/>')
    return "".join(parts)

def pat_icon_html(pat_str, swatch_color, cell=11, gap=3):
    sw = cell - 2
    if pat_str == "other":
        w = sw + gap + 4 * cell
        return (f'<svg width="{w}" height="{cell}" xmlns="http://www.w3.org/2000/svg">'
                f'<rect x="0" y="0" width="{sw}" height="{cell}" fill="{swatch_color}" rx="1"/>'
                f'<text x="{sw+gap+2}" y="{cell-2}" font-size="8" fill="#444" '
                f'font-family="monospace">other</text></svg>')
    states = sorted(pat_str.split("|"))
    N = len(states)
    w = sw + gap + 4 * cell
    h = N * cell
    rows = "".join(
        f'<g transform="translate({sw+gap},{si*cell})">{_state_row_svg(s, cell)}</g>'
        for si, s in enumerate(states)
    )
    return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="0" y="0" width="{sw}" height="{h}" fill="{swatch_color}" rx="1"/>'
            f'{rows}</svg>')

def build_bar_legend_html(top_pats, colors):
    all_pats   = list(top_pats) + ["other"]
    all_colors = list(colors) + ["#AAAAAA"]
    items = []
    for pat, col in zip(all_pats, all_colors):
        svg = pat_icon_html(pat, col)
        lbl = pat_label(pat) if pat != "other" else "other"
        items.append(
            f'<td style="text-align:center;vertical-align:top;padding:4px 8px;">'
            f'{svg}'
            f'<div style="font-size:8px;color:#333;max-width:60px;'
            f'word-wrap:break-word;margin-top:2px;">{lbl}</div></td>'
        )
    return ('<div style="overflow-x:auto;padding:6px 0;">'
            '<table style="border-collapse:collapse;"><tr>'
            + "".join(items) + '</tr></table></div>')

# ── Mini circuit PNG helpers (heatmap tick labels) ─────────────────────────────
_NP_MPL = {"F": (0.20, 0.80), "M": (0.80, 0.80),
           "T": (0.80, 0.20), "B": (0.20, 0.20)}
_NC_MPL = {"F": "#4C72B0", "M": "#DD8452", "T": "#55A868", "B": "#C44E52"}
_FWD_SD = [          # (source_xy, dest_xy) for F→T, M→T, F→B, M→B
    ((0.20, 0.80), (0.80, 0.20)),
    ((0.80, 0.80), (0.80, 0.20)),
    ((0.20, 0.80), (0.20, 0.20)),
    ((0.80, 0.80), (0.20, 0.20)),
]
_BWD_SD = [          # for T→F, T→M, B→F, B→M
    ((0.80, 0.20), (0.20, 0.80)),
    ((0.80, 0.20), (0.80, 0.80)),
    ((0.20, 0.20), (0.20, 0.80)),
    ((0.20, 0.20), (0.80, 0.80)),
]

def _circuit_png_b64(bits, edge_sds, size_in=0.42, dpi=120):
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0))
    for b, (src, dst) in zip(bits, edge_sds):
        if b:
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color="white",
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
def build_tick_images():
    fwd_imgs = [_circuit_png_b64(bb, _FWD_SD) for bb in fwd_combos]
    bwd_imgs = [_circuit_png_b64(bb, _BWD_SD) for bb in bwd_combos]
    return fwd_imgs, bwd_imgs

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

# ── Matrix builders (heatmap tab) ──────────────────────────────────────────────
@st.cache_data
def build_all_matrices():
    _HIER_CANONICAL = frozenset({"1000", "1100", "1111"})
    _HIER_EXACT_A   = frozenset({"0000", "1000", "1100", "1111"})
    _HIER_EXACT_B   = frozenset({"0000", "0011", "1000", "1100", "1111"})
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
    _ONE_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 1)
    _TWO_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 2)

    fns = [
        ("Canonical F→F+M→all (strict)", lambda s: _HIER_CANONICAL <= s and (s == _HIER_EXACT_A or s == _HIER_EXACT_B)),
        ("Contains F→F+M→all",           lambda s: _HIER_CANONICAL <= s),
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
        hier_lines = [
            f"  {bar(f, 6)}  {f:4.0%}  {name}"
            for (name, _), f in zip(fns, hier_freqs)
        ]
        # Note when all hierarchy criteria are satisfied by every sampled pattern
        if all(f >= 0.999 for f in hier_freqs):
            hier_lines.append("  (all sampled parameter sets are hierarchical)")

        pat_lines = [f"  {frac:5.1%}  {bar(frac)}  {pat_label(pat)}" for pat, frac in pats]
        hover[ri, ci] = "<br>".join([
            f"<b>Circuit {c}</b>  —  {n_distinct[c]} attractor type{'s' if n_distinct[c]>1 else ''}"
            f"  |  H = {entropy:.2f} bits",
            f"<span style='color:#333'>Fwd: {fwd_str}   Bwd: {bwd_str}</span>",
            "─" * 38,
            "<b>Hierarchy frequencies</b>  <span style='color:#555'>(fraction of 10k param samples)</span>",
            *hier_lines,
            "─" * 38,
            "<b>All attractor patterns</b>",
            *pat_lines,
        ])
    return hover

@st.cache_data
def build_tick_hover():
    """Row/column average statistics for tick-label hover tooltips."""
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
            f"Mean entropy:      {me:.2f} bits",
            f"Mean dominant share: {md:.0%}",
            f"Mean # attractors: {mn:.1f}",
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
            f"Mean entropy:      {me:.2f} bits",
            f"Mean dominant share: {md:.0%}",
            f"Mean # attractors: {mn:.1f}",
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

def build_heatmap_figure(view):
    z, zmin, zmax = mats[view["key"]], view["zmin"], view["zmax"]
    span = zmax - zmin if zmax > zmin else 1
    fmt  = view["ann_fmt"]
    annotations = [
        dict(x=ci, y=ri, text=format(z[ri, ci], fmt),
             showarrow=False, xref="x", yref="y",
             font=dict(size=7, color="white" if (z[ri,ci]-zmin)/span > 0.55 else "black"))
        for ri in range(16) for ci in range(16) if not np.isnan(z[ri, ci])
    ]

    fwd_imgs, bwd_imgs = build_tick_images()
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

    # Mini circuit images at y=-1 (x-axis ticks) and x=-1 (y-axis ticks)
    # Placed within the extended data-coordinate range [-1.5, 15.5]
    IMG_SZ = 0.82
    for ci, img in enumerate(fwd_imgs):
        fig.add_layout_image(dict(
            source=img, layer="above",
            xref="x", x=ci, yref="y", y=-1.0,
            xanchor="center", yanchor="middle",
            sizex=IMG_SZ, sizey=IMG_SZ,
        ))
    for ri, img in enumerate(bwd_imgs):
        fig.add_layout_image(dict(
            source=img, layer="above",
            xref="x", x=-1.0, yref="y", y=ri,
            xanchor="center", yanchor="middle",
            sizex=IMG_SZ, sizey=IMG_SZ,
        ))

    # Invisible scatter for column/row hover tooltips
    fig.add_trace(go.Scatter(
        x=list(range(16)), y=[-1.0] * 16,
        mode="markers",
        marker=dict(size=32, opacity=0, color="rgba(0,0,0,0)"),
        text=col_hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[-1.0] * 16, y=list(range(16)),
        mode="markers",
        marker=dict(size=32, opacity=0, color="rgba(0,0,0,0)"),
        text=row_hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        annotations=annotations,
        width=920, height=920,
        plot_bgcolor="black",
        margin=dict(l=60, r=60, t=20, b=60),
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa",
                        font=dict(size=11, family="monospace", color="black")),
    )
    fig.update_xaxes(
        tickmode="array", tickvals=list(range(16)), ticktext=[""] * 16,
        ticklen=0,
        range=[-1.5, 15.5],
        title=dict(text="FM→TB forward edges", font=dict(size=11)),
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(16)), ticktext=[""] * 16,
        ticklen=0,
        range=[-1.5, 15.5],
        title=dict(text="TB→FM backward edges", font=dict(size=11)),
        scaleanchor="x", scaleratio=1,
    )
    return fig

# ── Bar-plot figure ────────────────────────────────────────────────────────────
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

    all_pats = top_pats + ["other"]
    colors_used = [TAB10[pi % len(TAB10)] for pi in range(len(all_pats))]

    fig = go.Figure()
    for pi, (pat, col) in enumerate(zip(all_pats, colors_used)):
        label = pat_label(pat) if pat != "other" else "other"
        fig.add_trace(go.Bar(
            name=label,
            x=list(range(16)),
            y=bwd_pat_mean[:, pi],
            marker_color=col,
            showlegend=False,
            hovertemplate=f"<b>{label}</b><br>%{{y:.1%}}<extra></extra>",
        ))

    for sep in SEPS:
        fig.add_vline(x=sep, line_width=2, line_color="black")

    fig.update_layout(
        barmode="stack",
        title="Solution-type distribution by backward-edge combination<br>"
              "<sup>Each bar = average over 16 forward-edge variants</sup>",
        xaxis=dict(
            tickmode="array", tickvals=list(range(16)), ticktext=bwd_labels,
            tickangle=45, tickfont=dict(size=9),
            title="TB→FM backward edges",
        ),
        yaxis=dict(title="Fraction of samples", range=[0, 1]),
        height=550,
        margin=dict(l=60, r=40, t=80, b=140),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12),
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
             font=dict(size=10, color="white" if heat[nb,nf] > 0.65*vmax else "black",
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
        xaxis=dict(tickvals=list(range(5)), title="# Forward edges (FM→TB)", tickfont=dict(size=11)),
        yaxis=dict(tickvals=list(range(5)), title="# Backward edges (TB→FM)", tickfont=dict(size=11),
                   scaleanchor="x", scaleratio=1),
        width=460, height=460,
        margin=dict(l=60, r=60, t=50, b=60),
    )

    mask = data.fwd_frac.notna()
    rng  = np.random.default_rng(42)
    jx   = rng.uniform(-0.012, 0.012, mask.sum())

    grp   = data[mask].groupby("fwd_frac")["n_stable"]
    means = grp.mean()
    sems  = grp.sem().fillna(0)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=data.loc[mask, "fwd_frac"] + jx,
        y=data.loc[mask, "n_stable"],
        mode="markers",
        marker=dict(
            color=data.loc[mask, "n_total"],
            colorscale="Viridis",
            size=6, opacity=0.65,
            colorbar=dict(title="Total edges", thickness=14, x=1.02),
        ),
        hovertemplate="fwd_frac=%{x:.2f}<br>n_stable=%{y}<extra></extra>",
        showlegend=False,
    ))
    fig_scatter.add_trace(go.Scatter(
        x=means.index, y=means.values,
        error_y=dict(type="data", array=sems.values, visible=True),
        mode="markers", marker=dict(color="black", size=8),
        name="mean ± SEM",
    ))
    _ax = dict(color="black", linecolor="black", linewidth=1,
               tickcolor="black", tickfont=dict(color="black"),
               title_font=dict(color="black"),
               showgrid=True, gridcolor="#EEEEEE", zeroline=False)
    fig_scatter.update_layout(
        title=dict(text="Stable-state diversity vs forward fraction",
                   font=dict(color="black")),
        xaxis=dict(title="Forward fraction  (n_fwd / n_total)", range=[-0.05, 1.05], **_ax),
        yaxis=dict(title="# Distinct stable states", **_ax),
        height=460, width=540,
        margin=dict(l=60, r=80, t=50, b=60),
        legend=dict(x=0.02, y=0.98, font=dict(color="black")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
    )

    return fig_heat, fig_scatter

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("Four-node circuit attractor landscape")

tab_heat, tab_bar, tab_fwd = st.tabs([
    "🗺️ Topology heatmap",
    "📊 Solution types by backward edges",
    "🔍 Forward-edge analysis",
])

# ── Tab 1: heatmap ─────────────────────────────────────────────────────────────
with tab_heat:
    with st.sidebar:
        st.header("Display metric")
        view_groups = {
            "🔗 Hierarchy cascades": VIEWS[:4],
            "📊 Attractor statistics": VIEWS[4:7],
            "🔵 State frequencies": VIEWS[7:],
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
        st.caption("Hover cells for attractor breakdown · hover tick icons for row/column averages.")

    if "view_key" not in st.session_state:
        st.session_state["view_key"] = VIEWS[0]["key"]

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
    st.markdown("**Attractor patterns** (each column = one state; F / M / T / B left→right)")
    st.markdown(build_bar_legend_html(_bar_top_pats, _bar_colors), unsafe_allow_html=True)

# ── Tab 3: forward-edge analysis ───────────────────────────────────────────────
with tab_fwd:
    st.markdown(
        "**Left:** mean number of distinct stable states as a function of forward- and backward-edge count.  \n"
        "**Right:** scatter of stable-state diversity vs the fraction of edges that are forward (FM→TB)."
    )
    fig_heat, fig_scatter = build_forward_figure()
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.plotly_chart(fig_heat, use_container_width=False)
    with col2:
        st.plotly_chart(fig_scatter, use_container_width=False)
