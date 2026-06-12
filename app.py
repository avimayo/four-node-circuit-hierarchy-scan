"""
Streamlit app: Four-node circuit attractor landscape
Repo: avimayo/four-node-circuit-hierarchy-scan
"""

import csv
import numpy as np
import streamlit as st
import plotly.graph_objects as go
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

SEPS = [0.5, 4.5, 10.5, 14.5]

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
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

pat_freq_raw, n_distinct = load_data()
pat_freq = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in pat_freq_raw.items()})

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
    _ONE_NODE = frozenset(s for s in ALL_STATES if s.count("1") == 1)

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

    mats = {
        "hier_strict":  hier_mat(lambda s: _HIER_CANONICAL <= s and (
                            s == _HIER_EXACT_A or s == frozenset({"0000","0011","1000","1100","1111"}))),
        "hier_relaxed": hier_mat(lambda s: _HIER_CANONICAL <= s),
        "hier_1_2_all": hier_mat(lambda s: "1111" in s and bool(_ONE_NODE & s) and
                            bool(frozenset(x for x in ALL_STATES if x.count("1")==2) & s)),
        "hier_1_all":   hier_mat(lambda s: "1111" in s and bool(_ONE_NODE & s)),
        "ndistinct":    nd,
        "entropy":      ent,
        "all_active":   state_mat("1111"),
        "f_only":       state_mat("1000"),
        "dominant":     dom,
    }
    return mats

mats = build_all_matrices()

# ── Hover text ─────────────────────────────────────────────────────────────────
@st.cache_data
def build_hover():
    def bar(frac, w=10):
        return "█" * round(frac * w) + "░" * (w - round(frac * w))

    def pat_lbl(pat):
        return " + ".join(STATE_LABELS.get(s, s) for s in sorted(pat.split("|")))

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

        hier_lines = [
            f"  {bar(sum(frac for p,frac in pat_freq[c].items() if fn(frozenset(p.split('|')))),6)}"
            f"  {sum(frac for p,frac in pat_freq[c].items() if fn(frozenset(p.split('|')))):4.0%}"
            f"  {name}"
            for name, fn in fns
        ]
        pat_lines = [
            f"  {frac:5.1%}  {bar(frac)}  {pat_lbl(pat)}"
            for pat, frac in pats
        ]
        hover[ri, ci] = "<br>".join([
            f"<b>Circuit {c}</b>  —  {n_distinct[c]} attractor type{'s' if n_distinct[c]>1 else ''}"
            f"  |  H = {entropy:.2f} bits",
            f"<span style='color:#666'>Fwd: {fwd_str}   Bwd: {bwd_str}</span>",
            "─" * 38,
            "<b>Hierarchy frequencies</b>",
            *hier_lines,
            "─" * 38,
            "<b>All attractor patterns</b>",
            *pat_lines,
        ])
    return hover

hover_text = build_hover()

# ── View definitions ───────────────────────────────────────────────────────────
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

# ── Figure builder ─────────────────────────────────────────────────────────────
def build_figure(view):
    z     = mats[view["key"]]
    zmin  = view["zmin"]
    zmax  = view["zmax"]
    span  = zmax - zmin if zmax > zmin else 1
    fmt   = view["ann_fmt"]

    annotations = []
    for ri in range(16):
        for ci in range(16):
            v = z[ri, ci]
            if not np.isnan(v):
                norm = (v - zmin) / span
                annotations.append(dict(
                    x=ci, y=ri,
                    text=format(v, fmt),
                    showarrow=False, xref="x", yref="y",
                    font=dict(size=7, color="white" if norm > 0.55 else "black"),
                ))

    fig = go.Figure(go.Heatmap(
        z=z,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=view["colorscale"],
        zmin=zmin, zmax=zmax,
        colorbar=dict(
            title=dict(text=view["colorbar_title"], side="right"),
            tickformat=fmt if "%" in fmt else "",
            thickness=16, len=0.85,
        ),
    ))

    for sep in SEPS:
        fig.add_shape(type="line", x0=sep, x1=sep, y0=-0.5, y1=15.5,
                      line=dict(color="black", width=3))
        fig.add_shape(type="line", x0=-0.5, x1=15.5, y0=sep, y1=sep,
                      line=dict(color="black", width=3))

    fig.update_layout(
        annotations=annotations,
        width=900, height=900,
        plot_bgcolor="black",
        margin=dict(l=140, r=60, t=20, b=140),
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa",
                        font=dict(size=11, family="monospace")),
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
        scaleanchor="x", scaleratio=1,
    )
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Display metric")
    view_groups = {
        "🔗 Hierarchy cascades": VIEWS[:4],
        "📊 Attractor statistics": VIEWS[4:7],
        "🔵 State frequencies": VIEWS[7:],
    }

    selected_label = None
    for group_name, group_views in view_groups.items():
        st.markdown(f"**{group_name}**")
        for v in group_views:
            short = v["label"].replace("Hierarchy: ", "").replace(" (strict)", " ★").replace(" (relaxed)", " ○")
            if st.button(short, key=v["key"], use_container_width=True):
                selected_label = v["label"]
                st.session_state["view_key"] = v["key"]
        st.markdown("---")

    st.caption("Hover over any cell to see the full attractor breakdown for that circuit.")

# ── Main area ──────────────────────────────────────────────────────────────────
if "view_key" not in st.session_state:
    st.session_state["view_key"] = VIEWS[0]["key"]

view = next(v for v in VIEWS if v["key"] == st.session_state["view_key"])

st.title("Four-node circuit attractor landscape")
st.markdown(f"**{view['label']}** — {view['desc']}")

fig = build_figure(view)
st.plotly_chart(fig, use_container_width=False)
