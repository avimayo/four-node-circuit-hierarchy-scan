#!/usr/bin/env python3
"""
build_heteroclinic.py — Aggregate eigenvector data into heteroclinic graph edges.

For each (circuit, pattern) with positive eigenvalues, finds the dominant unstable
manifold direction in each sample and votes across samples to assign confidence.

Output: heteroclinic_edges.csv
  circuit_idx, src, tgt, n_src_samples, n_edge_samples, confidence, dominant_type, hamming
  - src/tgt  : 4-bit FMTB pattern string
  - confidence: fraction of pos-eval samples where this edge direction was detected
  - dominant_type: 'invasion' (absent type dominates evec) | 'bistable' (present type)
  - hamming  : bit distance src→tgt (1 = adjacent, >1 = skip connection)
"""

import csv, os, re
from collections import defaultdict
import numpy as np
from pathlib import Path

RES     = Path("~/circuit_hpc/results_evec").expanduser()
OUTPATH = Path("~/circuit_hpc/heteroclinic_edges.csv").expanduser()

EPS_EVAL = 0.05   # min eigenvalue re-part to count as positive
VEC_DOM  = 0.25   # min normalised eigenvec component to call it directional

def target_pattern(src, v):
    """
    Given src pattern (FMTB string) and unit eigenvector v (length 4),
    return implied target pattern, or None if no component clears VEC_DOM.

    Absent type i with v[i] >  VEC_DOM  → type i invades (bit turns on)
    Present type i with v[i] < -VEC_DOM → type i declines (bit turns off)
    """
    bits = list(src)
    changed = False
    for i in range(4):
        if bits[i] == '0' and v[i] > VEC_DOM:
            bits[i] = '1'; changed = True
        elif bits[i] == '1' and v[i] < -VEC_DOM:
            bits[i] = '0'; changed = True
    return ''.join(bits) if changed else None

def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

# ── gather circuit files ───────────────────────────────────────────────────────
circuits = sorted(
    int(m.group(1))
    for f in os.listdir(RES)
    if (m := re.match(r'circuit_(\d+)_evec\.csv', f))
       and (RES / f).stat().st_size > 0
)
print(f"Processing {len(circuits)} circuits ...")

rows_out = []

for ci, circ in enumerate(circuits):
    path = RES / f"circuit_{circ:03d}_evec.csv"

    # group: (sample_idx, state) → sorted list of rows
    groups = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            groups[(int(row['sample_idx']), row['state'])].append(row)

    n_pos  = defaultdict(int)          # src → # samples with ≥1 pos eval
    e_hits = defaultdict(int)          # (src, tgt) → # samples detecting edge
    e_type = {}                        # (src, tgt) → dominant_type

    for (_, state), rows in groups.items():
        rows.sort(key=lambda r: int(r['eval_idx']))
        evals_re = [float(r['e_re']) for r in rows]
        evecs    = [[float(r['vF']), float(r['vM']),
                     float(r['vT']), float(r['vB'])] for r in rows]

        pos_idx = [i for i, e in enumerate(evals_re) if e > EPS_EVAL]
        if not pos_idx:
            continue

        n_pos[state] += 1

        for pi in pos_idx:
            v = np.array(evecs[pi])
            nrm = np.linalg.norm(v)
            if nrm < 1e-10:
                continue
            v /= nrm

            tgt = target_pattern(state, v)
            if tgt is None:
                continue

            e_hits[(state, tgt)] += 1

            absent_max  = max((abs(v[i]) for i in range(4) if state[i]=='0'), default=0.0)
            present_max = max((abs(v[i]) for i in range(4) if state[i]=='1'), default=0.0)
            e_type[(state, tgt)] = 'invasion' if absent_max >= present_max else 'bistable'

    for (src, tgt), hits in e_hits.items():
        n = n_pos[src]
        rows_out.append({
            'circuit_idx':   circ,
            'src':           src,
            'tgt':           tgt,
            'n_src_samples': n,
            'n_edge_samples': hits,
            'confidence':    round(hits / n, 4) if n else 0.0,
            'dominant_type': e_type.get((src, tgt), 'unknown'),
            'hamming':       hamming(src, tgt),
        })

    if (ci + 1) % 50 == 0:
        print(f"  {ci+1}/{len(circuits)} done")

print(f"Writing {len(rows_out)} edges → {OUTPATH}")
with open(OUTPATH, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=[
        'circuit_idx','src','tgt','n_src_samples','n_edge_samples',
        'confidence','dominant_type','hamming',
    ])
    w.writeheader()
    w.writerows(rows_out)

print("Done.")
