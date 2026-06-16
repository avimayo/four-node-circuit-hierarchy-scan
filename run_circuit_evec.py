#!/usr/bin/env python3
"""
run_circuit_evec.py — Targeted eigenvector extraction for top circuits.

Same steady-state solver as run_circuit_v2c_np.py but saves eigenvectors
alongside eigenvalues. Runs fewer samples (N=5000 default) for tractability.

Output: results_evec/circuit_XXX_evec.csv
Columns (tidy format, one row per eigenvalue per steady state per sample):
  sample_idx, state, classification, eval_idx,
  e_re, e_im,          <- eigenvalue
  vF, vM, vT, vB,      <- eigenvector real parts (F=0,M=1,T=2,B=3)
  vF_im,vM_im,vT_im,vB_im  <- eigenvector imaginary parts

Usage: python3 run_circuit_evec.py <circIdx> [nSamples]
"""

import sys, os, csv
import numpy as np
from pathlib import Path

# ── Copy fixed parameters from run_circuit_v2c_np.py ─────────────────────────
Kb, Kf, Km, Kt = 2.0, 2.9, 4.7, 2.0
pff, pfm       = 1.49, 1.1
pmf, pmm       = 1.7,  1.76
ptt            = 2.37
ptb            = 2.5
pbt            = 1.04
pbb            = 1.7
rf, rm, rt, rb = 0.75, 2.5, 0.23, 1.5

K = np.array([Kf, Km, Kt, Kb], float)
r = np.array([rf, rm, rt, rb], float)

C_FIXED = np.array([
    [pff, pmf, 0.0, 0.0],
    [pfm, pmm, 0.0, 0.0],
    [0.0, 0.0, ptt, 0.0],
    [0.0, 0.0, ptb, 0.0],
], float)

EDGE_MAP = [(2,0),(0,2),(2,1),(1,2),(3,0),(0,3),(3,1),(1,3)]

CIRCUITS = np.array([
[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0],[0,0,0,0,0,1,0,0],
[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0],
[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,1,0,1],[0,0,0,0,0,1,1,0],
[0,0,0,0,1,0,0,1],[0,0,0,0,1,0,1,0],[0,0,0,0,1,1,0,0],[0,0,0,1,0,0,0,1],
[0,0,0,1,0,0,1,0],[0,0,0,1,0,1,0,0],[0,0,0,1,1,0,0,0],[0,0,1,0,0,0,0,1],
[0,0,1,0,0,0,1,0],[0,0,1,0,0,1,0,0],[0,0,1,0,1,0,0,0],[0,0,1,1,0,0,0,0],
[0,1,0,0,0,0,0,1],[0,1,0,0,0,0,1,0],[0,1,0,0,0,1,0,0],[0,1,0,0,1,0,0,0],
[0,1,0,1,0,0,0,0],[0,1,1,0,0,0,0,0],[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,1,0],
[1,0,0,0,0,1,0,0],[1,0,0,0,1,0,0,0],[1,0,0,1,0,0,0,0],[1,0,1,0,0,0,0,0],
[1,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,1],[0,0,0,0,1,0,1,1],[0,0,0,0,1,1,0,1],
[0,0,0,0,1,1,1,0],[0,0,0,1,0,0,1,1],[0,0,0,1,0,1,0,1],[0,0,0,1,0,1,1,0],
[0,0,0,1,1,0,0,1],[0,0,0,1,1,0,1,0],[0,0,0,1,1,1,0,0],[0,0,1,0,0,0,1,1],
[0,0,1,0,0,1,0,1],[0,0,1,0,0,1,1,0],[0,0,1,0,1,0,0,1],[0,0,1,0,1,0,1,0],
[0,0,1,0,1,1,0,0],[0,0,1,1,0,0,0,1],[0,0,1,1,0,0,1,0],[0,0,1,1,0,1,0,0],
[0,0,1,1,1,0,0,0],[0,1,0,0,0,0,1,1],[0,1,0,0,0,1,0,1],[0,1,0,0,0,1,1,0],
[0,1,0,0,1,0,0,1],[0,1,0,0,1,0,1,0],[0,1,0,0,1,1,0,0],[0,1,0,1,0,0,0,1],
[0,1,0,1,0,0,1,0],[0,1,0,1,0,1,0,0],[0,1,0,1,1,0,0,0],[0,1,1,0,0,0,0,1],
[0,1,1,0,0,0,1,0],[0,1,1,0,0,1,0,0],[0,1,1,0,1,0,0,0],[0,1,1,1,0,0,0,0],
[1,0,0,0,0,0,1,1],[1,0,0,0,0,1,0,1],[1,0,0,0,0,1,1,0],[1,0,0,0,1,0,0,1],
[1,0,0,0,1,0,1,0],[1,0,0,0,1,1,0,0],[1,0,0,1,0,0,0,1],[1,0,0,1,0,0,1,0],
[1,0,0,1,0,1,0,0],[1,0,0,1,1,0,0,0],[1,0,1,0,0,0,0,1],[1,0,1,0,0,0,1,0],
[1,0,1,0,0,1,0,0],[1,0,1,0,1,0,0,0],[1,0,1,1,0,0,0,0],[1,1,0,0,0,0,0,1],
[1,1,0,0,0,0,1,0],[1,1,0,0,0,1,0,0],[1,1,0,0,1,0,0,0],[1,1,0,1,0,0,0,0],
[1,1,1,0,0,0,0,0],[0,0,0,0,1,1,1,1],[0,0,0,1,0,1,1,1],[0,0,0,1,1,0,1,1],
[0,0,0,1,1,1,0,1],[0,0,0,1,1,1,1,0],[0,0,1,0,0,1,1,1],[0,0,1,0,1,0,1,1],
[0,0,1,0,1,1,0,1],[0,0,1,0,1,1,1,0],[0,0,1,1,0,0,1,1],[0,0,1,1,0,1,0,1],
[0,0,1,1,0,1,1,0],[0,0,1,1,1,0,0,1],[0,0,1,1,1,0,1,0],[0,0,1,1,1,1,0,0],
[0,1,0,0,0,1,1,1],[0,1,0,0,1,0,1,1],[0,1,0,0,1,1,0,1],[0,1,0,0,1,1,1,0],
[0,1,0,1,0,0,1,1],[0,1,0,1,0,1,0,1],[0,1,0,1,0,1,1,0],[0,1,0,1,1,0,0,1],
[0,1,0,1,1,0,1,0],[0,1,0,1,1,1,0,0],[0,1,1,0,0,0,1,1],[0,1,1,0,0,1,0,1],
[0,1,1,0,0,1,1,0],[0,1,1,0,1,0,0,1],[0,1,1,0,1,0,1,0],[0,1,1,0,1,1,0,0],
[0,1,1,1,0,0,0,1],[0,1,1,1,0,0,1,0],[0,1,1,1,0,1,0,0],[0,1,1,1,1,0,0,0],
[1,0,0,0,0,1,1,1],[1,0,0,0,1,0,1,1],[1,0,0,0,1,1,0,1],[1,0,0,0,1,1,1,0],
[1,0,0,1,0,0,1,1],[1,0,0,1,0,1,0,1],[1,0,0,1,0,1,1,0],[1,0,0,1,1,0,0,1],
[1,0,0,1,1,0,1,0],[1,0,0,1,1,1,0,0],[1,0,1,0,0,0,1,1],[1,0,1,0,0,1,0,1],
[1,0,1,0,0,1,1,0],[1,0,1,0,1,0,0,1],[1,0,1,0,1,0,1,0],[1,0,1,0,1,1,0,0],
[1,0,1,1,0,0,0,1],[1,0,1,1,0,0,1,0],[1,0,1,1,0,1,0,0],[1,0,1,1,1,0,0,0],
[1,1,0,0,0,0,1,1],[1,1,0,0,0,1,0,1],[1,1,0,0,0,1,1,0],[1,1,0,0,1,0,0,1],
[1,1,0,0,1,0,1,0],[1,1,0,0,1,1,0,0],[1,1,0,1,0,0,0,1],[1,1,0,1,0,0,1,0],
[1,1,0,1,0,1,0,0],[1,1,0,1,1,0,0,0],[1,1,1,0,0,0,0,1],[1,1,1,0,0,0,1,0],
[1,1,1,0,0,1,0,0],[1,1,1,0,1,0,0,0],[1,1,1,1,0,0,0,0],[0,0,0,1,1,1,1,1],
[0,0,1,0,1,1,1,1],[0,0,1,1,0,1,1,1],[0,0,1,1,1,0,1,1],[0,0,1,1,1,1,0,1],
[0,0,1,1,1,1,1,0],[0,1,0,0,1,1,1,1],[0,1,0,1,0,1,1,1],[0,1,0,1,1,0,1,1],
[0,1,0,1,1,1,0,1],[0,1,0,1,1,1,1,0],[0,1,1,0,0,1,1,1],[0,1,1,0,1,0,1,1],
[0,1,1,0,1,1,0,1],[0,1,1,0,1,1,1,0],[0,1,1,1,0,0,1,1],[0,1,1,1,0,1,0,1],
[0,1,1,1,0,1,1,0],[0,1,1,1,1,0,0,1],[0,1,1,1,1,0,1,0],[0,1,1,1,1,1,0,0],
[1,0,0,0,1,1,1,1],[1,0,0,1,0,1,1,1],[1,0,0,1,1,0,1,1],[1,0,0,1,1,1,0,1],
[1,0,0,1,1,1,1,0],[1,0,1,0,0,1,1,1],[1,0,1,0,1,0,1,1],[1,0,1,0,1,1,0,1],
[1,0,1,0,1,1,1,0],[1,0,1,1,0,0,1,1],[1,0,1,1,0,1,0,1],[1,0,1,1,0,1,1,0],
[1,0,1,1,1,0,0,1],[1,0,1,1,1,0,1,0],[1,0,1,1,1,1,0,0],[1,1,0,0,0,1,1,1],
[1,1,0,0,1,0,1,1],[1,1,0,0,1,1,0,1],[1,1,0,0,1,1,1,0],[1,1,0,1,0,0,1,1],
[1,1,0,1,0,1,0,1],[1,1,0,1,0,1,1,0],[1,1,0,1,1,0,0,1],[1,1,0,1,1,0,1,0],
[1,1,0,1,1,1,0,0],[1,1,1,0,0,0,1,1],[1,1,1,0,0,1,0,1],[1,1,1,0,0,1,1,0],
[1,1,1,0,1,0,0,1],[1,1,1,0,1,0,1,0],[1,1,1,0,1,1,0,0],[1,1,1,1,0,0,0,1],
[1,1,1,1,0,0,1,0],[1,1,1,1,0,1,0,0],[1,1,1,1,1,0,0,0],[0,0,1,1,1,1,1,1],
[0,1,0,1,1,1,1,1],[0,1,1,0,1,1,1,1],[0,1,1,1,0,1,1,1],[0,1,1,1,1,0,1,1],
[0,1,1,1,1,1,0,1],[0,1,1,1,1,1,1,0],[1,0,0,1,1,1,1,1],[1,0,1,0,1,1,1,1],
[1,0,1,1,0,1,1,1],[1,0,1,1,1,0,1,1],[1,0,1,1,1,1,0,1],[1,0,1,1,1,1,1,0],
[1,1,0,0,1,1,1,1],[1,1,0,1,0,1,1,1],[1,1,0,1,1,0,1,1],[1,1,0,1,1,1,0,1],
[1,1,0,1,1,1,1,0],[1,1,1,0,0,1,1,1],[1,1,1,0,1,0,1,1],[1,1,1,0,1,1,0,1],
[1,1,1,0,1,1,1,0],[1,1,1,1,0,0,1,1],[1,1,1,1,0,1,0,1],[1,1,1,1,0,1,1,0],
[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,1,0],[1,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,1],
[1,0,1,1,1,1,1,1],[1,1,0,1,1,1,1,1],[1,1,1,0,1,1,1,1],[1,1,1,1,0,1,1,1],
[1,1,1,1,1,0,1,1],[1,1,1,1,1,1,0,1],[1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1],
])

EPS_POS = 1e-9   # threshold for positive real part

def build_C(circ_idx, params):
    """Build coupling matrix for given circuit and parameter vector."""
    ev = CIRCUITS[circ_idx - 1]
    active = np.where(ev)[0]
    C = C_FIXED.copy()
    for k, aidx in enumerate(active):
        tgt, src = EDGE_MAP[aidx]
        C[tgt, src] += params[k]
    return C

def inner_and_grad(x, C):
    """Polynomial inner product and its Jacobian."""
    Kv = K[np.newaxis, :]
    xK = x / Kv
    xK2 = xK ** 2
    d = 1.0 + xK2
    sig = xK2 / d
    I = np.einsum('...ij,...j->...i', C, sig)
    inn = x * (I - r[np.newaxis, :])
    # Gradient w.r.t. x (diagonal terms only needed for Newton)
    dsig = 2.0 * xK / (d ** 2) / Kv
    dI = np.einsum('...ij,...j->...i', C, dsig)
    grd_diag = I - r[np.newaxis, :] + x * dI
    # Full Jacobian (off-diagonal from C * dsig contribution)
    grd = np.einsum('...i,...ij,...j->...ij', x, C, dsig)
    for i in range(4):
        grd[..., i, i] = grd_diag[..., i]
    return inn, grd

def classify_state(evals_re):
    n_pos = int(np.sum(evals_re > EPS_POS))
    if n_pos == 0:   return "stable"
    if n_pos == 4:   return "unstable"
    return f"semi{n_pos}"

# ── Main ─────────────────────────────────────────────────────────────────────
circ_idx = int(sys.argv[1])
N        = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

outdir = Path("~/circuit_hpc/results_evec").expanduser()
outdir.mkdir(exist_ok=True)
outpath = outdir / f"circuit_{circ_idx:03d}_evec.csv"

rng    = np.random.default_rng(circ_idx * 999983)
ev     = CIRCUITS[circ_idx - 1]
active = np.where(ev)[0]
n_act  = len(active)

SECTORS = [f"{i:04b}" for i in range(16)]
N_GRID  = {0: 1, 1: 8, 2: 12, 3: 16, 4: 8}   # k=4: 8^4=4096 starts (was 20^4=160k, too slow)
TOL     = 1e-8
MAXITER = 60

with open(outpath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_idx","state","classification","eval_idx",
                "e_re","e_im",
                "vF","vM","vT","vB",
                "vF_im","vM_im","vT_im","vB_im"])

    for sidx in range(1, N + 1):
        params = rng.uniform(0, 5, size=n_act) if n_act > 0 else np.array([])
        C = build_C(circ_idx, params)

        for sec in SECTORS:
            pres = [i for i, b in enumerate(sec) if b == "1"]
            ab   = [i for i, b in enumerate(sec) if b == "0"]
            k    = len(pres)
            if k == 0:
                # Zero state — eigenvalues are -r, eigenvectors are e_i
                evals = -r
                evecs = np.eye(4, dtype=complex)
                cls   = "stable"
                for ei in range(4):
                    w.writerow([sidx, sec, cls, ei+1,
                                float(evals[ei].real), 0.0,
                                float(evecs[0,ei].real), float(evecs[1,ei].real),
                                float(evecs[2,ei].real), float(evecs[3,ei].real),
                                0.0, 0.0, 0.0, 0.0])
                continue

            ng = N_GRID.get(k, 8)
            # Grid starts in this sector
            g  = np.linspace(0.1, 4.9, ng)
            starts = np.array(np.meshgrid(*([g]*k), indexing='ij')
                              ).reshape(k, -1).T
            S = starts.shape[0]

            x = np.zeros((S, 4))
            x[:, pres] = starts

            # Newton iteration
            C_rep = np.broadcast_to(C[np.newaxis], (S, 4, 4)).copy()
            for _ in range(MAXITER):
                x[:, ab] = 0.0
                inn, grd = inner_and_grad(x, C_rep)
                res = inn[:, pres]
                if np.max(np.abs(res)) < TOL:
                    break
                J = grd[:, pres, :][:, :, pres] + np.eye(k) * 1e-14
                try:
                    dx = np.linalg.solve(J, -res[..., np.newaxis]).squeeze(-1)
                except np.linalg.LinAlgError:
                    break
                x[:, pres] = np.clip(x[:, pres] + dx, -100, 100)
            x[:, ab] = 0.0

            inn_f, _ = inner_and_grad(x, C_rep)
            converged = np.max(np.abs(inn_f[:, pres]), axis=1) < TOL
            in_range  = np.all((x[:, pres] > 1e-6) & (x[:, pres] < 20), axis=1)
            valid     = converged & in_range

            # Deduplicate
            seen = []
            for xi in x[valid]:
                if not any(np.max(np.abs(xi - s)) < 1e-5 for s in seen):
                    seen.append(xi)

            for xi in seen:
                # Full 4×4 Jacobian at this steady state
                inn_i, J_full = inner_and_grad(xi[np.newaxis], C[np.newaxis])
                J4 = J_full[0]   # shape (4,4)
                try:
                    evals_c, evecs_c = np.linalg.eig(J4)
                except np.linalg.LinAlgError:
                    continue
                # Sort by real part descending (most unstable first)
                order = np.argsort(-evals_c.real)
                evals_c = evals_c[order]
                evecs_c = evecs_c[:, order]

                cls = classify_state(evals_c.real)
                for ei in range(4):
                    v = evecs_c[:, ei]
                    w.writerow([sidx, sec, cls, ei+1,
                                float(evals_c[ei].real), float(evals_c[ei].imag),
                                float(v[0].real), float(v[1].real),
                                float(v[2].real), float(v[3].real),
                                float(v[0].imag), float(v[1].imag),
                                float(v[2].imag), float(v[3].imag)])

print(f"Done: {outpath}")
