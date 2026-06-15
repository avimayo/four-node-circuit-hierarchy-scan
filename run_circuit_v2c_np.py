#!/usr/bin/env python3
"""
run_circuit_v2c_np.py — NumPy port of run_circuit_v2c.wls

Solves the degree-2 polynomial steady-state system for all 16 sectors via
batched Newton's method (grid of starts), then classifies each solution by
the 4x4 ODE Jacobian eigenvalues.

Same CLI and output as run_circuit_v2c.wls:
  results_v2c/circuit_XXX_chunkKK_r<range>_joint.csv
  results_v2c/circuit_XXX_chunkKK_r<range>_evals.csv

~50-100x faster than Mathematica on CPU.  Optional GPU: install torch.

Usage:
  python3 run_circuit_v2c_np.py <circIdx> [rangeMin rangeMax [nSamples [chunkIdx]]]
"""

import sys, os
import numpy as np
from pathlib import Path
from itertools import combinations

# Torch/GPU backend — set via run_circuit(device=...) or --device flag
_DEVICE = None   # None → pure NumPy;  'mps' or 'cuda' → torch on that device

# ── Fixed kinetic parameters ─────────────────────────────────────────────────
Kb, Kf, Km, Kt = 2.0, 2.9, 4.7, 2.0
pff, pfm       = 1.49, 1.1       # self-F, F→M
pmf, pmm       = 1.7,  1.76      # M→F, self-M
ptt            = 2.37            # self-T
ptb            = 2.5             # T→B (in I_B)
pbt            = 1.04            # B→T inhibition (separate term in inner_T)
pbb            = 1.7             # B self-inhibition (separate term in inner_B)
rf, rm, rt, rb = 0.75, 2.5, 0.23, 1.5

K = np.array([Kf, Km, Kt, Kb], float)   # F=0 M=1 T=2 B=3
r = np.array([rf, rm, rt, rb], float)

# Fixed part of coupling matrix C_FIXED[target, source]
C_FIXED = np.array([
    [pff, pmf, 0.0, 0.0],   # I_F: pff*F + pmf*M
    [pfm, pmm, 0.0, 0.0],   # I_M: pfm*F + pmm*M
    [0.0, 0.0, ptt, 0.0],   # I_T: ptt*T
    [0.0, 0.0, ptb, 0.0],   # I_B: ptb*T
], float)

# Free edge parameters in edgeVec order: [Pft, Ptf, Pmt, Ptm, Pfb, Pbf, Pmb, Pbm]
# EDGE_MAP[k] = (target_row, source_col) in C matrix
EDGE_MAP = [(2,0),(0,2),(2,1),(1,2),(3,0),(0,3),(3,1),(1,3)]

# ── Circuit data (256 combinatorial vectors from circuits.wl) ─────────────────
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
], dtype=np.int8)

# ── Core math ─────────────────────────────────────────────────────────────────

def build_C_free(P_full):
    """P_full (N,8) → C_free (N,4,4)"""
    N = P_full.shape[0]
    C = np.zeros((N, 4, 4))
    for k, (tgt, src) in enumerate(EDGE_MAP):
        C[:, tgt, src] = P_full[:, k]
    return C


def inner_and_grad(x, C_tot):
    """
    x:     (..., 4)  state [F,M,T,B]
    C_tot: (..., 4, 4)  full coupling = C_FIXED + C_free (broadcast over batch dims)

    Returns
    -------
    inner (..., 4)      inner-expression residuals
    grad  (..., 4, 4)   d(inner_i)/d(x_j)  [Newton / ODE Jacobian rows]
    """
    I = (C_tot @ x[..., np.newaxis]).squeeze(-1)         # (...,4)
    omxK = 1.0 - x / K                                    # (...,4)  1 - x_i/K_i

    inner = I * omxK - r                                   # (...,4)
    inner[..., 2] -= pbt * x[..., 3]                       # inner_T -= pbt*B
    inner[..., 3] -= pbb * x[..., 3]                       # inner_B -= pbb*B

    # grad[..., i, j] = C_tot[i,j] * (1 - x_i/K_i)
    grad = C_tot * omxK[..., :, np.newaxis]                # (...,4,4)
    # diagonal correction: grad[i,i] -= I_i / K_i
    grad[..., 0, 0] -= I[..., 0] / Kf
    grad[..., 1, 1] -= I[..., 1] / Km
    grad[..., 2, 2] -= I[..., 2] / Kt
    grad[..., 3, 3] -= I[..., 3] / Kb
    # extra off-diagonal terms from non-logistic parts
    grad[..., 2, 3] -= pbt     # d(inner_T)/dB
    grad[..., 3, 3] -= pbb     # d(inner_B)/dB  (extra self-inhib)

    return inner, grad


_NEWTON_CHUNK = 5000   # max samples per batch; bounds peak RAM to ~400 MB


def _numpy_newton(C_batch, pres, ab, starts, k, S, tol, max_iter):
    """NumPy Newton solver for one chunk of N samples."""
    N_b = C_batch.shape[0]
    NS  = N_b * S
    x   = np.zeros((N_b, S, 4))
    x[:, :, pres] = starts[np.newaxis, :, :]
    C_flat = np.broadcast_to(
        C_batch[:, np.newaxis, :, :], (N_b, S, 4, 4)
    ).reshape(NS, 4, 4).copy()
    x_flat = x.reshape(NS, 4)

    for _ in range(max_iter):
        x_flat[:, ab] = 0.0
        inn, grd = inner_and_grad(x_flat, C_flat)
        res = inn[:, pres]
        if np.max(np.abs(res)) < tol:
            break
        J  = grd[:, pres, :][:, :, pres]
        J += np.eye(k) * 1e-14
        try:
            dx = np.linalg.solve(J, -res[..., np.newaxis]).squeeze(-1)
        except np.linalg.LinAlgError:
            break
        x_flat[:, pres] += dx
        x_flat[:, pres]  = np.clip(x_flat[:, pres], -100, 100)

    x_flat[:, ab] = 0.0
    inn, _ = inner_and_grad(x_flat, C_flat)
    res_f  = inn[:, pres].reshape(N_b, S, k)
    x_out  = x_flat.reshape(N_b, S, 4)
    converged = np.max(np.abs(res_f), axis=-1) < tol
    in_range  = np.all((x_out[:, :, pres] > 1e-6) &
                       (x_out[:, :, pres] < 20), axis=-1)
    return x_out, converged & in_range


def sector_newton(C_tot, pres, n_grid=5, x_hi=19.9, tol=1e-8, max_iter=50):
    """
    Find all positive steady-state solutions for one sector across N samples.

    C_tot: (N, 4, 4)  full coupling matrices
    pres:  list of k variable indices present in this sector

    Returns x_out (N, n_starts, 4) with absent vars=0,
            valid  (N, n_starts)   bool: converged AND all present vars in (1e-6,20)

    Processes samples in chunks of _NEWTON_CHUNK to bound peak RAM usage.
    """
    N  = C_tot.shape[0]
    k  = len(pres)
    ab = [i for i in range(4) if i not in pres]

    pts1d = np.linspace(0.1, x_hi, n_grid)
    grids = np.meshgrid(*([pts1d] * k), indexing='ij')
    starts = np.stack([g.ravel() for g in grids], axis=-1)   # (S, k)
    S = starts.shape[0]
    NS = N * S

    if _DEVICE is not None:
        try:
            return _sector_newton_torch(C_tot, pres, ab, starts, N, S, NS, k, tol, max_iter)
        except Exception:
            pass  # fall through to NumPy path

    return _numpy_newton(C_tot, pres, ab, starts, k, S, tol, max_iter)


def _sector_newton_torch(C_tot, pres, ab, starts, N, S, NS, k, tol, max_iter):
    """Torch/GPU Newton solver — called by sector_newton when _DEVICE is set."""
    import torch
    dev   = _DEVICE
    # MPS only supports float32; CUDA and CPU support float64
    dtype = torch.float32 if (isinstance(dev, str) and dev == 'mps') else torch.float64
    tol   = max(tol, 1e-6)   # float32 machine eps ~1.2e-7; don't ask for more

    x = torch.zeros((N, S, 4), device=dev, dtype=dtype)
    x[:, :, pres] = torch.tensor(starts, device=dev, dtype=dtype)

    C_t    = torch.tensor(C_tot, device=dev, dtype=dtype)      # (N,4,4)
    C_flat = C_t.unsqueeze(1).expand(N, S, 4, 4).reshape(NS, 4, 4).contiguous()
    x_flat = x.reshape(NS, 4)

    K_t   = torch.tensor(K,   device=dev, dtype=dtype)         # (4,)
    r_t   = torch.tensor(r,   device=dev, dtype=dtype)
    eye_k = torch.eye(k, device=dev, dtype=dtype) * 1e-14

    for _ in range(max_iter):
        if ab:
            x_flat[:, ab] = 0.0

        # inner expressions and gradient (inlined for torch)
        I    = (C_flat @ x_flat.unsqueeze(-1)).squeeze(-1)     # (NS,4)
        omxK = 1.0 - x_flat / K_t                              # (NS,4)

        inn = I * omxK - r_t
        inn[:, 2] = inn[:, 2] - pbt * x_flat[:, 3]
        inn[:, 3] = inn[:, 3] - pbb * x_flat[:, 3]

        res = inn[:, pres]                                      # (NS,k)
        if res.abs().max().item() < tol:
            break

        grd = C_flat * omxK.unsqueeze(-1)                      # (NS,4,4)
        grd.diagonal(dim1=-2, dim2=-1).sub_(I / K_t)
        grd[:, 2, 3] -= pbt
        grd[:, 3, 3] -= pbb

        J = grd[:, pres, :][:, :, pres] + eye_k                # (NS,k,k)
        dx = torch.linalg.solve(J, -res.unsqueeze(-1)).squeeze(-1)
        x_flat[:, pres] += dx
        x_flat[:, pres]  = x_flat[:, pres].clamp(-100, 100)

    if ab:
        x_flat[:, ab] = 0.0
    I    = (C_flat @ x_flat.unsqueeze(-1)).squeeze(-1)
    omxK = 1.0 - x_flat / K_t
    inn  = I * omxK - r_t
    inn[:, 2] = inn[:, 2] - pbt * x_flat[:, 3]
    inn[:, 3] = inn[:, 3] - pbb * x_flat[:, 3]

    res_f = inn[:, pres].reshape(N, S, k)
    x_out = x_flat.reshape(N, S, 4).detach().cpu().numpy()

    converged = res_f.abs().amax(dim=-1).detach().cpu().numpy() < tol
    in_range  = np.all((x_out[:, :, pres] > 1e-6) &
                       (x_out[:, :, pres] < 20), axis=-1)
    return x_out, converged & in_range


def find_all_steady_states(C_tot, n_grids=None):
    """
    Returns list of N lists; each inner list has 1-D (4,) solution arrays.
    The zero state is always included.

    Optimisation: sectors whose equations contain no free-parameter terms
    (all inter-layer edges cancel because absent variables are 0) have
    solutions that are identical for every sample.  Solved once, broadcast.

    Free-param edges only appear when BOTH endpoint variables are present:
      Ptf→F(0), Pbf→F(0), Ptm→M(1), Pbm→M(1),
      Pft→T(2), Pmt→T(2), Pfb→B(3), Pmb→B(3)
    A sector has free params iff it contains at least one node from {F,M}
    AND at least one node from {T,B}.
    """
    N  = C_tot.shape[0]
    ng = n_grids or {1: 12, 2: 7, 3: 5, 4: 3}

    FM = {0, 1}   # F, M
    TB = {2, 3}   # T, B

    sol_stacks = [np.zeros((1, 4))] * N       # start with zero state (shared ref is fine)

    for k in range(1, 5):
        for pres in combinations(range(4), k):
            pres = list(pres)
            pres_set = set(pres)

            has_free_params = bool(pres_set & FM) and bool(pres_set & TB)

            if not has_free_params:
                # Solutions are the same for all N samples — solve once with N=1
                x_one, valid_one = sector_newton(C_tot[0:1], pres, n_grid=ng[k])
                cands_one = x_one[0, valid_one[0], :]   # (nc, 4)
                if cands_one.shape[0] == 0:
                    continue
                # Broadcast to all N samples
                for n in range(N):
                    stk = sol_stacks[n]
                    for cand in cands_one:
                        if np.max(np.abs(stk - cand), axis=1).min() >= 1e-6:
                            stk = np.vstack([stk, cand])
                    sol_stacks[n] = stk
            else:
                # Process in chunks so peak RAM = O(chunk × S) not O(N × S)
                for i in range(0, N, _NEWTON_CHUNK):
                    x_chunk, v_chunk = sector_newton(
                        C_tot[i:i+_NEWTON_CHUNK], pres, n_grid=ng[k])
                    for j in range(x_chunk.shape[0]):
                        cands = x_chunk[j, v_chunk[j], :]
                        if cands.shape[0] == 0:
                            continue
                        stk = sol_stacks[i + j]
                        for cand in cands:
                            if np.max(np.abs(stk - cand), axis=1).min() >= 1e-6:
                                stk = np.vstack([stk, cand])
                        sol_stacks[i + j] = stk

    return [list(stk) for stk in sol_stacks]


def ode_jacobian_batch(sols_flat, C_tot_flat):
    """
    sols_flat:   (M, 4)  steady-state solutions
    C_tot_flat:  (M, 4, 4)

    Returns J_ode (M, 4, 4):
      J[i,j] = x_i * d(inner_i)/d(x_j)  +  delta_ij * inner_i
    """
    inner, grad = inner_and_grad(sols_flat, C_tot_flat)
    # outer product: x_i * grad_inner[i,j]
    J = sols_flat[:, :, np.newaxis] * grad                    # (M, 4, 4)
    # add diagonal: inner_i on J[i,i]
    idx = np.arange(4)
    J[:, idx, idx] += inner
    return J


# ── Main ──────────────────────────────────────────────────────────────────────

def type_label(sol):
    return ''.join('1' if v > 1e-6 else '0' for v in sol)


def classify(sol, J_ode):
    """stable / semistable / unstable from pre-computed 4x4 ODE Jacobian."""
    pres = [i for i in range(4) if sol[i] > 1e-6]
    ab   = [i for i in range(4) if i not in pres]
    if len(pres) == 4:
        ev = np.linalg.eigvals(J_ode)
        return 'stable' if np.max(ev.real) <= 0 else 'unstable'
    pi = pres or []
    tangent_ev  = np.linalg.eigvals(J_ode[np.ix_(pi, pi)]) if pi else np.array([])
    invasion_ev = J_ode[ab, ab]                               # diagonal
    if len(tangent_ev) > 0 and np.max(tangent_ev.real) > 0:
        return 'unstable'
    if len(invasion_ev) > 0 and np.max(invasion_ev.real) > 0:
        return 'semistable'
    return 'stable'


def run_circuit(circ_idx, range_min=0.0, range_max=5.0,
                n_samples=1000, chunk_idx=1, out_dir='results_v2c', device=None):

    edge_vec = CIRCUITS[circ_idx - 1]                         # 1-indexed
    active   = np.where(edge_vec)[0]
    n_active = len(active)

    seed = circ_idx * 100000 + chunk_idx * 1000 + int(round(range_min * 100))
    rng  = np.random.default_rng(seed)
    P_raw = rng.uniform(range_min, range_max, (n_samples, n_active))

    P_full = np.zeros((n_samples, 8))
    P_full[:, active] = P_raw

    C_free = build_C_free(P_full)                             # (N,4,4)
    C_tot  = C_free + C_FIXED                                 # broadcast

    global _DEVICE
    _DEVICE = device
    if device is not None:
        try:
            import torch
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError("MPS not available")
            # Probe: try allocating a small tensor to catch OOM early
            _ = torch.zeros(1, device=device, dtype=torch.float32)
        except Exception as e:
            print(f"WARNING: torch device '{device}' unavailable ({e}), falling back to CPU")
            _DEVICE = None

    dev_label = f"device={_DEVICE}" if _DEVICE else "cpu/numpy"
    print(f"circuit {circ_idx}  chunk={chunk_idx}  "
          f"range=[{range_min},{range_max}]  nSamples={n_samples}  [{dev_label}]")

    all_sols = find_all_steady_states(C_tot)

    # ── Collect rows ─────────────────────────────────────────────────────────
    joint_rows = []
    eval_rows  = []

    for n in range(n_samples):
        sols_n = all_sols[n]
        if not sols_n:
            sols_n = [np.zeros(4)]

        sols_arr   = np.stack(sols_n)                          # (nsol, 4)
        C_rep      = np.tile(C_tot[n], (len(sols_n), 1, 1))   # (nsol,4,4)
        J_all      = ode_jacobian_batch(sols_arr, C_rep)       # (nsol,4,4)
        evals_all  = np.linalg.eigvals(J_all)                  # (nsol,4) complex

        stable_p, semi_p, unstable_p = [], [], []

        for si, (sol, J_ode, ev) in enumerate(zip(sols_n, J_all, evals_all)):
            label = type_label(sol)
            cls   = classify(sol, J_ode)
            # sort eigenvalues by Re descending
            ev_s  = ev[np.argsort(-ev.real)]
            eval_rows.append([
                n + 1, label, cls,
                ev_s[0].real, ev_s[0].imag,
                ev_s[1].real, ev_s[1].imag,
                ev_s[2].real, ev_s[2].imag,
                ev_s[3].real, ev_s[3].imag,
            ])
            if cls == 'stable':     stable_p.append(label)
            elif cls == 'semistable': semi_p.append(label)
            else:                   unstable_p.append(label)

        def fmt(lst):
            u = sorted(set(lst))
            return '|'.join(u) if u else 'none'

        joint_rows.append([n + 1, fmt(stable_p), fmt(semi_p), fmt(unstable_p)])

    # ── Write CSV ─────────────────────────────────────────────────────────────
    Path(out_dir).mkdir(exist_ok=True)
    def _rt(v):
        return str(int(v)) if v == int(v) else str(v).replace('.', 'p')
    tag  = f"{circ_idx:03d}"
    rtag = f"{_rt(range_min)}_{_rt(range_max)}"
    ctag = f"_chunk{chunk_idx:02d}"

    joint_file = f"{out_dir}/circuit_{tag}{ctag}_r{rtag}_joint.csv"
    eval_file  = f"{out_dir}/circuit_{tag}{ctag}_r{rtag}_evals.csv"

    import csv
    with open(joint_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'stable_pat', 'semi_pat', 'unstable_pat'])
        w.writerows(joint_rows)

    with open(eval_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx','state','classification',
                    'e1_re','e1_im','e2_re','e2_im',
                    'e3_re','e3_im','e4_re','e4_im'])
        w.writerows(eval_rows)

    print(f"Done — joint: {joint_file}  evals: {eval_file}  "
          f"({n_samples} samples, {len(eval_rows)} state records)")


def _run_one(job):
    """Worker for multiprocessing pool."""
    circ_idx, range_min, range_max, n_samples, chunk_idx, out_dir, device = job
    try:
        run_circuit(circ_idx, range_min, range_max, n_samples, chunk_idx, out_dir, device)
        return circ_idx, chunk_idx, True, ''
    except Exception as e:
        return circ_idx, chunk_idx, False, str(e)


if __name__ == '__main__':
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime, timedelta

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('circIdx', nargs='?', type=int,
                    help='single circuit index (1-256); omit with --all')
    ap.add_argument('rangeMin', nargs='?', type=float, default=0.0)
    ap.add_argument('rangeMax', nargs='?', type=float, default=5.0)
    ap.add_argument('nSamples', nargs='?', type=int,   default=1000)
    ap.add_argument('chunkIdx', nargs='?', type=int,   default=1)
    ap.add_argument('--all',     action='store_true',
                    help='run all 256 circuits (1 chunk each)')
    ap.add_argument('--workers', type=int, default=4,
                    help='parallel processes for --all mode (default: 4)')
    ap.add_argument('--out',     default='results_v2c',
                    help='output directory (default: results_v2c)')
    ap.add_argument('--device',  default=None,
                    help='torch device: mps, cuda, cpu (default: numpy/cpu)')
    args = ap.parse_args()

    if args.all:
        jobs = [
            (c, args.rangeMin, args.rangeMax, args.nSamples, 1, args.out, args.device)
            for c in range(1, 257)
        ]
        total = len(jobs)
        done_n, failed = 0, []
        t0 = datetime.now()
        print(f"Running {total} circuits with {args.workers} workers …")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, j): j for j in jobs}
            for fut in as_completed(futs):
                ci, ki, ok, err = fut.result()
                done_n += 1
                elapsed = (datetime.now() - t0).total_seconds()
                rate    = done_n / elapsed if elapsed > 0 else 1e-9
                eta     = int((total - done_n) / rate)
                sym     = '✓' if ok else '✗'
                print(f"[{done_n:3d}/{total}] {sym} circuit {ci:3d}  "
                      f"elapsed {str(timedelta(seconds=int(elapsed)))}  "
                      f"ETA {timedelta(seconds=eta)}", flush=True)
                if not ok:
                    failed.append((ci, err))
        print(f"\nDone. Succeeded: {total - len(failed)}  Failed: {len(failed)}")
        if failed:
            for ci, err in failed:
                print(f"  circuit {ci}: {err}")
    else:
        if args.circIdx is None:
            ap.print_help(); sys.exit(1)
        run_circuit(args.circIdx, args.rangeMin, args.rangeMax,
                    args.nSamples, args.chunkIdx, args.out, args.device)
