"""
compute_boundary_edges.py
=========================
Analytic boundary-state analysis for all 256 four-node (F,M,T,B) circuit topologies.

The core idea: several low-dimensional faces of the state space host fixed points
whose POSITIONS are parameter-independent (they depend only on the fixed kinetic
constants). Only the INVASION eigenvalues that control whether the system escapes
from these faces depend on the free edge-weight parameters.

Fixed points analysed here:
  * FM face  (T=B=0): FM_saddle, FM_stable
  * TB face  (F=M=0): TB_stable
  * F axis   (M=T=B=0): F_saddle, F_stable
  * T axis   (F=M=B=0): T_saddle, T_stable

Connections added:
  TB_stable  → FTB  : λ_F > 0  (needs Ptf, Pbf edges)
  TB_stable  → MTB  : λ_M > 0  (needs Ptm, Pbm edges)
  FM_stable  → FMT  : λ_T > 0  (needs Pft, Pmt edges)
  FM_stable  → FMB  : λ_B > 0  (needs Pfb, Pmb edges)
  F_stable   → FT   : λ_T > 0  (needs Pft edge)
  F_stable   → FB   : λ_B > 0  (needs Pfb edge)

Each confidence is estimated by Monte Carlo (200 000 draws from Uniform[0,5]).

Outputs:
  boundary_edges.csv               — new edges only
  heteroclinic_edges_augmented.csv — original + new, deduped on (circuit_idx, src, tgt)
"""

import re
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------
# Fixed kinetic parameters (from definitions.wl)
# ---------------------------------------------------------------------------
Kf = 2.9;  Km = 4.7;  Kt = 2.0;  Kb = 2.0
pff = 1.49; pmf = 1.7;  pfm = 1.1;  pmm = 1.76
ptt = 2.37; pbt = 1.04; ptb = 2.5;  pbb = 1.7
rf  = 0.75; rm  = 2.5;  rt  = 0.23; rb  = 1.5

N_MC = 200_000   # Monte Carlo samples
RNG  = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Edge vector ordering (from definitions.wl symbolEdges)
# index: 0=Pft, 1=Ptf, 2=Pmt, 3=Ptm, 4=Pfb, 5=Pbf, 6=Pmb, 7=Pbm
# ---------------------------------------------------------------------------
IDX_PFT = 0   # F→T
IDX_PTF = 1   # T→F
IDX_PMT = 2   # M→T
IDX_PTM = 3   # T→M
IDX_PFB = 4   # F→B
IDX_PBF = 5   # B→F
IDX_PMB = 6   # M→B
IDX_PBM = 7   # B→M

# ---------------------------------------------------------------------------
# State labels — 4-bit FMTB strings  (F=bit3, M=bit2, T=bit1, B=bit0)
# ---------------------------------------------------------------------------
EMPTY = "0000"
F_ST  = "1000"
M_ST  = "0100"
T_ST  = "0010"
B_ST  = "0001"
FM_ST = "1100"
FT_ST = "1010"
FB_ST = "1001"
MT_ST = "0110"
MB_ST = "0101"
TB_ST = "0011"
FMT_ST= "1110"
FMB_ST= "1101"
FTB_ST= "1011"
MTB_ST= "0111"
FMTB  = "1111"


def hamming(s1: str, s2: str) -> int:
    """Hamming distance between two 4-bit FMTB strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


# ---------------------------------------------------------------------------
# Parse circuits.wl
# ---------------------------------------------------------------------------
def parse_circuits(path: str) -> np.ndarray:
    """Return (256, 8) integer array of edge-presence vectors."""
    with open(path) as fh:
        text = fh.read()
    # Strip outer wrapper and extract the nested list
    inner = re.search(r"combinatorialVectors\s*=\s*\{(.+)\};", text, re.DOTALL).group(1)
    rows = re.findall(r"\{([0-9,\s]+)\}", inner)
    vectors = []
    for row in rows:
        vectors.append([int(x.strip()) for x in row.split(",")])
    arr = np.array(vectors, dtype=int)
    assert arr.shape == (256, 8), f"Expected (256,8), got {arr.shape}"
    return arr


# ---------------------------------------------------------------------------
# Parameter-independent fixed points (computed ONCE)
# ---------------------------------------------------------------------------

def fm_rhs(fv, Pft=0, Ptf=0, Pmt=0, Ptm=0, Pfb=0, Pbf=0, Pmb=0, Pbm=0):
    """
    RHS for F and M when T=B=0.
    We solve for nonzero steady state, so divide out the F and M factors:
      (pff*F + pmf*M)*(1 - F/Kf) - rf = 0
      (pfm*F + pmm*M)*(1 - M/Km) - rm = 0
    (cross-group terms vanish because T=B=0, so Ptf*ptf*T = 0 etc.)
    """
    F, M = fv
    eq1 = (pff * F + pmf * M) * (1 - F / Kf) - rf
    eq2 = (pfm * F + pmm * M) * (1 - M / Km) - rm
    return [eq1, eq2]


def tb_rhs(tv):
    """
    RHS for T and B when F=M=0.
      ptt*T*(1-T/Kt) - pbt*B - rt = 0      [from T eq, dividing by T]
      ptb*T*(1-B/Kb) - pbb*B - rb = 0      [from B eq, dividing by B]
    (cross-group terms Pft*pft*F = 0 etc. vanish)
    """
    T, B = tv
    eq1 = ptt * T * (1 - T / Kt) - pbt * B - rt
    eq2 = ptb * T * (1 - B / Kb) - pbb * B - rb
    return [eq1, eq2]


def f_axis_rhs(fv):
    """RHS for F when M=T=B=0:  pff*F*(1-F/Kf) - rf = 0  → pff*(1-F/Kf) - rf/F = 0"""
    F = fv[0]
    return [pff * F * (1 - F / Kf) - rf]


def t_axis_rhs(tv):
    """RHS for T when F=M=B=0:  ptt*T*(1-T/Kt) - rt = 0"""
    T = tv[0]
    return [ptt * T * (1 - T / Kt) - rt]


def compute_fixed_points():
    """
    Find all parameter-independent fixed points needed for boundary analysis.
    Returns a dict with entries for each face.
    """
    print("Computing parameter-independent fixed points...")

    # --- FM face: two non-trivial fixed points ---
    # Saddle (low F, low M near origin)
    FM_saddle = fsolve(fm_rhs, [0.65, 0.05], full_output=False)
    assert all(FM_saddle > 0), f"FM_saddle non-positive: {FM_saddle}"

    # Stable (high F, high M)
    FM_stable = fsolve(fm_rhs, [2.5, 3.0], full_output=False)
    assert all(FM_stable > 0), f"FM_stable non-positive: {FM_stable}"

    print(f"  FM_saddle: F={FM_saddle[0]:.4f}, M={FM_saddle[1]:.4f}")
    print(f"  FM_stable: F={FM_stable[0]:.4f}, M={FM_stable[1]:.4f}")

    # Verify residuals
    res_sad = fm_rhs(FM_saddle)
    res_sta = fm_rhs(FM_stable)
    assert max(abs(res_sad[0]), abs(res_sad[1])) < 1e-10, f"FM_saddle residual: {res_sad}"
    assert max(abs(res_sta[0]), abs(res_sta[1])) < 1e-10, f"FM_stable residual: {res_sta}"

    # --- TB face: one stable non-trivial fixed point ---
    TB_stable = fsolve(tb_rhs, [1.5, 0.6], full_output=False)
    assert all(TB_stable > 0), f"TB_stable non-positive: {TB_stable}"
    res_tb = tb_rhs(TB_stable)
    assert max(abs(res_tb[0]), abs(res_tb[1])) < 1e-10, f"TB_stable residual: {res_tb}"
    print(f"  TB_stable: T={TB_stable[0]:.4f}, B={TB_stable[1]:.4f}")

    # --- F axis: two non-trivial fixed points ---
    # Quadratic: pff*(1-F/Kf)*F - rf = 0 → pff*F - pff/Kf*F^2 - rf = 0
    # pff/Kf * F^2 - pff*F + rf = 0
    a = pff / Kf;  b_ = -pff;  c = rf
    disc = b_**2 - 4*a*c
    F_roots = [(-b_ + np.sqrt(disc)) / (2*a), (-b_ - np.sqrt(disc)) / (2*a)]
    F_roots = sorted([r for r in F_roots if r > 0])
    F_saddle_val = F_roots[0]   # smaller root = saddle (B always invades? — check separately)
    F_stable_val = F_roots[1]   # larger root = stable
    print(f"  F_saddle:  F={F_saddle_val:.4f}")
    print(f"  F_stable:  F={F_stable_val:.4f}")

    # --- T axis: two non-trivial fixed points ---
    # ptt*(1-T/Kt)*T - rt = 0 → ptt/Kt * T^2 - ptt*T + rt = 0
    a2 = ptt / Kt;  b2 = -ptt;  c2 = rt
    disc2 = b2**2 - 4*a2*c2
    T_roots = [(-b2 + np.sqrt(disc2)) / (2*a2), (-b2 - np.sqrt(disc2)) / (2*a2)]
    T_roots = sorted([r for r in T_roots if r > 0])
    T_saddle_val = T_roots[0]
    T_stable_val = T_roots[1]
    print(f"  T_saddle:  T={T_saddle_val:.4f}  (→ ∅ always; B always invades T_stable)")
    print(f"  T_stable:  T={T_stable_val:.4f}  (B invasion: λ_B = ptb*T* - rb = "
          f"{ptb * T_stable_val - rb:.4f} > 0 always)")

    return {
        "FM_saddle": FM_saddle,   # (F, M)
        "FM_stable": FM_stable,   # (F, M)
        "TB_stable": TB_stable,   # (T, B)
        "F_saddle":  F_saddle_val,
        "F_stable":  F_stable_val,
        "T_saddle":  T_saddle_val,
        "T_stable":  T_stable_val,
    }


# ---------------------------------------------------------------------------
# Monte Carlo invasion probability
# ---------------------------------------------------------------------------

def mc_invasion_prob(edge_vec: np.ndarray, param_indices: list[int],
                     lam_fn) -> float:
    """
    Estimate P(λ > 0) by Monte Carlo.

    Parameters
    ----------
    edge_vec     : (8,) array of 0/1 indicating which edges are present
    param_indices: list of edge indices that appear in the invasion eigenvalue
    lam_fn       : callable(params_array) → scalar λ, where params_array has
                   len(param_indices) columns (shape N × len(param_indices))

    Returns
    -------
    confidence ∈ [0, 1]
    """
    # Check if all required edges are present; if any is absent, P = 0
    for idx in param_indices:
        if edge_vec[idx] == 0:
            return 0.0

    # All required edges present — sample free parameters
    n_params = len(param_indices)
    samples = RNG.uniform(0, 5, size=(N_MC, n_params))
    lam_vals = lam_fn(samples)
    return float(np.mean(lam_vals > 0))


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(circuits_path: str, orig_csv_path: str,
                 out_csv_path: str, aug_csv_path: str):

    circuits = parse_circuits(circuits_path)   # (256, 8)
    fp = compute_fixed_points()

    F_FM, M_FM = fp["FM_stable"]
    T_TB, B_TB = fp["TB_stable"]
    F_star      = fp["F_stable"]
    T_stable_val = fp["T_stable"]  # noqa: F841 — used in 1D edge comment

    # Pre-draw Uniform[0,5] columns for each free parameter used
    # We do vectorised MC: draw all N_MC samples once per parameter index used.
    # Each invasion eigenvalue is linear in the free parameters, so:
    #   λ_F at TB = Ptf*ptf*T_TB + Pbf*pbf*B_TB - rf
    #   (ptf, pbf are fixed scalars; Ptf, Pbf ~ Unif[0,5])
    # Because the invasion eigenvalue is a *sum* of independent terms, we can
    # handle the "some edges absent" case by zeroing out their contribution.

    # Precompute all MC draws for each of the 8 parameter slots
    P_samples = RNG.uniform(0, 5, size=(N_MC, 8))  # column i = samples for edge i

    rows = []

    n_tb_f = 0   # circuits gaining TB→FTB
    n_tb_m = 0   # circuits gaining TB→MTB
    n_fm_t = 0   # circuits gaining FM→FMT
    n_fm_b = 0   # circuits gaining FM→FMB
    n_f_t  = 0   # circuits gaining F→FT
    n_f_b  = 0   # circuits gaining F→FB

    for circ_idx_0, ev in enumerate(circuits):
        circ_idx = circ_idx_0 + 1   # 1-based

        # --- Invasion eigenvalues (vectorised over N_MC parameter draws) ---
        # Each free parameter P_X uses its column from P_samples only if present;
        # if absent (ev[i]==0), its contribution is identically 0.

        Ptf_col = P_samples[:, IDX_PTF] * ev[IDX_PTF]
        Pbf_col = P_samples[:, IDX_PBF] * ev[IDX_PBF]
        Ptm_col = P_samples[:, IDX_PTM] * ev[IDX_PTM]
        Pbm_col = P_samples[:, IDX_PBM] * ev[IDX_PBM]
        Pft_col = P_samples[:, IDX_PFT] * ev[IDX_PFT]
        Pmt_col = P_samples[:, IDX_PMT] * ev[IDX_PMT]
        Pfb_col = P_samples[:, IDX_PFB] * ev[IDX_PFB]
        Pmb_col = P_samples[:, IDX_PMB] * ev[IDX_PMB]

        # ---- TB face invasions ----
        # λ_F at TB_stable = Ptf*ptf*T_TB + Pbf*pbf*B_TB - rf
        lam_F_TB = (Ptf_col * 1.0 * T_TB   # ptf=1.0 (lower indicator already in ev)
                    + Pbf_col * 1.0 * B_TB  # pbf=1.0
                    - rf)
        # Note: lower-case ptf and pbf from definitions.wl are the
        # *fixed* kinetic rates for the self/same-group interactions.
        # However, re-reading definitions.wl carefully:
        #   ẋ_F = F * ((Pbf*pbf*B + pff*F + pmf*M + Ptf*ptf*T)*(1-F/Kf) - rf)
        # Here Pbf and Ptf are the free uppercase edge weights (0=absent,5=max),
        # and pbf=1.04 [wait — check the actual values again from definitions.wl]
        # Actually from definitions.wl: pbt=1.04, ptb=2.5 — these are cross terms.
        # pbf and ptf are NOT listed in the fixed parameters, which means they are
        # the lowercase binary edge indicators (0 or 1) set by lowerRules!
        # Re-reading: lowerRules = {pft->ev[0], ptf->ev[1], ..., pbf->ev[5], pbm->ev[7]}
        # So lowercase ptf = ev[1] ∈ {0,1}, and Ptf = free Uniform[0,5].
        # The invasion eigenvalue λ_F = Ptf * ptf * T_TB + Pbf * pbf * B_TB - rf
        # where ptf=ev[IDX_PTF] and pbf=ev[IDX_PBF] are already baked into Ptf_col/Pbf_col.
        # So the formula above is correct — ptf and pbf multipliers are 1 (the binary
        # indicator is already folded in via ev[i] masking).

        # Recompute properly: the lowercase p values in the invasion eigenvalue ARE
        # the circuit edge indicators (0 or 1), not the fixed kinetic constants.
        # The fixed kinetic constants with lowercase p are only the self-loops:
        # pff, pmf, pfm, pmm, ptt, pbt, ptb, pbb, rf, rm, rt, rb
        # The cross-group terms use Pupper * plower where plower ∈ {0,1}.
        # Already handled above by multiplying P_samples[:, idx] * ev[idx].

        conf_TB_to_FTB = float(np.mean(lam_F_TB > 0))
        # Only meaningful if at least one of Ptf, Pbf is present
        has_tb_f = ev[IDX_PTF] or ev[IDX_PBF]
        if not has_tb_f:
            conf_TB_to_FTB = 0.0

        # λ_M at TB_stable = Ptm*ptm*T_TB + Pbm*pbm*B_TB - rm
        lam_M_TB = Ptm_col * T_TB + Pbm_col * B_TB - rm
        conf_TB_to_MTB = float(np.mean(lam_M_TB > 0))
        has_tb_m = ev[IDX_PTM] or ev[IDX_PBM]
        if not has_tb_m:
            conf_TB_to_MTB = 0.0

        # ---- FM face invasions ----
        # λ_T at FM_stable = Pft*pft*F_FM + Pmt*pmt*M_FM - rt
        lam_T_FM = Pft_col * F_FM + Pmt_col * M_FM - rt
        conf_FM_to_FMT = float(np.mean(lam_T_FM > 0))
        has_fm_t = ev[IDX_PFT] or ev[IDX_PMT]
        if not has_fm_t:
            conf_FM_to_FMT = 0.0

        # λ_B at FM_stable = Pfb*pfb*F_FM + Pmb*pmb*M_FM - rb
        lam_B_FM = Pfb_col * F_FM + Pmb_col * M_FM - rb
        conf_FM_to_FMB = float(np.mean(lam_B_FM > 0))
        has_fm_b = ev[IDX_PFB] or ev[IDX_PMB]
        if not has_fm_b:
            conf_FM_to_FMB = 0.0

        # ---- F axis invasions ----
        # λ_T at F_stable = Pft*pft*F_star - rt
        lam_T_F = Pft_col * F_star - rt
        conf_F_to_FT = float(np.mean(lam_T_F > 0))
        has_f_t = bool(ev[IDX_PFT])
        if not has_f_t:
            conf_F_to_FT = 0.0

        # λ_B at F_stable = Pfb*pfb*F_star - rb
        lam_B_F = Pfb_col * F_star - rb
        conf_F_to_FB = float(np.mean(lam_B_F > 0))
        has_f_b = bool(ev[IDX_PFB])
        if not has_f_b:
            conf_F_to_FB = 0.0

        # ---- Accumulate edges ----
        def add_edge(src, tgt, conf):
            if conf <= 0.0:
                return
            n_edge_samp = int(round(conf * N_MC))
            h = hamming(src, tgt)
            rows.append({
                "circuit_idx":   circ_idx,
                "src":           src,
                "tgt":           tgt,
                "n_src_samples": N_MC,
                "n_edge_samples":n_edge_samp,
                "confidence":    round(conf, 6),
                "dominant_type": "boundary_analytic",
                "hamming":       h,
            })

        # ---- 1D axis universal connections (exact, no MC needed) ----
        # F⁻ → ∅: F bistable saddle always flows to null state
        add_edge(F_ST, EMPTY, 1.0)
        # T⁻ → ∅: T bistable saddle always flows to null state
        add_edge(T_ST, EMPTY, 1.0)
        # T* → TB: B always invades T* (λ_B = ptb·T* − rb ≈ 3.24 > 0 for all circuits)
        add_edge(T_ST, TB_ST, 1.0)

        if conf_TB_to_FTB > 0:
            add_edge(TB_ST, FTB_ST, conf_TB_to_FTB)
            n_tb_f += 1
        if conf_TB_to_MTB > 0:
            add_edge(TB_ST, MTB_ST, conf_TB_to_MTB)
            n_tb_m += 1
        if conf_FM_to_FMT > 0:
            add_edge(FM_ST, FMT_ST, conf_FM_to_FMT)
            n_fm_t += 1
        if conf_FM_to_FMB > 0:
            add_edge(FM_ST, FMB_ST, conf_FM_to_FMB)
            n_fm_b += 1
        if conf_F_to_FT > 0:
            add_edge(F_ST, FT_ST, conf_F_to_FT)
            n_f_t += 1
        if conf_F_to_FB > 0:
            add_edge(F_ST, FB_ST, conf_F_to_FB)
            n_f_b += 1

    # -----------------------------------------------------------------------
    # Save boundary_edges.csv
    # -----------------------------------------------------------------------
    df_new = pd.DataFrame(rows, columns=[
        "circuit_idx", "src", "tgt", "n_src_samples", "n_edge_samples",
        "confidence", "dominant_type", "hamming"
    ])
    df_new.to_csv(out_csv_path, index=False)
    print(f"\nSaved {len(df_new)} boundary edges → {out_csv_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  Circuits with TB→FTB (λ_F>0 at TB face): {n_tb_f}/256")
    print(f"  Circuits with TB→MTB (λ_M>0 at TB face): {n_tb_m}/256")
    print(f"  Circuits with FM→FMT (λ_T>0 at FM face): {n_fm_t}/256")
    print(f"  Circuits with FM→FMB (λ_B>0 at FM face): {n_fm_b}/256")
    print(f"  Circuits with F→FT   (λ_T>0 at F axis):  {n_f_t}/256")
    print(f"  Circuits with F→FB   (λ_B>0 at F axis):  {n_f_b}/256")
    print(f"  Total new edges added: {len(df_new)}")

    # -----------------------------------------------------------------------
    # Merge with original heteroclinic_edges.csv
    # -----------------------------------------------------------------------
    df_orig = pd.read_csv(orig_csv_path, dtype={"src": str, "tgt": str})
    # Pad src/tgt to 4 chars (they may already be 4-char strings)
    df_orig["src"] = df_orig["src"].str.zfill(4)
    df_orig["tgt"] = df_orig["tgt"].str.zfill(4)
    df_new["src"]  = df_new["src"].str.zfill(4)
    df_new["tgt"]  = df_new["tgt"].str.zfill(4)

    # Concatenate: boundary_analytic rows come LAST so keep="last" lets them
    # override MC rows that share the same (circuit_idx, src, tgt).
    df_aug = pd.concat([df_orig, df_new], ignore_index=True)
    before = len(df_aug)
    df_aug = df_aug.drop_duplicates(subset=["circuit_idx", "src", "tgt"], keep="last")
    after = len(df_aug)
    print(f"\n  Original edges: {len(df_orig)}")
    print(f"  New boundary edges: {len(df_new)}")
    print(f"  Duplicates dropped: {before - after}")
    print(f"  Augmented total: {after}")

    df_aug.to_csv(aug_csv_path, index=False)
    print(f"  Saved augmented CSV → {aug_csv_path}")

    return df_new, df_aug


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    BASE = os.path.dirname(os.path.abspath(__file__))

    circuits_path = os.path.join(BASE, "circuits.wl")
    orig_csv_path = os.path.join(BASE, "heteroclinic_edges.csv")
    out_csv_path  = os.path.join(BASE, "boundary_edges.csv")
    aug_csv_path  = os.path.join(BASE, "heteroclinic_edges_augmented.csv")

    df_new, df_aug = run_analysis(
        circuits_path=circuits_path,
        orig_csv_path=orig_csv_path,
        out_csv_path=out_csv_path,
        aug_csv_path=aug_csv_path,
    )

    print("\nFirst 20 new boundary edges:")
    print(df_new.head(20).to_string(index=False))
