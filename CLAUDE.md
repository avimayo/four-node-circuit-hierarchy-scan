# Circuit Attractor Scan — Agent Reference

Read `/Users/Avimayo/CLAUDE.md` first for shared cluster constraints.

---

## Project Goal

Enumerate phenotype attractors (stable steady states) for all 256 four-node
cell-type circuit topologies (F=Fibroblasts, M=Macrophages, T=T-cells, B=B-cells are **cell types, NOT genes**; 8 possible directed edges between cell types).
Parameter space: edge weights sampled from `RandomReal[{0,5}, {nSamples, nActive}]`.

Central biological question: which circuit topologies support a **hierarchical activation
cascade** F → F+M → F+M+T+B, a pattern associated with productive anti-tumour immunity?

---

## Current Status (2026-06-14)

**Data collection: complete (stable attractors).**  
`final_results.csv` and `phenotype_table.csv` are present and up to date.
The Streamlit dashboard (`app.py`) is the primary analysis and visualisation tool.

**circV3 (5-type pilot): COMPLETE** — 256/256 circuits done, output in `results_v3/`.  
**circSemi (ghost attractors): IN PROGRESS** — 18% (160/891 missing chunks), job 398891 running.

GitHub repo: `avimayo/four-node-circuit-hierarchy-scan`
Deploy: Streamlit Community Cloud (auto-deploys on push to `main`)

```bash
streamlit run app.py          # local dev, port 8501
git push origin main          # triggers cloud redeploy (~1-3 min)
```

---

## Key Biological Finding (emerging)

Backward signaling from TB (T/B-cells) to FM (Fibroblasts/Macrophages) appears to be
a **necessary and sufficient condition** for hierarchical cascades. The right scatter in
the "Edge analysis" tab visualises this directly: hierarchy frequency (relaxed) vs n_bwd.

---

## Streamlit App — Tab Structure

| Tab | What it shows |
|-----|---------------|
| **About** | Biology intro, ODE equations (linear form), run statistics, simplex logo, tab guide |
| **Topology heatmap** | 16×16 grid (rows = backward combos, cols = forward combos); metric selector inline in tab |
| **Solution types** | Stacked bar chart of attractor-pattern distribution by backward-edge combo |
| **Edge analysis** | Left: mean # stable states heatmap by (n_fwd, n_bwd). Right: 2×2 scatter grid |
| **Take-home** | Summary of obvious, key/surprising, in-progress, and open-question findings |

Edge analysis 2×2 scatter grid:
- Row 1: # distinct stable states vs fwd_frac | # distinct stable states vs bwd_frac
- Row 2: hierarchy freq (relaxed) vs fwd_frac | hierarchy freq (relaxed) vs bwd_frac
- Color encodes the complementary edge count (n_bwd for fwd plots, n_fwd for bwd plots)

Key design decisions:
- Heatmap tick labels are mini circuit PNG diagrams (matplotlib, transparent, white arrows)
- Bar chart tick images use dark arrows (#333333) — white arrows invisible on white background
- Two-color state icons: yellow (#FDD835) = active, teal (#00897B) = inactive
- Bar chart fixed-width `_BAR_W, _BAR_H = 1160, 700`; tick images use `yref="y"` data coords with y-axis extended to `range=[-0.18, 1]` — same trick as heatmap (`range=[-1.5, 15.5]`)
- `N_SAMPLES = 10000` is the per-chunk normalization; 10 chunks → 100k samples/circuit
- ODE equations: multiplicative-linear (no logistic/carrying capacity term): `ẋ = x · (Σ inputs − r_x)`
- **Bar tooltip**: `pat_hover_grid()` renders ■ (U+25A0) spans at 16px, `&nbsp;` spacing, colored F/M/T/B headers, italic state labels. Matches dark-bg tooltip style.
- **Edge analysis tooltip**: uses `text=` param (not `customdata`) on go.Heatmap — Plotly reads `%{text}` from `text`, not `customdata`. Dark hoverlabel required.
- **Circuit topology inspector**: below edge analysis heatmap; two selectboxes (n_fwd, n_bwd); rendered as `st.markdown` HTML block (not `st.columns`+`st.image`) so CSS `:hover` tooltips work. Each card shows canonical + relaxed hierarchy freq + top-3 patterns. Render params: `size_in=2.0, dpi=120, ms=16, lw=2.0, mutation_scale=17, shrinkA=8, shrinkB=8`, `width=180`. Bump `_CACHE_V` inside `build_all_circuit_images` to bust cache when params change.
- **Scatter guide curve**: [2/1] Padé rational fit via numpy lstsq; falls back to quadratic if pole `x=-1/b1` is inside data range. `showlegend=False` on mean±SEM trace.
- **Cascade detection in `pat_label()`**: bitwise subset chain check — patterns like {∅,F,F+M,F+M+T+B} display with "→" instead of "+".

---

## Data Files

| File | Description |
|------|-------------|
| `final_results.csv` | `circuit_index, phenotype_pattern, count` — aggregated over 10 chunks |
| `phenotype_table.csv` | One row per circuit: `circuit_index, added_edges, n_fwd, n_bwd, solution_*` |

---

## HPC Run History (for reference)

- **256 circuits × 10 chunks = 2,560 jobs**, each 10,000 samples (100k/circuit total)
- LSF split: `circ3a[1-1280]` + `circ3b[1-1280]` (offset=1280; max array = 2000)
- Retries: `circ3r` (partial), `circ3r2` (aborted), `circS[1-891]` (stable-only, complete)
- All output in `results_v2/`; aggregated by `aggregate_v2.py`
- Seed formula: `circIdx * 100000 + chunkIdx * 1000 + Round[rangeMin * 100]`

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `app.py` | **Primary deliverable** — Streamlit dashboard |
| `run_circuit_stable.wls` | Mathematica: stable attractors only |
| `run_circuit_v2.wls` | Mathematica: stable + semistable (binary semi/not) |
| `run_circuit_v3.wls` | **New** — 5-type classification: stable/semi1/semi2/semi3/unstable; single-pass loop; `Cancel[]` not `Simplify[]`; 1000 samples default → `results_v3/` |
| `run_missing_semi.sh` | Wrapper: reads `missing_semi_jobs.txt`, dispatches v2 for incomplete chunks |
| `run_missing_v3.sh` | Wrapper: reads `missing_v3_circuits.txt`, dispatches v3 for missing circuits |
| `aggregate_v2.py` | Collates chunk CSVs → `_agg.csv` per circuit |
| `make_interactive_heatmap.py` | Standalone HTML heatmap (legacy) |

---

## Semistable / Ghost-Attractor Runs (updated 2026-06-14)

**v2 completion — `circSemi[1-891]%50`** (job 398891, `medium` queue)
- 891 missing (circuit, chunk) pairs identified from `results_v2/`; list in `missing_semi_jobs.txt`
- Wrapper: `run_missing_semi.sh` — reads line `$LSB_JOBINDEX` → dispatches v2
- Logs: `logs_semi/semi_%I.out`
- **Status: 18% complete (160/891)** as of 2026-06-14 afternoon; still running

**v3 pilot — COMPLETE**
- Job 401364 (`circV3[1-256]%50`) + job 485930 (`circV3r[1-68]%50`) resubmit
- Resubmit used wrapper `run_missing_v3.sh` reading `missing_v3_circuits.txt`
- **256/256 circuits complete** in `results_v3/`

**Memory fix (2026-06-14):**
- HPC team flagged jobs wasting 123 GB RAM — requested `mem=4000` but used ~215 MB (5%)
- All 9 submit scripts patched: `mem=4000` → `mem=600`
- Pending circSemi jobs updated live via `bmod -R 'rusage[mem=600]' 398891`
- **Always use `mem=600` for Mathematica circuit jobs** (actual peak ~215 MB)

**Classification definitions (v3):**
- `stable` — all tangent + invasion eigenvalues ≤ 0 (proper attractor)
- `semi1/2/3` — tangent evals ≤ 0 but n=1/2/3 diagonal Jacobian entries for absent variables > 0 (ghost/quasi-stable)
- `unstable` — at least one positive tangent eigenvalue (saddle)
- Zero state (0000) is always stable (Jacobian diagonal at zero = −r_x < 0)

**Key insight:** steady-state finding is purely algebraic (linear system per sector after
substituting parameter values — at most 4×4). `Cancel[]` suffices in place of `Simplify[]`
for the inner-expression factoring. No ODE integration.

---

## Job Naming

All circuit LSF jobs must use the `circ` prefix.
See `/Users/Avimayo/CLAUDE.md` for the full naming convention.
