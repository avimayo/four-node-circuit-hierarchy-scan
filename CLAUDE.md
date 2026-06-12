# Circuit Attractor Scan — Agent Reference

Read `/Users/Avimayo/CLAUDE.md` first for shared cluster constraints.

---

## Project Goal

Enumerate phenotype attractors (stable steady states) for all 256 four-node
gene-circuit topologies (F, M, T, B nodes; 8 possible directed edges).
Parameter space: edge weights sampled from `RandomReal[{0,5}, {nSamples, nActive}]`.

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_circuit_stable.wls` | **Primary script** — stable attractors only, fast. Range [0,5], 5000 samples default |
| `run_circuit_v2.wls` | Extended — stable + semistable. Slower (~3-4×). Use for targeted follow-up |
| `run_chunk.sh` | Decodes flat LSF array index → (circIdx, chunkIdx). Supports offset for split arrays |
| `run_stable_chunk.sh` | Reads `missing_stable.txt` line $LSB_JOBINDEX → calls `run_circuit_stable.wls` |
| `submit_v3.sh` | Submitted circ3a + circ3b (first full run, 2560 jobs × 10000 samples) |
| `aggregate_v2.py` | Collates chunk CSVs → `_agg.csv` per circuit. Run after all jobs complete |

---

## Scatter-Gather Structure

- **256 circuits × 10 chunks = 2560 jobs**, each 10 000 samples
- LSF max array 2000 → split: `circ3a[1-1280]` + `circ3b[1-1280]` (with offset=1280)
- Seed formula: `circIdx * 100000 + chunkIdx * 1000 + Round[rangeMin * 100]`
- Output: `results_v2/circuit_XXX_chunkKK_r0_5_stable.csv`
- Aggregated: `results_v2/circuit_XXX_r0_5_stable_agg.csv`

---

## Current State (2026-06-12)

| Array | Job name | Status |
|-------|----------|--------|
| First run | `circ3a`, `circ3b` | Complete (~1669/2560 succeeded) |
| First retry | `circ3r` | Complete (partial success) |
| Second retry | `circ3r2` | Aborted (exit bug, superseded) |
| **Active** | `circS[1-891]%200` | Running — stable-only retry for 891 missing chunks |

When `circS` completes (~2h wall time, short queue):
1. SSH to WEXAC
2. `cd ~/circuit_hpc && python3 aggregate_v2.py`
3. Regenerate all figures (see below)

---

## Figures to Regenerate (post-aggregation)

All figures must use the **[0,5] range** data from `results_v2/`:

```bash
python3 make_phenotype_table.py        # rebuild phenotype_table.csv
python3 hierarchy_heatmap.py           # Fig A — heatmap
python3 make_analysis_figures.py       # stacked bar + scatter
python3 make_3d_forward_analysis.py    # 3D forward-edge analysis
python3 make_combined_figure.py        # combined PDF
```

Run with `MPLBACKEND=Agg python3 <script>` to avoid display errors.

---

## Semistable States

- Available for ~1669 circuits from the first run (`*_semi.csv` files)
- Semistable = stable within its face (tangent eigenvalues ≤ 0) but positive
  invasion eigenvalue — ghost attractor / quasi-stable
- Detection is in `run_circuit_v2.wls` (block-triangular Jacobian)
- **Not included** in the circS retry (stable-only for speed)
- Targeted semistable follow-up: run `run_circuit_v2.wls` on circuits of interest

---

## Job Naming

All circuit LSF jobs must use the `circ` prefix: `circS`, `circ3a`, `circ3r`, etc.
See `/Users/Avimayo/CLAUDE.md` for the full naming convention.

---

## Logs

- First run: `logs_v2/`
- Retry runs: `logs_v3/`
- circS retry: `logs_v3/` (same directory)

---

## Other Active Projects (for cross-agent awareness)

See `/Users/Avimayo/CLAUDE.md` for the full list. The project running in parallel with this one:

**Geiger LR Network Dashboard** — Streamlit web app for liver-lung cell-cell communication
network analysis (melanoma metastasis, Geiger Lab).
- Local files: `/Users/Avimayo/Library/CloudStorage/Box-Box/Geiger/dashboard/`
- GitHub: `avimayo/geiger-lr-network` (private), deployed on Streamlit Community Cloud
- **No cluster involvement** — entirely local/cloud, no LSF jobs, no WEXAC
- Current state: GO/KEGG enrichment tab + drug actionability overlays deployed
