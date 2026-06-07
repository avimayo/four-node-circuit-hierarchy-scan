# Four-Node Circuit Hierarchy Scan

Exhaustive parameter scan of all 256 possible cross-edge topologies in a four-node
gene-regulatory circuit, identifying which topologies robustly produce a sequential
hierarchical activation pattern F → F+M → F+M+T+B.

---

## Biological context

The circuit has four nodes arranged in two modules:

- **FM module** (source): F and M, with fixed mutual activation and self-regulation
- **TB module** (target): T and B, with fixed mutual regulation and self-regulation

The eight tunable cross-edges connect the two modules:

| Direction | Edges |
|---|---|
| FM → TB (feedforward) | F→T, M→T, F→B, M→B |
| TB → FM (feedback) | T→F, T→M, B→F, B→M |

Each edge is either present (1) or absent (0), giving 2⁸ = 256 distinct circuit topologies.

The **canonical hierarchy** is the stable-state set `{0000, 1000, 1100, 1111}`:
- `0000` — all off
- `1000` — F alone active
- `1100` — F + M active
- `1111` — all active (F + M + T + B)

This represents a stepwise, ordered activation sequence where each state is a stable
attractor, reflecting a developmental or differentiation hierarchy.

---

## Computational approach

### ODE model

Each node follows a logistic growth equation with cross-regulation:

```
0 = F · [(Pbf·B + pff·F + pmf·M + Ptf·T)(1 − F/Kf) − rf]
0 = M · [(Pbm·B + pfm·F + pmm·M + Ptm·T)(1 − M/Km) − rm]
0 = T · [(Pft·F + Pmt·M + ptt·T)(1 − T/Kt) − pbt·B − rt]
0 = B · [(Pfb·F + Pmb·M + ptb·T)(1 − B/Kb) − pbb·B − rb]
```

Fixed kinetic parameters: `Kb=2.0, Kf=2.9, Km=4.7, Kt=2.0`,
`pbb=1.7, pbt=1.04, pff=1.49, pfm=1.1, pmf=1.7, pmm=1.76, ptb=2.5, ptt=2.37`,
`rb=1.5, rf=0.75, rm=2.5, rt=0.23`.

Present cross-edges contribute free parameters (uppercase `Pij`) sampled uniformly
from [0, 2]. Absent edges are multiplied out to zero.

### Steady-state enumeration

For each parameter sample, all 2⁴ = 16 sector subsets (which nodes are positive)
are enumerated. Within each sector the system of equations reduces to a polynomial
system solved with Mathematica's `NSolve`. Stability is assessed via Jacobian
eigenvalues (stable ↔ all Re(λ) ≤ 0).

### Parameter sampling

For each circuit: 10,000 independent parameter samples. Each sample yields a set of
stable steady states encoded as a 4-bit binary string (e.g. `1100` = F+M active).
The full set of stable states for a sample is its **phenotype pattern**.

### Cluster execution

Jobs ran on the Weizmann WEXAC cluster (LSF scheduler) via `run_circuit.wls`,
one LSF array task per circuit, using `wolframscript` with Mathematica 14.3.

---

## Key findings

1. **Feedforward edges destroy hierarchy.** Any FM→TB edge creates a competing
   bistable attractor `{0000, 1111}` (all-off / all-on), which crowds out the
   stepwise hierarchy.

2. **Feedback edges build hierarchy.** TB→FM edges are necessary and sufficient
   to produce the canonical hierarchy; the fraction of samples in the hierarchy
   rises monotonically with the feedback fraction n(TB→FM) / n(total cross-edges).

3. **T→F is the canonical hierarchy edge.** It produces the pure `{0000,1000,1100,1111}`
   set. B→M also builds hierarchy but always in the generalized form
   `{0000,0011,1000,1100,1111}` (T+B co-attractor present alongside the hierarchy).

4. **Attractor diversity increases with feedforward edges.** More FM→TB edges →
   more distinct phenotype patterns per circuit (up to 27 observed).

5. **15 circuits achieve ≥ 95% canonical hierarchy** (all have zero feedforward edges
   and include T→F among their backward edges).

---

## Files

### Mathematica (cluster)
| File | Description |
|---|---|
| `definitions.wl` | Fixed parameters and `CircModel[]` function |
| `circuits.wl` | All 256 edge vectors (`combinatorialVectors`) |
| `run_circuit.wls` | Main LSF job script: samples parameters, enumerates steady states, writes results |
| `aggregate.wls` | Aggregates per-circuit CSV files into `final_results.csv` |
| `submit.sh` | LSF submission script |

### Python (local analysis)
| File | Description |
|---|---|
| `make_frequency_vectors.py` | Per-circuit per-state marginal frequencies → `frequency_vectors.csv` |
| `make_phenotype_table.py` | Full phenotype distribution table → `phenotype_table.csv` |
| `hierarchy_heatmap.py` | 16×16 hierarchy heatmap (forward × backward edge combinations) |
| `make_analysis_figures.py` | Per-state heatmaps, diversity heatmap, backward-edge collapse bar |
| `make_combined_figure.py` | Combined 3-panel figure (hierarchy heatmap, diversity, bwd collapse, feedback-fraction scatter) |

### Data
| File | Description |
|---|---|
| `final_results.csv` | Raw results: circuit, phenotype pattern, count (3012 rows, 256 circuits × 10k samples) |
| `frequency_vectors.csv` | Per-circuit frequency of each of the 16 possible states |
| `hierarchy_breakdown.csv` | Per-circuit canonical and generalized hierarchy fractions |
| `phenotype_table.csv` | One row per circuit: all solution types and frequencies |

### Figures
| File | Description |
|---|---|
| `combined_figure.png` | Main combined figure (panels A–D) |
| `fig_feedback_fraction_comparison.png` | Hierarchy frequency vs. feedback fraction (canonical vs. generalized) |
| `fig_state_heatmaps.png` | Per-state frequency heatmaps |
| `fig_diversity_heatmap.png` | Attractor diversity heatmap |
| `fig_bwd_collapse.png` | Solution-type distribution by backward-edge combination |
| `hierarchy_heatmap.png` | Standalone hierarchy heatmap |

---

## Requirements

**Cluster:** Mathematica 14.3 (`wolframscript`), LSF scheduler  
**Local:** Python 3.x, `numpy`, `matplotlib`
