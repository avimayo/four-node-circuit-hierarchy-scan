#!/usr/bin/env python3
"""
mine_oscillatory.py — Extract oscillatory-dynamics statistics from evals files.

For every (circuit, state, classification) triplet, computes:
  - n_obs          : number of independent parameter samples
  - frac_complex   : fraction of samples where at least one eigenvalue has |Im| > IM_THRESH
  - per-eigenvalue  : mean Re and mean |Im| (sorted most-unstable-first, matching evals CSV)
  - max_absim_mean  : mean of max(|Im|) across the 4 eigenvalues per sample

Output: oscillatory_stats.csv
Run on the cluster — reads all *_evals.csv in results_v2c/ (~48 GB total).
"""

import csv, os, re
from collections import defaultdict

RES      = os.path.expanduser("~/circuit_hpc/results_v2c")
OUTDIR   = os.path.expanduser("~/circuit_hpc")
OUTPATH  = os.path.join(OUTDIR, "oscillatory_stats.csv")
IM_THRESH = 0.01   # threshold for "meaningfully complex" imaginary part

# accumulator per (circuit, state, classification):
#   [n, n_complex, sum_e1re, sum_e1aim, sum_e2re, sum_e2aim,
#                  sum_e3re, sum_e3aim, sum_e4re, sum_e4aim, sum_maxaim]
ACC = defaultdict(lambda: [0]*11)

all_circs = sorted(set(
    int(re.match(r"circuit_(\d+)_chunk\d+_r0_5_evals\.csv", f).group(1))
    for f in os.listdir(RES)
    if re.match(r"circuit_(\d+)_chunk\d+_r0_5_evals\.csv", f)
))
print(f"Processing {len(all_circs)} circuits ...")

for ci, circ in enumerate(all_circs):
    for chunk in (1, 2):
        path = os.path.join(RES,
               f"circuit_{circ:03d}_chunk{chunk:02d}_r0_5_evals.csv")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                state = row["state"]
                cls   = row["classification"]
                e1r = float(row["e1_re"]); e1i = abs(float(row["e1_im"]))
                e2r = float(row["e2_re"]); e2i = abs(float(row["e2_im"]))
                e3r = float(row["e3_re"]); e3i = abs(float(row["e3_im"]))
                e4r = float(row["e4_re"]); e4i = abs(float(row["e4_im"]))
                max_aim = max(e1i, e2i, e3i, e4i)
                is_cpx  = 1 if max_aim > IM_THRESH else 0
                key = (circ, state, cls)
                a = ACC[key]
                a[0]  += 1
                a[1]  += is_cpx
                a[2]  += e1r;  a[3]  += e1i
                a[4]  += e2r;  a[5]  += e2i
                a[6]  += e3r;  a[7]  += e3i
                a[8]  += e4r;  a[9]  += e4i
                a[10] += max_aim

    if (ci + 1) % 20 == 0:
        print(f"  {ci+1}/{len(all_circs)} done")

print("Writing output ...")
with open(OUTPATH, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow([
        "circuit_idx", "state", "classification", "n_obs",
        "frac_complex",
        "e1_mean_re", "e1_mean_absim",
        "e2_mean_re", "e2_mean_absim",
        "e3_mean_re", "e3_mean_absim",
        "e4_mean_re", "e4_mean_absim",
        "max_absim_mean",
    ])
    for (circ, state, cls), a in sorted(ACC.items()):
        n = a[0]
        if n == 0:
            continue
        w.writerow([
            circ, state, cls, n,
            f"{a[1]/n:.6f}",
            f"{a[2]/n:.6f}",  f"{a[3]/n:.6f}",
            f"{a[4]/n:.6f}",  f"{a[5]/n:.6f}",
            f"{a[6]/n:.6f}",  f"{a[7]/n:.6f}",
            f"{a[8]/n:.6f}",  f"{a[9]/n:.6f}",
            f"{a[10]/n:.6f}",
        ])

print(f"Written: {OUTPATH}")
print("Done.")
