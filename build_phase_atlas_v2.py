#!/usr/bin/env python3
"""
build_phase_atlas_v2.py — Phase portrait atlas with semi_1/2/3 distinguished.

Reads both _joint.csv (stable_pat, semi_pat, unstable_pat) and _evals.csv
(eigenvalues per steady state) to split the lumped semi_pat into:
  semi1_pat  — n=1 positive real eigenvalue  (codim-3 stable manifold)
  semi2_pat  — n=2 positive real eigenvalues (codim-2 stable manifold)
  semi3_pat  — n=3 positive real eigenvalues (codim-1 stable manifold)

Outputs:
  phase_atlas_v2.csv     — one row per (circuit, phase-type)
  circuit_summary_v2.csv — one row per circuit
"""

import csv, os, re
from collections import defaultdict, Counter

RES    = os.path.expanduser("~/circuit_hpc/results_v2c")
OUTDIR = os.path.expanduser("~/circuit_hpc")

ALLOWED  = {"0000", "1000", "1100", "1111"}
REQUIRED = {"1000", "1100", "1111"}
EPS      = 1e-9   # threshold for "positive" real part

def has_hierarchy(stable_pat):
    if not stable_pat or stable_pat == "none":
        return False
    pats = set(stable_pat.split("|"))
    return REQUIRED.issubset(pats) and pats.issubset(ALLOWED)

def pat_str(s):
    return "|".join(sorted(s)) if s else "none"

# ── Process one circuit (both chunks) ────────────────────────────────────────
def process_circuit(circ):
    """Return Counter of (stable,semi1,semi2,semi3,unstable) string keys."""
    counter = Counter()
    total   = 0

    for chunk in (1, 2):
        jname = f"circuit_{circ:03d}_chunk{chunk:02d}_r0_5_joint.csv"
        ename = f"circuit_{circ:03d}_chunk{chunk:02d}_r0_5_evals.csv"
        jpath = os.path.join(RES, jname)
        epath = os.path.join(RES, ename)
        if not os.path.exists(jpath) or not os.path.exists(epath):
            continue

        # ── Load evals: build {sample_idx: {state: n_pos}} ──────────────────
        n_pos = defaultdict(dict)   # n_pos[sample_idx][state] = count of pos Re evals
        with open(epath) as f:
            for row in csv.DictReader(f):
                sid   = int(row["sample_idx"])
                state = row["state"]
                n = sum(1 for k in ("e1_re","e2_re","e3_re","e4_re")
                        if float(row[k]) > EPS)
                # keep worst (highest n) if multiple solutions in same sector
                if state not in n_pos[sid] or n > n_pos[sid][state]:
                    n_pos[sid][state] = n

        # ── Load joint: re-classify semi states using evals ─────────────────
        with open(jpath) as f:
            for row in csv.DictReader(f):
                sid = int(row["sample_idx"])
                sp  = row.get("stable_pat",   "none") or "none"
                smp = row.get("semi_pat",     "none") or "none"
                usp = row.get("unstable_pat", "none") or "none"

                semi1, semi2, semi3 = set(), set(), set()
                if smp and smp != "none":
                    for pat in smp.split("|"):
                        n = n_pos[sid].get(pat, 1)   # default to 1 if missing
                        if n == 1:
                            semi1.add(pat)
                        elif n == 2:
                            semi2.add(pat)
                        else:
                            semi3.add(pat)

                key = "||".join([sp,
                                 pat_str(semi1), pat_str(semi2), pat_str(semi3),
                                 usp])
                counter[key] += 1
                total += 1

    return counter, total

# ── Main ─────────────────────────────────────────────────────────────────────
all_circs = sorted(set(
    int(re.match(r"circuit_(\d+)_chunk\d+_r0_5_joint\.csv", f).group(1))
    for f in os.listdir(RES)
    if re.match(r"circuit_(\d+)_chunk\d+_r0_5_joint\.csv", f)
))
print(f"Processing {len(all_circs)} circuits...")

atlas_path   = os.path.join(OUTDIR, "phase_atlas_v2.csv")
summary_path = os.path.join(OUTDIR, "circuit_summary_v2.csv")

with open(atlas_path, "w", newline="") as fa, \
     open(summary_path, "w", newline="") as fs:

    wa = csv.writer(fa)
    wa.writerow(["circuit_idx","rank","count","freq_pct","is_canonical_hier",
                 "n_stable","n_semi1","n_semi2","n_semi3","n_unstable",
                 "stable_pat","semi1_pat","semi2_pat","semi3_pat","unstable_pat"])

    ws = csv.writer(fs)
    ws.writerow(["circuit_idx","n_samples","n_phase_types",
                 "dominant_type_freq_pct","canonical_hier_freq_pct",
                 "n_phase_types_canon"])

    for i, circ in enumerate(all_circs):
        counter, total = process_circuit(circ)
        if total == 0:
            continue

        canon_cnt  = 0
        n_types_c  = 0
        dom_cnt, _ = counter.most_common(1)[0]

        for rank, (key, cnt) in enumerate(counter.most_common(), start=1):
            sp, sm1, sm2, sm3, usp = key.split("||")
            n_st  = 0 if sp  == "none" else len(sp.split("|"))
            n_sm1 = 0 if sm1 == "none" else len(sm1.split("|"))
            n_sm2 = 0 if sm2 == "none" else len(sm2.split("|"))
            n_sm3 = 0 if sm3 == "none" else len(sm3.split("|"))
            n_un  = 0 if usp == "none" else len(usp.split("|"))
            canon = int(has_hierarchy(sp))
            if canon:
                canon_cnt += cnt
                n_types_c += 1
            wa.writerow([circ, rank, cnt, f"{cnt/total*100:.4f}",
                         canon, n_st, n_sm1, n_sm2, n_sm3, n_un,
                         sp, sm1, sm2, sm3, usp])

        dom_cnt = counter.most_common(1)[0][1]
        ws.writerow([circ, total, len(counter),
                     f"{dom_cnt/total*100:.4f}",
                     f"{canon_cnt/total*100:.4f}",
                     n_types_c])

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(all_circs)} done")

print(f"Written: {atlas_path}")
print(f"Written: {summary_path}")
print("Done.")
