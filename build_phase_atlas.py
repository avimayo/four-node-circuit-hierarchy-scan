#!/usr/bin/env python3
"""
build_phase_atlas.py — Phase portrait atlas for all 256 circuits.

For each circuit, enumerates every distinct (stable_pat, semi_pat, unstable_pat)
triplet found across all 100k parameter samples. Each unique triplet is one
"phase portrait type" — a qualitatively distinct attractor landscape.
Different types are separated by bifurcations in parameter space.

Outputs:
  phase_atlas.csv        — one row per (circuit, phase-type), sorted by frequency
  circuit_summary.csv    — one row per circuit: n_types, dominant type freq, etc.

Both files go to ~/circuit_hpc/  and are small enough for the GitHub repo.
"""

import csv, os, re
from collections import defaultdict, Counter

RES     = os.path.expanduser("~/circuit_hpc/results_v2c")
OUTDIR  = os.path.expanduser("~/circuit_hpc")

ALLOWED  = {"0000", "1000", "1100", "1111"}
REQUIRED = {"1000", "1100", "1111"}

def has_hierarchy(stable_pat):
    if not stable_pat or stable_pat == "none":
        return False
    pats = set(stable_pat.split("|"))
    return REQUIRED.issubset(pats) and pats.issubset(ALLOWED)

# ── Aggregate ─────────────────────────────────────────────────────────────────
# circ → Counter of (stable_pat, semi_pat, unstable_pat) strings
phase_counter = defaultdict(Counter)
circ_total    = defaultdict(int)

for fname in sorted(os.listdir(RES)):
    m = re.match(r"circuit_(\d+)_chunk\d+_r0_5_joint\.csv", fname)
    if not m:
        continue
    circ = int(m.group(1))
    with open(os.path.join(RES, fname)) as f:
        for row in csv.DictReader(f):
            sp  = row.get("stable_pat",   "none") or "none"
            smp = row.get("semi_pat",     "none") or "none"
            usp = row.get("unstable_pat", "none") or "none"
            key = f"{sp}||{smp}||{usp}"
            phase_counter[circ][key] += 1
            circ_total[circ] += 1

print(f"Loaded {len(circ_total)} circuits.")

# ── Write phase_atlas.csv ─────────────────────────────────────────────────────
atlas_path = os.path.join(OUTDIR, "phase_atlas.csv")
with open(atlas_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "circuit_idx", "rank", "count", "freq_pct",
        "is_canonical_hier",
        "n_stable", "n_semi", "n_unstable",
        "stable_pat", "semi_pat", "unstable_pat",
    ])
    for circ in sorted(phase_counter):
        total = circ_total[circ]
        for rank, (key, cnt) in enumerate(
            phase_counter[circ].most_common(), start=1
        ):
            sp, smp, usp = key.split("||")
            n_st  = 0 if sp  == "none" else len(sp.split("|"))
            n_sm  = 0 if smp == "none" else len(smp.split("|"))
            n_un  = 0 if usp == "none" else len(usp.split("|"))
            canon = int(has_hierarchy(sp))
            w.writerow([
                circ, rank, cnt, f"{cnt/total*100:.4f}",
                canon, n_st, n_sm, n_un, sp, smp, usp,
            ])

print(f"Written: {atlas_path}")

# ── Write circuit_summary.csv ─────────────────────────────────────────────────
summary_path = os.path.join(OUTDIR, "circuit_summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "circuit_idx",
        "n_samples",
        "n_phase_types",
        "dominant_type_freq_pct",       # freq of the single most common type
        "canonical_hier_freq_pct",       # total freq of all canonical-hierarchy types
        "non_canon_stable_freq_pct",     # freq of types with wrong stable attractors
        "n_phase_types_canon",           # how many distinct types have canonical stable set
        "dominant_n_stable",             # n_stable in the dominant type
        "dominant_n_semi",
        "dominant_n_unstable",
    ])
    for circ in sorted(phase_counter):
        total   = circ_total[circ]
        counter = phase_counter[circ]
        types   = counter.most_common()

        n_types = len(types)
        dom_key, dom_cnt = types[0]
        dom_sp, dom_smp, dom_usp = dom_key.split("||")

        canon_cnt     = sum(cnt for key, cnt in types
                            if has_hierarchy(key.split("||")[0]))
        non_canon_cnt = sum(cnt for key, cnt in types
                            if not has_hierarchy(key.split("||")[0]))
        n_types_canon = sum(1 for key, _ in types
                            if has_hierarchy(key.split("||")[0]))

        dom_n_st = 0 if dom_sp  == "none" else len(dom_sp.split("|"))
        dom_n_sm = 0 if dom_smp == "none" else len(dom_smp.split("|"))
        dom_n_un = 0 if dom_usp == "none" else len(dom_usp.split("|"))

        w.writerow([
            circ, total, n_types,
            f"{dom_cnt/total*100:.4f}",
            f"{canon_cnt/total*100:.4f}",
            f"{non_canon_cnt/total*100:.4f}",
            n_types_canon,
            dom_n_st, dom_n_sm, dom_n_un,
        ])

print(f"Written: {summary_path}")
print("Done.")
