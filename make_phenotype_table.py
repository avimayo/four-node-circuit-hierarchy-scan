"""
Generate a table of all phenotype-pattern frequencies per circuit,
with each circuit identified by its active cross-edges (not a number).

Output: phenotype_table.csv
Columns: added_edges, n_edges, phenotype_pattern, fraction
"""

import csv
from itertools import product
from collections import defaultdict

N_SAMPLES = 10000

EDGE_NAMES = ["F->T", "T->F", "M->T", "T->M", "F->B", "B->F", "M->B", "B->M"]

# Reconstruct the sorted-by-sum edge-vector ordering (same as Mathematica)
all_vecs = sorted(product([0, 1], repeat=8), key=sum)
idx_to_vec = {i + 1: v for i, v in enumerate(all_vecs)}  # 1-based

def edges_label(vec):
    active = [EDGE_NAMES[i] for i, b in enumerate(vec) if b]
    return "{" + ",".join(active) + "}" if active else "{}"

# Read final_results.csv
rows = defaultdict(list)  # circuit_index → [(pattern, count), ...]
with open("final_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        c = int(row["circuit_index"])
        rows[c].append((row["phenotype_pattern"], int(row["count"])))

# Sort each circuit's rows by count descending
for c in rows:
    rows[c].sort(key=lambda x: -x[1])

# Write output
MAX_SOLS = 27
header = ["circuit_index", "added_edges", "n_edges"]
for i in range(1, MAX_SOLS + 1):
    header += [f"solution_{i}", f"freq_{i}"]

with open("phenotype_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for c in range(1, 257):
        if c not in rows:
            continue
        vec = idx_to_vec[c]
        label = edges_label(vec)
        n = sum(vec)
        row = [c, label, n]
        for pat, cnt in rows[c]:
            row += [pat, round(cnt / N_SAMPLES, 4)]
        writer.writerow(row)

print("Wrote phenotype_table.csv")

# Also print a compact human-readable version (circuits with >0 hierarchy)
HIER1 = "0000|1000|1100|1111"
HIER2 = "0000|0011|1000|1100|1111"

hier_circuits = set()
for c, pats in rows.items():
    for pat, cnt in pats:
        if pat in (HIER1, HIER2):
            hier_circuits.add(c)

print(f"\nFull distribution for circuits with any hierarchy ({len(hier_circuits)} circuits):\n")
for c in sorted(hier_circuits):
    vec = idx_to_vec[c]
    label = edges_label(vec)
    print(f"  {label}")
    for pat, cnt in rows[c]:
        marker = " ◄" if pat in (HIER1, HIER2) else ""
        print(f"    {pat}{marker}")
    print()
