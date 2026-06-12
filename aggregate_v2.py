#!/usr/bin/env python3
"""
aggregate_v2.py — collate per-chunk CSVs from the v3 scatter run into
                   one aggregated file per (circuit, range, type).

Reads:  results_v2/circuit_XXX_chunkKK_rRANGE_{stable,semi}.csv
Writes: results_v2/circuit_XXX_rRANGE_{stable,semi}_agg.csv
        (same format as v1: phenotype_pattern, count)

Run after all circ3 jobs have completed.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results_v2"


def collect_chunks(results_dir: Path):
    """
    Returns a dict keyed by (circuit_tag, range_tag, kind) ->
    list of (phenotype_pattern, count) tuples from all matching chunk files.
    """
    pattern = re.compile(
        r"^circuit_(\d{3})_chunk\d{2}_r(.+?)_(stable|semi)\.csv$"
    )
    groups = defaultdict(list)
    for f in sorted(results_dir.glob("circuit_*_chunk*_r*_*.csv")):
        m = pattern.match(f.name)
        if not m:
            continue
        key = (m.group(1), m.group(2), m.group(3))
        groups[key].append(f)
    return groups


def aggregate(files):
    """Sum counts across all chunk files for one (circuit, range, kind)."""
    totals = defaultdict(int)
    for f in files:
        lines = f.read_text().splitlines()
        for line in lines[1:]:          # skip header
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            pattern, count = parts[0].strip('"'), parts[1].strip('"')
            try:
                totals[pattern] += int(count)
            except ValueError:
                pass
    return totals


def write_agg(totals, out_path: Path):
    rows = sorted(totals.items(), key=lambda x: -x[1])
    lines = ["phenotype_pattern,count"]
    for pat, cnt in rows:
        lines.append(f"{pat},{cnt}")
    out_path.write_text("\n".join(lines) + "\n")
    return sum(totals.values())


def main():
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} does not exist — run jobs first.")
        sys.exit(1)

    groups = collect_chunks(RESULTS)
    if not groups:
        print(f"No chunk files found in {RESULTS}. Jobs may still be running.")
        sys.exit(1)

    print(f"Found {len(groups)} (circuit, range, kind) groups to aggregate.\n")

    for (circ, rng, kind), files in sorted(groups.items()):
        totals = aggregate(files)
        out = RESULTS / f"circuit_{circ}_r{rng}_{kind}_agg.csv"
        total_samples = write_agg(totals, out)
        n_pheno = sum(1 for p in totals if p != "none")
        print(f"  circuit {circ}  range={rng}  {kind:8s}  "
              f"{len(files):2d} chunks  {total_samples:7d} samples  "
              f"{n_pheno} phenotypes  → {out.name}")

    print("\nDone. Aggregated files written to results_v2/.")
    print("Next: run make_phenotype_table_v2.py to rebuild the phenotype table.")


if __name__ == "__main__":
    main()
