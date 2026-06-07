import csv
from collections import defaultdict

N_SAMPLES = 10000

# All 16 possible 4-species sign patterns, ordered canonically
ALL_STATES = [format(i, "04b") for i in range(16)]  # 0000 .. 1111

# Read raw results
rows = []
with open("final_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "circuit": int(row["circuit_index"]),
            "pattern": row["phenotype_pattern"],
            "count":   int(row["count"]),
        })

# Build per-circuit frequency for each individual state
# freq[circuit][state] = fraction of samples where that state is stable
freq = defaultdict(lambda: defaultdict(float))

for row in rows:
    c   = row["circuit"]
    cnt = row["count"]
    for state in row["pattern"].split("|"):
        freq[c][state] += cnt / N_SAMPLES

# Write frequency vectors (one row per circuit, one column per state)
with open("frequency_vectors.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["circuit_index"] + ALL_STATES)
    for c in range(1, 257):
        writer.writerow([c] + [round(freq[c].get(s, 0.0), 6) for s in ALL_STATES])

# Also write a tidy summary: which states are ever stable, across how many circuits
print("State frequencies across 256 circuits:")
print(f"{'state':>6}  {'label':>20}  {'n_circuits':>10}  {'mean_freq':>10}")
labels = {"0000":"none","0001":"B","0010":"T","0011":"B+T",
          "0100":"M","0101":"M+B","0110":"M+T","0111":"M+T+B",
          "1000":"F","1001":"F+B","1010":"F+T","1011":"F+T+B",
          "1100":"F+M","1101":"F+M+B","1110":"F+M+T","1111":"all"}
for s in ALL_STATES:
    freqs = [freq[c].get(s, 0.0) for c in range(1, 257)]
    n = sum(1 for v in freqs if v > 0)
    mean = sum(freqs) / 256
    print(f"{s:>6}  {labels[s]:>20}  {n:>10}  {mean:>10.3f}")

print("\nWrote frequency_vectors.csv")
