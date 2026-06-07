#!/usr/bin/env bash
# submit.sh — submit 256 circuit jobs as an LSF job array
#
# Usage:
#   cd /path/to/circuit_hpc
#   bash submit.sh
#
# Each array task runs one circuit (1-256) with 1000 parameter samples.
# %100 cap means at most 100 jobs run simultaneously.
#
# Prerequisites:
#   - circuits.wl must exist in this directory (exported from the notebook)
#   - module load Mathematica  (or wolfram is in PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f circuits.wl ]]; then
  echo "ERROR: circuits.wl not found in $SCRIPT_DIR"
  echo "Export combinatorialVectors from the notebook first (see README at bottom of submit.sh)"
  exit 1
fi

bsub \
  -J "circuit[1-256]%100" \
  -q short \
  -n 1 \
  -R "rusage[mem=4000]" \
  -W 02:00 \
  -o "$SCRIPT_DIR/logs/circuit_%I.out" \
  -e "$SCRIPT_DIR/logs/circuit_%I.err" \
  "wolframscript -script $SCRIPT_DIR/run_circuit.wls \$LSB_JOBINDEX"

echo "Submitted 256-task job array. Monitor with: bjobs -J circuit"

# ---------------------------------------------------------------------------
# HOW TO EXPORT circuits.wl FROM THE MATHEMATICA NOTEBOOK
# ---------------------------------------------------------------------------
# In Mathematica, after running the cell that defines combinatorialVectors:
#
#   DumpSave["/path/to/circuit_hpc/circuits.wl", combinatorialVectors]
#
# Or if you prefer a human-readable format:
#
#   Export["/path/to/circuit_hpc/circuits.wl",
#     "combinatorialVectors = " <> ToString[combinatorialVectors, InputForm] <> ";",
#     "Text"]
#
# Then scp the file to the cluster:
#   scp circuits.wl avimayo@access1.wexac.weizmann.ac.il:/path/to/circuit_hpc/
# ---------------------------------------------------------------------------
