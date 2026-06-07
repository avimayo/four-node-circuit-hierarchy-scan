#!/usr/bin/env wolframscript
(* aggregate.wls — collect all 256 circuit CSVs into final_results.csv *)
(* Usage: wolframscript -script aggregate.wls *)
(* Run from the circuit_hpc directory after all array jobs complete.   *)

scriptDir = DirectoryName @ $InputFileName;
SetDirectory[scriptDir];

nCircuits = 256;
missing   = {};
allRows   = {};

Do[
  file = "results/circuit_" <> IntegerString[c, 10, 3] <> ".csv";
  If[!FileExistsQ[file],
    AppendTo[missing, c];
    Continue[]
  ];
  data = Import[file, "CSV"];
  (* data = {{"phenotype_pattern","count"}, {pattern, n}, ...} *)
  (* skip header row *)
  Do[
    AppendTo[allRows, Prepend[data[[r]], c]],
    {r, 2, Length[data]}
  ],
  {c, nCircuits}
];

If[missing =!= {},
  Print["WARNING: missing results for circuits: " <> ToString[missing]]
];

(* Aggregate: sum counts for each (circuit, phenotype) pair *)
header = {{"circuit_index", "phenotype_pattern", "count"}};
Export["final_results.csv", Join[header, allRows], "CSV"];

Print["Wrote final_results.csv — " <>
      ToString[Length[allRows]] <> " rows from " <>
      ToString[nCircuits - Length[missing]] <> "/" <>
      ToString[nCircuits] <> " circuits"];
Exit[0];
