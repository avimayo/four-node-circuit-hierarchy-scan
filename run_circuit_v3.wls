#!/usr/bin/env wolframscript
(*
  run_circuit_v3.wls — full attractor classification in a single pass.

  Usage:
    wolframscript -script run_circuit_v3.wls <circuitIndex> [rangeMin rangeMax [nSamples [chunkIdx]]]

  Defaults: range [0,5], nSamples=1000, chunkIdx=Null

  Outputs (in results_v3/):
    circuit_XXX[_chunkKK]_r0_5_stable.csv   — all tangent + invasion evals <= 0
    circuit_XXX[_chunkKK]_r0_5_semi1.csv    — 1 positive invasion eigenvalue
    circuit_XXX[_chunkKK]_r0_5_semi2.csv    — 2 positive invasion eigenvalues
    circuit_XXX[_chunkKK]_r0_5_semi3.csv    — 3 positive invasion eigenvalues
    circuit_XXX[_chunkKK]_r0_5_unstable.csv — at least one positive tangent eigenvalue

  Notes:
  - No Simplify calls: Cancel[] handles the v-factor cancellation in InnerExpr.
  - Single main loop: steady states + Jacobian computed once per sample.
  - The zero state (0000) is always stable in this model (Jacobian diagonal at 0
    equals -r_x for each variable; r_x > 0 by construction). Included in stable output.
  - Semi grade = number of positive invasion eigenvalues on the diagonal of J
    for absent variables (at most 4 - k for a k-node state).
*)

(* ── Command-line arguments ────────────────────────────────────────── *)
args = $ScriptCommandLine;

circIdx = ToExpression @ args[[2]];
If[!IntegerQ[circIdx] || circIdx < 1,
  Print["ERROR: first argument must be a positive integer (circuit index)"]; Exit[1]];

rangeMin = If[Length[args] >= 4, ToExpression @ args[[3]], 0];
rangeMax = If[Length[args] >= 4, ToExpression @ args[[4]], 5];
nSamples = If[Length[args] >= 5, ToExpression @ args[[5]], 1000];
chunkIdx = If[Length[args] >= 6, ToExpression @ args[[6]], Null];

If[!NumberQ[rangeMin] || !NumberQ[rangeMax] || rangeMin >= rangeMax,
  Print["ERROR: invalid range [" <> ToString[rangeMin] <> "," <> ToString[rangeMax] <> "]"];
  Exit[1]];

(* ── Load model ────────────────────────────────────────────────────── *)
scriptDir = DirectoryName @ $InputFileName;
SetDirectory[scriptDir];
Get["definitions.wl"];
Get["circuits.wl"];

nCircuits = Length[combinatorialVectors];
If[circIdx > nCircuits,
  Print["ERROR: circIdx=" <> ToString[circIdx] <> " > " <> ToString[nCircuits]]; Exit[1]];

edgeVec           = combinatorialVectors[[circIdx]];
{eqs, activeSyms} = CircModel[edgeVec];
vars              = {F, M, T, B};
sectors           = Subsets[vars];

Print["circuit " <> ToString[circIdx] <>
      "  chunk=" <> If[chunkIdx === Null, "single", ToString[chunkIdx]] <>
      "  range=[" <> ToString[rangeMin] <> "," <> ToString[rangeMax] <>
      "]  nSamples=" <> ToString[nSamples]];

(* ── Parameter matrix ──────────────────────────────────────────────── *)
SeedRandom[circIdx * 100000 + If[chunkIdx === Null, 0, chunkIdx] * 1000 + Round[rangeMin * 100]];
nActive     = Length[activeSyms];
paramMatrix = RandomReal[{rangeMin, rangeMax}, {nSamples, nActive}];

(* ── Inner expressions (symbolic, computed once) ────────────────────
   Cancel[] handles the v-factor cancellation; no Simplify needed since
   the system is solved numerically by NSolve.                          *)
InnerExpr[eq_, v_] := Cancel[Last[eq] / v];
innerExprs = Table[InnerExpr[eqs[[i]], vars[[i]]], {i, 4}];

(* ── Steady-state finder ────────────────────────────────────────────── *)
FindAllSteadyStates[paramVals_] := Module[
  {subs, allSols, sector, absentVars, zeroSubs,
   presentVars, innerSub, sysEqs, sols, fullSols},

  subs    = Thread[activeSyms -> paramVals];
  allSols = {};

  Do[
    sector     = sectors[[s]];
    absentVars = Complement[vars, sector];
    zeroSubs   = Thread[absentVars -> 0.];

    If[sector === {},
      (* Zero state: always a fixed point; diagonal of J at 0 = -r_x < 0, always stable *)
      AppendTo[allSols, Thread[vars -> 0.]];
      Continue[]
    ];

    presentVars = sector;
    innerSub    = innerExprs /. subs /. zeroSubs;
    sysEqs      = Table[
      innerSub[[First @ First @ Position[vars, presentVars[[j]]]]] == 0,
      {j, Length[presentVars]}
    ];

    sols = Quiet @ NSolve[
      And @@ sysEqs && And @@ Thread[presentVars > 0] &&
        And @@ Thread[presentVars < 20],
      presentVars, Reals, WorkingPrecision -> MachinePrecision
    ];

    fullSols = Join[#, zeroSubs] & /@ sols;
    allSols  = Join[allSols, fullSols],
    {s, Length[sectors]}
  ];

  DeleteDuplicates[allSols,
    Max[Abs[(vars /. #1) - (vars /. #2)]] < 1*^-6 &]
];

(* ── Stability classifier ───────────────────────────────────────────
   Returns "stable" | "semi1" | "semi2" | "semi3" | "unstable".

   For a state with presentVars:
     tangent block  = J restricted to present×present indices
     invasion evals = diagonal of J at absent-variable indices

   Classification:
     - any tangent eval Re > 0         → "unstable"   (saddle in its face)
     - all tangent + invasion Re ≤ 0   → "stable"     (proper attractor)
     - otherwise: n = #{invasion evals Re > 0}  → "semi{n}"
*)
odeRHS      = Last /@ eqs;
jacobianSym = D[odeRHS, {vars}];

ClassifyState[sol_, jacNum_, presentVars_] := Module[
  {absentVars, jEval, presentIdx, absentIdx,
   tangentBlock, tangentEvals, invasionEvals, nPos},

  jEval = N[jacNum /. sol];

  (* Full 4-D state: use all eigenvalues *)
  If[Length[presentVars] == 4,
    With[{ev = Quiet @ Eigenvalues[jEval]},
      Return[If[Max[Re @ ev] <= 0, "stable", "unstable"]]]
  ];

  (* Zero state: always stable (see note above) *)
  If[Length[presentVars] == 0, Return["stable"]];

  absentVars    = Complement[vars, presentVars];
  presentIdx    = Flatten[Position[vars, #] & /@ presentVars];
  absentIdx     = Flatten[Position[vars, #] & /@ absentVars];
  tangentBlock  = jEval[[presentIdx, presentIdx]];
  tangentEvals  = Quiet @ Eigenvalues[tangentBlock];
  invasionEvals = Diagonal[jEval][[absentIdx]];

  Which[
    Max[Re @ tangentEvals] > 0, "unstable",
    Max[Re @ invasionEvals] <= 0, "stable",
    True,
      nPos = Count[Re @ invasionEvals, x_ /; x > 0];
      "semi" <> ToString[nPos]
  ]
];

TypeLabel[sol_] :=
  StringJoin[ToString /@ (Sign[Round[vars /. sol, 1*^-6]] /. {-1 -> 0})];

(* ── Single-pass main loop ─────────────────────────────────────────── *)
types = {"stable", "semi1", "semi2", "semi3", "unstable"};

allResults = Table[
  Module[{paramVals, pSubs, jacNum, allSS, buckets, cls, lbl, present},
    paramVals = paramMatrix[[k]];
    pSubs     = Thread[activeSyms -> paramVals];
    jacNum    = jacobianSym /. pSubs;
    allSS     = FindAllSteadyStates[paramVals];
    buckets   = AssociationThread[types, ConstantArray[{}, 5]];

    Scan[Function[s,
      present = Select[vars, (# /. s) > 1*^-6 &];
      cls = ClassifyState[s, jacNum, present];
      lbl = TypeLabel[s];
      buckets[cls] = Append[buckets[cls], lbl]
    ], allSS];

    Map[
      If[# === {}, "none", StringJoin[Riffle[Sort @ DeleteDuplicates @ #, "|"]]] &,
      buckets
    ]
  ],
  {k, nSamples}
];

(* ── Write outputs ──────────────────────────────────────────────────── *)
If[!DirectoryQ["results_v3"], CreateDirectory["results_v3"]];

tag      = IntegerString[circIdx, 10, 3];
rangeTag = StringReplace[ToString[rangeMin] <> "_" <> ToString[rangeMax], "." -> "p"];
chunkTag = If[chunkIdx === Null, "", "_chunk" <> IntegerString[chunkIdx, 10, 2]];
prefix   = "results_v3/circuit_" <> tag <> chunkTag <> "_r" <> rangeTag <> "_";

Do[
  outFile = prefix <> t <> ".csv";
  col     = Values[allResults][[All, Position[types, t][[1, 1]]]];
  Export[outFile, Prepend[Tally[col], {"phenotype_pattern", "count"}], "CSV"],
  {t, types}
];

Print["Done — " <> prefix <> "{stable,semi1,semi2,semi3,unstable}.csv"];
Print["  stable phenotypes:   " <> ToString[Length @ DeleteCases[Tally[Values[allResults][[All,1]]], {"none",_}]]];
Print["  semi1 phenotypes:    " <> ToString[Length @ DeleteCases[Tally[Values[allResults][[All,2]]], {"none",_}]]];
Print["  semi2 phenotypes:    " <> ToString[Length @ DeleteCases[Tally[Values[allResults][[All,3]]], {"none",_}]]];
Print["  semi3 phenotypes:    " <> ToString[Length @ DeleteCases[Tally[Values[allResults][[All,4]]], {"none",_}]]];
Print["  unstable phenotypes: " <> ToString[Length @ DeleteCases[Tally[Values[allResults][[All,5]]], {"none",_}]]];
Exit[0];
