#!/usr/bin/env wolframscript
(* run_circuit.wls — one LSF array task = one circuit, N=1000 parameter samples *)
(* Usage: wolframscript -script run_circuit.wls <circuitIndex>  (1-based) *)

circIdx = ToExpression @ $ScriptCommandLine[[2]];
If[!IntegerQ[circIdx] || circIdx < 1,
  Print["ERROR: first argument must be a positive integer (circuit index)"];
  Exit[1]];

(* Paths are relative to the directory containing this script *)
scriptDir = DirectoryName @ $InputFileName;
SetDirectory[scriptDir];

Get["definitions.wl"];
Get["circuits.wl"];   (* defines combinatorialVectors — list of 256 edge vectors *)

nCircuits = Length[combinatorialVectors];
If[circIdx > nCircuits,
  Print["ERROR: circIdx=" <> ToString[circIdx] <> " > " <> ToString[nCircuits]];
  Exit[1]];

edgeVec = combinatorialVectors[[circIdx]];
{eqs, activeSyms} = CircModel[edgeVec];

(* ------------------------------------------------------------------ *)
(* Parameter sampling                                                   *)
(* Sample the active edge weights uniformly in [0.5, 3.0]             *)
(* ------------------------------------------------------------------ *)
SeedRandom[circIdx];   (* reproducible per circuit *)
nSamples    = 10000;
nActive     = Length[activeSyms];
paramMatrix = RandomReal[{0, 2}, {nSamples, nActive}];

(* ------------------------------------------------------------------ *)
(* Sector enumeration — find ALL real non-negative steady states        *)
(* ------------------------------------------------------------------ *)
vars    = {F, M, T, B};
sectors = Subsets[vars];   (* 2^4 = 16 subsets including {} *)

(* Extract the "inner" expression for variable v from equation 0==v*inner *)
InnerExpr[eq_, v_] := Module[{rhs},
  rhs = Last[eq];   (* rhs of 0 == ... *)
  Simplify[rhs / v, v > 0]
];

innerExprs = Table[InnerExpr[eqs[[i]], vars[[i]]], {i, 4}];

FindAllSteadyStates[paramVals_] := Module[
  {subs, allSols, sector, absentVars, zeroSubs,
   presentVars, innerSub, sysEqs, sols, fullSols},

  subs = Thread[activeSyms -> paramVals];
  allSols = {};

  Do[
    sector     = sectors[[s]];
    absentVars = Complement[vars, sector];
    zeroSubs   = Thread[absentVars -> 0.];

    If[sector === {},
      (* trivial zero state *)
      AppendTo[allSols, Thread[vars -> 0.]];
      Continue[]
    ];

    presentVars = sector;
    (* For each present variable, inner expression must = 0 *)
    innerSub = innerExprs /. subs /. zeroSubs;
    sysEqs   = Table[
      innerSub[[First @ First @ Position[vars, presentVars[[j]]]]] == 0,
      {j, Length[presentVars]}
    ];

    (* Bound search to biologically reasonable range *)
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

(* ------------------------------------------------------------------ *)
(* Stability filter via Jacobian eigenvalues                           *)
(* ------------------------------------------------------------------ *)
odeRHS      = Last /@ eqs;          (* list of rhs expressions *)
jacobianSym = D[odeRHS, {vars}];    (* symbolic Jacobian, computed once *)

IsStable[sol_, jacNum_] := Module[{jEval, evals},
  jEval  = jacNum /. sol;
  evals  = Quiet @ Eigenvalues[N @ jEval];
  Max[Re @ evals] <= 0
];

(* Encode a solution as a sign-pattern string: e.g. "1010" *)
TypeLabel[sol_] :=
  StringJoin[ToString /@ (Sign[Round[vars /. sol, 1*^-6]] /. {-1 -> 0})];

(* ------------------------------------------------------------------ *)
(* Main loop over parameter samples                                    *)
(* ------------------------------------------------------------------ *)
results = Table[
  Module[{paramVals, pSubs, jacNum, allSS, stableTypes},
    paramVals   = paramMatrix[[k]];
    pSubs       = Thread[activeSyms -> paramVals];
    jacNum      = jacobianSym /. pSubs;

    allSS       = FindAllSteadyStates[paramVals];
    stableTypes = DeleteDuplicates @ Map[
      Function[s, If[IsStable[s, jacNum], TypeLabel[s], Nothing]],
      allSS
    ];

    If[stableTypes === {}, "none",
       StringJoin[Riffle[Sort[stableTypes], "|"]]]
  ],
  {k, nSamples}
];

(* ------------------------------------------------------------------ *)
(* Write output                                                         *)
(* ------------------------------------------------------------------ *)
outFile = "results/circuit_" <> IntegerString[circIdx, 10, 3] <> ".csv";

tallied = Tally[results];
rows    = Prepend[tallied, {"phenotype_pattern", "count"}];
Export[outFile, rows, "CSV"];

Print["circuit " <> ToString[circIdx] <> " done — " <>
      ToString[Length[tallied]] <> " distinct phenotypes across " <>
      ToString[nSamples] <> " samples"];
Exit[0];
