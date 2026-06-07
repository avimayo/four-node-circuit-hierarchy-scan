(* definitions.wl — fixed biological constants and CircModel *)
(* Load this with Get["definitions.wl"] from run_circuit.wls *)

(* Fixed kinetic parameters *)
Kb = 2.0; Kf = 2.9; Km = 4.7; Kt = 2.0;
pbb = 1.7; pbt = 1.04; pff = 1.49; pfm = 1.1;
pmf = 1.7; pmm = 1.76; ptb = 2.5; ptt = 2.37;
rb = 1.5; rf = 0.75; rm = 2.5; rt = 0.23;

(* Symbolic edge-presence variables — set to 0 to silence an edge *)
symbolEdges = {Pft, Ptf, Pmt, Ptm, Pfb, Pbf, Pmb, Pbm};

(*
  CircModel[edgeVec] builds the ODE system for one circuit topology.
  edgeVec = {pft, ptf, pmt, ptm, pfb, pbf, pmb, pbm}
    where each entry is 0 (edge absent) or 1 (edge present).
  Returns {equations, activeEdgeSymbols} where equations is a list of
  four 0 == ... expressions in F, M, T, B.
*)
CircModel[edgeVec_List] := Module[{lowerRules, eqs, activeSyms},
  (* Set lowercase binary indicators from edgeVec — absent edges multiply out to 0 *)
  lowerRules = Thread[{pft, ptf, pmt, ptm, pfb, pbf, pmb, pbm} -> edgeVec];

  eqs = {
    0 == F * ((Pbf pbf B + pff F + pmf M + Ptf ptf T) (1 - F/Kf) - rf),
    0 == M * ((Pbm pbm B + pfm F + pmm M + Ptm ptm T) (1 - M/Km) - rm),
    0 == T * ((Pft pft F + Pmt pmt M + ptt T) (1 - T/Kt) - pbt B - rt),
    0 == B * ((Pfb pfb F + Pmb pmb M + ptb T) (1 - B/Kb) - pbb B - rb)
  } /. lowerRules;

  (* After lowerRules: absent edges → 0*Pupper = 0; present → 1*Pupper = Pupper (free) *)
  activeSyms = Pick[symbolEdges, edgeVec, 1];  (* uppercase symbols, genuinely free *)
  {eqs, activeSyms}
];
