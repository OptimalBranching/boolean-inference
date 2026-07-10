#!/usr/bin/env python3
"""Solve a native .csp (our tensor format) with OR-Tools CP-SAT — the SOTA CP
solver — using AddAllowedAssignments per tensor, i.e. the SAME relations our
solver sees, modelled natively (no CNF handicap). This is the fair SOTA baseline
for goal D: if our region contraction can't beat CP-SAT here, there is no CSP
advantage over the state of the art.

Reads the .csp format: line 1 = n_vars; then per tensor
"<v0> <v1> ... : <cfg0> <cfg1> ...", cfg = bitmask over the scope.

Prints: result, wall_seconds, branches (CP-SAT's search-node analogue).
Usage: cp_sat_solve.py <problem.csp>
"""
import sys
import time

from ortools.sat.python import cp_model


def main():
    path = sys.argv[1]
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]

    for line in lines[1:]:
        lhs, rhs = line.split(":")
        scope = [int(s) for s in lhs.split()]
        allowed_cfgs = [int(s) for s in rhs.split()]
        tuples = []
        for c in allowed_cfgs:
            tuples.append([(c >> i) & 1 for i in range(len(scope))])
        model.AddAllowedAssignments([x[v] for v in scope], tuples)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1  # single-threaded, fair vs our solver
    t0 = time.perf_counter()
    status = solver.Solve(model)
    dt = time.perf_counter() - t0
    res = {cp_model.OPTIMAL: "SAT", cp_model.FEASIBLE: "SAT",
           cp_model.INFEASIBLE: "UNSAT"}.get(status, "?")
    print(f"result={res} time={dt:.3f}s branches={solver.NumBranches()} conflicts={solver.NumConflicts()}")


if __name__ == "__main__":
    main()
