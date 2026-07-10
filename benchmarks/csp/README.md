# CSP benchmark generators

Instance generators probing **where region-contraction (structural, local
inference) can beat flat CDCL** — decision families whose structure our solver
sees but a CNF-only solver does not. Each generator emits the same system two
ways where applicable: a native `.csp` (one tensor per constraint, relation seen
whole) and a DIMACS `.cnf` (the flattened encoding a CDCL solver gets).

| Script | Family | Why it might favor structural inference |
|---|---|---|
| `gen_extcsp.py` | Bounded-treewidth extensional (random table) CSP | Random tables make unit propagation weak; bounded width keeps exact local inference cheap. |
| `gen_xor.py` | Random k-XOR-SAT | Parity is exponentially hard for resolution/CDCL (no Gaussian elimination), linear-algebra-easy. |
| `gen_tseitin.py` | Tseitin parity formulas | Resolution-hard on expanders; poly via local elimination on bounded-treewidth graphs. |
| `gen_coloring.py` | k-coloring, banded vs random graphs | Low-treewidth (banded) graphs keep regions local; tests width exploitation. |
| `gen_qcp.py` | Quasigroup completion (Latin square) | Classic CP-beats-SAT; GAC on wide exactly-one tensors. |
| `gen_x3c.py` | Exact Cover by 3-Sets | Home family for the exact-cover γ branching rules. |
| `cp_sat_solve.py` | OR-Tools CP-SAT baseline | Fair SOTA CP baseline: solves the native `.csp` relations directly (no CNF handicap). |

Native `.csp` format: line 1 = `n_vars`; then per tensor
`<v0> <v1> ... : <cfg0> <cfg1> ...`, where each `cfg` is a bitmask over the scope
listing an allowed assignment.

Status: exploratory. The current paper direction converged on **uniform cubing
for parallel Cube-and-Conquer on arithmetic circuits**, not on a CSP decision
win — these generators are retained as reproducible probes for revisiting the
CSP question, not as a claimed result.
