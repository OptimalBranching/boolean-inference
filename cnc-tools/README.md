# CnC baseline tools (for the boolean-inference paper comparison)

Third-party cubers / conquer solvers for the Cube-and-Conquer comparison, kept
inside the repo but git-ignored (`.git/info/exclude`). `bin/` holds the actual
binaries (self-contained: they link only libc++/libSystem), reachable by bare
name once `bin/` is on `PATH`:

```
export PATH=cnc-tools/bin:$PATH
```

| Tool | Role | Version | Origin (binary copied into `bin/`) |
|---|---|---|---|
| `cadical` | conquer solver + DRAT emitter (Proofix needs it) | 3.0.0 | built from github.com/arminbiere/cadical (source tree removed; only the binary kept) |
| `kissat`  | conquer solver (the fixed conquerer) | 4.0.4 | homebrew |
| `march_cu`| cuber (lookahead) | — | arm64 binary from BooleanInference/benchmarks/artifacts/bin |
| Proofix   | cuber (proof-prefix) | SAT 2025 | github.com/zaxioms0/proofix clone in `proofix/` (pure Python ≥3.13) |

## Cube generation (all emit march_cu's `a <lits> 0` iCNF over the SAME numbering)

The shared CNF must be COMMENT-FREE with `p cnf` as line 1 (Proofix's header
parser is strict): `grep -v '^c' full.cnf > clean.cnf`.

```bash
# our cuber (bi): from the CircuitSAT JSON, numbering = export_dimacs v+1
target/release/examples/gen_cubes inst.json <theta> out.cubes

# march_cu: default lookahead, dynamic cutoff (sweep -n/-d to match granularity)
march_cu clean.cnf -o out.cubes

# Proofix: static proof-prefix partition, cube-size = tree depth, cutoff = proof-prefix length
cd proofix && python3 proofix.py --cnf clean.cnf --cube-size 10 --cutoff 100000 \
    --log run.log --icnf out.cubes --tmp-dir tmp
```

## Conquer + distribution (one harness for every cuber)

```bash
python3 benchmarks/conquer_cubes.py clean.cnf out.cubes --out res.csv \
    --contract-lock experiments/cnc-study.lock
# reports per-cube difficulty distribution (CV/P95/P99 = uniformity) + cutoff-proxy Spearman
```

The lock is mandatory: every CSV row starts with the frozen contract digest so
results cannot be detached from the protocol that produced them.

Note (Proofix): its static partition fixes cube depth = `--cube-size`, so every
cube has the same `sigma_dec` by construction — the cutoff proxy has no variance
across Proofix cubes (rho undefined), unlike bi/march whose cube depth varies.

## Fairness protocol / must-cite refs

See `../BooleanInference_Paper/ref/cnc_fairness_baselines.md`.
