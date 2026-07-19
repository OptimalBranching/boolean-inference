# CnC baseline tools (for the boolean-inference paper comparison)

Third-party cubers and conquer solvers for the Cube-and-Conquer comparison.
`bin/` is git-ignored and holds machine-specific builds, reachable by bare name
once it is on `PATH`:

```
export PATH=cnc-tools/bin:$PATH
```

`make` also keeps revision-qualified experiment binaries such as
`kissat-8af8e56f...` and `march_cu-705b60c...`. Hard-regime toolchain locks must
name those qualified paths; the bare names are convenience copies only.

| Tool | Role | Version | Source/build policy |
|---|---|---|---|
| `cadical` | conquer solver + DRAT emitter | record `--version` and executable hash | build with the pinned Makefile target |
| `kissat` | fixed conquer solver | `8af8e56f174b778aef3aa45af9f739b2a5f492c2`; also record `--version` and executable hash | build with the pinned Makefile target |
| `march_cu` | external lookahead cuber | `705b60c6491ef2b61988b3ce6ac674be1b90571d`; also record executable hash | build upstream source with the Makefile target |
| Proofix | optional proof-prefix cuber | SAT 2025 | pinned clone in `proofix/` |

## Primary cube generation

The conquer CNF must be COMMENT-FREE with `p cnf` as line 1 (Proofix's header
parser is strict): `grep -v '^c' full.cnf > clean.cnf`.

```bash
# Region cuber: online classical CC difficulty on structured CircuitSAT.
cargo build --release --bin cnc_cuber
target/release/cnc_cuber instance.json --cc-threshold <threshold> \
    -o region.cubes --max-rows 512

# External baseline: upstream default dynamic cutoff on the chosen global CNF.
march_cu full.cnf -o march.cubes

# Proofix: static proof-prefix partition, cube-size = tree depth, cutoff = proof-prefix length
cd proofix && python3 proofix.py --cnf clean.cnf --cube-size 10 --cutoff 100000 \
    --log run.log --icnf out.cubes --tmp-dir tmp
```

## Conquer + distribution (one harness for every cuber)

```bash
python3 benchmarks/conquer_cubes.py clean.cnf out.cubes --jobs 16 --out res.csv
# reports per-cube difficulty distribution (CV/P95/P99 = uniformity)
# and the common BCP residual-size audit
```

Do not force a shared numeric cutoff in the primary comparison: the structured
cuber and `march_cu` observe different representations. Sweep frontier regimes
and keep all conquer settings fixed. The Rust `-n` mode remains only as a small
implementation-level ablation; there is no maintained static-cutoff workflow.

## Fairness protocol / must-cite refs

See `../BooleanInference_Paper/ref/cnc_fairness_baselines.md`.
