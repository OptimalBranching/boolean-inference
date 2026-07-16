# HPC2 benchmark environment

The persistent checkout lives at `~/Codes/boolean-inference`. Generated data
and build products live under `~/jhspoolers/boolean-inference` on SSD; archival
manifests and compressed final corpora belong under
`~/jhaidata/boolean-inference` on HDD.

Pinned tools:

- Boolean Inference branch commit: recorded by each generation job and dataset manifest
- Multgen commit: `215fe0a77b2f3e61f6757a39323afa13bbe8e13f`
- Yosys tag/commit: `v0.66` / `86f2ddebce7e98ce7cacc27e8a5c14cb53b51b51`
- Compiler module: `compilers/gcc-12.2.0`
- Yosys build dependencies: `~/envs/yosys-0.66-build`, reproducible from
  `yosys-build-env.lock.txt`
- Python: uv-managed CPython 3.12

Recreate the build environment with the cluster `anaconda3` module and the
explicit lock file, then run `build_yosys.sh --check-deps` before submitting a
build job.

With no arguments, `generate_scope_v3_pilot.sh` reproduces the 24/32-bit
six-architecture smoke corpus. That corpus validates the pipeline and is not
benchmark evidence. Explicit `--dataset-id`, repeatable `--width`, `--count`,
and `--expected-targets-sha256` options support larger scale probes and prevent
an unreviewed target set from being generated accidentally. Every run uses a
job-specific SSD staging directory, refuses to overwrite existing data, and
writes a deterministic HDD archive only after manifest collection,
multiplication-witness simulation, and scope validation pass.

The completed scale calibration used one deterministic target per factor
width. Scope v4 retains all three widths in the formal ladder; these one-target
runs establish feasibility and provenance, not the final sample count:

| factor width | target-file SHA-256 | elapsed | MaxRSS |
|---:|---|---:|---:|
| 64 | `40da54c65500d78aefa8e784bbb7a0ae1b70381d40eba886b3d9ed0aa5c97ae1` | 2:52 | 1,556,784 KiB |
| 96 | `6e20978e00decb1ceb7b05f4e34730231eec609e5b85797fbd2afe76d091f03c` | 6:33 | 3,674,560 KiB |
| 128 | `0ef813c0e50bf15382a36efec9f34a3ec2882828f0449970b8faea4ed4c1ad41` | 11:39 | 6,925,116 KiB |

The tracked manifests record elapsed time, peak memory, extracted bytes, and
per-architecture variables/assignments. The probes do not define an evaluation
split, sample count, or success criterion.

`build_yosys.sh` defaults to a free debug-partition build and smoke test; its
Slurm directives can be overridden explicitly at submission. Per the HPC2
submission policy, inspect and approve every Slurm resource and command before
each `sbatch`; the script being present is not submission authorization.
