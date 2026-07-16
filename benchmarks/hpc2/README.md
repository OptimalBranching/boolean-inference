# HPC2 benchmark environment

The persistent checkout lives at `~/Codes/boolean-inference`. Generated data
and build products live under `~/jhspoolers/boolean-inference` on SSD; archival
manifests and compressed final corpora belong under
`~/jhaidata/boolean-inference` on HDD.

Pinned tools:

- Boolean Inference branch commit: `b790dae43b74549c2c5aadb852f944f96aee3274`
- Multgen commit: `215fe0a77b2f3e61f6757a39323afa13bbe8e13f`
- Yosys tag/commit: `v0.66` / `86f2ddebce7e98ce7cacc27e8a5c14cb53b51b51`
- Compiler module: `compilers/gcc-12.2.0`
- Yosys Readline support: disabled (batch use does not need it)
- Python: uv-managed CPython 3.12

`build_yosys.sh` defaults to a free debug-partition build and smoke test; its
Slurm directives can be overridden explicitly at submission. Per the HPC2
submission policy, inspect and approve every Slurm resource and command before
each `sbatch`; the script being present is not submission authorization.
