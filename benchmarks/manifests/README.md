# Materialized benchmark manifests

Manifest JSONL files freeze generated instance identities, architectures,
targets, provenance, relative artifact paths, and SHA-256 digests. The
corresponding YAML file records the generation recipe, artifact base, scope
digest, validation evidence, and whether the corpus is a smoke test, scale
probe, or final study dataset. Scale probes also track a measurements JSON
with per-architecture structural counts and artifact sizes.

Generated CircuitSAT, CNF, source, and private-oracle files remain under
`benchmarks/data/` and are not committed. Reproduce them with the commands in
`benchmarks/pipeline/README.md`, then run `collect_manifest.py` with the YAML's
declared `artifact_base`.
