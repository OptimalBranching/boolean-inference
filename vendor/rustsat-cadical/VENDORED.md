# Vendored rustsat-cadical

This is the crates.io source for `rustsat-cadical` 0.7.5 (upstream RustSAT
commit `457d6d7bf27998947edc45fa2200d6a5fef6c389`) with its bundled CaDiCaL
2.2.1 source.

Local changes:

- remove an accidental `dbg!(res)` call from the CaDiCaL 2.1+ `Propagate`
  implementation, which otherwise writes one line to stderr per query;
- expose a narrow `maintain_learned_clauses` hook that runs CaDiCaL's own
  scheduled `reducing()`/`reduce()` policy after assumption conflicts. The
  assumption-propagation entry point performs conflict analysis outside the
  normal search loop, so without this hook its persistent learned database is
  never reduced.

No branching, restart, or clause-quality policy is replaced locally.
