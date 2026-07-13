//! GammaCover: the γ-OPTIMAL exact-cover counting branch arm (issue #35).
//!
//! `BlockMerge` (see `blockmerge.rs`) partitions the region's feasible set `S`
//! into perfect subcubes with a GREEDY rule — take the lowest config, grow it as
//! far as it goes, repeat. That is A legal partition, not the γ-minimal one. The
//! Optimal-Branching (OB) mechanism replaces a greedy rule with a small
//! optimizer that picks the branching rule minimizing the γ-measure (the root of
//! `Σ γ^{-Δρ_i} = 1` over the branches). The DECISION engine already does this
//! (`adapter.rs`, `bbsat`); the COUNTING engine did not. This module closes that
//! gap: `gamma_cover` selects a γ-minimal EXACT-COVER partition of `S` into
//! subcubes.
//!
//! Why exact cover, not OB's set cover. Decision-OB solves `Σ x ≥ 1` (a cover:
//! branches may overlap because a decision only needs SOME branch to hit a
//! solution). Counting SUMS every branch, so an overlap double-counts and a gap
//! undercounts — the partition must be EXACT: `Σ x = 1` for every config of `S`.
//! That single constraint change is the whole "decision-OB : set-cover ::
//! counting-OB : exact-cover" claim.
//!
//! What we reuse from the `optimal-branching` crate, and what we deliberately do
//! NOT. We reuse `complexity_bv` (γ from a branching vector by bisection) and the
//! `good_lp`/HiGHS stack (a DIRECT dependency here). We deliberately do NOT call
//! `minimize_gamma`: it runs `remove_dominated` before solving, which is valid
//! for a ≥1 cover but INVALID for an exact cover (a "dominated" small cube can be
//! REQUIRED to tile `S` exactly), and its IP builds `coverage.geq(1.0)`. So the
//! ~40-line γ fixed-point loop is reimplemented here over an exact-cover IP.
//!
//! Why minimizing `Σ γ^{-Δρ}` under `= 1` prefers BIG cubes (the elegant part).
//! `Δρ_i = popcount(mask_i)` = vars the cube fixes (`Measure::NumUnfixedVars`).
//! A big cube fixes FEW coords (small Δρ, weight near `γ^0 = 1`); a singleton
//! fixes ALL (large Δρ, tiny weight). Naively the tiny singleton weight looks
//! cheaper — but the EXACT-cover constraint forces you to tile every point:
//! covering one `2^f`-point block costs `γ^{-(w-f)}` as one big cube versus
//! `2^f · γ^{-w}` as `2^f` singletons, and `γ^f < 2^f` for `γ < 2`, so the big
//! cube wins. The `= 1` tiling requirement is what turns the γ objective into a
//! preference for few, big cubes.
//!
//! Totality. The candidate pool always includes every singleton, so the exact
//! cover is always feasible (the per-config partition). If the prime-implicant
//! pool overflows its cap, `|S|` exceeds the guard, or HiGHS fails, the arm falls
//! back to `block_merge` (a legal partition) and flags `fell_back`, so it is
//! total and never silently degrades to a greedy γ heuristic.

use std::collections::HashMap;

use good_lp::{
    default_solver, variable, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use optimal_branching_core::complexity_bv;

use crate::blockmerge::{block_merge, cube_points, partition_is_valid, Subcube};

/// Size guards for `gamma_cover`. `max_rows` caps `|S|` (the region row budget,
/// ≤ 128 as everywhere in the engine); `pool_cap` caps the candidate-cube pool
/// (prime implicants + singletons); `max_itr` caps the γ fixed-point iterations
/// (mirrors `IPSolver::max_itr = 20`). Overflow of either size guard ⇒ fallback.
#[derive(Clone, Copy, Debug)]
pub struct GammaCoverLimits {
    pub max_rows: usize,
    pub pool_cap: usize,
    pub max_itr: usize,
}

impl GammaCoverLimits {
    /// Defaults tied to the region row budget `max_rows`: pool cap 4096 (plan
    /// §3.1), 20 γ iterations (OB's `IPSolver::max_itr`).
    pub fn from_max_rows(max_rows: usize) -> GammaCoverLimits {
        GammaCoverLimits {
            max_rows,
            pool_cap: 4096,
            max_itr: 20,
        }
    }
}

/// The chosen partition: `cubes` exactly partition `S`, `gamma` is the γ-measure
/// of that partition (`1.0` when one cube covers all of `S`), and `fell_back` is
/// `true` iff a size guard or a HiGHS failure forced the `block_merge` fallback.
#[derive(Clone, Debug)]
pub struct GammaCoverResult {
    pub cubes: Vec<Subcube>,
    pub gamma: f64,
    pub fell_back: bool,
}

fn full_mask(width: usize) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// A `block_merge` partition wrapped as a fallback result (γ recomputed from the
/// cubes so the diagnostic stays meaningful). Total: `block_merge` is always a
/// legal partition of `S`.
fn fallback(configs: &[u64], width: usize) -> GammaCoverResult {
    let cubes = block_merge(configs, width);
    let bv: Vec<f64> = cubes.iter().map(|c| c.mask.count_ones() as f64).collect();
    let gamma = if bv.is_empty() {
        0.0
    } else {
        complexity_bv(&bv)
    };
    GammaCoverResult {
        cubes,
        gamma,
        fell_back: true,
    }
}

/// All PRIME IMPLICANTS of the on-set `S` (maximal subcubes fully contained in
/// `S`), by Quine–McCluskey merging on the minterm list. Level 0 is every config
/// as a fully-fixed cube; two cubes with the SAME fixed set that differ in one
/// fixed bit merge into a cube with that bit freed (their union stays ⊆ `S`,
/// since both halves are). A cube that never merges is prime. Returns `None` if
/// the working set ever exceeds `cap` (⇒ caller falls back). `configs` must be
/// sorted, deduped, and each `< 2^width`.
fn prime_implicants(configs: &[u64], width: usize, cap: usize) -> Option<Vec<Subcube>> {
    let full = full_mask(width);
    let mut current: Vec<(u64, u64)> = configs.iter().map(|&c| (full, c & full)).collect();
    let mut primes: Vec<Subcube> = Vec::new();

    loop {
        let level: std::collections::HashSet<(u64, u64)> = current.iter().copied().collect();
        let mut used: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
        let mut next: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();

        for &(mask, val) in &current {
            let mut fixed = mask;
            while fixed != 0 {
                let bit = fixed & fixed.wrapping_neg();
                fixed ^= bit;
                let neighbor = (mask, val ^ bit);
                if level.contains(&neighbor) {
                    used.insert((mask, val));
                    used.insert(neighbor);
                    let nmask = mask & !bit;
                    next.insert((nmask, val & nmask));
                }
            }
        }
        for &(mask, val) in &current {
            if !used.contains(&(mask, val)) {
                primes.push(Subcube {
                    mask,
                    val: val & mask,
                });
            }
        }
        if next.is_empty() {
            break;
        }
        current = next.into_iter().collect();
        if current.len() + primes.len() > cap {
            return None;
        }
    }
    if primes.len() > cap {
        return None;
    }
    Some(primes)
}

/// Solve the weighted EXACT cover: pick cubes minimizing `Σ weights[i] x_i`
/// subject to `Σ_{i ∋ c} x_i = 1` for every config `c`, `x_i` binary. `nsc` is the
/// cube (variable) count; `by_config[c]` lists the cube indices containing config
/// `c` (the constraint incidence, weight-independent so the caller builds it once
/// across the γ iterations). Returns the chosen cube indices, or `None` on a HiGHS
/// failure / infeasibility (never hit while singletons are in the pool, but total
/// either way). Weights are normalized by their max before solving — the argmin is
/// scale-invariant and the raw `γ^{-Δρ}` values span many orders of magnitude,
/// which HiGHS's MIP handles poorly (the same reason OB's `solve_cover_model`
/// rescales).
fn solve_exact_cover(nsc: usize, by_config: &[Vec<usize>], weights: &[f64]) -> Option<Vec<usize>> {
    if nsc == 0 {
        return if by_config.is_empty() {
            Some(Vec::new())
        } else {
            None
        };
    }
    let wmax = weights.iter().cloned().fold(0.0_f64, f64::max);
    let scale = if wmax > 0.0 { wmax } else { 1.0 };

    let mut problem = ProblemVariables::new();
    let vars: Vec<Variable> = (0..nsc)
        .map(|_| problem.add(variable().min(0).max(1).integer()))
        .collect();
    let objective: Expression = vars
        .iter()
        .enumerate()
        .map(|(i, &v)| (weights[i] / scale) * v)
        .sum();
    let mut model = problem.minimise(objective).using(default_solver);
    for cands in by_config {
        let coverage: Expression = cands.iter().map(|&i| vars[i]).sum();
        model = model.with(coverage.eq(1.0));
    }
    match model.solve() {
        Ok(sol) => Some((0..nsc).filter(|&i| sol.value(vars[i]) > 0.5).collect()),
        Err(_) => None,
    }
}

/// Select a γ-minimal EXACT-COVER partition of the region's feasible set
/// `configs` (`S ⊆ {0,1}^width`) into subcubes (issue #35, plan §3.1). Pipeline:
/// build the candidate pool (prime implicants of `S` ∪ every singleton), set
/// `Δρ_i = popcount(mask_i)`, then run the local γ fixed-point loop — weight each
/// cube `γ_old^{-Δρ_i}`, solve the weighted exact cover, recompute `γ` from the
/// chosen cubes via `complexity_bv`, iterate to a fixed point keeping the best γ.
/// Falls back to `block_merge` (flagging `fell_back`) on any size-guard overflow
/// or HiGHS failure. The returned partition is asserted valid on every build.
pub fn gamma_cover(configs: &[u64], width: usize, limits: &GammaCoverLimits) -> GammaCoverResult {
    if configs.is_empty() {
        return GammaCoverResult {
            cubes: Vec::new(),
            gamma: 0.0,
            fell_back: false,
        };
    }
    // Normalize S: sorted + deduped (count_component already does this, but the
    // pool/exact-cover indexing depends on it, so make it total).
    let mut s = configs.to_vec();
    s.sort_unstable();
    s.dedup();

    if s.len() > limits.max_rows {
        return fallback(&s, width);
    }

    let primes = match prime_implicants(&s, width, limits.pool_cap) {
        Some(p) => p,
        None => return fallback(&s, width),
    };

    // Pool = prime implicants + every singleton (singletons guarantee the exact
    // cover is feasible). Dedup by (mask, val).
    let full = full_mask(width);
    let mut pool = primes;
    let mut seen: std::collections::HashSet<(u64, u64)> =
        pool.iter().map(|c| (c.mask, c.val)).collect();
    for &c in &s {
        let key = (full, c & full);
        if seen.insert(key) {
            pool.push(Subcube {
                mask: full,
                val: c & full,
            });
        }
    }
    if pool.len() > limits.pool_cap {
        return fallback(&s, width);
    }

    // Short-circuit: one cube covers all of S (S is itself a full subcube) ⇒
    // single branch, γ = 1. Detectable from masks alone — every pool cube is ⊆ S,
    // so a cube whose `2^free` point count equals |S| IS S — so we can skip
    // building `covered`/`idx` (thousands of `cube_points` allocs) in this common
    // compressible case. Also sidesteps the degenerate all-covering MIP that trips
    // HiGHS.
    if let Some(pos) = pool.iter().position(|c| {
        let fixed = (c.mask & full).count_ones() as usize;
        1u128 << (width - fixed) == s.len() as u128
    }) {
        let cubes = vec![pool[pos]];
        assert!(
            partition_is_valid(&s, &cubes, width),
            "gammacover full-cover short-circuit is not a partition"
        );
        return GammaCoverResult {
            cubes,
            gamma: 1.0,
            fell_back: false,
        };
    }

    // Config value -> index in S; each cube's covered config indices, and the
    // weight-INDEPENDENT constraint incidence `by_config` (built once, reused
    // across every γ iteration).
    let idx: HashMap<u64, usize> = s.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let covered: Vec<Vec<usize>> = pool
        .iter()
        .map(|cube| {
            cube_points(cube.mask, cube.val, width)
                .iter()
                .map(|p| idx[p])
                .collect()
        })
        .collect();
    let mut by_config: Vec<Vec<usize>> = vec![Vec::new(); s.len()];
    for (i, cov) in covered.iter().enumerate() {
        for &c in cov {
            by_config[c].push(i);
        }
    }

    let delta_rho: Vec<f64> = pool.iter().map(|c| c.mask.count_ones() as f64).collect();

    // γ fixed-point loop (local exact-cover reimplementation of minimize_gamma's
    // iteration; see module docs for why minimize_gamma itself is unusable).
    let mut gamma_old = 2.0_f64;
    let mut best_gamma = f64::INFINITY;
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..limits.max_itr {
        let weights: Vec<f64> = delta_rho.iter().map(|&d| gamma_old.powf(-d)).collect();
        let chosen = match solve_exact_cover(pool.len(), &by_config, &weights) {
            Some(c) => c,
            None => return fallback(&s, width),
        };
        let bv: Vec<f64> = chosen.iter().map(|&i| delta_rho[i]).collect();
        let gamma_new = complexity_bv(&bv);
        if gamma_new < best_gamma {
            best_gamma = gamma_new;
            best = chosen;
        }
        if (gamma_new - gamma_old).abs() < 1e-6 * gamma_old.abs().max(1.0) {
            break;
        }
        gamma_old = gamma_new;
    }

    if best.is_empty() {
        // No iteration produced a cover (should not happen with singletons in the
        // pool); stay total.
        return fallback(&s, width);
    }
    let cubes: Vec<Subcube> = best.iter().map(|&i| pool[i]).collect();

    // Always-on partition check: a stray overlap/gap silently corrupts the count.
    assert!(
        partition_is_valid(&s, &cubes, width),
        "gammacover result is not an exact partition of S"
    );

    GammaCoverResult {
        cubes,
        gamma: best_gamma,
        fell_back: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> GammaCoverLimits {
        GammaCoverLimits::from_max_rows(128)
    }

    /// Independent re-derivation of coverage (each config in exactly one cube).
    fn covers_exactly(configs: &[u64], cubes: &[Subcube]) -> bool {
        configs.iter().all(|&s| {
            cubes
                .iter()
                .filter(|c| (s & c.mask) == (c.val & c.mask))
                .count()
                == 1
        })
    }

    /// Σ 2^free over the cubes = the number of points they tile (|S| for a valid
    /// partition). Used by the negative control to show a corruption moves it.
    fn tiled_points(cubes: &[Subcube], width: usize) -> u128 {
        cubes
            .iter()
            .map(|c| 1u128 << (width - c.mask.count_ones() as usize))
            .sum()
    }

    #[test]
    fn full_cube_is_single_branch_gamma_one() {
        // S = all 4 over 2 vars ⇒ one free-everything cube, γ = 1 short-circuit.
        let res = gamma_cover(&[0, 1, 2, 3], 2, &limits());
        assert!(!res.fell_back);
        assert_eq!(res.cubes, vec![Subcube { mask: 0, val: 0 }]);
        assert!((res.gamma - 1.0).abs() < 1e-9);
    }

    #[test]
    fn half_space_is_single_cube() {
        // S = {x1 = 1} over 2 vars = {2,3}: coord 0 free, 1 fixed ⇒ one cube.
        let res = gamma_cover(&[2, 3], 2, &limits());
        assert!(!res.fell_back);
        assert_eq!(res.cubes.len(), 1);
        assert_eq!(res.cubes[0].mask, 0b10);
        assert!((res.gamma - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parity_degenerates_to_singletons() {
        // Even parity over 3 vars {000,011,101,110}: no two configs differ in one
        // bit, so the only cubes are singletons ⇒ per-config partition (γ arm
        // cannot beat naive here — the pre-registered negative-control regime).
        let s = vec![0b000, 0b011, 0b101, 0b110];
        let res = gamma_cover(&s, 3, &limits());
        assert!(!res.fell_back);
        assert_eq!(res.cubes.len(), 4);
        assert!(res.cubes.iter().all(|c| c.mask == 0b111)); // all fully fixed
        assert!(covers_exactly(&s, &res.cubes));
        // γ = |S|^{1/width} = 4^{1/3} for 4 singletons of Δρ = 3.
        assert!((res.gamma - 4.0_f64.powf(1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn compressible_set_beats_perconfig() {
        // S = {00,01,10} over 2 vars: {00,01} is a perfect block (coord 0 free),
        // leaving {10}. γ arm must find 2 cubes, not 3 singletons.
        let s = vec![0b00, 0b01, 0b10];
        let res = gamma_cover(&s, 2, &limits());
        assert!(!res.fell_back);
        assert_eq!(res.cubes.len(), 2);
        assert!(covers_exactly(&s, &res.cubes));
        assert!(res.cubes.len() < s.len());
    }

    #[test]
    fn never_more_cubes_than_blockmerge_greedy() {
        // On a two-block union the γ optimum is at least as compact as greedy: S
        // = the 2-cube {bit2=0} plus one stray point over 3 vars.
        let s = vec![0b000, 0b001, 0b010, 0b011, 0b100];
        let res = gamma_cover(&s, 3, &limits());
        let greedy = block_merge(&s, 3);
        assert!(!res.fell_back);
        assert!(covers_exactly(&s, &res.cubes));
        assert!(res.cubes.len() <= greedy.len());
    }

    #[test]
    fn empty_set_is_no_cubes() {
        let res = gamma_cover(&[], 3, &limits());
        assert!(res.cubes.is_empty());
        assert!(!res.fell_back);
    }

    #[test]
    fn width_64_single_config() {
        // A lone config over the full 64-bit width ⇒ one singleton cube.
        let res = gamma_cover(&[0x00ff_00ff_00ff_00ff], 64, &limits());
        assert!(!res.fell_back);
        assert_eq!(res.cubes.len(), 1);
        assert_eq!(res.cubes[0].mask, u64::MAX);
        assert_eq!(res.cubes[0].val, 0x00ff_00ff_00ff_00ff);
    }

    #[test]
    fn pool_cap_overflow_falls_back() {
        // A tiny pool cap forces the fallback path; the result must still be a
        // legal partition (from block_merge) and flag fell_back.
        let s = vec![0b00, 0b01, 0b10];
        let tight = GammaCoverLimits {
            max_rows: 128,
            pool_cap: 1,
            max_itr: 20,
        };
        let res = gamma_cover(&s, 2, &tight);
        assert!(res.fell_back);
        assert_eq!(res.cubes, block_merge(&s, 2));
        assert!(covers_exactly(&s, &res.cubes));
    }

    #[test]
    fn max_rows_guard_falls_back() {
        // |S| over the max_rows guard ⇒ fallback without building the pool.
        let s = vec![0b00, 0b01, 0b10, 0b11];
        let tight = GammaCoverLimits {
            max_rows: 2,
            pool_cap: 4096,
            max_itr: 20,
        };
        let res = gamma_cover(&s, 2, &tight);
        assert!(res.fell_back);
    }

    #[test]
    fn corrupted_partition_is_detected() {
        // NEGATIVE CONTROL (plan §3.3): a valid γ partition passes the always-on
        // check; duplicating one emitted cube (the exact corruption the three-arm
        // test guards against) must FAIL it and change the tiled-point count. This
        // is why an overlap in the real partition would turn the count wrong and
        // the invariance test red.
        let s = vec![0b00, 0b01, 0b10];
        let res = gamma_cover(&s, 2, &limits());
        assert!(partition_is_valid(&s, &res.cubes, 2));
        assert_eq!(tiled_points(&res.cubes, 2), s.len() as u128);

        let mut corrupted = res.cubes.clone();
        corrupted.push(corrupted[0]); // duplicate one cube ⇒ overlap
        assert!(
            !partition_is_valid(&s, &corrupted, 2),
            "duplicated cube must be caught by the partition check"
        );
        assert_ne!(
            tiled_points(&corrupted, 2),
            s.len() as u128,
            "the corruption must move the counted point total"
        );
    }

    #[test]
    fn always_a_valid_partition_over_random_sets() {
        // 200 random config subsets over 3..=6 vars: gamma_cover must always
        // return an exact partition of S (the internal assert would fire
        // otherwise), never more cubes than |S|, and never spuriously fall back.
        fn next(s: &mut u64) -> u64 {
            let mut x = *s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            x
        }
        for seed in 1u64..=200 {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let width = 3 + (next(&mut s) % 4) as usize; // 3..=6
            let mut set: Vec<u64> = (0u64..(1u64 << width))
                .filter(|_| next(&mut s).is_multiple_of(2))
                .collect();
            set.sort_unstable();
            if set.is_empty() {
                continue;
            }
            let res = gamma_cover(&set, width, &limits());
            assert!(!res.fell_back, "seed {seed}: unexpected fallback");
            assert!(covers_exactly(&set, &res.cubes), "seed {seed}: coverage");
            assert!(
                res.cubes.len() <= set.len(),
                "seed {seed}: more cubes than configs"
            );
            assert_eq!(
                tiled_points(&res.cubes, width),
                set.len() as u128,
                "seed {seed}: tiled point count != |S|"
            );
        }
    }
}
