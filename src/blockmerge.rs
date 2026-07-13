//! BlockMerge: the counting-specific branching rule (design doc §3.1).
//!
//! At a counting branch node the region has `width` (≤ 64) variables and a set
//! `S` of GAC-feasible configs (u64 rows, one bit per region var). The default
//! counting arm branches per-config — a partition of `S` into singletons. Both
//! arms must be an exact PARTITION of the solution space (branches mutually
//! exclusive and exhaustive), because counting SUMS every branch: a covering
//! rule that let two branches overlap would double-count.
//!
//! BlockMerge partitions `S` into pairwise-DISJOINT SUBCUBES, each of them FULLY
//! contained in `S` ("perfect blocks"). A subcube fixes some coordinates and
//! leaves the rest free; "fully contained in `S`" means EVERY one of its
//! `2^free` points is in `S`. Because the blocks are disjoint and their union is
//! exactly `S`, applying a block as a plain masked assignment (only its fixed
//! coordinates, no guard machinery) keeps the branches an exact partition — so
//! BlockMerge is STRICTLY correct for counting, whatever the recursion does with
//! the free coordinates. Per-config is the special case where every block is a
//! singleton (all coordinates fixed).
//!
//! The payoff is that a perfect block leaves region coordinates free, so a region
//! tensor often becomes a CONSTANT function under the partial fixing and the
//! existing delta accounting folds the whole `2^free`-point block in one node —
//! per-config can never do that. Correctness never DEPENDS on this collapse; it
//! comes from the partition property alone.

use std::collections::HashSet;

/// A subcube over a region's `width` coordinates. `mask` bit `i` set means
/// coordinate `i` is FIXED to `val`'s bit `i`; a clear `mask` bit means the
/// coordinate is FREE (ranges over both values). The cube's points are every
/// `x` with `x & mask == val & mask`; there are `2^(free coords)` of them. Fed
/// to `apply_masked_assignment` exactly like a per-config clause, only with a
/// non-full mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Subcube {
    pub mask: u64,
    pub val: u64,
}

/// The `2^free` points of the subcube `(mask, val)` over `width` coordinates:
/// the fixed coordinates hold `val`'s bits, the free coordinates enumerate every
/// combination. `free` is bounded by `log2(|S|)` in practice (a cube ⊆ `S`), so
/// this is cheap.
pub(crate) fn cube_points(mask: u64, val: u64, width: usize) -> Vec<u64> {
    let free: Vec<usize> = (0..width).filter(|&i| (mask >> i) & 1 == 0).collect();
    let base = val & mask;
    let mut pts = Vec::with_capacity(1usize << free.len());
    for combo in 0u64..(1u64 << free.len()) {
        let mut p = base;
        for (b, &fp) in free.iter().enumerate() {
            if (combo >> b) & 1 == 1 {
                p |= 1u64 << fp;
            }
        }
        pts.push(p);
    }
    pts
}

/// Partition `configs` (the region's feasible set `S`, assumed sorted+deduped)
/// into perfect blocks. Greedy MVP (design §3.1): repeatedly take the lowest
/// unclaimed config as a singleton cube, then free one coordinate at a time as
/// long as EVERY point the freeing adds is still unclaimed (so the grown cube
/// stays ⊆ `S` and disjoint from the blocks already emitted), retrying until no
/// coordinate can be freed; emit the maximal cube and remove its points from the
/// unclaimed set. Leftover configs that grow no further are singleton cubes.
///
/// `width` is the region size (≤ 64). Returns the blocks in seed order. Invariant
/// maintained across the greedy growth: the current cube's points are all
/// unclaimed, so verifying only the FLIPPED copies (the points a freeing adds)
/// suffices. Debug builds assert the strict partition of `S`; every build runs
/// the cheap `O(|S|·#cubes)` release check.
pub fn block_merge(configs: &[u64], width: usize) -> Vec<Subcube> {
    if configs.is_empty() {
        return Vec::new();
    }
    let full_mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let mut unclaimed: HashSet<u64> = configs.iter().copied().collect();
    let mut cubes: Vec<Subcube> = Vec::new();

    for &seed in configs {
        if !unclaimed.contains(&seed) {
            continue; // already absorbed into an earlier block
        }
        let mut mask = full_mask;
        let val = seed & full_mask;
        // Greedily free coordinates. Free the first coordinate whose doubled cube
        // is still entirely unclaimed, then rescan from the start (retry until no
        // coordinate can be freed), which yields a maximal cube.
        loop {
            let mut freed = false;
            for i in 0..width {
                let bit = 1u64 << i;
                if mask & bit == 0 {
                    continue; // already free
                }
                // Freeing `i` adds exactly the current cube points with bit `i`
                // flipped; the current points are unclaimed by the invariant, so
                // only the flipped copies need checking.
                let all_in = cube_points(mask, val, width)
                    .iter()
                    .all(|&p| unclaimed.contains(&(p ^ bit)));
                if all_in {
                    mask &= !bit;
                    freed = true;
                    break;
                }
            }
            if !freed {
                break;
            }
        }
        for p in cube_points(mask, val, width) {
            unclaimed.remove(&p);
        }
        cubes.push(Subcube {
            mask,
            val: val & mask,
        });
    }

    debug_assert!(unclaimed.is_empty(), "blockmerge left configs unclaimed");
    debug_assert_partition(configs, &cubes, width);
    // Always-on count-safety guard (shared with the γ arm).
    assert!(
        partition_is_valid(configs, &cubes, width),
        "blockmerge result is not an exact partition of S"
    );
    cubes
}

/// Debug-only strict partition proof: the cubes' points are exactly `S`, with no
/// overlap and no escape outside `S`.
#[cfg(debug_assertions)]
fn debug_assert_partition(configs: &[u64], cubes: &[Subcube], width: usize) {
    let s: HashSet<u64> = configs.iter().copied().collect();
    let mut seen: HashSet<u64> = HashSet::new();
    for c in cubes {
        for p in cube_points(c.mask, c.val, width) {
            assert!(s.contains(&p), "blockmerge cube escapes S at point {p:#x}");
            assert!(seen.insert(p), "blockmerge cubes overlap at point {p:#x}");
        }
    }
    assert_eq!(seen.len(), s.len(), "blockmerge cubes do not cover S");
}

#[cfg(not(debug_assertions))]
fn debug_assert_partition(_configs: &[u64], _cubes: &[Subcube], _width: usize) {}

/// Cheap `O(|S|·#cubes)` exact-partition check: `cubes`' total point count
/// equals `|S|` AND every config of `S` is matched by exactly one cube. Together
/// these force a strict partition with full containment (|S| matched points, all
/// distinct, and no extra points), guarding the counting sum from a stray overlap
/// or escape without the debug pass's `2^free` enumeration. Shared by both
/// subcube-partition arms (`block_merge` here and `gammacover::gamma_cover`) so
/// the load-bearing count-safety invariant lives in one place.
pub(crate) fn partition_is_valid(configs: &[u64], cubes: &[Subcube], width: usize) -> bool {
    let full_mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let mut total_points: u128 = 0;
    for c in cubes {
        let fixed = (c.mask & full_mask).count_ones() as usize;
        total_points += 1u128 << (width - fixed);
    }
    if total_points != configs.len() as u128 {
        return false;
    }
    configs.iter().all(|&s| {
        cubes
            .iter()
            .filter(|c| (s & c.mask) == (c.val & c.mask))
            .count()
            == 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every config of `S` is covered by exactly one returned cube (an
    /// independent re-derivation of the internal partition check).
    fn covers_exactly(configs: &[u64], cubes: &[Subcube]) -> bool {
        configs.iter().all(|&s| {
            cubes
                .iter()
                .filter(|c| (s & c.mask) == (c.val & c.mask))
                .count()
                == 1
        })
    }

    #[test]
    fn full_cube_merges_into_one_block() {
        // S = all 4 configs over 2 vars ⇒ one free-everything cube (mask 0).
        let cubes = block_merge(&[0, 1, 2, 3], 2);
        assert_eq!(cubes, vec![Subcube { mask: 0, val: 0 }]);
    }

    #[test]
    fn half_space_is_one_block() {
        // S = {x1 = 1} over 2 vars = configs {2,3}: coordinate 0 free, 1 fixed.
        let cubes = block_merge(&[2, 3], 2);
        assert_eq!(cubes.len(), 1);
        assert_eq!(cubes[0].mask, 0b10);
        assert_eq!(cubes[0].val, 0b10);
    }

    #[test]
    fn parity_set_does_not_compress() {
        // Even-parity over 2 vars = {00,11}: no two differ in exactly one bit, so
        // no coordinate can be freed — two singletons, compression ratio 1.
        let cubes = block_merge(&[0b00, 0b11], 2);
        assert_eq!(cubes.len(), 2);
        assert!(cubes.iter().all(|c| c.mask == 0b11)); // all fixed
        assert!(covers_exactly(&[0, 3], &cubes));
    }

    #[test]
    fn three_of_four_splits_into_line_plus_point() {
        // S = {00,01,10} over 2 vars: {00,01} is a perfect block (var0 free),
        // leaving {10} a singleton — 2 cubes for 3 configs.
        let cubes = block_merge(&[0b00, 0b01, 0b10], 2);
        assert_eq!(cubes.len(), 2);
        assert!(covers_exactly(&[0, 1, 2], &cubes));
    }

    #[test]
    fn three_var_full_cube() {
        // All 8 over 3 vars ⇒ single fully-free cube.
        let all: Vec<u64> = (0..8).collect();
        let cubes = block_merge(&all, 3);
        assert_eq!(cubes, vec![Subcube { mask: 0, val: 0 }]);
    }

    #[test]
    fn empty_set_is_no_blocks() {
        assert!(block_merge(&[], 3).is_empty());
    }

    #[test]
    fn every_config_covered_over_random_sets() {
        // 200 random config subsets over 4..6 vars: block_merge must always return
        // a strict partition (the internal asserts would fire otherwise) whose
        // cubes cover each config exactly once, and never MORE cubes than configs.
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
            let width = 4 + (next(&mut s) % 3) as usize; // 4..=6
            let mut set: Vec<u64> = (0u64..(1u64 << width))
                .filter(|_| next(&mut s) % 2 == 0)
                .collect();
            set.sort_unstable();
            if set.is_empty() {
                continue;
            }
            let cubes = block_merge(&set, width);
            assert!(covers_exactly(&set, &cubes), "seed {seed}: coverage");
            assert!(
                cubes.len() <= set.len(),
                "seed {seed}: more cubes than configs"
            );
        }
    }
}
