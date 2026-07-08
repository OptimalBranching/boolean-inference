use crate::ct::{
    apply_masked_assignment, ct_propagate, enqueue_var_change, RSparseBitSet, TableMasks,
};
use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::trail::Trail;

/// Domination fixing to a fixpoint (the MIS domination rule; the SAT pure
/// literal generalized to arbitrary tables). Variable `v` is dominated toward
/// value `b` when in EVERY incident tensor, every live row with `v = ¬b` has
/// its bit-flipped partner (same config, `v = b`) live too: any solution with
/// `v = ¬b` then maps to a solution with `v = b`, so `v = b` is fixed
/// (trailed) without branching. Satisfiability-preserving, never
/// verdict-changing.
///
/// One sweep collects all dominated vars before propagating: fixes are
/// mutually invariant (slicing a table on another var's fix removes a row and
/// its partner together, so a valid domination stays valid). After each
/// propagation the sweep repeats — shrunken tables can only create new
/// dominations. Precondition: `(doms, tables)` at a GAC fixpoint with
/// `buffer` drained; on return the same holds, or `doms[0] == NONE`.
pub fn dominate_fixpoint(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
) {
    loop {
        let mut fixed_any = false;
        #[allow(clippy::needless_range_loop)]
        'vars: for v in 0..doms.len() {
            if doms[v] != DomainMask::BOTH {
                continue;
            }
            // ok1: fixing v=1 is sound (every live v=0 row flips to a live
            // v=1 row in every incident tensor); ok0 symmetric.
            let (mut ok0, mut ok1) = (true, true);
            for &tid in &cn.v2t[v] {
                let t = &cn.tensors[tid];
                let m = &masks[t.table_idx];
                let mut axis = None;
                for (j, &u) in t.var_axes.iter().enumerate() {
                    if u == v {
                        if axis.is_some() {
                            // v on two axes: a single-bit flip is not a value
                            // flip of v — treat as non-dominatable.
                            continue 'vars;
                        }
                        axis = Some(j);
                    }
                }
                let j = axis.expect("v2t[v] tensor must contain v");
                let flip = m.flip_slice(j);
                if ok1 {
                    ok1 = tables[tid].flipped_supported(m.support_slice(j, 0), flip);
                }
                if ok0 {
                    ok0 = tables[tid].flipped_supported(m.support_slice(j, 1), flip);
                }
                if !ok0 && !ok1 {
                    continue 'vars;
                }
            }
            let nd = if ok1 {
                DomainMask::D1
            } else if ok0 {
                DomainMask::D0
            } else {
                continue;
            };
            trail.record_dom(v, doms[v]);
            doms[v] = nd;
            enqueue_var_change(cn, buffer, v);
            fixed_any = true;
        }
        if !fixed_any {
            return;
        }
        ct_propagate(cn, doms, masks, tables, buffer, trail);
        if doms[0] == DomainMask::NONE {
            return;
        }
    }
}

/// Returns (mask0, mask1): bit i set in mask0/mask1 iff var_axes[i] is fixed to 0/1.
#[inline]
pub fn compute_query_masks(doms: &[DomainMask], var_axes: &[usize]) -> (u32, u32) {
    debug_assert!(var_axes.len() <= 32);
    let mut mask0: u32 = 0;
    let mut mask1: u32 = 0;
    for (i, &var_id) in var_axes.iter().enumerate() {
        let bit = 1u32 << i;
        match doms[var_id] {
            DomainMask::D0 => mask0 |= bit,
            DomainMask::D1 => mask1 |= bit,
            _ => {}
        }
    }
    (mask0, mask1)
}

/// Scan a tensor's support for configs compatible with the fixed vars.
/// Returns (valid_or, valid_and, found_any). Used only by the rescan test
/// oracle (`propagate_core_rescan`) — the live propagator is CT.
#[cfg(test)]
fn scan_supports(support: &[u32], mask0: u32, mask1: u32) -> (u32, u32, bool) {
    let m = mask0 | mask1;
    let mut valid_or: u32 = 0;
    let mut valid_and: u32 = 0xFFFF_FFFF;
    let mut found = false;
    for &config in support {
        if config & m == mask1 {
            valid_or |= config;
            valid_and &= config;
            found = true;
        }
    }
    (valid_or, valid_and, found)
}

#[cfg(test)]
#[inline]
fn enqueue_neighbors(queue: &mut Vec<usize>, in_queue: &mut [bool], neighbors: &[usize]) {
    for &t_idx in neighbors {
        if !in_queue[t_idx] {
            in_queue[t_idx] = true;
            queue.push(t_idx);
        }
    }
}

#[cfg(test)]
#[inline]
fn apply_updates(
    doms: &mut [DomainMask],
    cn: &ConstraintNetwork,
    var_axes: &[usize],
    valid_or: u32,
    valid_and: u32,
    queue: &mut Vec<usize>,
    in_queue: &mut [bool],
) {
    for (i, &var_id) in var_axes.iter().enumerate() {
        let old = doms[var_id];
        if old == DomainMask::D0 || old == DomainMask::D1 {
            continue;
        }
        let bit = 1u32 << i;
        let can_be_1 = valid_or & bit != 0;
        let must_be_1 = valid_and & bit != 0;
        let new_dom = if must_be_1 {
            DomainMask::D1
        } else if can_be_1 {
            DomainMask::BOTH
        } else {
            DomainMask::D0
        };
        debug_assert!(
            new_dom != DomainMask::NONE,
            "apply_updates must never narrow to NONE"
        );
        if new_dom != old {
            doms[var_id] = new_dom;
            enqueue_neighbors(queue, in_queue, &cn.v2t[var_id]);
        }
    }
}

/// Fork from the current `(doms, tables)`: apply `vars`/`mask`/`val`, propagate
/// with Compact-Table, hand the live domains to `read`, then restore in place.
/// Every write to `doms` and to the live-row sets is trailed, so on return the
/// base `(doms, tables)` are exactly as they were before the call.
#[allow(clippy::too_many_arguments)]
pub fn probe<R>(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    vars: &[usize],
    mask: u64,
    val: u64,
    read: impl FnOnce(&[DomainMask]) -> R,
) -> R {
    trail.open();
    let m = trail.mark();
    buffer.reset_worklist();
    apply_masked_assignment(cn, doms, buffer, trail, vars, mask, val);
    ct_propagate(cn, doms, masks, tables, buffer, trail);
    let r = read(doms);
    trail.restore_to(m, doms, tables);
    r
}

/// MSB-first bit key reading `c`'s bits at the positions in `order`
/// (order[0] = most significant). Used to sort configs so trie subtrees are
/// contiguous ranges. `order.len()` <= 64 (region size fits in a u64 config).
#[inline]
fn key_of(c: u64, order: &[usize]) -> u64 {
    let mut k = 0u64;
    for &pos in order {
        k = (k << 1) | ((c >> pos) & 1);
    }
    k
}

/// DFS one trie level. `range` is the contiguous slice of the sorted config
/// list that agrees with the current path on `order[..level]`. Precondition:
/// `buffer` is drained clean and `doms`/`tables` reflect the parent prefix.
#[allow(clippy::too_many_arguments)]
fn descend(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    region_vars: &[usize],
    order: &[usize],
    range: &[u64],
    level: usize,
    out: &mut Vec<u64>,
) {
    if level == order.len() {
        // Leaf: the whole prefix is fixed and no ancestor contradicted, so every
        // config in `range` is feasible. (range is a single config in practice.)
        out.extend_from_slice(range);
        return;
    }
    let pos = order[level];
    let var = region_vars[pos];
    // `range` is sorted MSB-first by `order`, so at this level the configs with
    // bit `pos` == 0 form a contiguous run followed by those with bit `pos` == 1.
    let mut i = 0usize;
    while i < range.len() {
        let value = ((range[i] >> pos) & 1) as u8;
        let mut j = i;
        while j < range.len() && (((range[j] >> pos) & 1) as u8) == value {
            j += 1;
        }
        let sub = &range[i..j];
        i = j;

        trail.open(); // fresh epoch per trie edge — required for nested restore
        let m = trail.mark();
        debug_assert!(
            buffer.queue.is_empty(),
            "worklist must be drained per sibling"
        );
        let nd = if value == 1 {
            DomainMask::D1
        } else {
            DomainMask::D0
        };
        // `var` is unfixed here (order holds only unfixed positions) => real change.
        trail.record_dom(var, doms[var]);
        doms[var] = nd;
        enqueue_var_change(cn, buffer, var);
        ct_propagate(cn, doms, masks, tables, buffer, trail);
        if doms[0] != DomainMask::NONE {
            descend(
                cn,
                doms,
                masks,
                tables,
                buffer,
                trail,
                region_vars,
                order,
                sub,
                level + 1,
                out,
            );
        }
        trail.restore_to(m, doms, tables);
    }
}

/// Return the subset of `configs` that are GAC-feasible from the current
/// `(doms, tables)`, sharing the propagation of common prefixes via a trie DFS
/// over the region's UNFIXED variables. Set-identical to probing each config
/// independently with `probe(.., |d| d[0] != DomainMask::NONE)`.
///
/// Precondition: each config agrees with `doms` on already-fixed region vars
/// (caller applies the consistency filter). On return `(doms, tables)` and
/// `buffer` are exactly as on entry.
#[allow(clippy::too_many_arguments)]
pub fn feasible_configs(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
    region_vars: &[usize],
    configs: &[u64],
) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    if configs.is_empty() {
        return out;
    }
    // Trie levels = unfixed region-var positions, in region_vars (ascending) order.
    let order: Vec<usize> = (0..region_vars.len())
        .filter(|&pos| !doms[region_vars[pos]].is_fixed())
        .collect();
    if order.is_empty() {
        // All region vars fixed: each config equals the current assignment, so
        // feasibility is the (live) base state. Matches probing each as a no-op.
        if doms[0] != DomainMask::NONE {
            out.extend_from_slice(configs);
        }
        return out;
    }
    let mut sorted: Vec<u64> = configs.to_vec();
    sorted.sort_by_key(|&c| key_of(c, &order));

    // Clean the worklist once, as `probe` does.
    buffer.reset_worklist();
    descend(
        cn,
        doms,
        masks,
        tables,
        buffer,
        trail,
        region_vars,
        &order,
        &sorted,
        0,
        &mut out,
    );
    out
}

/// Pre-CT linear-rescan GAC propagator. Now used only as the differential oracle
/// in `ct::engine_tests` and this module's tests — the ob-core adapter path
/// (`apply_branch`) moved to CT — so it is `#[cfg(test)]`.
#[cfg(test)]
pub(crate) fn propagate_core_rescan(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    buffer: &mut SolverBuffer,
) {
    let mut head = 0usize;
    while head < buffer.queue.len() {
        let tensor_id = buffer.queue[head];
        head += 1;
        buffer.in_queue[tensor_id] = false;

        let tensor = &cn.tensors[tensor_id];
        let (m0, m1) = compute_query_masks(doms, &tensor.var_axes);
        let td = cn.table(tensor);
        let (valid_or, valid_and, found) = scan_supports(&td.support, m0, m1);
        if !found {
            doms[0] = DomainMask::NONE;
            // Leave the buffer consistent on the contradiction path: reset the
            // in_queue flags of the undrained worklist and clear the queue, so a
            // caller that reuses this buffer is not poisoned by stale state.
            for &t in &buffer.queue[head..] {
                buffer.in_queue[t] = false;
            }
            buffer.queue.clear();
            return;
        }
        apply_updates(
            doms,
            cn,
            &tensor.var_axes,
            valid_or,
            valid_and,
            &mut buffer.queue,
            &mut buffer.in_queue,
        );
    }
    buffer.queue.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DomainMask;

    #[test]
    fn domination_fixes_pure_literals_to_a_fixpoint() {
        // (x∨y) ∧ (x∨z): every literal is pure-positive, so domination alone
        // fixes all three vars to 1 — no branching, no contradiction.
        let or2 = vec![false, true, true, true];
        let cn =
            crate::network::setup_problem(3, vec![vec![0, 1], vec![0, 2]], vec![or2.clone(), or2]);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buffer = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        trail.open();
        dominate_fixpoint(&cn, &mut doms, &masks, &mut tables, &mut buffer, &mut trail);
        assert_eq!(doms, vec![DomainMask::D1; 3]);
    }

    #[test]
    fn domination_respects_polarity() {
        // (¬x∨¬y): both literals pure-negative — dominated to 0.
        let nand = vec![true, true, true, false];
        let cn = crate::network::setup_problem(2, vec![vec![0, 1]], vec![nand]);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 2];
        let mut buffer = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        trail.open();
        dominate_fixpoint(&cn, &mut doms, &masks, &mut tables, &mut buffer, &mut trail);
        assert_eq!(doms, vec![DomainMask::D0; 2]);
    }

    #[test]
    fn domination_leaves_xor_untouched() {
        // x⊕y: flipping either bit of a satisfying row leaves the support —
        // neither direction dominates, nothing is fixed.
        let xor = vec![false, true, true, false];
        let cn = crate::network::setup_problem(2, vec![vec![0, 1]], vec![xor]);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 2];
        let mut buffer = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        trail.open();
        dominate_fixpoint(&cn, &mut doms, &masks, &mut tables, &mut buffer, &mut trail);
        assert_eq!(doms, vec![DomainMask::BOTH; 2]);
    }

    #[test]
    fn domination_is_undone_by_the_trail() {
        let or2 = vec![false, true, true, true];
        let cn = crate::network::setup_problem(2, vec![vec![0, 1]], vec![or2]);
        let (masks, mut tables) = crate::ct::build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 2];
        let mut buffer = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        trail.open();
        let m = trail.mark();
        dominate_fixpoint(&cn, &mut doms, &masks, &mut tables, &mut buffer, &mut trail);
        assert_eq!(doms, vec![DomainMask::D1; 2]);
        trail.restore_to(m, &mut doms, &mut tables);
        assert_eq!(doms, vec![DomainMask::BOTH; 2]);
    }

    #[test]
    fn query_masks_pick_up_fixed_vars() {
        // var_axes = [0,1,2]; doms: v0=D1, v1=D0, v2=BOTH
        let doms = vec![DomainMask::D1, DomainMask::D0, DomainMask::BOTH];
        let (m0, m1) = compute_query_masks(&doms, &[0, 1, 2]);
        assert_eq!(m1, 0b001); // bit0 set (v0 = D1)
        assert_eq!(m0, 0b010); // bit1 set (v1 = D0)
    }

    #[test]
    fn scan_supports_filters_by_compatibility() {
        // support configs {0b01, 0b11}; require bit0 = 1 (mask1=0b01, mask0=0)
        let support = vec![0b01u32, 0b11u32];
        let (vor, vand, found) = scan_supports(support.as_slice(), 0, 0b01);
        assert!(found);
        assert_eq!(vor, 0b11); // 01 | 11
        assert_eq!(vand, 0b01); // 01 & 11
    }

    #[test]
    fn scan_supports_no_match() {
        // require bit1 = 1 but no support config has it
        let support = vec![0b01u32];
        let (_, _, found) = scan_supports(support.as_slice(), 0, 0b10);
        assert!(!found);
    }

    use crate::network::setup_problem;
    use crate::problem::{has_contradiction, SolverBuffer};

    #[test]
    fn unit_propagation_fixes_implied_var() {
        // single clause (x0 OR x1): tensor over [0,1], dense [F,T,T,T].
        // Fix x0 = 0 (D0). GAC must force x1 = 1 (D1).
        let cn = setup_problem(2, vec![vec![0, 1]], vec![vec![false, true, true, true]]);
        let mut doms = vec![DomainMask::D0, DomainMask::BOTH];
        let mut buf = SolverBuffer::new(&cn);
        // seed the queue with tensor 0
        buf.queue.push(0);
        buf.in_queue[0] = true;
        propagate_core_rescan(&cn, &mut doms, &mut buf);
        assert!(!has_contradiction(&doms));
        assert_eq!(doms[1], DomainMask::D1);
    }

    #[test]
    fn conflicting_assignment_yields_contradiction() {
        // clause (x0 OR x1); fix x0=0 and x1=0 -> unsatisfiable.
        let cn = setup_problem(2, vec![vec![0, 1]], vec![vec![false, true, true, true]]);
        let mut doms = vec![DomainMask::D0, DomainMask::D0];
        let mut buf = SolverBuffer::new(&cn);
        buf.queue.push(0);
        buf.in_queue[0] = true;
        propagate_core_rescan(&cn, &mut doms, &mut buf);
        assert!(has_contradiction(&doms));
    }

    #[test]
    fn propagate_core_leaves_clean_buffer_on_contradiction() {
        // clause (x0 OR x1); fix both to 0 -> contradiction. Buffer must be left clean.
        let cn = setup_problem(2, vec![vec![0, 1]], vec![vec![false, true, true, true]]);
        let mut doms = vec![DomainMask::D0, DomainMask::D0];
        let mut buf = SolverBuffer::new(&cn);
        buf.queue.push(0);
        buf.in_queue[0] = true;
        propagate_core_rescan(&cn, &mut doms, &mut buf);
        assert!(has_contradiction(&doms));
        assert!(
            buf.queue.is_empty(),
            "queue must be cleared on contradiction"
        );
        assert!(
            buf.in_queue.iter().all(|&q| !q),
            "no in_queue flag may remain set"
        );
    }

    #[test]
    fn probe_propagates_from_base_and_restores() {
        use crate::ct::build_tables;
        use crate::trail::Trail;
        let cn = setup_problem(2, vec![vec![0, 1]], vec![vec![false, true, true, true]]);
        let (masks, mut tables) = build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH, DomainMask::BOTH];
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        // probe x0 = 0 (mask bit0=1, value bit0=0)
        let (c0, c1) = probe(
            &cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buf,
            &mut trail,
            &[0],
            1u64,
            0u64,
            |d| (d[0], d[1]),
        );
        assert_eq!(c0, DomainMask::D0);
        assert_eq!(c1, DomainMask::D1); // forced
                                        // probe restores the base state
        assert_eq!(doms, vec![DomainMask::BOTH, DomainMask::BOTH]);
    }

    #[test]
    fn feasible_configs_matches_known_set_on_or_chain() {
        use crate::ct::build_tables;
        use crate::trail::Trail;
        // (x0 OR x1) AND (x1 OR x2): feasible assignments over [0,1,2] are
        // {010,011,101,110,111} = {2,3,5,6,7} (bit i = value of var i).
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![or2.clone(), or2]);
        let (masks, mut tables) = build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 3];
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let region_vars = vec![0usize, 1, 2];
        let all: Vec<u64> = (0u64..8).collect();
        let mut got = feasible_configs(
            &cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buf,
            &mut trail,
            &region_vars,
            &all,
        );
        got.sort_unstable();
        assert_eq!(got, vec![2, 3, 5, 6, 7]);
        // base state fully restored
        assert_eq!(doms, vec![DomainMask::BOTH; 3]);
        assert!(buf.queue.is_empty());
        assert!(buf.in_queue.iter().all(|&q| !q));
    }

    #[test]
    fn feasible_configs_empty_input_is_empty() {
        use crate::ct::build_tables;
        use crate::trail::Trail;
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(2, vec![vec![0, 1]], vec![or2]);
        let (masks, mut tables) = build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 2];
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let got = feasible_configs(
            &cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buf,
            &mut trail,
            &[0, 1],
            &[],
        );
        assert!(got.is_empty());
    }

    #[test]
    fn feasible_configs_prunes_infeasible_prefix() {
        use crate::ct::build_tables;
        use crate::trail::Trail;
        // (x0 OR x1): assignment 00 (=0) is the only infeasible one.
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(2, vec![vec![0, 1]], vec![or2]);
        let (masks, mut tables) = build_tables(&cn);
        let mut doms = vec![DomainMask::BOTH; 2];
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let mut got = feasible_configs(
            &cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buf,
            &mut trail,
            &[0, 1],
            &[0, 1, 2, 3],
        );
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3]); // 00 pruned
        assert_eq!(doms, vec![DomainMask::BOTH; 2]);
    }

    #[test]
    fn feasible_configs_all_region_vars_fixed_returns_all() {
        use crate::ct::build_tables;
        use crate::trail::Trail;
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(2, vec![vec![0, 1]], vec![or2]);
        let (masks, mut tables) = build_tables(&cn);
        // Fix both vars consistently (x0=1,x1=0 => config bit0=1 => value 1).
        let mut doms = vec![DomainMask::D1, DomainMask::D0];
        let mut buf = SolverBuffer::new(&cn);
        let mut trail = Trail::new();
        let got = feasible_configs(
            &cn,
            &mut doms,
            &masks,
            &mut tables,
            &mut buf,
            &mut trail,
            &[0, 1],
            &[1],
        );
        assert_eq!(got, vec![1]);
        assert_eq!(doms, vec![DomainMask::D1, DomainMask::D0]);
    }

    #[test]
    fn feasible_configs_matches_probe_oracle_randomized() {
        use crate::ct::build_tables;
        use crate::trail::Trail;

        // Tiny deterministic PRNG (xorshift64) — no external dep.
        fn next(s: &mut u64) -> u64 {
            let mut x = *s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            x
        }

        for seed in 1u64..=300 {
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            let n_vars = 3 + (next(&mut s) % 3) as usize; // 3..=5 vars
            let n_tensors = 2 + (next(&mut s) % 3) as usize; // 2..=4 tensors

            // Build tensors over random distinct-var scopes with random non-empty support.
            let mut scopes: Vec<Vec<usize>> = Vec::new();
            let mut tables_dense: Vec<Vec<bool>> = Vec::new();
            for _ in 0..n_tensors {
                let arity = 2 + (next(&mut s) % 2) as usize; // 2 or 3
                let mut vs: Vec<usize> = Vec::new();
                while vs.len() < arity {
                    let v = (next(&mut s) % n_vars as u64) as usize;
                    if !vs.contains(&v) {
                        vs.push(v);
                    }
                }
                let rows = 1usize << arity;
                let mut support = vec![false; rows];
                let mut any = false;
                for r in support.iter_mut() {
                    if next(&mut s) % 100 < 60 {
                        *r = true;
                        any = true;
                    }
                }
                if !any {
                    support[(next(&mut s) as usize) % rows] = true; // ensure non-empty
                }
                scopes.push(vs);
                tables_dense.push(support);
            }
            let cn = setup_problem(n_vars, scopes, tables_dense);
            // setup_problem compresses out vars that appear in no tensor; use
            // the compressed count for everything after construction.
            let n_cvars = cn.n_vars;
            let (masks, mut tables) = build_tables(&cn);
            let mut doms = vec![DomainMask::BOTH; n_cvars];
            let mut buf = SolverBuffer::new(&cn);
            let mut trail = Trail::new();

            // Establish a base fixpoint: randomly fix ~1/3 of vars, propagate from base.
            buf.queue.clear();
            for b in buf.in_queue.iter_mut() {
                *b = false;
            }
            for v in 0..n_cvars {
                if next(&mut s) % 3 == 0 {
                    let nd = if next(&mut s) & 1 == 1 {
                        DomainMask::D1
                    } else {
                        DomainMask::D0
                    };
                    trail.record_dom(v, doms[v]);
                    doms[v] = nd;
                    crate::ct::enqueue_var_change(&cn, &mut buf, v);
                }
            }
            ct_propagate(&cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail);
            if doms[0] == DomainMask::NONE {
                continue; // base already contradictory — skip this seed
            }

            // Region = all (compressed) vars. Candidate configs = all 2^n_cvars,
            // filtered to those consistent with the fixed vars (the caller's contract).
            let region_vars: Vec<usize> = (0..n_cvars).collect();
            let (check_mask, check_value) = mask_value_bits(&doms, &region_vars);
            let full_mask: u64 = if n_cvars >= 64 {
                u64::MAX
            } else {
                (1u64 << n_cvars) - 1
            };
            let all: Vec<u64> = (0u64..(1u64 << n_cvars))
                .filter(|c| (c & check_mask) == check_value)
                .collect();

            // Snapshot base for restore-integrity check.
            let doms_before = doms.clone();
            let words_before: Vec<Vec<u64>> = tables.iter().map(|t| t.words.clone()).collect();
            let limit_before: Vec<u32> = tables.iter().map(|t| t.limit).collect();

            // Oracle: probe each config independently.
            let mut want: Vec<u64> = Vec::new();
            for &c in &all {
                let feas = probe(
                    &cn,
                    &mut doms,
                    &masks,
                    &mut tables,
                    &mut buf,
                    &mut trail,
                    &region_vars,
                    full_mask,
                    c,
                    |d| d[0] != DomainMask::NONE,
                );
                if feas {
                    want.push(c);
                }
            }
            want.sort_unstable();

            // System under test.
            let mut got = feasible_configs(
                &cn,
                &mut doms,
                &masks,
                &mut tables,
                &mut buf,
                &mut trail,
                &region_vars,
                &all,
            );
            got.sort_unstable();

            assert_eq!(
                got, want,
                "seed {seed}: feasible set mismatch vs probe oracle"
            );

            // Restore integrity.
            assert_eq!(doms, doms_before, "seed {seed}: doms not restored");
            for (i, t) in tables.iter().enumerate() {
                assert_eq!(
                    t.words, words_before[i],
                    "seed {seed}: table {i} words not restored"
                );
                assert_eq!(
                    t.limit, limit_before[i],
                    "seed {seed}: table {i} limit not restored"
                );
            }
            assert!(buf.queue.is_empty(), "seed {seed}: worklist leaked");
            assert!(
                buf.in_queue.iter().all(|&q| !q),
                "seed {seed}: in_queue leaked"
            );
        }
    }

    /// (mask, value): bit i set in `mask` iff region_vars[i] is fixed; the same bit
    /// in `value` is its fixed value. Mirrors the call-site consistency filter.
    fn mask_value_bits(doms: &[DomainMask], region_vars: &[usize]) -> (u64, u64) {
        let mut mask = 0u64;
        let mut value = 0u64;
        for (i, &v) in region_vars.iter().enumerate() {
            match doms[v] {
                DomainMask::D0 => {
                    mask |= 1 << i;
                }
                DomainMask::D1 => {
                    mask |= 1 << i;
                    value |= 1 << i;
                }
                _ => {}
            }
        }
        (mask, value)
    }
}
