use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;

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
/// Returns (valid_or, valid_and, found_any).
#[inline]
pub fn scan_supports(
    support: &[u32],
    support_or: u32,
    support_and: u32,
    mask0: u32,
    mask1: u32,
) -> (u32, u32, bool) {
    let m = mask0 | mask1;
    if m == 0 {
        return (support_or, support_and, !support.is_empty());
    }
    let mut valid_or: u32 = 0;
    let mut valid_and: u32 = 0xFFFF_FFFF;
    let mut found = false;
    for &config in support {
        if config & m == mask1 {
            valid_or |= config;
            valid_and &= config;
            found = true;
            if valid_or == 0xFFFF_FFFF && valid_and == 0 {
                break;
            }
        }
    }
    (valid_or, valid_and, found)
}

#[inline]
fn enqueue_neighbors(queue: &mut Vec<usize>, in_queue: &mut [bool], neighbors: &[usize]) {
    for &t_idx in neighbors {
        if !in_queue[t_idx] {
            in_queue[t_idx] = true;
            queue.push(t_idx);
        }
    }
}

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
        debug_assert!(new_dom != DomainMask::NONE, "apply_updates must never narrow to NONE");
        if new_dom != old {
            doms[var_id] = new_dom;
            enqueue_neighbors(queue, in_queue, &cn.v2t[var_id]);
        }
    }
}

/// Drain the worklist seeded in `buffer.queue` / `buffer.in_queue`.
/// On an unsatisfiable tensor, sets `doms[0] = NONE` (contradiction sentinel).
pub fn propagate_core(cn: &ConstraintNetwork, doms: &mut [DomainMask], buffer: &mut SolverBuffer) {
    let mut head = 0usize;
    while head < buffer.queue.len() {
        let tensor_id = buffer.queue[head];
        head += 1;
        buffer.in_queue[tensor_id] = false;

        let tensor = &cn.tensors[tensor_id];
        let (m0, m1) = compute_query_masks(doms, &tensor.var_axes);
        let td = cn.data(tensor);
        let (valid_or, valid_and, found) =
            scan_supports(&td.support, td.support_or, td.support_and, m0, m1);
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
        let (vor, vand, found) = scan_supports(support.as_slice(), 0b11, 0b01, 0, 0b01);
        assert!(found);
        assert_eq!(vor, 0b11); // 01 | 11
        assert_eq!(vand, 0b01); // 01 & 11
    }

    #[test]
    fn scan_supports_no_match() {
        // require bit1 = 1 but no support config has it
        let support = vec![0b01u32];
        let (_, _, found) = scan_supports(support.as_slice(), 0b01, 0b01, 0, 0b10);
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
        propagate_core(&cn, &mut doms, &mut buf);
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
        propagate_core(&cn, &mut doms, &mut buf);
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
        propagate_core(&cn, &mut doms, &mut buf);
        assert!(has_contradiction(&doms));
        assert!(buf.queue.is_empty(), "queue must be cleared on contradiction");
        assert!(buf.in_queue.iter().all(|&q| !q), "no in_queue flag may remain set");
    }
}
