use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;

/// `(mask, value)` over `vars`: bit *i* of `mask` is set iff `vars[i]` is fixed;
/// bit *i* of `value` is that fixed bit's value. Port of `utils.jl::mask_value`.
pub fn mask_value_u64(doms: &[DomainMask], vars: &[usize]) -> (u64, u64) {
    debug_assert!(vars.len() <= 64, "mask_value_u64 supports at most 64 vars");
    let mut mask = 0u64;
    let mut value = 0u64;
    for (i, &v) in vars.iter().enumerate() {
        match doms[v] {
            DomainMask::D1 => {
                mask |= 1u64 << i;
                value |= 1u64 << i;
            }
            DomainMask::D0 => mask |= 1u64 << i,
            _ => {}
        }
    }
    (mask, value)
}

/// Tensor ids that still have at least one unfixed variable, ascending.
/// Port of `utils.jl::get_active_tensors`.
pub fn get_active_tensors(cn: &ConstraintNetwork, doms: &[DomainMask]) -> Vec<usize> {
    let mut active = Vec::with_capacity(cn.tensors.len());
    for (tid, t) in cn.tensors.iter().enumerate() {
        if t.var_axes.iter().any(|&v| !doms[v].is_fixed()) {
            active.push(tid);
        }
    }
    active
}

/// True iff every tensor has at most 2 unfixed variables (the residual is 2-SAT).
/// Port of `utils.jl::is_two_sat` — iterates ALL tensors, not just active ones.
pub fn is_two_sat(cn: &ConstraintNetwork, doms: &[DomainMask]) -> bool {
    for t in &cn.tensors {
        let mut unfixed = 0usize;
        for &v in &t.var_axes {
            if !doms[v].is_fixed() {
                unfixed += 1;
                if unfixed > 2 {
                    return false;
                }
            }
        }
    }
    true
}

/// Number of unfixed domains. Port of `utils.jl::count_unfixed`.
pub fn count_unfixed(doms: &[DomainMask]) -> usize {
    doms.iter().filter(|d| !d.is_fixed()).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    #[test]
    fn mask_value_marks_fixed_bits() {
        // vars [0,1,2]; doms: v0=D1, v1=D0, v2=BOTH.
        let doms = vec![DomainMask::D1, DomainMask::D0, DomainMask::BOTH];
        let (mask, value) = mask_value_u64(&doms, &[0, 1, 2]);
        assert_eq!(mask, 0b011); // bits 0,1 fixed; bit 2 free
        assert_eq!(value, 0b001); // v0=1 -> bit0 set; v1=0 -> bit1 clear
    }

    #[test]
    fn active_tensors_have_an_unfixed_var() {
        // T0 over [0,1], T1 over [2]. Fix v2 -> T1 inactive, T0 active.
        let cn = setup_problem(
            3,
            vec![vec![0, 1], vec![2]],
            vec![vec![false, true, true, true], vec![false, true]],
        );
        let doms = vec![DomainMask::BOTH, DomainMask::BOTH, DomainMask::D1];
        assert_eq!(get_active_tensors(&cn, &doms), vec![0]);
    }

    #[test]
    fn is_two_sat_detects_a_hard_tensor() {
        let or2 = vec![false, true, true, true];
        let t3 = vec![false, true, true, true, true, true, true, true]; // degree-3 OR
        let bin_only = setup_problem(
            3,
            vec![vec![0, 1], vec![1, 2]],
            vec![or2.clone(), or2.clone()],
        );
        let with_hard = setup_problem(3, vec![vec![0, 1, 2], vec![0, 1]], vec![t3, or2]);
        let doms = vec![DomainMask::BOTH; 3];
        assert!(is_two_sat(&bin_only, &doms));
        assert!(!is_two_sat(&with_hard, &doms));
        // Fixing v2 drops the degree-3 tensor to degree 2 -> 2-SAT again.
        let doms2 = vec![DomainMask::BOTH, DomainMask::BOTH, DomainMask::D0];
        assert!(is_two_sat(&with_hard, &doms2));
    }

    #[test]
    fn count_unfixed_counts_free_vars() {
        let doms = vec![
            DomainMask::D1,
            DomainMask::BOTH,
            DomainMask::D0,
            DomainMask::BOTH,
        ];
        assert_eq!(count_unfixed(&doms), 2);
    }
}
