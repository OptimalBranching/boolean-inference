use crate::domain::DomainMask;

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
}
