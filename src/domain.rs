/// A 2-bit domain over a boolean variable: bit0 = "can be 0", bit1 = "can be 1".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DomainMask(pub u8);

impl DomainMask {
    pub const NONE: DomainMask = DomainMask(0b00);
    pub const D0: DomainMask = DomainMask(0b01);
    pub const D1: DomainMask = DomainMask(0b10);
    pub const BOTH: DomainMask = DomainMask(0b11);

    #[inline]
    pub fn has0(self) -> bool {
        self.0 & 0b01 != 0
    }
    #[inline]
    pub fn has1(self) -> bool {
        self.0 & 0b10 != 0
    }
    #[inline]
    pub fn is_fixed(self) -> bool {
        self == DomainMask::D0 || self == DomainMask::D1
    }
    #[inline]
    pub fn value(self) -> Option<bool> {
        match self {
            DomainMask::D0 => Some(false),
            DomainMask::D1 => Some(true),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_mask_semantics() {
        assert!(DomainMask::D0.is_fixed());
        assert!(DomainMask::D1.is_fixed());
        assert!(!DomainMask::BOTH.is_fixed());
        assert!(!DomainMask::NONE.is_fixed());

        assert!(DomainMask::D0.has0() && !DomainMask::D0.has1());
        assert!(!DomainMask::D1.has0() && DomainMask::D1.has1());
        assert!(DomainMask::BOTH.has0() && DomainMask::BOTH.has1());

        assert_eq!(DomainMask::D0.value(), Some(false));
        assert_eq!(DomainMask::D1.value(), Some(true));
        assert_eq!(DomainMask::BOTH.value(), None);
    }
}
