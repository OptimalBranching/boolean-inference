//! Counting weights: the sum-product semiring the counting engine accumulates in.
//!
//! The decision engine returns `bool` (OR over branches, AND over components,
//! short-circuit on the first success). The counting engine (`solver::bbcount`)
//! returns a `Weight` instead: branches SUM, components MULTIPLY, and there is no
//! short-circuit on success — every branch is traversed (see the counting section
//! of `docs/design/counting-solver.md`, §1/§4). The weight type lives only in the
//! coordination layer; the hot propagation path (CT/region/OB) never sees it.
//!
//! `free_factor(var)` is the factor an UNFIXED, TENSORLESS variable contributes
//! at a leaf: it ranges over both values with no constraint or weight biasing it,
//! so the factor is 2 (`= 1 + 1`, unit weight on each polarity). M2.2 literal
//! weights `w(v), w(¬v)` do NOT ride on `free_factor`; they enter as a 1-ary
//! WEIGHTED tensor per variable (design doc M2.2 / §2), which the weighted VE
//! front-end folds in and the search consumes like any other tensor. A variable
//! carrying a literal weight is therefore never tensorless, so its `w(v) + w(¬v)`
//! contribution is realised by SUMMING that 1-ary tensor's two rows — which is
//! exactly `w(v) + w(¬v)`, tested with NON-normalized weights (a normalized
//! `w + w̄ = 1` would silently mask a dropped weighted factor; §7 trait note).
//! `free_factor` stays the tensorless fallback: 2, the unit-weight value count.

use std::fmt::Debug;

use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::{One, Zero};

/// A commutative sum-product semiring over which the counting engine accumulates
/// model weights. `zero`/`one` are the additive/multiplicative identities;
/// `add` is branch aggregation (in place), `mul` is component/table aggregation.
///
/// `PartialEq` is a supertrait so the counting front-end can test whether a
/// sliced weighted tensor is a CONSTANT function (all rows equal) — the M2.1
/// entailment redefinition (design doc §7 blocker 1). It is `PartialEq` rather
/// than `Eq` so `f64` (which is not `Eq`) can be a weight.
pub trait Weight: Clone + Debug + PartialEq {
    /// Additive identity — the weight of an UNSAT branch/component.
    fn zero() -> Self;
    /// Multiplicative identity — the weight of an empty product (a fully-fixed,
    /// contradiction-free leaf with no free variables).
    fn one() -> Self;
    /// Branch aggregation: `self += rhs`.
    fn add(&mut self, rhs: &Self);
    /// Component/table aggregation: `self * rhs`.
    fn mul(&self, rhs: &Self) -> Self;
    /// The factor an unfixed, otherwise-unconstrained variable contributes at a
    /// leaf. Plain counting: 2. M2 literal weights: `w(v) + w(¬v)`.
    fn free_factor(var: usize) -> Self;
    /// `true` iff this is the additive identity — lets the component loop
    /// short-circuit a zero product without traversing sibling components.
    fn is_zero(&self) -> bool;
}

/// The DEFAULT exact-counting weight: arbitrary-precision unsigned integers.
/// #CSP solution counts routinely exceed 2^128 (thousands of bits), so a fixed
/// width would silently wrap; `BigUint` never does. Newtyped so the crate owns
/// the `Weight` impl (orphan rule) and so `free_factor`/`Debug` read cleanly.
#[derive(Clone, PartialEq, Eq)]
pub struct BigCount(pub BigUint);

impl BigCount {
    /// The counting result as a decimal string — the `models=<N>` field the
    /// counting front-ends print.
    pub fn to_decimal(&self) -> String {
        self.0.to_str_radix(10)
    }
}

impl Debug for BigCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Weight for BigCount {
    fn zero() -> Self {
        BigCount(BigUint::from(0u32))
    }
    fn one() -> Self {
        BigCount(BigUint::from(1u32))
    }
    fn add(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }
    fn mul(&self, rhs: &Self) -> Self {
        BigCount(&self.0 * &rhs.0)
    }
    fn free_factor(_var: usize) -> Self {
        BigCount(BigUint::from(2u32))
    }
    fn is_zero(&self) -> bool {
        self.0 == BigUint::from(0u32)
    }
}

/// A CHECKED `u128` weight for small instances where the count is known to fit.
/// Every `add`/`mul` uses the checked arithmetic and PANICS on overflow in ALL
/// build profiles — a `debug_assert` would let release silently wrap and emit a
/// wrong count, the single worst failure mode for a counter (§7 blocker 7). Use
/// `BigCount` when a fit cannot be guaranteed; this exists only as a fast path
/// with a loud, unmissable failure at the boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CheckedU128(pub u128);

impl CheckedU128 {
    pub fn value(&self) -> u128 {
        self.0
    }
}

impl Weight for CheckedU128 {
    fn zero() -> Self {
        CheckedU128(0)
    }
    fn one() -> Self {
        CheckedU128(1)
    }
    fn add(&mut self, rhs: &Self) {
        self.0 = self
            .0
            .checked_add(rhs.0)
            .expect("CheckedU128 overflow on add — the count exceeds u128; use BigCount");
    }
    fn mul(&self, rhs: &Self) -> Self {
        CheckedU128(
            self.0
                .checked_mul(rhs.0)
                .expect("CheckedU128 overflow on mul — the count exceeds u128; use BigCount"),
        )
    }
    fn free_factor(_var: usize) -> Self {
        CheckedU128(2)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

/// The WEIGHTED model-counting weight (M2.2): EXACT rationals. Literal weights
/// and partition functions are rationals; `BigRational` (arbitrary-precision
/// `p/q`) carries them with no rounding — the reviewed trap of using bare `f64`
/// for weighted counts (§7 blocker 7: "M2 加权用有理数,不用裸 f64"). Newtyped so
/// the crate owns the impl and `Debug`/`free_factor` read cleanly.
#[derive(Clone, PartialEq, Eq)]
pub struct RationalWeight(pub BigRational);

impl RationalWeight {
    /// Construct from an integer numerator/denominator pair (unnormalized literal
    /// weights are integers, so this is the common constructor).
    pub fn ratio(num: i64, den: i64) -> Self {
        RationalWeight(BigRational::new(BigInt::from(num), BigInt::from(den)))
    }
    /// Construct from an integer.
    pub fn int(n: i64) -> Self {
        RationalWeight(BigRational::from_integer(BigInt::from(n)))
    }
    /// Decimal-ish `p/q` string — the `models=<W>` field weighted front-ends print.
    pub fn to_ratio_string(&self) -> String {
        format!("{}/{}", self.0.numer(), self.0.denom())
    }

    /// Parse a weight literal EXACTLY: an integer (`3`), a decimal (`0.75` ⇒
    /// `3/4`), or an explicit ratio (`3/4`). Front-ends parse MCC weight strings
    /// this way so no `f64` rounding sneaks into an exact weighted count.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if let Some((num, den)) = s.split_once('/') {
            let n: BigInt = num
                .trim()
                .parse()
                .map_err(|_| format!("bad numerator {num:?}"))?;
            let d: BigInt = den
                .trim()
                .parse()
                .map_err(|_| format!("bad denominator {den:?}"))?;
            if d.is_zero() {
                return Err(format!("zero denominator in {s:?}"));
            }
            return Ok(RationalWeight(BigRational::new(n, d)));
        }
        if let Some((int, frac)) = s.split_once('.') {
            let sign = if int.trim_start().starts_with('-') {
                -1
            } else {
                1
            };
            let int_abs = int.trim().trim_start_matches('-');
            let ipart: BigInt = if int_abs.is_empty() {
                BigInt::from(0)
            } else {
                int_abs
                    .parse()
                    .map_err(|_| format!("bad integer part {int:?}"))?
            };
            let fpart: BigInt = if frac.is_empty() {
                BigInt::from(0)
            } else {
                frac.parse()
                    .map_err(|_| format!("bad fractional part {frac:?}"))?
            };
            let den = BigInt::from(10).pow(frac.len() as u32);
            let num = &ipart * &den + &fpart;
            let num = if sign < 0 { -num } else { num };
            return Ok(RationalWeight(BigRational::new(num, den)));
        }
        let n: BigInt = s.parse().map_err(|_| format!("bad integer {s:?}"))?;
        Ok(RationalWeight(BigRational::from_integer(n)))
    }
}

impl Debug for RationalWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.0.numer(), self.0.denom())
    }
}

impl Weight for RationalWeight {
    fn zero() -> Self {
        RationalWeight(BigRational::zero())
    }
    fn one() -> Self {
        RationalWeight(BigRational::one())
    }
    fn add(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }
    fn mul(&self, rhs: &Self) -> Self {
        RationalWeight(&self.0 * &rhs.0)
    }
    fn free_factor(_var: usize) -> Self {
        RationalWeight::int(2)
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

/// A FAST, APPROXIMATE weighted-counting weight: IEEE `f64`. No newtype is needed
/// — the `Weight` trait is local, so the orphan rule permits `impl Weight for
/// f64` directly. Use `RationalWeight` when exactness matters; `f64` trades it for
/// speed (and for the constant-tensor check, exact bit equality — a hazard the
/// exact path does not have, so the oracle exercises `RationalWeight`).
impl Weight for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn add(&mut self, rhs: &Self) {
        *self += *rhs;
    }
    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
    fn free_factor(_var: usize) -> Self {
        2.0
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_weight_is_a_sum_product_semiring() {
        // (2/3) + (2/3) = 4/3; (4/3) * (3/2) = 2.
        let mut a = RationalWeight::ratio(2, 3);
        a.add(&RationalWeight::ratio(2, 3));
        assert_eq!(a, RationalWeight::ratio(4, 3));
        let b = a.mul(&RationalWeight::ratio(3, 2));
        assert_eq!(b, RationalWeight::int(2));
        assert!(RationalWeight::zero().is_zero());
        // free_factor is the tensorless unit-weight value count = 2, NOT a literal
        // weight sum (those ride on 1-ary tensors, not free_factor).
        assert_eq!(RationalWeight::free_factor(3), RationalWeight::int(2));
    }

    #[test]
    fn f64_weight_is_a_sum_product_semiring() {
        let mut a = 2.0f64;
        <f64 as Weight>::add(&mut a, &3.0);
        assert_eq!(a, 5.0);
        assert_eq!(<f64 as Weight>::mul(&a, &2.0), 10.0);
        assert!(<f64 as Weight>::is_zero(&0.0));
        assert_eq!(<f64 as Weight>::free_factor(0), 2.0);
    }

    #[test]
    fn bigcount_is_a_sum_product_semiring() {
        let mut a = BigCount::one();
        a.add(&BigCount::one()); // 1 + 1 = 2
        assert_eq!(a.to_decimal(), "2");
        let b = a.mul(&BigCount(BigUint::from(3u32))); // 2 * 3 = 6
        assert_eq!(b.to_decimal(), "6");
        assert!(BigCount::zero().is_zero());
        assert!(!BigCount::one().is_zero());
        // free_factor is 2 for plain counting, regardless of var id.
        assert_eq!(BigCount::free_factor(7).to_decimal(), "2");
    }

    #[test]
    fn bigcount_exceeds_u128_without_wrapping() {
        // 2^200 has no u128 representation; BigUint holds it exactly.
        let mut w = BigCount::one();
        for _ in 0..200 {
            w = w.mul(&BigCount(BigUint::from(2u32)));
        }
        assert_eq!(w.0, BigUint::from(2u32).pow(200));
    }

    #[test]
    fn checked_u128_matches_bigcount_within_range() {
        let mut a = CheckedU128::one();
        a.add(&CheckedU128::one());
        let b = a.mul(&CheckedU128(3));
        assert_eq!(b.value(), 6);
        assert!(CheckedU128::zero().is_zero());
    }

    #[test]
    #[should_panic(expected = "overflow on mul")]
    fn checked_u128_panics_on_overflow_in_every_profile() {
        // 2^127 * 4 overflows u128 — must panic loudly even in release, never wrap.
        let big = CheckedU128(1u128 << 127);
        let _ = big.mul(&CheckedU128(4));
    }

    #[test]
    #[should_panic(expected = "overflow on add")]
    fn checked_u128_add_overflow_panics() {
        let mut m = CheckedU128(u128::MAX);
        m.add(&CheckedU128(1));
    }
}
