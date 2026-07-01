/// Reversible sparse bit-set for Compact Table propagation.
/// Fields `words` and `limit` are written by `Trail::restore_to`; all other
/// methods and fields are defined by Task 3.
pub struct RSparseBitSet {
    pub words: Vec<u64>,
    pub limit: u32,
}

use crate::domain::DomainMask;

/// Static, shared per-unique-tensor masks. `supports[(axis*2+value)*n_words + w]`
/// is word `w` of the bit-set over support rows where the config's `axis` bit == value.
pub struct TableMasks {
    pub n_rows: usize,
    pub n_words: usize,
    pub supports: Vec<u64>,
}

impl TableMasks {
    pub fn build(support: &[u32], arity: usize) -> TableMasks {
        let n_rows = support.len();
        let n_words = (n_rows + 63) / 64; // n_rows==0 => 0
        let mut supports = vec![0u64; arity * 2 * n_words];
        for (r, &config) in support.iter().enumerate() {
            let w = r / 64;
            let bit = 1u64 << (r % 64);
            for i in 0..arity {
                let v = ((config >> i) & 1) as usize;
                supports[(i * 2 + v) * n_words + w] |= bit;
            }
        }
        // High bits beyond n_rows in the last word are never set (loop bound = n_rows).
        TableMasks {
            n_rows,
            n_words,
            supports,
        }
    }

    #[inline]
    pub fn support_slice(&self, axis: usize, value: usize) -> &[u64] {
        let base = (axis * 2 + value) * self.n_words;
        &self.supports[base..base + self.n_words]
    }
}

#[cfg(test)]
mod masks_tests {
    use super::*;

    #[test]
    fn masks_index_rows_by_literal() {
        // arity 2, support = configs {0b01, 0b11} (rows 0,1). bit0==1 in both;
        // bit1==1 only in row1 (0b11).
        let support = vec![0b01u32, 0b11u32];
        let m = TableMasks::build(&support, 2);
        assert_eq!(m.n_rows, 2);
        assert_eq!(m.n_words, 1);
        assert_eq!(m.support_slice(0, 1)[0], 0b11); // axis0==1: rows 0,1
        assert_eq!(m.support_slice(0, 0)[0], 0b00); // axis0==0: none
        assert_eq!(m.support_slice(1, 1)[0], 0b10); // axis1==1: row1
        assert_eq!(m.support_slice(1, 0)[0], 0b01); // axis1==0: row0
    }

    #[test]
    fn empty_support_has_zero_words() {
        let m = TableMasks::build(&[], 2);
        assert_eq!(m.n_rows, 0);
        assert_eq!(m.n_words, 0);
        assert!(m.supports.is_empty());
    }
}
