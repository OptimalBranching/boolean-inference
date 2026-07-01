#[cfg(test)]
use crate::domain::DomainMask;
use crate::trail::Trail;

/// Reversible sparse bit-set over a tensor's support rows (Demeulenaere et al.,
/// CP 2016). `words` is physical-indexed and never reordered; `index[0..limit]`
/// is the active (possibly-nonzero) subset. Word contents are trailed
/// save-on-first-write per epoch; `limit` is trailed on shrink.
pub struct RSparseBitSet {
    pub words: Vec<u64>,
    saved_epoch: Vec<u64>,
    index: Vec<u32>,
    pub limit: u32,
    residue: Vec<u32>, // [axis*2+value] -> physical word id last seen supporting
}

impl RSparseBitSet {
    pub fn new(masks: &TableMasks, arity: usize) -> RSparseBitSet {
        let nw = masks.n_words;
        let mut words = vec![u64::MAX; nw];
        if nw > 0 {
            let rem = masks.n_rows % 64;
            if rem != 0 {
                words[nw - 1] = (1u64 << rem) - 1; // zero high bits beyond n_rows
            }
        }
        RSparseBitSet {
            words,
            saved_epoch: vec![0u64; nw],
            index: (0..nw as u32).collect(),
            limit: nw as u32,
            residue: vec![0u32; arity * 2],
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.limit == 0
    }

    #[inline]
    fn save_word(&mut self, table_id: usize, w: usize, trail: &mut Trail) {
        if self.saved_epoch[w] != trail.epoch() {
            trail.record_word(table_id, w, self.words[w]);
            self.saved_epoch[w] = trail.epoch();
        }
    }

    /// `words &= mask` over active words; swap out any word that becomes 0.
    pub fn intersect_with_mask(&mut self, table_id: usize, mask: &[u64], trail: &mut Trail) {
        let mut p = 0usize;
        while p < self.limit as usize {
            let w = self.index[p] as usize;
            let nw = self.words[w] & mask[w];
            if nw != self.words[w] {
                self.save_word(table_id, w, trail);
                self.words[w] = nw;
            }
            if nw == 0 {
                // swap-out: move w past the active prefix, shrink limit (trailed once)
                trail.record_limit(table_id, self.limit);
                self.limit -= 1;
                self.index.swap(p, self.limit as usize);
                // do not advance p: the swapped-in word must be examined
            } else {
                p += 1;
            }
        }
    }

    /// Does any live row satisfy `mask`? Uses the residue hint (physical word id).
    /// `key` = axis*2+value, used to cache/refresh the residue.
    pub fn intersect_index(&mut self, mask: &[u64], key: usize) -> bool {
        let r = self.residue[key] as usize;
        if r < self.words.len() && (self.words[r] & mask[r]) != 0 {
            return true;
        }
        for p in 0..self.limit as usize {
            let w = self.index[p] as usize;
            if (self.words[w] & mask[w]) != 0 {
                self.residue[key] = w as u32;
                return true;
            }
        }
        false
    }
}

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
mod rsbs_tests {
    use super::*;

    fn masks() -> TableMasks {
        // arity 2, support {01, 11}
        TableMasks::build(&[0b01u32, 0b11u32], 2)
    }

    #[test]
    fn new_has_all_rows_live() {
        let m = masks();
        let s = RSparseBitSet::new(&m, 2);
        assert_eq!(s.limit, 1);
        assert_eq!(s.words[0], 0b11);
        assert!(!s.is_empty());
    }

    #[test]
    fn intersect_prunes_and_restores() {
        let m = masks();
        let mut s = RSparseBitSet::new(&m, 2);
        let mut tr = Trail::new();
        tr.open();
        let mk = tr.mark();
        // Require axis1==1 -> mask = support_slice(1,1) = row1 only (0b10).
        let want = m.support_slice(1, 1).to_vec();
        s.intersect_with_mask(0, &want, &mut tr);
        assert_eq!(s.words[0], 0b10);
        assert_eq!(s.limit, 1);
        assert!(s.intersect_index(m.support_slice(0, 1), 2)); // axis0==1 still supported (row1)
                                                              // restore
        let mut tables = vec![RSparseBitSet::new(&m, 2)];
        std::mem::swap(&mut tables[0], &mut s);
        let mut doms: Vec<DomainMask> = Vec::new();
        tr.restore_to(mk, &mut doms, &mut tables);
        assert_eq!(tables[0].words[0], 0b11, "word restored");
        assert_eq!(tables[0].limit, 1, "limit restored");
    }

    #[test]
    fn intersect_to_empty_sets_is_empty() {
        let m = masks();
        let mut s = RSparseBitSet::new(&m, 2);
        let mut tr = Trail::new();
        tr.open();
        // axis0==0 -> no rows (support_slice(0,0) == 0) -> empties the set.
        let mask0 = m.support_slice(0, 0).to_vec();
        s.intersect_with_mask(0, &mask0, &mut tr);
        assert!(s.is_empty());
    }

    #[test]
    fn empty_table_is_empty() {
        let m = TableMasks::build(&[], 2);
        let s = RSparseBitSet::new(&m, 2);
        assert!(s.is_empty());
    }

    /// Exercises the `Undo::Limit` restore-replay path end-to-end.
    ///
    /// Intersecting with an all-zero mask zeroes the single active word,
    /// pushing both `Undo::Word` and `Undo::Limit` onto the trail.
    /// `restore_to` must replay them LIFO so that `limit` is raised back to 1
    /// AND the word's original bits are reinstated — fully reactivating the tensor.
    #[test]
    fn limit_restore_reactivates_swapped_out_word() {
        let m = masks();
        let mut tables = vec![RSparseBitSet::new(&m, 2)];
        let mut doms: Vec<DomainMask> = Vec::new();
        let mut tr = Trail::new();
        tr.open();
        let mk = tr.mark();

        // axis0==0 support is empty (no row has bit0==0) → mask = [0] → zeros word 0
        // and swaps it out: trail receives Undo::Word{old:0b11} then Undo::Limit{old:1}.
        let mask0 = m.support_slice(0, 0).to_vec();
        tables[0].intersect_with_mask(0, &mask0, &mut tr);
        assert!(
            tables[0].is_empty(),
            "after zeroing the only word the set must be empty (limit==0)"
        );

        // restore_to replays LIFO: Undo::Limit first (limit→1), then Undo::Word (words[0]→0b11).
        tr.restore_to(mk, &mut doms, &mut tables);

        assert_eq!(tables[0].limit, 1, "Limit arm must restore limit to 1");
        assert!(
            !tables[0].is_empty(),
            "set must be non-empty after Limit arm replayed"
        );
        assert_eq!(
            tables[0].words[0], 0b11,
            "Word arm must reinstate the original word bits"
        );
    }

    /// Two-word variant: one word is swapped out while the other stays live.
    ///
    /// 65 rows (→ n_words==2): rows 0..63 have config 0b10 (bit1=1, bit0=0),
    /// row 64 has config 0b01 (bit1=0, bit0=1).
    /// mask = support_slice(1, 0) = [0, 1]: zeros word 0 (rows 0..63), keeps word 1.
    /// After restore, both words must be live (limit==2) with correct bit patterns,
    /// confirming that the Limit arm raises the count and the Word arm restores bits.
    #[test]
    fn limit_restore_two_words_active_set_correct() {
        let mut support: Vec<u32> = vec![0b10u32; 64]; // rows 0..63: bit1=1 bit0=0
        support.push(0b01u32); // row 64: bit1=0 bit0=1
        let m = TableMasks::build(&support, 2);
        assert_eq!(m.n_words, 2, "65 rows must yield 2 words");

        let mut tables = vec![RSparseBitSet::new(&m, 2)];
        let mut doms: Vec<DomainMask> = Vec::new();
        let mut tr = Trail::new();
        tr.open();
        let mk = tr.mark();

        // axis1==0 rows: only row 64 → mask = [0x0000…0, 0x1].
        // word 0 is zeroed and swapped out; word 1 (value 1) stays active.
        let mask = m.support_slice(1, 0).to_vec();
        assert_eq!(mask[0], 0u64, "word0 of axis1==0 mask must be zero");
        assert_eq!(mask[1], 1u64, "word1 of axis1==0 mask must be 1");

        tables[0].intersect_with_mask(0, &mask, &mut tr);
        assert_eq!(tables[0].limit, 1, "word0 swapped out → limit must be 1");
        assert!(!tables[0].is_empty());
        assert_eq!(
            tables[0].words[0], 0u64,
            "word0 must be zeroed after intersect"
        );
        assert_eq!(
            tables[0].words[1], 1u64,
            "word1 must be unchanged after intersect"
        );

        // Restore: Undo::Limit → limit back to 2; Undo::Word → words[0] back to u64::MAX.
        tr.restore_to(mk, &mut doms, &mut tables);

        assert_eq!(
            tables[0].limit, 2,
            "Limit arm must restore limit to 2 (both words live)"
        );
        assert!(!tables[0].is_empty());
        assert_eq!(
            tables[0].words[0],
            u64::MAX,
            "Word arm must restore word0 to all-64-bits-set"
        );
        assert_eq!(
            tables[0].words[1], 1u64,
            "word1 was never modified; must still be 1"
        );
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
