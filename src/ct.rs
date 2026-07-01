use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::problem::SolverBuffer;
use crate::trail::Trail;

/// Reversible sparse bit-set over a tensor's support rows (Demeulenaere et al.,
/// CP 2016). `words` is physical-indexed and never reordered; `index[0..limit]`
/// is the active (possibly-nonzero) subset. Word contents are trailed
/// save-on-first-write per epoch; `limit` is trailed on shrink.
#[derive(Clone)]
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

/// Build per-unique-tensor `TableMasks` and one `RSparseBitSet` per instance
/// tensor. Masks are indexed like `cn.unique_tensors`; each instance table
/// points at its unique masks via `data_idx`.
pub fn build_tables(cn: &ConstraintNetwork) -> (Vec<TableMasks>, Vec<RSparseBitSet>) {
    let n_unique = cn.unique_tensors.len();
    let mut masks_opt: Vec<Option<TableMasks>> = (0..n_unique).map(|_| None).collect();
    for t in &cn.tensors {
        let uid = t.data_idx;
        if masks_opt[uid].is_none() {
            masks_opt[uid] = Some(TableMasks::build(
                &cn.unique_tensors[uid].support,
                t.var_axes.len(),
            ));
        }
    }
    let masks: Vec<TableMasks> = masks_opt
        .into_iter()
        .map(|m| m.expect("every unique tensor is used by some instance"))
        .collect();
    let tables: Vec<RSparseBitSet> = cn
        .tensors
        .iter()
        .map(|t| RSparseBitSet::new(&masks[t.data_idx], t.var_axes.len()))
        .collect();
    (masks, tables)
}

/// Compact-Table GAC propagation. Drains `buffer.queue`, mutating `doms` and the
/// live-row sets in `tables` in place, recording undo into `trail`. On any
/// contradiction sets the sentinel `doms[0] = NONE`, cleans the worklist, and
/// returns.
pub fn ct_propagate(
    cn: &ConstraintNetwork,
    doms: &mut [DomainMask],
    masks: &[TableMasks],
    tables: &mut [RSparseBitSet],
    buffer: &mut SolverBuffer,
    trail: &mut Trail,
) {
    let mut head = 0usize;
    while head < buffer.queue.len() {
        let tid = buffer.queue[head];
        head += 1;
        buffer.in_queue[tid] = false;

        let t = &cn.tensors[tid];
        let m = &masks[t.data_idx];
        if m.n_words == 0 {
            // empty support => unsatisfiable
            trail.record_dom(0, doms[0]);
            doms[0] = DomainMask::NONE;
            for &q in &buffer.queue[head..] {
                buffer.in_queue[q] = false;
            }
            buffer.queue.clear();
            return;
        }

        // 1. updateTable: restrict live rows to those consistent with current domains.
        let n_words = m.n_words;
        for (i, &var) in t.var_axes.iter().enumerate() {
            let d = doms[var];
            if d == DomainMask::BOTH {
                continue;
            }
            if d == DomainMask::NONE {
                trail.record_dom(0, doms[0]);
                doms[0] = DomainMask::NONE;
                for &q in &buffer.queue[head..] {
                    buffer.in_queue[q] = false;
                }
                buffer.queue.clear();
                return;
            }
            // Build the union of supports for the in-domain value(s) directly into
            // buffer.mask_scratch; pass the slice to intersect_with_mask.
            // buffer and tables are distinct parameters — no borrow conflict.
            for s in buffer.mask_scratch[..n_words].iter_mut() {
                *s = 0;
            }
            if d.has0() {
                for (w, &b) in m.support_slice(i, 0).iter().enumerate() {
                    buffer.mask_scratch[w] |= b;
                }
            }
            if d.has1() {
                for (w, &b) in m.support_slice(i, 1).iter().enumerate() {
                    buffer.mask_scratch[w] |= b;
                }
            }
            tables[tid].intersect_with_mask(tid, &buffer.mask_scratch[..n_words], trail);
        }

        // 2. contradiction
        if tables[tid].is_empty() {
            trail.record_dom(0, doms[0]);
            doms[0] = DomainMask::NONE;
            for &q in &buffer.queue[head..] {
                buffer.in_queue[q] = false;
            }
            buffer.queue.clear();
            return;
        }

        // 3. filterDomains
        for (i, &var) in t.var_axes.iter().enumerate() {
            if doms[var].is_fixed() {
                continue;
            }
            let can0 = tables[tid].intersect_index(m.support_slice(i, 0), i * 2);
            let can1 = tables[tid].intersect_index(m.support_slice(i, 1), i * 2 + 1);
            let new = match (can0, can1) {
                (true, true) => DomainMask::BOTH,
                (true, false) => DomainMask::D0,
                (false, true) => DomainMask::D1,
                (false, false) => DomainMask::NONE,
            };
            if new != doms[var] {
                trail.record_dom(var, doms[var]);
                doms[var] = new;
                if new == DomainMask::NONE {
                    trail.record_dom(0, doms[0]);
                    doms[0] = DomainMask::NONE;
                    for &q in &buffer.queue[head..] {
                        buffer.in_queue[q] = false;
                    }
                    buffer.queue.clear();
                    return;
                }
                for &nt in &cn.v2t[var] {
                    if !buffer.in_queue[nt] {
                        buffer.in_queue[nt] = true;
                        buffer.queue.push(nt);
                    }
                }
            }
        }
    }
    buffer.queue.clear();
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

#[cfg(test)]
mod engine_tests {
    use super::*;
    use crate::dimacs::network_from_dimacs;
    use crate::domain::DomainMask;
    use crate::problem::SolverBuffer;
    use crate::propagate::propagate_core_rescan;
    use crate::trail::Trail;

    // deterministic xorshift, no rng dep
    fn xs(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn rand_3sat(n: usize, m: usize, seed: u64) -> String {
        let mut s = seed;
        let mut out = format!("p cnf {n} {m}\n");
        for _ in 0..m {
            let mut lits = Vec::new();
            while lits.len() < 3 {
                let v = (xs(&mut s) as usize % n) + 1;
                if !lits.iter().any(|l: &i64| l.unsigned_abs() as usize == v) {
                    let sign = if xs(&mut s) & 1 == 0 { 1i64 } else { -1 };
                    lits.push(sign * v as i64);
                }
            }
            out.push_str(&format!("{} {} {} 0\n", lits[0], lits[1], lits[2]));
        }
        out
    }

    fn ct_state_invariant(
        cn: &ConstraintNetwork,
        doms: &[DomainMask],
        masks: &[TableMasks],
        tables: &[RSparseBitSet],
    ) {
        // row live in currTable <=> config consistent with current domains on every axis
        for (tid, t) in cn.tensors.iter().enumerate() {
            let _m = &masks[t.data_idx];
            for (r, &config) in cn.unique_tensors[t.data_idx].support.iter().enumerate() {
                let live = (tables[tid].words[r / 64] >> (r % 64)) & 1 == 1;
                let consistent = t.var_axes.iter().enumerate().all(|(i, &v)| {
                    let bit = ((config >> i) & 1) == 1;
                    if bit {
                        doms[v].has1()
                    } else {
                        doms[v].has0()
                    }
                });
                assert_eq!(live, consistent, "tensor {tid} row {r}");
            }
        }
    }

    // GAC fixpoint from scratch given an explicit set of (var, value) fixes.
    fn oracle(cn: &ConstraintNetwork, n: usize, fixes: &[(usize, bool)]) -> Vec<DomainMask> {
        let mut buf = SolverBuffer::new(cn);
        let mut doms = vec![DomainMask::BOTH; n];
        for &(v, b) in fixes {
            let nd = if b { DomainMask::D1 } else { DomainMask::D0 };
            if doms[v] != DomainMask::BOTH && doms[v] != nd {
                doms[0] = DomainMask::NONE;
            }
            doms[v] = nd;
            for &t in &cn.v2t[v] {
                if !buf.in_queue[t] {
                    buf.in_queue[t] = true;
                    buf.queue.push(t);
                }
            }
        }
        if doms[0] != DomainMask::NONE {
            propagate_core_rescan(cn, &mut doms, &mut buf);
        }
        doms
    }

    #[test]
    fn ct_matches_rescan_under_multivar_paths_and_backtrack() {
        // Coverage counters — proven to reach the hard paths across 300 seeds.
        let mut n_contradictions = 0usize; // (a) contradiction-exit paths hit
        let mut n_cascades = 0usize; // (b) NON-fixed var narrowed by propagation
        let mut n_backtrack_asserts = 0usize; // (c) restore_to round-trip asserts run

        for seed in 0..300u64 {
            // Dense near-phase-transition 3-SAT so multi-var fixes cascade and hit UNSAT.
            let cnf = rand_3sat(12, 50, 0x9E3779B97F4A7C15 ^ seed.wrapping_mul(2654435761));
            let cn = network_from_dimacs(&cnf).expect("parse");
            let (masks, mut tables) = build_tables(&cn);
            let n = cn.vars.len();

            let mut buf = SolverBuffer::new(&cn);
            let mut trail = Trail::new();
            let mut doms = vec![DomainMask::BOTH; n];

            let mut fixes: Vec<(usize, bool)> = Vec::new();
            let mut marks: Vec<usize> = Vec::new();
            let mut snaps: Vec<Vec<DomainMask>> = Vec::new();
            let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xABCDEF;

            for _ in 0..6 {
                if doms[0] == DomainMask::NONE {
                    break;
                }
                let unfixed: Vec<usize> = (0..n).filter(|&v| !doms[v].is_fixed()).collect();
                if unfixed.is_empty() {
                    break;
                }
                let var = unfixed[(xs(&mut s) as usize) % unfixed.len()];
                let val = (xs(&mut s) & 1) == 1;

                snaps.push(doms.clone());
                marks.push(trail.mark());
                trail.open();
                buf.queue.clear();
                for b in buf.in_queue.iter_mut() {
                    *b = false;
                }
                let nd = if val { DomainMask::D1 } else { DomainMask::D0 };
                trail.record_dom(var, doms[var]);
                doms[var] = nd;
                for &t in &cn.v2t[var] {
                    if !buf.in_queue[t] {
                        buf.in_queue[t] = true;
                        buf.queue.push(t);
                    }
                }
                ct_propagate(&cn, &mut doms, &masks, &mut tables, &mut buf, &mut trail);
                fixes.push((var, val));

                let od = oracle(&cn, n, &fixes);
                let cc = doms[0] == DomainMask::NONE;
                let co = od[0] == DomainMask::NONE;
                assert_eq!(
                    cc,
                    co,
                    "seed {seed} depth {}: contradiction agree",
                    fixes.len()
                );
                if cc {
                    n_contradictions += 1;
                } else {
                    assert_eq!(
                        doms,
                        od,
                        "seed {seed} depth {}: domains bit-identical",
                        fixes.len()
                    );
                    ct_state_invariant(&cn, &doms, &masks, &tables);
                    // Cascade check: any non-BOTH domain beyond the vars we explicitly
                    // fixed was narrowed by propagation.
                    let narrowed = (0..n)
                        .filter(|&v| doms[v] != DomainMask::BOTH)
                        .any(|v| !fixes.iter().any(|&(fv, _)| fv == v));
                    if narrowed {
                        n_cascades += 1;
                    }
                }
            }

            // Backtrack round-trip: undo each fix, recover each snapshot exactly.
            while let Some(mark) = marks.pop() {
                trail.restore_to(mark, &mut doms, &mut tables);
                let snap = snaps.pop().unwrap();
                assert_eq!(
                    doms,
                    snap,
                    "seed {seed}: doms restored to snapshot at depth {}",
                    marks.len()
                );
                n_backtrack_asserts += 1;
                fixes.pop();
                if doms[0] != DomainMask::NONE {
                    ct_state_invariant(&cn, &doms, &masks, &tables);
                }
            }
            assert!(
                doms.iter().all(|&d| d == DomainMask::BOTH),
                "seed {seed}: full restore to base"
            );
        }

        eprintln!(
            "coverage: contradictions={n_contradictions} cascades={n_cascades} backtrack_asserts={n_backtrack_asserts}"
        );
        assert!(
            n_contradictions > 0,
            "expected >=1 contradiction across 300 seeds"
        );
        assert!(
            n_cascades > 0,
            "expected >=1 propagation cascade across 300 seeds"
        );
        assert!(
            n_backtrack_asserts > 0,
            "expected backtrack asserts to execute"
        );
    }
}
