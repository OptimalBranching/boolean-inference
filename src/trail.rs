use crate::ct::RSparseBitSet;
use crate::domain::DomainMask;

/// One reversible mutation. `restore_to` replays these LIFO.
enum Undo {
    Dom { var: u32, old: DomainMask },
    Word { table: u32, word: u32, old: u64 },
    Limit { table: u32, old: u32 },
}

/// Undo log for in-place search. Records the OLD value before every write to
/// `doms` or to a tensor's `RSparseBitSet`; `restore_to(mark)` reverts LIFO.
/// A monotonic `epoch` (bumped on `open()` and `restore_to()`, never reused)
/// drives CT's save-word-once-per-scope stamping.
pub struct Trail {
    entries: Vec<Undo>,
    epoch: u64,
}

impl Trail {
    pub fn new() -> Trail {
        Trail {
            entries: Vec::new(),
            epoch: 1,
        }
    }

    #[inline]
    pub fn mark(&self) -> usize {
        self.entries.len()
    }
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Enter a new reversible scope (branch descent or probe). Never reused.
    #[inline]
    pub fn open(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    #[inline]
    pub fn record_dom(&mut self, var: usize, old: DomainMask) {
        self.entries.push(Undo::Dom {
            var: var as u32,
            old,
        });
    }
    #[inline]
    pub fn record_word(&mut self, table: usize, word: usize, old: u64) {
        self.entries.push(Undo::Word {
            table: table as u32,
            word: word as u32,
            old,
        });
    }
    #[inline]
    pub fn record_limit(&mut self, table: usize, old: u32) {
        self.entries.push(Undo::Limit {
            table: table as u32,
            old,
        });
    }

    /// Pop back to `mark`, restoring each recorded old value (LIFO), then bump
    /// `epoch` so the post-restore state is a fresh, never-reused epoch.
    pub fn restore_to(
        &mut self,
        mark: usize,
        doms: &mut [DomainMask],
        tables: &mut [RSparseBitSet],
    ) {
        while self.entries.len() > mark {
            match self.entries.pop().expect("len > mark") {
                Undo::Dom { var, old } => doms[var as usize] = old,
                Undo::Word { table, word, old } => {
                    tables[table as usize].words[word as usize] = old
                }
                Undo::Limit { table, old } => tables[table as usize].limit = old,
            }
        }
        self.epoch += 1;
    }

    /// Drop all entries without restoring (clone-and-return path).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for Trail {
    fn default() -> Self {
        Trail::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dom_record_and_restore_is_lifo() {
        let mut doms = vec![DomainMask::BOTH, DomainMask::BOTH];
        let mut tables: Vec<RSparseBitSet> = Vec::new();
        let mut tr = Trail::new();
        let m = tr.mark();
        tr.record_dom(0, doms[0]);
        doms[0] = DomainMask::D1;
        tr.record_dom(1, doms[1]);
        doms[1] = DomainMask::D0;
        tr.record_dom(0, doms[0]);
        doms[0] = DomainMask::NONE; // overwrite same var
        tr.restore_to(m, &mut doms, &mut tables);
        assert_eq!(doms, vec![DomainMask::BOTH, DomainMask::BOTH]);
    }

    #[test]
    fn epoch_is_monotonic_across_open_and_restore() {
        let mut doms: Vec<DomainMask> = Vec::new();
        let mut tables: Vec<RSparseBitSet> = Vec::new();
        let mut tr = Trail::new();
        let e0 = tr.epoch();
        let e1 = tr.open();
        assert!(e1 > e0);
        let m = tr.mark();
        tr.restore_to(m, &mut doms, &mut tables);
        assert!(
            tr.epoch() > e1,
            "restore must bump epoch so scopes never share one"
        );
    }
}
