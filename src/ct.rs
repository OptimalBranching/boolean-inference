/// Reversible sparse bit-set for Compact Table propagation.
/// Fields `words` and `limit` are written by `Trail::restore_to`; all other
/// methods and fields are defined by Task 3.
pub struct RSparseBitSet {
    pub words: Vec<u64>,
    pub limit: u32,
}
