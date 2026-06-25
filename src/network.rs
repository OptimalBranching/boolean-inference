/// Shared truth-table data for a constraint tensor (flyweight; deduplicated).
#[derive(Clone, Debug)]
pub struct TensorData {
    /// dense[config] == true iff `config` satisfies the constraint.
    pub dense: Vec<bool>,
    /// Satisfied configs (0-indexed), the sparse "support".
    pub support: Vec<u32>,
    /// OR over all support configs (fast feasibility scan).
    pub support_or: u32,
    /// AND over all support configs (fast feasibility scan).
    pub support_and: u32,
}

impl TensorData {
    pub fn from_dense(dense: Vec<bool>) -> TensorData {
        let mut support = Vec::new();
        let mut support_or: u32 = 0;
        let mut support_and: u32 = 0xFFFF_FFFF;
        for (i, &sat) in dense.iter().enumerate() {
            if sat {
                let config = i as u32;
                support.push(config);
                support_or |= config;
                support_and &= config;
            }
        }
        TensorData {
            dense,
            support,
            support_or,
            support_and,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensordata_extracts_support_and_aggregates() {
        // 2-var tensor, satisfied configs = {01, 11} (i.e. index 1 and 3)
        let dense = vec![false, true, false, true];
        let td = TensorData::from_dense(dense);
        assert_eq!(td.support, vec![1u32, 3u32]);
        assert_eq!(td.support_or, 0b11); // 1 | 3
        assert_eq!(td.support_and, 0b01); // 1 & 3
    }

    #[test]
    fn tensordata_empty_support_aggregates() {
        let td = TensorData::from_dense(vec![false, false]);
        assert!(td.support.is_empty());
        assert_eq!(td.support_or, 0);
        assert_eq!(td.support_and, 0xFFFF_FFFF);
    }
}
