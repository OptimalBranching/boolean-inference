use crate::domain::DomainMask;
use crate::network::ConstraintNetwork;
use crate::util::active_tensors;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measure {
    NumUnfixedVars,
    NumUnfixedTensors,
    NumHardTensors,
}

impl Measure {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "vars" => Ok(Self::NumUnfixedVars),
            "tensors" => Ok(Self::NumUnfixedTensors),
            "hard-tensors" => Ok(Self::NumHardTensors),
            _ => Err(format!(
                "invalid --measure value: {value}; expected vars, tensors, or hard-tensors"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::NumUnfixedVars => "vars",
            Self::NumUnfixedTensors => "tensors",
            Self::NumHardTensors => "hard-tensors",
        }
    }
}

/// Port of `measure.jl::measure_core` (the non-Gurobi measures). Returns `f64`
/// because `minimize_gamma` consumes the size reductions as `f64`.
pub fn measure_core(cn: &ConstraintNetwork, doms: &[DomainMask], m: Measure) -> f64 {
    match m {
        Measure::NumUnfixedVars => doms.iter().filter(|d| !d.is_fixed()).count() as f64,
        Measure::NumUnfixedTensors => active_tensors(cn, doms).count() as f64,
        Measure::NumHardTensors => {
            let mut excess = 0usize;
            for tid in active_tensors(cn, doms) {
                let degree = cn.tensors[tid]
                    .var_axes
                    .iter()
                    .filter(|&&v| !doms[v].is_fixed())
                    .count();
                if degree > 2 {
                    excess += degree - 2;
                }
            }
            excess as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    #[test]
    fn measures_count_as_julia_does() {
        // T0 hard: degree-3 OR over [0,1,2]; T1 binary over [2,3].
        let t0 = vec![false, true, true, true, true, true, true, true];
        let or2 = vec![false, true, true, true];
        let cn = setup_problem(4, vec![vec![0, 1, 2], vec![2, 3]], vec![t0, or2]);

        // All unfixed.
        let doms = vec![DomainMask::BOTH; 4];
        assert_eq!(measure_core(&cn, &doms, Measure::NumUnfixedVars), 4.0);
        assert_eq!(measure_core(&cn, &doms, Measure::NumUnfixedTensors), 2.0);
        // T0 degree 3 -> excess 1; T1 degree 2 -> excess 0.
        assert_eq!(measure_core(&cn, &doms, Measure::NumHardTensors), 1.0);

        // Fix v3 -> T1 still active (v2 free), T0 still degree 3.
        let doms2 = vec![
            DomainMask::BOTH,
            DomainMask::BOTH,
            DomainMask::BOTH,
            DomainMask::D0,
        ];
        assert_eq!(measure_core(&cn, &doms2, Measure::NumUnfixedVars), 3.0);
        assert_eq!(measure_core(&cn, &doms2, Measure::NumUnfixedTensors), 2.0);
        assert_eq!(measure_core(&cn, &doms2, Measure::NumHardTensors), 1.0);

        // Fix v0,v1 -> T0 degree 1 (no longer hard); T1 degree 2.
        let doms3 = vec![
            DomainMask::D1,
            DomainMask::D0,
            DomainMask::BOTH,
            DomainMask::BOTH,
        ];
        assert_eq!(measure_core(&cn, &doms3, Measure::NumHardTensors), 0.0);
    }

    #[test]
    fn cli_labels_round_trip() {
        for (label, measure) in [
            ("vars", Measure::NumUnfixedVars),
            ("tensors", Measure::NumUnfixedTensors),
            ("hard-tensors", Measure::NumHardTensors),
        ] {
            assert_eq!(Measure::parse(label), Ok(measure));
            assert_eq!(measure.label(), label);
        }
        assert!(Measure::parse("unknown").is_err());
    }
}
