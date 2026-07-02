use crate::domain::DomainMask;
use crate::network::{BoolTensor, ConstraintNetwork};
use crate::util::get_active_tensors;

#[inline]
fn lit_true(v: usize) -> usize {
    2 * v
}
#[inline]
fn lit_false(v: usize) -> usize {
    2 * v + 1
}

/// Is `(var@pos1=val1, var@pos2=val2)` a satisfying configuration of `tensor`,
/// with all other vars taken from their fixed domains? Port of
/// `twosat.jl::is_valid_assignment`.
fn is_valid_assignment(
    cn: &ConstraintNetwork,
    tensor: &BoolTensor,
    doms: &[DomainMask],
    pos1: usize,
    val1: bool,
    pos2: usize,
    val2: bool,
) -> bool {
    let mut config = 0u32;
    for (i, &var) in tensor.var_axes.iter().enumerate() {
        let bit = if doms[var].is_fixed() {
            doms[var].value().expect("fixed var has a value")
        } else if i == pos1 {
            val1
        } else if i == pos2 {
            val2
        } else {
            false // unreachable for a binary constraint; matches Julia's default
        };
        if bit {
            config |= 1 << i;
        }
    }
    cn.is_sat(tensor, config)
}

/// Add 2-SAT implications for a binary constraint over `var1`,`var2`. Port of
/// `twosat.jl::add_binary_implications!`.
fn add_binary_implications(
    cn: &ConstraintNetwork,
    graph: &mut [Vec<usize>],
    tensor: &BoolTensor,
    doms: &[DomainMask],
    var1: usize,
    var2: usize,
) {
    let pos1 = tensor
        .var_axes
        .iter()
        .position(|&v| v == var1)
        .expect("var1 in tensor");
    let pos2 = tensor
        .var_axes
        .iter()
        .position(|&v| v == var2)
        .expect("var2 in tensor");

    let valid_00 = is_valid_assignment(cn, tensor, doms, pos1, false, pos2, false);
    let valid_01 = is_valid_assignment(cn, tensor, doms, pos1, false, pos2, true);
    let valid_10 = is_valid_assignment(cn, tensor, doms, pos1, true, pos2, false);
    let valid_11 = is_valid_assignment(cn, tensor, doms, pos1, true, pos2, true);

    let (ut, uf) = (lit_true(var1), lit_false(var1));
    let (vt, vf) = (lit_true(var2), lit_false(var2));

    if !valid_00 {
        graph[uf].push(vt);
        graph[vf].push(ut);
    }
    if !valid_01 {
        graph[uf].push(vf);
        graph[vt].push(ut);
    }
    if !valid_10 {
        graph[ut].push(vt);
        graph[vf].push(uf);
    }
    if !valid_11 {
        graph[ut].push(vf);
        graph[vt].push(uf);
    }
}

/// Tarjan SCC. Returns components in reverse-topological order (component index
/// = position in the returned Vec). Port of `twosat.jl::tarjan_scc`.
struct Tarjan {
    index: Vec<i64>,
    lowlink: Vec<i64>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    next: i64,
    sccs: Vec<Vec<usize>>,
}

impl Tarjan {
    fn run(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
        let n = graph.len();
        let mut t = Tarjan {
            index: vec![-1; n],
            lowlink: vec![-1; n],
            on_stack: vec![false; n],
            stack: Vec::new(),
            next: 0,
            sccs: Vec::new(),
        };
        for v in 0..n {
            if t.index[v] == -1 {
                t.strongconnect(graph, v);
            }
        }
        t.sccs
    }

    fn strongconnect(&mut self, graph: &[Vec<usize>], v: usize) {
        self.index[v] = self.next;
        self.lowlink[v] = self.next;
        self.next += 1;
        self.stack.push(v);
        self.on_stack[v] = true;

        for &w in &graph[v] {
            if self.index[w] == -1 {
                self.strongconnect(graph, w);
                self.lowlink[v] = self.lowlink[v].min(self.lowlink[w]);
            } else if self.on_stack[w] {
                self.lowlink[v] = self.lowlink[v].min(self.index[w]);
            }
        }

        if self.lowlink[v] == self.index[v] {
            let mut scc = Vec::new();
            loop {
                let w = self.stack.pop().expect("non-empty Tarjan stack");
                self.on_stack[w] = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            self.sccs.push(scc);
        }
    }
}

/// Solve the residual 2-SAT. Returns a full assignment (every unfixed var set)
/// if satisfiable, else `None`. Port of `twosat.jl::solve_2sat`.
pub fn solve_2sat(cn: &ConstraintNetwork, doms: &[DomainMask]) -> Option<Vec<DomainMask>> {
    let n = cn.vars.len();
    let mut graph: Vec<Vec<usize>> = vec![Vec::new(); 2 * n];

    for tid in get_active_tensors(cn, doms) {
        let t = &cn.tensors[tid];
        let unfixed: Vec<usize> = t
            .var_axes
            .iter()
            .copied()
            .filter(|&v| !doms[v].is_fixed())
            .collect();
        debug_assert!(unfixed.len() <= 2, "solve_2sat needs a 2-SAT residual");
        // unit (len 1) clauses are already propagated; len 2 -> implications.
        if unfixed.len() == 2 {
            add_binary_implications(cn, &mut graph, t, doms, unfixed[0], unfixed[1]);
        }
    }

    let sccs = Tarjan::run(&graph);
    let mut scc_id = vec![0usize; 2 * n];
    for (id, comp) in sccs.iter().enumerate() {
        for &vert in comp {
            scc_id[vert] = id;
        }
    }

    // UNSAT iff x and ¬x share an SCC.
    for v in 0..n {
        if doms[v].is_fixed() {
            continue;
        }
        if scc_id[lit_true(v)] == scc_id[lit_false(v)] {
            return None;
        }
    }

    // Assignment rule (Tarjan yields reverse-topo order): x=true iff its
    // true-literal SCC has the lower index. Matches `twosat.jl`.
    let mut sol = doms.to_vec();
    for v in 0..n {
        if doms[v].is_fixed() {
            continue;
        }
        sol[v] = if scc_id[lit_true(v)] < scc_id[lit_false(v)] {
            DomainMask::D1
        } else {
            DomainMask::D0
        };
    }
    Some(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::setup_problem;

    // Does a fully-fixed assignment satisfy every tensor?
    fn satisfies(cn: &ConstraintNetwork, sol: &[DomainMask]) -> bool {
        cn.tensors.iter().all(|t| {
            let mut cfg = 0u32;
            for (i, &v) in t.var_axes.iter().enumerate() {
                if sol[v].value().expect("fully assigned") {
                    cfg |= 1 << i;
                }
            }
            cn.is_sat(t, cfg)
        })
    }

    const OR2: [bool; 4] = [false, true, true, true]; // x ∨ y

    #[test]
    fn satisfiable_2sat_returns_a_model() {
        // (x0∨x1) ∧ (¬x1∨x2): a binary chain, satisfiable.
        // ¬x1∨x2 over [1,2]: forbids (x1=1,x2=0) = config 0b01 -> dense[1]=false.
        let imp = vec![true, false, true, true];
        let cn = setup_problem(3, vec![vec![0, 1], vec![1, 2]], vec![OR2.to_vec(), imp]);
        let doms = vec![DomainMask::BOTH; 3];
        let sol = solve_2sat(&cn, &doms).expect("should be SAT");
        assert!(sol.iter().all(|d| d.is_fixed()));
        assert!(satisfies(&cn, &sol));
    }

    #[test]
    fn unsatisfiable_2sat_returns_none() {
        // All four binary patterns of (x0,x1) forbidden -> UNSAT.
        let f00 = vec![false, true, true, true]; // forbids (0,0)
        let f10 = vec![true, false, true, true]; // forbids (1,0)  [config bit0=x0]
        let f01 = vec![true, true, false, true]; // forbids (0,1)
        let f11 = vec![true, true, true, false]; // forbids (1,1)
        let cn = setup_problem(
            2,
            vec![vec![0, 1], vec![0, 1], vec![0, 1], vec![0, 1]],
            vec![f00, f10, f01, f11],
        );
        let doms = vec![DomainMask::BOTH; 2];
        assert!(solve_2sat(&cn, &doms).is_none());
    }
}
