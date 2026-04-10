//! Practice NOCQ (No Opponent Cycle) propagator.
//!
//! Current simplified behavior:
//! - Do a DFS from `game.start` following edges that are NOT fixed to false.
//! - If we detect a cycle, compute whether the cycle satisfies a winning condition
//!   (here: `parity_even_max`).
//! - If the cycle is "bad", cut it by fixing the closing edge to `false`.

use std::sync::Arc;

use crate::{
	// --- Boolean variable actions used by this propagator ---
	actions::{BoolInitActions, BoolInspectionActions, BoolPropagationActions, ReasoningEngine},

	// --- Propagator trait (initialize + propagate interface) ---
	constraints::Propagator,

	// --- Parity-game data structure & winning condition helper ---
	parity::{game::Game, winner_condition::parity_even_max},

	// --- Concrete engine contexts & solver view type ---
	solver::{
		decision::Decision,
		engine::Engine,
		initialization_context::InitializationContext,
		solving_context::SolvingContext,
		view::View,
	},
};

/// Concrete conflict type used in this practice propagator:
/// conflicts are expressed over Boolean decisions.
type BoolConflict = crate::constraints::Conflict<Decision<bool>>;

/// NOCQ propagator state.
/// - `game`: immutable parity game structure (shared with `Arc`)
#[derive(Clone, Debug)]
pub struct NocqProp {
	e: Vec<View<bool>>,
	v: Vec<View<bool>>,
	game: Arc<Game>,
}

impl NocqProp {
	pub fn new(e: Vec<View<bool>>, v: Vec<View<bool>>, game: Arc<Game>) -> Self {
		Self { e, v, game }
	}

	fn noceager(
        &self,
        ctx: &mut SolvingContext<'_>,
        path_v: &mut Vec<usize>,
        path_e: &mut Vec<usize>,
        vtx: usize,
        last_edge: Option<usize>,
        defined_edge: bool,
    ) -> Result<(), BoolConflict>{
        // --- 1) Cycle detection: is `vtx` already on the current path? ---
        if let Some(cycle_start_idx) = path_v.iter().position(|&x| x == vtx) {
            // Evaluate the cycle using the winning condition helper.
			// `parity_even_max == true`  => cycle acceptable
			// `parity_even_max == false` => cycle violates condition => cut
			let ok = parity_even_max(self.game.as_ref(), path_v, cycle_start_idx);
            if !ok {
                if let Some(e_close) = last_edge {
                    let mut reason = Vec::new();
                    let cycle_edges = &path_e[cycle_start_idx..];
                    let upto = cycle_edges.len().saturating_sub(1);
                    // Cut the cycle by forcing the closing edge to false.
                    for &ei in &cycle_edges[..upto] {
                        reason.push(self.e[ei].clone().into());
                    }
					// This call uses BoolPropagationActions::fix, and requires:
					// - reading happens via BoolInspectionActions::val
					// - fixing happens via BoolPropagationActions::fix
					self.e[e_close].fix(ctx, false, reason)?;
                }
            }
            return Ok(());
        }
        // --- 2) Continue DFS only if the incoming edge is defined true ---
        if defined_edge {
            path_v.push(vtx);

            for &e2 in &self.game.outs[vtx] {
                // Read current assignment of edge var (None/Some(true)/Some(false))
				let val = self.e[e2].val(ctx);
                // Skip edges already fixed to false.
                if val == Some(false) {
                    continue;
                }

                path_e.push(e2);

                let w = self.game.targets[e2];
                let def = val == Some(true);
                self.noceager(ctx, path_v, path_e, w, Some(e2), def)?;

                path_e.pop();
            }

            path_v.pop();
        }

        Ok(())
    }
}

impl Propagator<Engine> for NocqProp {
    /// Initialization stage: subscribe this propagator to “fixed events” of vars.
	///
	/// This uses BoolInitActions::enqueue_when_fixed(ctx) on each variable view.
	/// Meaning: if any of these vars becomes fixed, the propagator is scheduled.
	fn initialize(&mut self, ctx: &mut InitializationContext<'_>) {
        for b in &self.e {
            b.enqueue_when_fixed(ctx);
        }
        for b in &self.v {
            b.enqueue_when_fixed(ctx);
        }
    }

    /// Main propagation step: start a DFS from the start vertex.
    fn propagate(
        &mut self,
        ctx: &mut SolvingContext<'_>,
    ) -> Result<(), BoolConflict> {
        let mut path_v = Vec::new();
        let mut path_e = Vec::new();
        // Start DFS at game.start.
		// `last_edge=None` because start has no incoming edge.
		// `defined_edge=true` to allow DFS to expand from start.
		self.noceager(ctx, &mut path_v, &mut path_e, self.game.start, None, true)
    }
}

/// Build a solver + lowering map with our NOCQ propagator installed.
/// Returns (solver, map, views for vertices, views for edges).
fn build_solver_with_nocq(
    game: Arc<Game>,
    force_edges_true: &[usize],
) -> (crate::solver::Solver<pindakaas::solver::cadical::Cadical>, impl Clone, Vec<View<bool>>, Vec<View<bool>>) {
    use pindakaas::propositional_logic::Formula;
    use pindakaas::solver::cadical::Cadical;
    use crate::{Model, lower::InitConfig};

    // 1) model decisions
    let mut prb = Model::default();

    // vertex decisions
    let mut v_dec = Vec::new();
    for _ in 0..game.nvertices {
        v_dec.push(prb.new_bool_decision());
    }

    // edge decisions
    let mut e_dec = Vec::new();
    for _ in 0..game.nedges {
        e_dec.push(prb.new_bool_decision());
    }

    // 2) force some edges to true (like Gecode model forcing)
    for &eid in force_edges_true {
        prb.proposition(Formula::Or(vec![e_dec[eid].into()])).post();
    }

    // 3) lower
    let (mut slv, map): (crate::solver::Solver<Cadical>, _) =
        prb.to_solver::<Cadical>(&InitConfig::default()).unwrap();

    // 4) map to solver views
    let sv_v: Vec<View<bool>> = v_dec.into_iter().map(|d| map.get(&mut slv, d)).collect();
    let sv_e: Vec<View<bool>> = e_dec.into_iter().map(|d| map.get(&mut slv, d)).collect();

    // 5) install propagator
    slv.add_propagator(
        Box::new(NocqProp::new(sv_e.clone(), sv_v.clone(), game)),
        true,
    );

    (slv, map, sv_v, sv_e)
}

/// Solve once; return Status.
fn solve_status(mut slv: crate::solver::Solver<pindakaas::solver::cadical::Cadical>) -> crate::solver::Status {
    slv.solve(|_sol| {})
}

#[cfg(test)]
mod tests {
	use super::*;
	use pindakaas::{
		propositional_logic::Formula,
		solver::cadical::Cadical,
	};

	use crate::{lower::InitConfig, solver::Status, Model};

	#[test]
	fn nocq_cuts_odd_cycle_makes_unsat_when_edges_forced_true() {
		// game: 0 -> 1 -> 0
		let mut g = Game::new(2, 0);
		let _e01 = g.add_edge(0, 1);
		let _e10 = g.add_edge(1, 0);

		g.set_owner(0, 0);
		g.set_owner(1, 0);

		// max priority on cycle is odd -> parity_even_max should be false -> cut edge
		g.set_priority(0, 1);
		g.set_priority(1, 3);

		let game = Arc::new(g);

        // Build a model with Boolean decisions for vertices & edges.
		let mut prb = Model::default();
		let v0 = prb.new_bool_decision();
		let v1 = prb.new_bool_decision();
		let e0 = prb.new_bool_decision();
		let e1 = prb.new_bool_decision();

		// force edges true
		prb.proposition(Formula::Or(vec![e0.into()])).post();
		prb.proposition(Formula::Or(vec![e1.into()])).post();

        // Lower to a concrete solver using Cadical SAT backend.
        let (mut slv, map): (crate::solver::Solver<Cadical>, _) =
            prb.to_solver::<Cadical>(&InitConfig::default()).unwrap();

        // Map model decisions to solver views.
		let sv_v0 = map.get(&mut slv, v0);
		let sv_v1 = map.get(&mut slv, v1);
		let sv_e0 = map.get(&mut slv, e0);
		let sv_e1 = map.get(&mut slv, e1);

		// Install the propagator.
        slv.add_propagator(
			Box::new(NocqProp::new(vec![sv_e0, sv_e1], vec![sv_v0, sv_v1], game)),
			true,
		);

        // Solve: should be UNSAT because we force both edges true,
		// but NOCQ cuts the closing edge to false.
		let st = slv.solve(|_sol| {});
		assert_eq!(st, Status::Unsatisfiable);
	}

    #[test]
    fn nocq_odd_cycle_forced_true_is_unsat() {
        // 0 -> 1 -> 0
        let mut g = Game::new(2, 0);
        let _e01 = g.add_edge(0, 1); // eid 0
        let _e10 = g.add_edge(1, 0); // eid 1

        // make cycle "bad": max priority odd
        g.set_priority(0, 1);
        g.set_priority(1, 3);

        let game = Arc::new(g);

        // force both edges true => propagator must cut one => conflict => UNSAT
        let (slv, _map, _sv_v, _sv_e) = build_solver_with_nocq(game, &[0, 1]);
        let st = solve_status(slv);
        assert_eq!(st, Status::Unsatisfiable);
    }

    #[test]
    fn nocq_even_cycle_forced_true_is_sat() {
        // 0 -> 1 -> 0
        let mut g = Game::new(2, 0);
        let _e01 = g.add_edge(0, 1); // eid 0
        let _e10 = g.add_edge(1, 0); // eid 1

        // make cycle "good": max priority even
        g.set_priority(0, 2);
        g.set_priority(1, 0);

        let game = Arc::new(g);

        // force both edges true, but cycle is acceptable => should be SAT
        let (slv, _map, _sv_v, _sv_e) = build_solver_with_nocq(game, &[0, 1]);
        let st = solve_status(slv);
        assert_eq!(st, Status::Satisfied);
    }

    #[test]
    fn nocq_bad_cycle_exists_but_not_forced_can_be_sat() {
        // vertices: 0(start),1,2
        // edges:
        // 0->1 (e0), 1->0 (e1) forms bad cycle
        // 0->2 (e2) escape
        let mut g = Game::new(3, 0);
        let _e01 = g.add_edge(0, 1); // 0
        let _e10 = g.add_edge(1, 0); // 1
        let _e02 = g.add_edge(0, 2); // 2

        // bad cycle by priority
        g.set_priority(0, 1);
        g.set_priority(1, 3);
        g.set_priority(2, 0);

        let game = Arc::new(g);

        // only force one cycle edge true; leave the other free.
        // solver can set the closing edge false to satisfy nocq => SAT expected.
        let (slv, _map, _sv_v, _sv_e) = build_solver_with_nocq(game, &[0]);
        let st = solve_status(slv);
        assert_eq!(st, Status::Satisfied);
    }

    #[test]
    fn nocq_forcing_non_cycle_edge_true_should_not_make_unsat_by_itself() {
        // 0->1->0 is bad, but do NOT force both
        // also add a non-cycle edge 1->2 and force it true
        let mut g = Game::new(3, 0);
        let _e01 = g.add_edge(0, 1); // 0
        let _e10 = g.add_edge(1, 0); // 1
        let _e12 = g.add_edge(1, 2); // 2

        g.set_priority(0, 1);
        g.set_priority(1, 3);
        g.set_priority(2, 0);

        let game = Arc::new(g);

        // force only e12 true (not part of closing cycle from start exploration unless chosen),
        // should still be SAT.
        let (slv, _map, _sv_v, _sv_e) = build_solver_with_nocq(game, &[2]);
        let st = solve_status(slv);
        assert_eq!(st, Status::Satisfied);
    }
}