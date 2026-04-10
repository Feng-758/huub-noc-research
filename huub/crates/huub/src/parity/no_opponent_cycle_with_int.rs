use crate::{
	IntVal, actions::{InitActions, PostingActions, ReasoningEngine}, constraints::{Constraint, IntModelActions, IntSolverActions, Propagator, SimplificationStatus}, lower::{LoweringContext, LoweringError}
};

use crate::solver::activation_list::IntPropCond;
use crate::solver::IntLitMeaning;

use super::game::Game;

// JD Note: This contains all the data that the NoOpponentCycleWithInt
// constraint/propagator needs to function.
#[derive(Debug, Clone)]
pub struct NoOpponentCycleWithInt<IV> {
	pub(crate) vertices: Vec<IV>,
	pub(crate) game: Game,
}


impl<I, E> Propagator<E> for NoOpponentCycleWithInt<I>
where
	I: IntSolverActions<E>,
	E: ReasoningEngine,
{
	fn initialize(&mut self, ctx: &mut E::InitializationCtx<'_>) {
		for v in &self.vertices {
			v.enqueue_when(ctx, IntPropCond::Fixed);
		}
		ctx.enqueue_now(true);
	}

	fn propagate(
		&mut self,
		ctx: &mut E::PropagationCtx<'_>,
	) -> Result<(), <E as ReasoningEngine>::Conflict> {
		let n = self.game.num_vertices();

		// Cache current vertex assignments (fixed => Some(IntVal), else None)
		let mut v_val: Vec<Option<IntVal>> = vec![None; n];
		for v in 0..n {
			v_val[v] = self.vertices[v].val(ctx);
		}

		let mut path_v: Vec<usize> = Vec::new();
		let mut path_e: Vec<usize> = Vec::new();
		let mut in_stack: Vec<bool> = vec![false; n];

		// Start DFS from every vertex that is currently active.
		for v in 0..n {
			if self.is_active(v, v_val[v]) {
				self.noceager_dfs_int::<E>(
					ctx,
					&v_val,
					&mut path_v,
					&mut path_e,
					&mut in_stack,
					v,
					None,
				)?;
			}
		}

		Ok(())
	}
}


impl<I> NoOpponentCycleWithInt<I> {
	#[inline]
	fn is_even(&self, v: usize) -> bool {
		self.game.owner(v) == self.game.player_sat()
	}

	#[inline]
	fn is_active(&self, v: usize, v_val: Option<IntVal>) -> bool {
		let Some(x) = v_val else { return false };
		let x: i64 = x.into();

		if self.is_even(v) {
			x != 0
		} else {
			x == 1
		}
	}

	#[inline]
	fn value_for_edge(&self, e: usize) -> IntVal {
		(e as i64 + 1).into()
	}

	#[inline]
	fn chosen_edge_of_even(&self, v: usize, vv: Option<IntVal>) -> Option<usize> {
		let val = vv?;
		let raw: i64 = val.into();

		if raw <= 0 {
			return None;
		}

		let e = (raw - 1) as usize;

		// Sanity check: the chosen edge must really leave v
		if self.game.source(e) != v {
			return None;
		}

		Some(e)
	}

	fn edge_literal_for_reason<E: ReasoningEngine>(
		&self,
		ctx: &mut <E as ReasoningEngine>::PropagationCtx<'_>,
		e: usize,
	) -> E::Atom
	where
		I: IntSolverActions<E>,
	{
		let u = self.game.source(e);

		if self.is_even(u) {
			// even vertex chose edge e  <=>  V[u] = e + 1
			self.vertices[u].lit(ctx, IntLitMeaning::Eq(self.value_for_edge(e)))
		} else {
			// odd vertex active  <=>  V[u] = 1
			self.vertices[u].lit(ctx, IntLitMeaning::Eq(1.into()))
		}
	}

	#[inline]
	fn cycle_is_forbidden_by_parity_vertices(&self, cycle_vertices: &[usize]) -> bool {
		let mut maxp = i64::MIN;
		for &u in cycle_vertices {
			maxp = maxp.max(self.game.prior(u));
		}

		let parity = (maxp & 1) as usize;
		parity != self.game.player_sat()
	}

	// ---------------------------------
	// NOCEager DFS
	// ---------------------------------
	fn noceager_dfs_int<E: ReasoningEngine>(
		&self,
		ctx: &mut <E as ReasoningEngine>::PropagationCtx<'_>,
		v_val: &[Option<IntVal>],
		path_v: &mut Vec<usize>,
		path_e: &mut Vec<usize>,
		in_stack: &mut [bool],
		v: usize,
		parent_edge: Option<usize>,
	) -> Result<(), <E as ReasoningEngine>::Conflict>
	where
		I: IntSolverActions<E>,
	{
		// 1 cycle check
		if in_stack[v] {
			let start = path_v
				.iter()
				.position(|&x| x == v)
				.expect("in_stack[v] implies v is on path_v");

			let cycle_vertices = &path_v[start..];

			// path_e currently stores the edges already pushed on the path.
			// The edge that just re-enters v is parent_edge, which is not yet pushed,
			// so we must append it explicitly to obtain the full cycle.
			let mut cycle_edges: Vec<usize> = path_e[start..].to_vec();
			if let Some(e_in) = parent_edge {
				cycle_edges.push(e_in);
			}

			if self.cycle_is_forbidden_by_parity_vertices(cycle_vertices) {
				let mut reason: Vec<E::Atom> = Vec::new();

				// Build reason only from the already-fixed edge choices on the stack suffix.
				// Do NOT include parent_edge itself, otherwise the propagation is self-justifying.
				for &e in &path_e[start..] {
					reason.push(self.edge_literal_for_reason::<E>(ctx, e));
				}
				if reason.is_empty() {
					if let Some(e_in) = parent_edge {
						reason.push(self.edge_literal_for_reason::<E>(ctx, e_in));
					}
				}

				// Forbid the incoming choice that closes the cycle
				if let Some(e_in) = parent_edge {
					let p = self.game.source(e_in);

					if self.is_even(p) {
						// even parent chose edge e_in  <=>  V[p] = e_in + 1
						let forbid_val = self.value_for_edge(e_in);

						for &e in &path_e[start..] {
							let u = self.game.source(e);
							let val = self.vertices[u].val(ctx);
						}
						self.vertices[p].remove_val(ctx, forbid_val, reason)?;
					} else {
						// odd parent active  <=>  V[p] = 1

						for &e in &path_e[start..] {
							let u = self.game.source(e);
							let val = self.vertices[u].val(ctx);
						}
						self.vertices[p].remove_val(ctx, 1.into(), reason)?;
					}
				}
			}
			return Ok(());
		}

		// 2 eager expansion requires v to be fixed and active
		let vv = match v_val[v] {
			Some(x) => x,
			None => return Ok(()),
		};
		if !self.is_active(v, Some(vv)) {
			return Ok(());
		}

		// push current vertex
		in_stack[v] = true;
		path_v.push(v);
		if let Some(e) = parent_edge {
			path_e.push(e);
		}

		if self.is_even(v) {
			// even: follow the chosen edge only
			if let Some(e) = self.chosen_edge_of_even(v, Some(vv)) {
				let w = self.game.target(e);
				self.noceager_dfs_int::<E>(
					ctx,
					v_val,
					path_v,
					path_e,
					in_stack,
					w,
					Some(e),
				)?;
			}
		} else {
			// odd: active => traverse all outgoing edges
			for &e in self.game.out_edges(v) {
				let w = self.game.target(e);
				self.noceager_dfs_int::<E>(
					ctx,
					v_val,
					path_v,
					path_e,
					in_stack,
					w,
					Some(e),
				)?;
			}
		}

		// pop
		if parent_edge.is_some() {
			path_e.pop();
		}
		path_v.pop();
		in_stack[v] = false;

		Ok(())
	}
}


impl<I, E> Constraint<E> for NoOpponentCycleWithInt<I>
where
	E: ReasoningEngine,
	I: IntModelActions<E>,
{
	fn simplify(
		&mut self,
		ctx: &mut E::PropagationCtx<'_>,
	) -> Result<SimplificationStatus, <E as ReasoningEngine>::Conflict> {
		self.propagate(ctx)?;
		Ok(SimplificationStatus::NoFixpoint)
	}

	fn to_solver(&self, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
		let vertices: Vec<_> = self
			.vertices
			.iter()
			.map(|v| ctx.solver_view(v.clone().into()))
			.collect();

		ctx.add_propagator(Box::new(NoOpponentCycleWithInt {
			vertices,
			game: self.game.clone(),
		}));
		Ok(())
	}
}