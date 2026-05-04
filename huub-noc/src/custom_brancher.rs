use huub::{
	actions::{
		BrancherInitActions, DecisionActions, IntDecisionActions, IntInspectionActions,
		ReasoningContext, Trailed,
	},
	solver::{
		IntLitMeaning, View,
		branchers::{Brancher, Directive},
	},
};

#[derive(Clone, Debug)]
pub(crate) struct CustomSearchBrancher {
	vars: Vec<View<i64>>,
	/// The start of the unfixed variables in `vars`.
	next: Trailed<usize>,
}

impl CustomSearchBrancher {
	/// Create a new [`CustomSearchBrancher`] brancher and add to the end of the
	/// branching queue in the solver.
	pub fn new_in(solver: &mut impl BrancherInitActions, vars: Vec<View<i64>>) {
		// let vars: Vec<_> = vars
		// 	.into_iter()
		// 	.filter(|i| i.val(solver).is_none())
		// 	.collect();

		for &v in &vars {
			solver.ensure_decidable(v);
		}

		let next = solver.new_trailed(0);
		solver.push_brancher(Box::new(CustomSearchBrancher { vars, next }));
	}
}

impl<D> Brancher<D> for CustomSearchBrancher
where
	D: DecisionActions + ReasoningContext<Atom = View<bool>>,
	View<i64>: IntDecisionActions<D>,
{
	fn decide(&mut self, ctx: &mut D) -> Directive {
		let next = ctx.trailed(self.next);
		let mut unfixed = None;
		for i in next..self.vars.len() {
			let v = &self.vars[i];
			let (lb, ub) = v.bounds(ctx);
			if lb != ub {
				if lb > 0 {
					return Directive::Select(v.lit(ctx, IntLitMeaning::Less(lb + 1)));
				}
				if unfixed.is_none() {
					ctx.set_trailed(self.next, i);
					unfixed = Some(v);
				}
			}
		}
		if let Some(v) = unfixed {
			debug_assert!(v.min(ctx) <= 0);
			return Directive::Select(v.lit(ctx, IntLitMeaning::GreaterEq(1)));
		}
		Directive::Exhausted
	}
}
