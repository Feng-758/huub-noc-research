mod custom_brancher;
pub(crate) mod game;
mod loader;
mod no_opponent_cycle;
mod no_opponent_cycle_with_int;

use std::{path::PathBuf, time::Instant};

use clap::{Parser, ValueEnum};
use huub::{
	actions::{BoolPropagationActions, IntPropagationActions, IntSimplificationActions},
	lower::InitConfig,
	model::{
		Model,
		expressions::{BoolFormula, IntLinearExp},
	},
	solver::{Solver, Status},
};
use no_opponent_cycle::NoOpponentCycle;
use pindakaas::solver::cadical::Cadical;
use rangelist::RangeList;

pub(crate) use crate::game::Game;
use crate::{
	custom_brancher::CustomSearchBrancher, loader::load_game_from_dzn,
	no_opponent_cycle_with_int::NoOpponentCycleWithInt,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
	Bool,
	Int,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
	#[arg(long, value_name = "FILE")]
	mode: Mode,
	#[arg(long, default_value_t = false)]
	custom_brancher: bool,
	#[arg(long, default_value_t = 255)]
	int_eager_limit: usize,
	instance: PathBuf,
}

fn main() {
	let cli = Cli::parse();

	// -- Suggested workflow --
	// 1. Read a new Game object (file should be provided as command-line argument).
	let game: Game = load_game_from_dzn(&cli.instance);

	// 2. Construct a new huub::model::Model, add clausal constraint and
	//    Constraint/Propagator.
	let mut prb = Model::default();
	let mut search_vars = Vec::new();
	match cli.mode {
		Mode::Bool => {
			let vertices = prb.new_bool_decisions(game.num_vertices());
			let edges = prb.new_bool_decisions(game.num_edges());

			// constraint V[init] = true;
			vertices[game.init].require(&mut prb, []).unwrap();

			// % ------------------ Structural 1 ------------------
			// constraint
			//   forall(v in VERTS where owners[v] == player_sat /\
			// card(outs(v)) > 0) (     V[v] -> sum(e in
			// outs(v))(bool2int(E[e])) = 1   );
			// % ------------------ Structural 2 ------------------
			// constraint
			//   forall(v in VERTS where owners[v] != player_sat /\
			// card(outs(v)) > 0) (     forall(e in outs(v)) ( V[v] -> E[e] )
			//   );

			for v in 0..game.num_vertices() {
				let outs: Vec<usize> = (0..game.num_edges())
					.filter(|&e| game.sources[e] == v)
					.collect();
				if outs.len() > 0 {
					if game.owners[v] == game.player_sat() {
						// Structural 1:
						prb.linear(outs.iter().map(|&e| edges[e]).sum::<IntLinearExp>())
							.eq(1)
							.implied_by(vertices[v])
							.post();
					} else {
						// Structural 2:
						for &e in &outs {
							prb.proposition(BoolFormula::Implies(
								Box::new(BoolFormula::Atom(vertices[v])),
								Box::new(BoolFormula::Atom(edges[e])),
							))
							.post();
						}
					}
				}
			}

			// % ------------------ Structural 3 ------------------
			// constraint
			//   forall(e in EDGES where targets[e] != init) (
			//     E[e] -> V[targets[e]]
			//   );

			for e in 0..game.num_edges() {
				if game.targets[e] != game.init {
					prb.proposition(BoolFormula::Implies(
						Box::new(BoolFormula::Atom(edges[e])),
						Box::new(BoolFormula::Atom(vertices[game.targets[e]])),
					))
					.post();
				}
			}

			// % ------------------ NOC propagator call ------------------
			// constraint noc(V, E, owners, priors, sources0, targets0, weights,
			// init0, reward);

			prb.post_constraint(NoOpponentCycle {
				vertices,
				edges,
				game,
			});
		}
		Mode::Int => {
			// % -----------------------------
			// % Decision
			// % -----------------------------

			// % array[VERTS] of var 0..nvertices: V;
			// array[VERTS] of var 0..nedges: V;

			let vertices = prb.new_int_decisions(game.num_vertices(), 0..=game.num_edges() as i64);

			// % helper: outgoing edges of v (by scanning all edges)
			// function set of int: outs(int: v) =
			//   { e | e in EDGES where sources[e] == v };

			// % init must be active (odd:1, even: pick some successor)
			// constraint
			//   if owners[init] == player_sat then
			//     V[init] != 0
			//   else
			//     V[init] = 1
			//   endif;

			if game.owners[game.init] == game.player_sat() {
				prb.linear(vertices[game.init]).ne(0).post();
				vertices[game.init].remove_val(&mut prb, 0, []).unwrap();
			} else {
				vertices[game.init].fix(&mut prb, 1, []).unwrap();
			}

			// % odd nodes domain restriction
			// constraint forall(v in VERTS where owners[v] != player_sat)(
			//   V[v] <= 1
			// );

			for v in 0..game.num_vertices() {
				if game.owners[v] != game.player_sat() {
					vertices[v].tighten_max(&mut prb, 1, []).unwrap();
				}
			}

			// constraint forall(v in VERTS where owners[v] == player_sat) (
			//   V[v] in outs(v) union {0}
			// );

			for v in 0..game.num_vertices() {
				if game.owners[v] == game.player_sat() {
					let outs = (0..game.num_edges()).filter(|&e| game.sources[e] == v);
					let domain: RangeList<i64> =
						outs.chain([0]).map(|v| v as i64..=v as i64).collect();
					vertices[v].restrict_domain(&mut prb, &domain, []).unwrap();
				}
			}

			// constraint forall(v in VERTS where owners[v] == player_sat) (
			//   forall(e in outs(v)) (
			//     V[v] = e -> V[targets[e]] != 0
			//   )
			// );
			for v in 0..game.num_vertices() {
				let outs: Vec<_> = (0..game.num_edges())
					.filter(|&e| game.sources[e] == v)
					.collect();
				if game.owners[v] == game.player_sat() {
					for e in outs {
						prb.proposition(BoolFormula::Implies(
							Box::new(BoolFormula::Atom(vertices[v].eq(e as i64))),
							Box::new(BoolFormula::Atom(vertices[game.targets[e]].ne(0))),
						))
						.post();
					}
				} else {
					// % if an odd node is active, all successors must be active
					// constraint forall(v in VERTS where owners[v] != player_sat) (
					//   V[v] != 0 -> forall(e in outs(v)) (V[targets[e]] != 0)
					// );
					for e in outs {
						prb.proposition(BoolFormula::Implies(
							Box::new(BoolFormula::Atom(vertices[v].ne(0))),
							Box::new(BoolFormula::Atom(vertices[game.targets[e]].ne(0))),
						))
						.post();
					}
				}
			}

			// % solve satisfy;
			// solve :: custom_search(V) satisfy;
			if cli.custom_brancher {
				search_vars = vertices.clone()
			}

			// % ------------------------------------------------------------
			// % Call your custom constraint (FlatZinc name must match your
			// flatzinc.rs) constraint noc_int(V, owners, priors, sources0,
			// targets0, weights, init0, reward);
			//
			prb.post_constraint(NoOpponentCycleWithInt { vertices, game });

			// output ["=====SATISFIABLE===== \n"];
		}
	};

	// 3. Transform into huub::solver::Solver and solve problem.
	let start_solve = Instant::now();
	let config = InitConfig::default().with_int_eager_limit(cli.int_eager_limit);
	let Ok((mut slv, map)): Result<(Solver<Cadical>, _), _> = prb.to_solver(&config) else {
		let finish_solve = Instant::now();
		println!("UNSATISFIABLE!");
		println!("conflicts=1");
		println!("solveTime={}", &(finish_solve - start_solve).as_secs_f64());
		return;
	};

	if cli.custom_brancher {
		if search_vars.is_empty() {
			println!("CUSTOM BRANCHER: cannot be used with bool.");
		}
		let vars = search_vars
			.into_iter()
			.map(|v| map.get(&mut slv, v))
			.collect();
		CustomSearchBrancher::new_in(&mut slv, vars);
	}

	let status = slv.solve(|_| {});
	let finish_solve = Instant::now();
	match status {
		Status::Satisfied => {
			println!("SATISFIED!");
		}
		Status::Unsatisfiable => {
			println!("UNSATISFIABLE!");
		}
		Status::Unknown => {
			println!("TIMEOUT!")
		}
		Status::Complete => unreachable!(),
	}

	// 4. Check statistics
	let stats = slv.solver_statistics();
	println!("solveTime={}", &(finish_solve - start_solve).as_secs_f64());
	println!("eagerLits={}", stats.eager_literals);
	println!("lazyLits={}", stats.lazy_literals);
	println!("conflicts={}", stats.conflicts);
	println!("cpPropagatorCalls={}", stats.cp_propagator_calls)
}
