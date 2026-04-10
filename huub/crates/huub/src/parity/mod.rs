pub mod game;
pub mod no_opponent_cycle;
pub mod no_opponent_cycle_with_int;

pub(crate) use game::{Game, RewardType};
pub(crate) use no_opponent_cycle::NoOpponentCycle;
pub(crate) use no_opponent_cycle_with_int::NoOpponentCycleWithInt;

#[cfg(test)]
mod tests {
    use super::*; // brings Game, RewardType, NoOpponentCycle into scope

    use crate::{
        lower::InitConfig,
        model::Model,
        solver::Solver,
    };

    use pindakaas::propositional_logic::Formula;
    use pindakaas::solver::cadical::Cadical;

    /// Solve a small instance under NoOpponentCycle.
    /// If `force_cycle` is true, we force edges (0,1,2) to be active.
    fn solve_instance(game: Game, force_cycle: bool) -> bool {
        let mut prb = Model::default();

        let edges: Vec<_> = (0..game.num_edges())
            .map(|_| prb.new_bool_decision())
            .collect();

        let vertices: Vec<_> = (0..game.num_vertices())
            .map(|_| prb.new_bool_decision())
            .collect();

        // Ensure the initial vertex is active, so the instance is non-trivial.
        prb.proposition(Formula::Atom(vertices[game.init])).post();

        if force_cycle {
            prb.proposition(Formula::Atom(edges[0])).post();
            prb.proposition(Formula::Atom(edges[1])).post();
            prb.proposition(Formula::Atom(edges[2])).post();
        }

        prb.post_constraint(NoOpponentCycle {
            vertices,
            edges,
            game,
        });

        let Ok((mut slv, _map)): Result<(Solver<Cadical>, _), _> =
            prb.to_solver(&InitConfig::default())
        else {
            return false;
        };

        let mut sat = false;
        slv.solve(|_sol| sat = true);
        sat
    }

    #[test]
    fn even_max_priority_cycle_is_allowed() {
        let game = Game::new(
            vec![0, 1, 0],
            vec![1, 2, 4], // max priority = 4 (even)
            vec![0, 1, 2],
            vec![1, 2, 0],
            vec![0, 0, 0],
            0,
            RewardType::Min,
        );

        assert!(solve_instance(game, true));
    }

    #[test]
    fn odd_max_priority_cycle_is_forbidden() {
        let game = Game::new(
            vec![0, 1, 0],
            vec![1, 2, 3], // max priority = 3 (odd)
            vec![0, 1, 2],
            vec![1, 2, 0],
            vec![0, 0, 0],
            0,
            RewardType::Min,
        );

        assert!(!solve_instance(game, true));
    }

    #[test]
    fn solver_avoids_odd_cycle_when_free_to_choose_edges() {
        let game = Game::new(
            vec![0, 1, 0],
            vec![1, 2, 3],      // odd max if 3-cycle used
            vec![0, 1, 2, 2],   // extra edge e3: 2->2
            vec![1, 2, 0, 2],
            vec![0, 0, 0, 0],
            0,
            RewardType::Min,
        );
    }
}