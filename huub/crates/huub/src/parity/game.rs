#[derive(Clone, Debug)]
pub struct Game {
    pub owners: Vec<usize>,
    pub priors: Vec<i64>,
    pub sources: Vec<usize>,
    pub targets: Vec<usize>,
    pub weights: Vec<i64>,
    pub outs: Vec<Vec<usize>>,
    pub ins: Vec<Vec<usize>>,
    pub nvertices: usize,
    pub nedges: usize,
    pub init: usize,
    pub reward: RewardType,
}

#[derive(Clone, Copy, Debug)]
pub enum RewardType {
    Min,
    Max,
}

impl Game {
    pub(crate) fn new(
        owners: Vec<usize>,
        priors: Vec<i64>,
        sources: Vec<usize>,
        targets: Vec<usize>,
        weights: Vec<i64>,
        init: usize,
        reward: RewardType,
    ) -> Self {
        let nvertices = owners.len();
        let nedges = sources.len();
        assert_eq!(priors.len(), nvertices);
        assert_eq!(targets.len(), nedges);
        assert_eq!(weights.len(), nedges);

        let mut outs = vec![Vec::new(); nvertices];
        let mut ins = vec![Vec::new(); nvertices];
        for e in 0..nedges {
            let s = sources[e];
            let t = targets[e];
            outs[s].push(e);
            ins[t].push(e);
        }

        Self {
            owners,
            priors,
            sources,
            targets,
            weights,
            outs,
            ins,
            nvertices,
            nedges,
            init,
            reward,
        }
    }

    /// Returns the player we are solving for.
    /// Equivalent to C++ playerSAT.
    pub fn player_sat(&self) -> usize {
        match self.reward {
            RewardType::Min => 0,
            RewardType::Max => 1,
        }
    }
    /// Returns the opponent player.
    pub fn opponent(&self) -> usize {
        1 - self.player_sat()
    }

    pub(crate) fn num_vertices(&self) -> usize { self.nvertices }
    pub(crate) fn num_edges(&self) -> usize { self.nedges }
    pub(crate) fn out_edges(&self, v: usize) -> &[usize] { &self.outs[v] }
    pub(crate) fn in_edges(&self, v: usize) -> &[usize] { &self.ins[v] }
    pub(crate) fn source(&self, e: usize) -> usize { self.sources[e] }
    pub(crate) fn target(&self, e: usize) -> usize { self.targets[e] }
    pub(crate) fn owner(&self, v: usize) -> usize { self.owners[v] }
    pub(crate) fn prior(&self, v: usize) -> i64 { self.priors[v] }
}

// helper for noc_int
impl Game {
    /// true if there is an edge v -> w
    pub(crate) fn has_edge(&self, v: usize, w: usize) -> bool {
        self.out_edges(v).iter().any(|&e| self.target(e) == w)
    }

    /// all successor vertices of v (duplicates unlikely, but could exist)
    pub(crate) fn succs(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        self.out_edges(v).iter().map(|&e| self.target(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_construction_builds_outs_and_ins() {
        // Graph:
        // e0: 0 -> 1
        // e1: 1 -> 2
        // e2: 2 -> 0
        let owners = vec![0, 1, 0];          // arbitrary owners
        let priors = vec![1_i64, 2, 3];      // priorities per vertex
        let sources = vec![0_usize, 1, 2];
        let targets = vec![1_usize, 2, 0];
        let weights = vec![10_i64, 20, 30];  // optional 

        let init = 0_usize;
        let reward = RewardType::Min;

        let game = Game::new(
            owners.clone(),
            priors.clone(),
            sources.clone(),
            targets.clone(),
            weights.clone(),
            init,
            reward,
        );

        // sizes
        assert_eq!(game.num_vertices(), 3);
        assert_eq!(game.num_edges(), 3);
        assert_eq!(game.nvertices, 3);
        assert_eq!(game.nedges, 3);

        // vertex attributes
        assert_eq!(game.owner(0), 0);
        assert_eq!(game.owner(1), 1);
        assert_eq!(game.prior(2), 3);

        // edge endpoints
        assert_eq!(game.source(0), 0);
        assert_eq!(game.target(0), 1);
        assert_eq!(game.source(2), 2);
        assert_eq!(game.target(2), 0);

        // weights
        assert_eq!(game.weights[1], 20);

        // outs adjacency: outs[0] should contain edge 0 only
        assert_eq!(game.out_edges(0), &[0]);

        // ins adjacency: ins[0] should contain edge 2 only (2 -> 0)
        assert_eq!(game.in_edges(0), &[2]);

        // init & reward
        assert_eq!(game.init, 0);
        assert!(matches!(game.reward, RewardType::Min));
    }

    #[test]
    fn adjacency_multiple_edges_from_same_source() {
        // Graph:
        // e0: 0 -> 1
        // e1: 0 -> 2
        let owners = vec![0, 0, 1];
        let priors = vec![0_i64, 1, 2];
        let sources = vec![0_usize, 0];
        let targets = vec![1_usize, 2];
        let weights = vec![1_i64, 1];

        let game = Game::new(owners, priors, sources, targets, weights, 0, RewardType::Max);

        // outs[0] should have both edges in order of construction
        assert_eq!(game.out_edges(0), &[0, 1]);

        // ins[1] has edge 0, ins[2] has edge 1
        assert_eq!(game.in_edges(1), &[0]);
        assert_eq!(game.in_edges(2), &[1]);
    }
}