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

	pub(crate) fn num_vertices(&self) -> usize {
		self.nvertices
	}
	pub(crate) fn num_edges(&self) -> usize {
		self.nedges
	}
	pub(crate) fn out_edges(&self, v: usize) -> &[usize] {
		&self.outs[v]
	}
	pub(crate) fn in_edges(&self, v: usize) -> &[usize] {
		&self.ins[v]
	}
	pub(crate) fn source(&self, e: usize) -> usize {
		self.sources[e]
	}
	pub(crate) fn target(&self, e: usize) -> usize {
		self.targets[e]
	}
	pub(crate) fn owner(&self, v: usize) -> usize {
		self.owners[v]
	}
	pub(crate) fn prior(&self, v: usize) -> i64 {
		self.priors[v]
	}
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
