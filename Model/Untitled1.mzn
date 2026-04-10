% Use this editor as a MiniZinc scratch book
int: nvertices;
int: nedges;

set of int: VERTS = 1..nvertices;
set of int: EDGES = 1..nedges;

int: init =1;
int: player_sat = 0;
array[VERTS] of int: owners;
array[VERTS] of int: priors;
array[EDGES] of int: sources;
array[EDGES] of int: targets;
array[VERTS] of int: priors_shifted = [priors[v] + 1 | v in VERTS];

% array[EDGES] of int: weights;
array[EDGES] of int: weights = [ 0 | e in EDGES ];

int: reward = 0;

array[VERTS] of var bool: V;
array[EDGES] of var bool: E;

% array[VERTS] of var 0..nvertices: V;

% --- convert to 0-based ids for the solver-side Game ---
int: init0 = init - 1;
array[EDGES] of int: sources0 = [ sources[e] - 1 | e in EDGES ];
array[EDGES] of int: targets0 = [ targets[e] - 1 | e in EDGES ];

constraint V[init] = true;

% helper: outgoing edges of v (by scanning all edges)
function set of int: outs(int: v) =
  { e | e in EDGES where sources[e] == v };

% ------------------ Structural 1 ------------------
constraint
  forall(v in VERTS where owners[v] == player_sat /\ card(outs(v)) > 0) (
    V[v] -> sum(e in outs(v))(bool2int(E[e])) = 1
  );

% ------------------ Structural 2 ------------------
constraint
  forall(v in VERTS where owners[v] != player_sat /\ card(outs(v)) > 0) (
    forall(e in outs(v)) ( V[v] -> E[e] )
  );

% ------------------ Structural 3 ------------------
constraint
  forall(e in EDGES where targets[e] != init) (
    E[e] -> V[targets[e]]
  );

% ------------------ NOC propagator call ------------------
constraint noc(V, E, owners, priors_shifted, sources0, targets0, weights, init0, reward);


solve satisfy;

% --- SATISFIABLE if the <player> of interest wins; 
% --- UNSATISFIABLE otherwise

output ["=====SATISFIABLE===== \n"]; 