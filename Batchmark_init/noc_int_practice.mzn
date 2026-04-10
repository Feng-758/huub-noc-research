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

% array[EDGES] of int: weights;
array[EDGES] of int: weights = [ 0 | e in EDGES ];
int: reward = 0;

% -----------------------------
% Decision
% -----------------------------

% array[VERTS] of var 0..nvertices: V;
array[VERTS] of var 0..nedges: V;

% helper: outgoing edges of v (by scanning all edges)
function set of int: outs(int: v) =
  { e | e in EDGES where sources[e] == v };

% init must be active (odd:1, even: pick some successor)
constraint
  if owners[init] == player_sat then
    V[init] != 0
  else
    V[init] = 1
  endif;

% odd nodes domain restriction
constraint forall(v in VERTS where owners[v] != player_sat)(
  V[v] <= 1
);

% even vertex chooses a successor that is actually reachable
% constraint forall(v in VERTS where owners[v] == player_sat) (
%   V[v] in { targets[e] | e in outs(v) } union {0}
% );
constraint forall(v in VERTS where owners[v] == player_sat) (
  V[v] in outs(v) union {0}
);

% if even is active then its successor must be active
% constraint forall(v in VERTS where owners[v] == player_sat) (
%   V[v] != 0 -> V[V[v]] != 0
% );
% constraint forall(v in VERTS where owners[v] == player_sat) (
%   V[v] != 0 -> V[targets[V[v]]] != 0
% );
constraint forall(v in VERTS where owners[v] == player_sat) (
  forall(e in outs(v)) (
    V[v] = e -> V[targets[e]] != 0
  )
);

% if an odd node is active, all successors must be active
constraint forall(v in VERTS where owners[v] != player_sat) (
  V[v] != 0 -> forall(e in outs(v)) (V[targets[e]] != 0)
);


% ------------------------------------------------------------
% Convert to 0-based for the solver-side Game
int: init0 = init - 1;
array[EDGES] of int: sources0 = [ sources[e] - 1 | e in EDGES ];
array[EDGES] of int: targets0 = [ targets[e] - 1 | e in EDGES ];

% ------------------------------------------------------------
% Call your custom constraint (FlatZinc name must match your flatzinc.rs)
constraint noc_int(V, owners, priors, sources0, targets0, weights, init0, reward);

solve satisfy;

output ["=====SATISFIABLE===== \n"]; 