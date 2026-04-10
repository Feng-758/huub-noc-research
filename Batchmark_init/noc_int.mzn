% Use this editor as a MiniZinc scratch book
int: nvertices;
int: nedges;

set of int: VERTS = 1..nvertices;
set of int: EDGES = 1..nedges;

int: init =2;
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
array[VERTS] of var 0..nvertices: V;

% helper: outgoing edges of v (by scanning all edges)
function set of int: outs(int: v) =
  { e | e in EDGES where sources[e] == v };

% helper: checking if v is active
function var bool: active(int v) = V[v] != 0;

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

% ------------------ Structural 1 ------------------
% even/player: if active, choose exactly one outgoing target; if no outs -> cannot be active
constraint forall(v in VERTS where owners[v] == player_sat)(
  if card(outs(v)) = 0 then
    V[v] = 0
  else
    (V[v] = 0) \/ exists(e in outs(v))(V[v] = targets[e])
  endif
);
% ------------------ Structural 2 ------------------
% odd/opponent: if active, all successors must be active
constraint forall(v in VERTS where owners[v] != player_sat)(
  (V[v] = 0) \/
  (V[v] = 1 /\ forall(e in outs(v))(active(targets[e])))
);
% ------------------ Structural 3 ------------------
% even/player: if choose successor t, then t must be active
constraint forall(v in VERTS where owners[v] == player_sat)(
  if V[v] = 0 then true else active(V[v]) endif
);

% ------------------ NOC propagator call ------------------
constraint noc_int(V, owners, priors, sources, targets, weights, init, reward);

solve satisfy;

% --- SATISFIABLE if the <player> of interest wins; 
% --- UNSATISFIABLE otherwise

output ["=====SATISFIABLE===== \n"]; 