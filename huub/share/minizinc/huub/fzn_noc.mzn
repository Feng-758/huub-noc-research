predicate huub_noc(
    array[int] of var bool: vertices,
    array[int] of var bool: edges,
    array[int] of int: owners,
    array[int] of int: priors,
    array[int] of int: sources,
    array[int] of int: targets,
    array[int] of int: weights,
    int: init,
    int: reward
);

predicate fzn_noc(
    array[int] of var bool: vertices,
    array[int] of var bool: edges,
    array[int] of int: owners,
    array[int] of int: priors,
    array[int] of int: sources,
    array[int] of int: targets,
    array[int] of int: weights,
    int: init,
    int: reward
) =
    huub_noc(vertices, edges, owners, priors, sources, targets, weights,
              init, reward);