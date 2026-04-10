predicate huub_noc_int(
    array[int] of var int: vertices,
    array[int] of int: owners,
    array[int] of int: priors,
    array[int] of int: sources,
    array[int] of int: targets,
    array[int] of int: weights,
    int: init,
    int: reward
);

predicate fzn_noc_int(
    array[int] of var int: vertices,
    array[int] of int: owners,
    array[int] of int: priors,
    array[int] of int: sources,
    array[int] of int: targets,
    array[int] of int: weights,
    int: init,
    int: reward
) =
    huub_noc_int(vertices, owners, priors, sources, targets, weights,
              init, reward);