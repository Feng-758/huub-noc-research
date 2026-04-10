predicate huub_bool_clause_reif(
	array[int] of var bool: as,
	array[int] of var bool: bs,
	var bool: b,
);
predicate bool_clause_reif(
	array[int] of var bool: as,
	array[int] of var bool: bs,
	var bool: b
) = huub_bool_clause_reif(as, bs, b);

predicate huub_array_int_maximum(var int: m, array[int] of var int: x);
predicate array_int_maximum(var int: m, array[int] of var int: x)
= huub_array_int_maximum(m, x);

predicate array_float_maximum(var float: m, array[int] of var float: x)
= abort("float variables are not supported by huub");

predicate huub_array_int_minimum(var int: m, array[int] of var int: x);
predicate array_int_minimum(var int: m, array[int] of var int: x)
= huub_array_int_minimum(m, x);

predicate array_float_minimum(var float: m, array[int] of var float: x)
= abort("float variables are not supported by huub");

include "fzn_noc.mzn";

predicate noc(
    array[int] of var bool: vertices,
    array[int] of var bool: edges,
    array[int] of int: owners,
    array[int] of int: priors,
    array[int] of int: sources,
    array[int] of int: targets,
    array[int] of int: weights,
    int: init,
    int: reward
) = fzn_noc(vertices, edges, owners, priors, sources, targets, weights, init, reward);

include "fzn_noc_int.mzn";

predicate noc_int(
    array[int] of var int: vertices,
    array[int] of int: owners,
    array[int] of int: priors,
    array[int] of int: sources,
    array[int] of int: targets,
    array[int] of int: weights,
    int: init,
    int: reward
) = fzn_noc_int(vertices, owners, priors, sources, targets, weights, init, reward);