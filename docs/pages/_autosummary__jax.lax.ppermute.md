- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.ppermute

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.ppermute.rst "Download source file")
-  .pdf

# jax.lax.ppermute

## Contents

- [`ppermute()`](#jax.lax.ppermute)

# jax.lax.ppermute[\#](#jax-lax-ppermute "Link to this heading")

jax.lax.ppermute(*x*, *axis_name*, *perm*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L356-L382)[\#](#jax.lax.ppermute "Link to this definition")  
Perform a collective permutation according to the permutation `perm`.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

This function is an analog of the CollectivePermute HLO.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **perm** – list of pairs of ints, representing `(source_index,`` ``destination_index)` pairs that encode how the mapped axis named `axis_name` should be shuffled. The integer values are treated as indices into the mapped axis `axis_name`. Any two pairs should not have the same source index or the same destination index. For each index of the axis `axis_name` that does not correspond to a destination index in `perm`, the corresponding values in the result are filled with zeros of the appropriate type.

Returns:  
Array(s) with the same shape as `x` with slices along the axis `axis_name` gathered from `x` according to the permutation `perm`.

[](jax.lax.pmean.html "previous page")

previous

jax.lax.pmean

[](jax.lax.pshuffle.html "next page")

next

jax.lax.pshuffle

Contents

- [`ppermute()`](#jax.lax.ppermute)

By The JAX authors

© Copyright 2024, The JAX Authors.\
