- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pshuffle

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pshuffle.rst "Download source file")
-  .pdf

# jax.lax.pshuffle

## Contents

- [`pshuffle()`](#jax.lax.pshuffle)

# jax.lax.pshuffle[\#](#jax-lax-pshuffle "Link to this heading")

jax.lax.pshuffle(*x*, *axis_name*, *perm*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L466-L488)[\#](#jax.lax.pshuffle "Link to this definition")  
Convenience wrapper of jax.lax.ppermute with alternate permutation encoding

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **perm** – list of ints encoding sources for the permutation to be applied to the axis named `axis_name`, so that the output at axis index i comes from the input at axis index perm\[i\]. Every integer in \[0, N) should be included exactly once for axis size N.

Returns:  
Array(s) with the same shape as `x` with slices along the axis `axis_name` gathered from `x` according to the permutation `perm`.

[](jax.lax.ppermute.html "previous page")

previous

jax.lax.ppermute

[](jax.lax.pswapaxes.html "next page")

next

jax.lax.pswapaxes

Contents

- [`pshuffle()`](#jax.lax.pshuffle)

By The JAX authors

© Copyright 2024, The JAX Authors.\
