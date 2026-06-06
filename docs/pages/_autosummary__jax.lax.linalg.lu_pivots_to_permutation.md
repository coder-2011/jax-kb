- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.lu_pivots_to_permutation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.lu_pivots_to_permutation.rst "Download source file")
-  .pdf

# jax.lax.linalg.lu_pivots_to_permutation

## Contents

- [`lu_pivots_to_permutation()`](#jax.lax.linalg.lu_pivots_to_permutation)

# jax.lax.linalg.lu_pivots_to_permutation[\#](#jax-lax-linalg-lu-pivots-to-permutation "Link to this heading")

jax.lax.linalg.lu_pivots_to_permutation(*pivots*, *permutation_size*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L363-L378)[\#](#jax.lax.linalg.lu_pivots_to_permutation "Link to this definition")  
Converts the pivots (row swaps) returned by LU to a permutation.

We build a permutation rather than applying pivots directly to the rows of a matrix because lax loops aren’t differentiable.

Parameters:  
- **pivots** (*ArrayLike*) – an int32 array of shape (…, k) of row swaps to perform

- **permutation_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the size of the output permutation. Has to be \>= k.

Returns:  
An int32 array of shape (…, permutation_size).

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.linalg.lu.html "previous page")

previous

jax.lax.linalg.lu

[](jax.lax.linalg.ormqr.html "next page")

next

jax.lax.linalg.ormqr

Contents

- [`lu_pivots_to_permutation()`](#jax.lax.linalg.lu_pivots_to_permutation)

By The JAX authors

© Copyright 2024, The JAX Authors.\
