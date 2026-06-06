- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.permute_dims

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.permute_dims.rst "Download source file")
-  .pdf

# jax.numpy.permute_dims

## Contents

- [`permute_dims()`](#jax.numpy.permute_dims)

# jax.numpy.permute_dims[\#](#jax-numpy-permute-dims "Link to this heading")

jax.numpy.permute_dims(*a*, */*, *axes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1167-L1195)[\#](#jax.numpy.permute_dims "Link to this definition")  
Permute the axes/dimensions of an array.

JAX implementation of [`array_api.permute_dims()`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.permute_dims.html#array_api.permute_dims "(in Python array API standard)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axes** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – tuple of integers in range `[0,`` ``a.ndim)` specifying the axes permutation.

Returns:  
a copy of `a` with axes permuted.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose")

- [`jax.numpy.matrix_transpose()`](jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose")

Examples

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.permute_dims(a, (1, 0))
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

[](jax.numpy.percentile.html "previous page")

previous

jax.numpy.percentile

[](jax.numpy.piecewise.html "next page")

next

jax.numpy.piecewise

Contents

- [`permute_dims()`](#jax.numpy.permute_dims)

By The JAX authors

© Copyright 2024, The JAX Authors.\
