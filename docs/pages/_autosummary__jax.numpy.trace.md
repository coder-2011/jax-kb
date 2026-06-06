- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.trace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.trace.rst "Download source file")
-  .pdf

# jax.numpy.trace

## Contents

- [`trace()`](#jax.numpy.trace)

# jax.numpy.trace[\#](#jax-numpy-trace "Link to this heading")

jax.numpy.trace(*a*, *offset=0*, *axis1=0*, *axis2=1*, *dtype=None*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6704-L6769)[\#](#jax.numpy.trace "Link to this definition")  
Calculate sum of the diagonal of input along the given axes.

JAX implementation of [`numpy.trace()`](https://numpy.org/doc/stable/reference/generated/numpy.trace.html#numpy.trace "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array. Must have `a.ndim`` ``>=`` ``2`.

- **offset** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike*) – optional, int, default=0. Diagonal offset from the main diagonal. Can be positive or negative.

- **axis1** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, default=0. The first axis along which to take the sum of diagonal. Must be a static integer value.

- **axis2** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, default=1. The second axis along which to take the sum of diagonal. Must be a static integer value.

- **dtype** (*DTypeLike* *\|* *None*) – optional. The dtype of the output array. Should be provided as static argument in JIT compilation.

- **out** (*None*) – Not used by JAX.

Returns:  
An array of dimension x.ndim-2 containing the sum of the diagonal elements along axes (axis1, axis2)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.diag()`](jax.numpy.diag.html#jax.numpy.diag "jax.numpy.diag"): Returns the specified diagonal or constructs a diagonal array

- [`jax.numpy.diagonal()`](jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal"): Returns the specified diagonal of an array.

- [`jax.numpy.diagflat()`](jax.numpy.diagflat.html#jax.numpy.diagflat "jax.numpy.diagflat"): Returns a 2-D array with the flattened input array laid out on the diagonal.

Examples

    >>> x = jnp.arange(1, 9).reshape(2, 2, 2)
    >>> x
    Array([[[1, 2],
            [3, 4]],

           [[5, 6],
            [7, 8]]], dtype=int32)
    >>> jnp.trace(x)
    Array([ 8, 10], dtype=int32)
    >>> jnp.trace(x, offset=1)
    Array([3, 4], dtype=int32)
    >>> jnp.trace(x, axis1=1, axis2=2)
    Array([ 5, 13], dtype=int32)
    >>> jnp.trace(x, offset=1, axis1=1, axis2=2)
    Array([2, 6], dtype=int32)

[](jax.numpy.tile.html "previous page")

previous

jax.numpy.tile

[](jax.numpy.trapezoid.html "next page")

next

jax.numpy.trapezoid

Contents

- [`trace()`](#jax.numpy.trace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
