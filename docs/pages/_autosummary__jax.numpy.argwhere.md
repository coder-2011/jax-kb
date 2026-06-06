- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.argwhere

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.argwhere.rst "Download source file")
-  .pdf

# jax.numpy.argwhere

## Contents

- [`argwhere()`](#jax.numpy.argwhere)

# jax.numpy.argwhere[\#](#jax-numpy-argwhere "Link to this heading")

jax.numpy.argwhere(*a*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8151-L8216)[\#](#jax.numpy.argwhere "Link to this definition")  
Find the indices of nonzero array elements

JAX implementation of [`numpy.argwhere()`](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html#numpy.argwhere "(in NumPy v2.4)").

`jnp.argwhere(x)` is essentially equivalent to `jnp.column_stack(jnp.nonzero(x))` with special handling for zero-dimensional (i.e. scalar) inputs.

Because the size of the output of `argwhere` is data-dependent, the function is not typically compatible with JIT. The JAX version adds the optional `size` argument, which specifies the size of the leading dimension of the output - it must be specified statically for `jnp.argwhere` to be compiled with non-static operands. See [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") for a full discussion of `size` and its semantics.

Parameters:  
- **a** (*ArrayLike*) – array for which to find nonzero elements

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer specifying statically the number of expected nonzero elements. This must be specified in order to use `argwhere` within JAX transformations like [`jax.jit()`](jax.jit.html#jax.jit "jax.jit"). See [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") for more information.

- **fill_value** (*ArrayLike* *\|* *None*) – optional array specifying the fill value when `size` is specified. See [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") for more information.

Returns:  
a two-dimensional array of shape `[size,`` ``x.ndim]`. If `size` is not specified as an argument, it is equal to the number of nonzero elements in `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.where()`](jax.numpy.where.html#jax.numpy.where "jax.numpy.where")

- [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")

Examples

Two-dimensional array:

    >>> x = jnp.array([[1, 0, 2],
    ...                [0, 3, 0]])
    >>> jnp.argwhere(x)
    Array([[0, 0],
           [0, 2],
           [1, 1]], dtype=int32)

Equivalent computation using [`jax.numpy.column_stack()`](jax.numpy.column_stack.html#jax.numpy.column_stack "jax.numpy.column_stack") and [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero"):

    >>> jnp.column_stack(jnp.nonzero(x))
    Array([[0, 0],
           [0, 2],
           [1, 1]], dtype=int32)

Special case for zero-dimensional (i.e. scalar) inputs:

    >>> jnp.argwhere(1)
    Array([], shape=(1, 0), dtype=int32)
    >>> jnp.argwhere(0)
    Array([], shape=(0, 0), dtype=int32)

[](jax.numpy.argsort.html "previous page")

previous

jax.numpy.argsort

[](jax.numpy.around.html "next page")

next

jax.numpy.around

Contents

- [`argwhere()`](#jax.numpy.argwhere)

By The JAX authors

© Copyright 2024, The JAX Authors.\
