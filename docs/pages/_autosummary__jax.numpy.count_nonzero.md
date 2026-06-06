- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.count_nonzero

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.count_nonzero.rst "Download source file")
-  .pdf

# jax.numpy.count_nonzero

## Contents

- [`count_nonzero()`](#jax.numpy.count_nonzero)

# jax.numpy.count_nonzero[\#](#jax-numpy-count-nonzero "Link to this heading")

jax.numpy.count_nonzero(*a*, *axis=None*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1336-L1378)[\#](#jax.numpy.count_nonzero "Link to this definition")  
Return the number of nonzero elements along a given axis.

JAX implementation of [`numpy.count_nonzero()`](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the number of nonzeros are counted. If None, counts within the flattened array.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

Returns:  
An array with number of nonzeros elements along specified axis of the input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

By default, `jnp.count_nonzero` counts the nonzero values along all axes.

    >>> x = jnp.array([[1, 0, 0, 0],
    ...                [0, 0, 1, 0],
    ...                [1, 1, 1, 0]])
    >>> jnp.count_nonzero(x)
    Array(5, dtype=int32)

If `axis=1`, counts along axis 1.

    >>> jnp.count_nonzero(x, axis=1)
    Array([1, 1, 3], dtype=int32)

To preserve the dimensions of input, you can set `keepdims=True`.

    >>> jnp.count_nonzero(x, axis=1, keepdims=True)
    Array([[1],
           [1],
           [3]], dtype=int32)

[](jax.numpy.cosh.html "previous page")

previous

jax.numpy.cosh

[](jax.numpy.cov.html "next page")

next

jax.numpy.cov

Contents

- [`count_nonzero()`](#jax.numpy.count_nonzero)

By The JAX authors

© Copyright 2024, The JAX Authors.\
