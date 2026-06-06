- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.mask_indices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.mask_indices.rst "Download source file")
-  .pdf

# jax.numpy.mask_indices

## Contents

- [`mask_indices()`](#jax.numpy.mask_indices)

# jax.numpy.mask_indices[\#](#jax-numpy-mask-indices "Link to this heading")

jax.numpy.mask_indices(*n*, *mask_func*, *k=0*, *\**, *size=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6771-L6819)[\#](#jax.numpy.mask_indices "Link to this definition")  
Return indices of a mask of an (n, n) array.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – static integer array dimension.

- **mask_func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[ArrayLike,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – a function that takes a shape `(n,`` ``n)` array and an optional offset `k`, and returns a shape `(n,`` ``n)` mask. Examples of functions with this signature are [`triu()`](jax.numpy.triu.html#jax.numpy.triu "jax.numpy.triu") and [`tril()`](jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril").

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – a scalar value passed to `mask_func`.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional argument specifying the static size of the output arrays. This is passed to [`nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") when generating the indices from the mask.

Returns:  
a tuple of indices where `mask_func` is nonzero.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.triu_indices()`](jax.numpy.triu_indices.html#jax.numpy.triu_indices "jax.numpy.triu_indices"): compute `mask_indices` for [`triu()`](jax.numpy.triu.html#jax.numpy.triu "jax.numpy.triu").

- [`jax.numpy.tril_indices()`](jax.numpy.tril_indices.html#jax.numpy.tril_indices "jax.numpy.tril_indices"): compute `mask_indices` for [`tril()`](jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril").

Examples

Calling `mask_indices` on built-in masking functions:

    >>> jnp.mask_indices(3, jnp.triu)
    (Array([0, 0, 0, 1, 1, 2], dtype=int32), Array([0, 1, 2, 1, 2, 2], dtype=int32))

    >>> jnp.mask_indices(3, jnp.tril)
    (Array([0, 1, 1, 2, 2, 2], dtype=int32), Array([0, 0, 1, 0, 1, 2], dtype=int32))

Calling `mask_indices` on a custom masking function:

    >>> def mask_func(x, k=0):
    ...   i = jnp.arange(x.shape[0])[:, None]
    ...   j = jnp.arange(x.shape[1])
    ...   return (i + 1) % (j + 1 + k) == 0
    >>> mask_func(jnp.ones((3, 3)))
    Array([[ True, False, False],
           [ True,  True, False],
           [ True, False,  True]], dtype=bool)
    >>> jnp.mask_indices(3, mask_func)
    (Array([0, 1, 1, 2, 2], dtype=int32), Array([0, 0, 1, 0, 2], dtype=int32))

[](jax.numpy.logspace.html "previous page")

previous

jax.numpy.logspace

[](jax.numpy.matmul.html "next page")

next

jax.numpy.matmul

Contents

- [`mask_indices()`](#jax.numpy.mask_indices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
