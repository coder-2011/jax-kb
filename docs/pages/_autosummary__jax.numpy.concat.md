- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.concat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.concat.rst "Download source file")
-  .pdf

# jax.numpy.concat

## Contents

- [`concat()`](#jax.numpy.concat)

# jax.numpy.concat[\#](#jax-numpy-concat "Link to this heading")

jax.numpy.concat(*arrays*, */*, *\**, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4581-L4620)[\#](#jax.numpy.concat "Link to this definition")  
Join arrays along an existing axis.

JAX implementation of [`array_api.concat()`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.concat.html#array_api.concat "(in Python array API standard)").

Parameters:  
- **arrays** (*Sequence\[ArrayLike\]*) – a sequence of arrays to concatenate; each must have the same shape except along the specified axis. If a single array is given it will be treated equivalently to arrays = unstack(arrays), but the implementation will avoid explicit unstacking.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – specify the axis along which to concatenate.

Returns:  
the concatenated result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.concatenate()`](jax.lax.concatenate.html#jax.lax.concatenate "jax.lax.concatenate"): XLA concatenation API.

- [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate"): NumPy version of this function.

- [`jax.numpy.stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"): concatenate arrays along a new axis.

Examples

One-dimensional concatenation:

    >>> x = jnp.arange(3)
    >>> y = jnp.zeros(3, dtype=int)
    >>> jnp.concat([x, y])
    Array([0, 1, 2, 0, 0, 0], dtype=int32)

Two-dimensional concatenation:

    >>> x = jnp.ones((2, 3))
    >>> y = jnp.zeros((2, 1))
    >>> jnp.concat([x, y], axis=1)
    Array([[1., 1., 1., 0.],
           [1., 1., 1., 0.]], dtype=float32)

[](jax.numpy.compress.html "previous page")

previous

jax.numpy.compress

[](jax.numpy.concatenate.html "next page")

next

jax.numpy.concatenate

Contents

- [`concat()`](#jax.numpy.concat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
