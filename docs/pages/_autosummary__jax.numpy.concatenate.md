- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.concatenate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.concatenate.rst "Download source file")
-  .pdf

# jax.numpy.concatenate

## Contents

- [`concatenate()`](#jax.numpy.concatenate)

# jax.numpy.concatenate[\#](#jax-numpy-concatenate "Link to this heading")

jax.numpy.concatenate(*arrays*, *axis=0*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4523-L4579)[\#](#jax.numpy.concatenate "Link to this definition")  
Join arrays along an existing axis.

JAX implementation of [`numpy.concatenate()`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate "(in NumPy v2.4)").

Parameters:  
- **arrays** (*np.ndarray* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a sequence of arrays to concatenate; each must have the same shape except along the specified axis. If a single array is given it will be treated equivalently to arrays = unstack(arrays), but the implementation will avoid explicit unstacking.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – specify the axis along which to concatenate. If None, the arrays are flattened before concatenation.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of the resulting array. If not specified, the dtype will be determined via type promotion rules described in [Type promotion semantics](../type_promotion.html#type-promotion).

Returns:  
the concatenated result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.concatenate()`](jax.lax.concatenate.html#jax.lax.concatenate "jax.lax.concatenate"): XLA concatenation API.

- [`jax.numpy.concat()`](jax.numpy.concat.html#jax.numpy.concat "jax.numpy.concat"): Array API version of this function.

- [`jax.numpy.stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"): concatenate arrays along a new axis.

Examples

One-dimensional concatenation:

    >>> x = jnp.arange(3)
    >>> y = jnp.zeros(3, dtype=int)
    >>> jnp.concatenate([x, y])
    Array([0, 1, 2, 0, 0, 0], dtype=int32)

Two-dimensional concatenation:

    >>> x = jnp.ones((2, 3))
    >>> y = jnp.zeros((2, 1))
    >>> jnp.concatenate([x, y], axis=1)
    Array([[1., 1., 1., 0.],
           [1., 1., 1., 0.]], dtype=float32)

[](jax.numpy.concat.html "previous page")

previous

jax.numpy.concat

[](jax.numpy.conj.html "next page")

next

jax.numpy.conj

Contents

- [`concatenate()`](#jax.numpy.concatenate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
