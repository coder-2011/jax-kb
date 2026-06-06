- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.stack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.stack.rst "Download source file")
-  .pdf

# jax.numpy.stack

## Contents

- [`stack()`](#jax.numpy.stack)

# jax.numpy.stack[\#](#jax-numpy-stack "Link to this heading")

jax.numpy.stack(*arrays*, *axis=0*, *out=None*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4359-L4421)[\#](#jax.numpy.stack "Link to this definition")  
Join arrays along a new axis.

JAX implementation of [`numpy.stack()`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html#numpy.stack "(in NumPy v2.4)").

Parameters:  
- **arrays** (*np.ndarray* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a sequence of arrays to stack; each must have the same shape. If a single array is given it will be treated equivalently to arrays = unstack(arrays), but the implementation will avoid explicit unstacking.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – specify the axis along which to stack.

- **out** (*None*) – unused by JAX

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of the resulting array. If not specified, the dtype will be determined via type promotion rules described in [Type promotion semantics](../type_promotion.html#type-promotion).

Returns:  
the stacked result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.unstack()`](jax.numpy.unstack.html#jax.numpy.unstack "jax.numpy.unstack"): inverse of `stack`.

- [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate"): concatenation along existing axes.

- [`jax.numpy.vstack()`](jax.numpy.vstack.html#jax.numpy.vstack "jax.numpy.vstack"): stack vertically, i.e. along axis 0.

- [`jax.numpy.hstack()`](jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack"): stack horizontally, i.e. along axis 1.

- [`jax.numpy.dstack()`](jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack"): stack depth-wise, i.e. along axis 2.

- [`jax.numpy.column_stack()`](jax.numpy.column_stack.html#jax.numpy.column_stack "jax.numpy.column_stack"): stack columns.

Examples

    >>> x = jnp.array([1, 2, 3])
    >>> y = jnp.array([4, 5, 6])
    >>> jnp.stack([x, y])
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)
    >>> jnp.stack([x, y], axis=1)
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

[`unstack()`](jax.numpy.unstack.html#jax.numpy.unstack "jax.numpy.unstack") performs the inverse operation:

    >>> arr = jnp.stack([x, y], axis=1)
    >>> x, y = jnp.unstack(arr, axis=1)
    >>> x
    Array([1, 2, 3], dtype=int32)
    >>> y
    Array([4, 5, 6], dtype=int32)

[](jax.numpy.squeeze.html "previous page")

previous

jax.numpy.squeeze

[](jax.numpy.std.html "next page")

next

jax.numpy.std

Contents

- [`stack()`](#jax.numpy.stack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
