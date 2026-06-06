- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.dstack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.dstack.rst "Download source file")
-  .pdf

# jax.numpy.dstack

## Contents

- [`dstack()`](#jax.numpy.dstack)

# jax.numpy.dstack[\#](#jax-numpy-dstack "Link to this heading")

jax.numpy.dstack(*tup*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4742-L4801)[\#](#jax.numpy.dstack "Link to this definition")  
Stack arrays depth-wise.

JAX implementation of [`numpy.dstack()`](https://numpy.org/doc/stable/reference/generated/numpy.dstack.html#numpy.dstack "(in NumPy v2.4)").

For arrays of three or more dimensions, this is equivalent to [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate") with `axis=2`.

Parameters:  
- **tup** (*np.ndarray* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a sequence of arrays to stack; each must have the same shape along all but the third axis. Input arrays will be promoted to at least rank 3. If a single array is given it will be treated equivalently to tup = unstack(tup), but the implementation will avoid explicit unstacking.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of the resulting array. If not specified, the dtype will be determined via type promotion rules described in [Type promotion semantics](../type_promotion.html#type-promotion).

Returns:  
the stacked result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"): stack along arbitrary axes

- [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate"): concatenation along existing axes.

- [`jax.numpy.vstack()`](jax.numpy.vstack.html#jax.numpy.vstack "jax.numpy.vstack"): stack vertically, i.e. along axis 0.

- [`jax.numpy.hstack()`](jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack"): stack horizontally, i.e. along axis 1.

Examples

Scalar values:

    >>> jnp.dstack([1, 2, 3])
    Array([[[1, 2, 3]]], dtype=int32, weak_type=True)

1D arrays:

    >>> x = jnp.arange(3)
    >>> y = jnp.ones(3)
    >>> jnp.dstack([x, y])
    Array([[[0., 1.],
            [1., 1.],
            [2., 1.]]], dtype=float32)

2D arrays:

    >>> x = x.reshape(1, 3)
    >>> y = y.reshape(1, 3)
    >>> jnp.dstack([x, y])
    Array([[[0., 1.],
            [1., 1.],
            [2., 1.]]], dtype=float32)

[](jax.numpy.dsplit.html "previous page")

previous

jax.numpy.dsplit

[](jax.numpy.dtype.html "next page")

next

jax.numpy.dtype

Contents

- [`dstack()`](#jax.numpy.dstack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
