- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vstack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vstack.rst "Download source file")
-  .pdf

# jax.numpy.vstack

## Contents

- [`vstack()`](#jax.numpy.vstack)

# jax.numpy.vstack[\#](#jax-numpy-vstack "Link to this heading")

jax.numpy.vstack(*tup*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4622-L4680)[\#](#jax.numpy.vstack "Link to this definition")  
Vertically stack arrays.

JAX implementation of [`numpy.vstack()`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack "(in NumPy v2.4)").

For arrays of two or more dimensions, this is equivalent to [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate") with `axis=0`.

Parameters:  
- **tup** (*np.ndarray* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a sequence of arrays to stack; each must have the same shape along all but the first axis. If a single array is given it will be treated equivalently to tup = unstack(tup), but the implementation will avoid explicit unstacking.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of the resulting array. If not specified, the dtype will be determined via type promotion rules described in [Type promotion semantics](../type_promotion.html#type-promotion).

Returns:  
the stacked result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"): stack along arbitrary axes

- [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate"): concatenation along existing axes.

- [`jax.numpy.hstack()`](jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack"): stack horizontally, i.e. along axis 1.

- [`jax.numpy.dstack()`](jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack"): stack depth-wise, i.e. along axis 2.

Examples

Scalar values:

    >>> jnp.vstack([1, 2, 3])
    Array([[1],
           [2],
           [3]], dtype=int32, weak_type=True)

1D arrays:

    >>> x = jnp.arange(4)
    >>> y = jnp.ones(4)
    >>> jnp.vstack([x, y])
    Array([[0., 1., 2., 3.],
           [1., 1., 1., 1.]], dtype=float32)

2D arrays:

    >>> x = x.reshape(1, 4)
    >>> y = y.reshape(1, 4)
    >>> jnp.vstack([x, y])
    Array([[0., 1., 2., 3.],
           [1., 1., 1., 1.]], dtype=float32)

[](jax.numpy.vsplit.html "previous page")

previous

jax.numpy.vsplit

[](jax.numpy.where.html "next page")

next

jax.numpy.where

Contents

- [`vstack()`](#jax.numpy.vstack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
