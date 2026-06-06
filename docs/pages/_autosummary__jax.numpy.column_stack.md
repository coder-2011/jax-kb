- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.column_stack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.column_stack.rst "Download source file")
-  .pdf

# jax.numpy.column_stack

## Contents

- [`column_stack()`](#jax.numpy.column_stack)

# jax.numpy.column_stack[\#](#jax-numpy-column-stack "Link to this heading")

jax.numpy.column_stack(*tup*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4803-L4861)[\#](#jax.numpy.column_stack "Link to this definition")  
Stack arrays column-wise.

JAX implementation of [`numpy.column_stack()`](https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html#numpy.column_stack "(in NumPy v2.4)").

For arrays of two or more dimensions, this is equivalent to [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate") with `axis=1`.

Parameters:  
- **tup** (*np.ndarray* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a sequence of arrays to stack; each must have the same leading dimension. Input arrays will be promoted to at least rank 2. If a single array is given it will be treated equivalently to tup = unstack(tup), but the implementation will avoid explicit unstacking.

- **dtype** – optional dtype of the resulting array. If not specified, the dtype will be determined via type promotion rules described in [Type promotion semantics](../type_promotion.html#type-promotion).

Returns:  
the stacked result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"): stack along arbitrary axes

- [`jax.numpy.concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate"): concatenation along existing axes.

- [`jax.numpy.vstack()`](jax.numpy.vstack.html#jax.numpy.vstack "jax.numpy.vstack"): stack vertically, i.e. along axis 0.

- [`jax.numpy.hstack()`](jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack"): stack horizontally, i.e. along axis 1.

- [`jax.numpy.dstack()`](jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack"): stack depth-wise, i.e. along axis 2.

Examples

Scalar values:

    >>> jnp.column_stack([1, 2, 3])
    Array([[1, 2, 3]], dtype=int32, weak_type=True)

1D arrays:

    >>> x = jnp.arange(3)
    >>> y = jnp.ones(3)
    >>> jnp.column_stack([x, y])
    Array([[0., 1.],
           [1., 1.],
           [2., 1.]], dtype=float32)

2D arrays:

    >>> x = x.reshape(3, 1)
    >>> y = y.reshape(3, 1)
    >>> jnp.column_stack([x, y])
    Array([[0., 1.],
           [1., 1.],
           [2., 1.]], dtype=float32)

[](jax.numpy.clip.html "previous page")

previous

jax.numpy.clip

[](jax.numpy.complex_.html "next page")

next

jax.numpy.complex\_

Contents

- [`column_stack()`](#jax.numpy.column_stack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
