- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.where

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.where.rst "Download source file")
-  .pdf

# jax.numpy.where

## Contents

- [`where()`](#jax.numpy.where)

# jax.numpy.where[\#](#jax-numpy-where "Link to this heading")

jax.numpy.where(*condition*, *x=None*, *y=None*, */*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2726-L2800)[\#](#jax.numpy.where "Link to this definition")  
Select elements from two arrays based on a condition.

JAX implementation of [`numpy.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where "(in NumPy v2.4)").

Note

when only `condition` is provided, `jnp.where(condition)` is equivalent to `jnp.nonzero(condition)`. For that case, refer to the documentation of [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero"). The docstring below focuses on the case where `x` and `y` are specified.

The three-term version of `jnp.where` lowers to [`jax.lax.select()`](jax.lax.select.html#jax.lax.select "jax.lax.select").

Parameters:  
- **condition** – boolean array. Must be broadcast-compatible with `x` and `y` when they are specified.

- **x** – arraylike. Should be broadcast-compatible with `condition` and `y`, and typecast-compatible with `y`.

- **y** – arraylike. Should be broadcast-compatible with `condition` and `x`, and typecast-compatible with `x`.

- **size** – integer, only referenced when `x` and `y` are `None`. For details, see [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero").

- **fill_value** – only referenced when `x` and `y` are `None`. For details, see [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero").

Returns:  
An array of dtype `jnp.result_type(x,`` ``y)` with values drawn from `x` where `condition` is True, and from `y` where condition is `False`. If `x` and `y` are `None`, the function behaves differently; see [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero") for a description of the return type.

See also

- [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")

- [`jax.numpy.argwhere()`](jax.numpy.argwhere.html#jax.numpy.argwhere "jax.numpy.argwhere")

- [`jax.lax.select()`](jax.lax.select.html#jax.lax.select "jax.lax.select")

Notes

Special care is needed when the `x` or `y` input to [`jax.numpy.where()`](#jax.numpy.where "jax.numpy.where") could have a value of NaN. Specifically, when a gradient is taken with [`jax.grad()`](jax.grad.html#jax.grad "jax.grad") (reverse-mode differentiation), a NaN in either `x` or `y` will propagate into the gradient, regardless of the value of `condition`. More information on this behavior and workarounds is available in the [JAX FAQ](https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where).

Examples

When `x` and `y` are not provided, `where` behaves equivalently to [`jax.numpy.nonzero()`](jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero"):

    >>> x = jnp.arange(10)
    >>> jnp.where(x > 4)
    (Array([5, 6, 7, 8, 9], dtype=int32),)
    >>> jnp.nonzero(x > 4)
    (Array([5, 6, 7, 8, 9], dtype=int32),)

When `x` and `y` are provided, `where` selects between them based on the specified condition:

    >>> jnp.where(x > 4, x, 0)
    Array([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype=int32)

[](jax.numpy.vstack.html "previous page")

previous

jax.numpy.vstack

[](jax.numpy.zeros.html "next page")

next

jax.numpy.zeros

Contents

- [`where()`](#jax.numpy.where)

By The JAX authors

© Copyright 2024, The JAX Authors.\
