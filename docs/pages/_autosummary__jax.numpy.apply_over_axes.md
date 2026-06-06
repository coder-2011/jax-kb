- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.apply_over_axes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.apply_over_axes.rst "Download source file")
-  .pdf

# jax.numpy.apply_over_axes

## Contents

- [`apply_over_axes()`](#jax.numpy.apply_over_axes)

# jax.numpy.apply_over_axes[\#](#jax-numpy-apply-over-axes "Link to this heading")

jax.numpy.apply_over_axes(*func*, *a*, *axes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7885-L7939)[\#](#jax.numpy.apply_over_axes "Link to this definition")  
Apply a function repeatedly over specified axes.

JAX implementation of [`numpy.apply_over_axes()`](https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html#numpy.apply_over_axes "(in NumPy v2.4)").

Parameters:  
- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[ArrayLike,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – the function to apply, with signature `func(Array,`` ``int)`` ``->`` ``Array`, and where `y`` ``=`` ``func(x,`` ``axis)` must satisfy `y.ndim`` ``in`` ``[x.ndim,`` ``x.ndim`` ``-`` ``1]`.

- **a** (*ArrayLike*) – N-dimensional array over which to apply the function.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the sequence of axes over which to apply the function.

Returns:  
An N-dimensional array containing the result of the repeated function application.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.apply_along_axis()`](jax.numpy.apply_along_axis.html#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis"): apply a 1D function along a single axis.

Examples

This function is designed to have similar semantics to typical associative [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy") reductions over one or more axes with `keepdims=True`. For example:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])

    >>> jnp.apply_over_axes(jnp.sum, x, [0])
    Array([[5, 7, 9]], dtype=int32)
    >>> jnp.sum(x, [0], keepdims=True)
    Array([[5, 7, 9]], dtype=int32)

    >>> jnp.apply_over_axes(jnp.min, x, [1])
    Array([[1],
           [4]], dtype=int32)
    >>> jnp.min(x, [1], keepdims=True)
    Array([[1],
           [4]], dtype=int32)

    >>> jnp.apply_over_axes(jnp.prod, x, [0, 1])
    Array([[720]], dtype=int32)
    >>> jnp.prod(x, [0, 1], keepdims=True)
    Array([[720]], dtype=int32)

[](jax.numpy.apply_along_axis.html "previous page")

previous

jax.numpy.apply_along_axis

[](jax.numpy.arange.html "next page")

next

jax.numpy.arange

Contents

- [`apply_over_axes()`](#jax.numpy.apply_over_axes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
