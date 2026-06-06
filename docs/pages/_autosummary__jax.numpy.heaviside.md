- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.heaviside

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.heaviside.rst "Download source file")
-  .pdf

# jax.numpy.heaviside

## Contents

- [`heaviside()`](#jax.numpy.heaviside)

# jax.numpy.heaviside[\#](#jax-numpy-heaviside "Link to this heading")

jax.numpy.heaviside(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3662-L3711)[\#](#jax.numpy.heaviside "Link to this definition")  
Compute the heaviside step function.

JAX implementation of [`numpy.heaviside`](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html#numpy.heaviside "(in NumPy v2.4)").

The heaviside step function is defined by:

\\\begin{split}\mathrm{heaviside}(x1, x2) = \begin{cases} 0, & x1 \< 0\\ x2, & x1 = 0\\ 1, & x1 \> 0. \end{cases}\end{split}\\

Parameters:  
- **x1** (*ArrayLike*) – input array or scalar. `complex` dtype are not supported.

- **x2** (*ArrayLike*) – scalar or array. Specifies the return values when `x1` is `0`. `complex` dtype are not supported. `x1` and `x2` must either have same shape or broadcast compatible.

Returns:  
An array containing the heaviside step function of `x1`, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x1 = jnp.array([[-2, 0, 3],
    ...                 [5, -1, 0],
    ...                 [0, 7, -3]])
    >>> x2 = jnp.array([2, 0.5, 1])
    >>> jnp.heaviside(x1, x2)
    Array([[0. , 0.5, 1. ],
           [1. , 0. , 1. ],
           [2. , 1. , 0. ]], dtype=float32)
    >>> jnp.heaviside(x1, 0.5)
    Array([[0. , 0.5, 1. ],
           [1. , 0. , 0.5],
           [0.5, 1. , 0. ]], dtype=float32)
    >>> jnp.heaviside(-3, x2)
    Array([0., 0., 0.], dtype=float32)

[](jax.numpy.hanning.html "previous page")

previous

jax.numpy.hanning

[](jax.numpy.histogram.html "next page")

next

jax.numpy.histogram

Contents

- [`heaviside()`](#jax.numpy.heaviside)

By The JAX authors

© Copyright 2024, The JAX Authors.\
