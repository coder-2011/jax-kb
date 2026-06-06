- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.copy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.copy.rst "Download source file")
-  .pdf

# jax.numpy.copy

## Contents

- [`copy()`](#jax.numpy.copy)

# jax.numpy.copy[\#](#jax-numpy-copy "Link to this heading")

jax.numpy.copy(*a*, *order=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5322-L5369)[\#](#jax.numpy.copy "Link to this definition")  
Return a copy of the array.

JAX implementation of [`numpy.copy()`](https://numpy.org/doc/stable/reference/generated/numpy.copy.html#numpy.copy "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – arraylike object to copy

- **order** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – not implemented in JAX

Returns:  
a copy of the input array `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.array()`](jax.numpy.array.html#jax.numpy.array "jax.numpy.array"): create an array with or without a copy.

- [`jax.Array.copy()`](jax.Array.copy.html#jax.Array.copy "jax.Array.copy"): same function accessed as an array method.

Examples

Since JAX arrays are immutable, in most cases explicit array copies are not necessary. One exception is when using a function with donated arguments (see the `donate_argnums` argument to [`jax.jit()`](jax.jit.html#jax.jit "jax.jit")).

    >>> f = jax.jit(lambda x: 2 * x, donate_argnums=0)
    >>> x = jnp.arange(4)
    >>> y = f(x)
    >>> print(y)
    [0 2 4 6]

Because we marked `x` as being donated, the original array is no longer available:

    >>> print(x)  
    Traceback (most recent call last):
    RuntimeError: Array has been deleted with shape=int32[4].

In situations like this, an explicit copy will let you keep access to the original buffer:

    >>> x = jnp.arange(4)
    >>> y = f(x.copy())
    >>> print(y)
    [0 2 4 6]
    >>> print(x)
    [0 1 2 3]

[](jax.numpy.convolve.html "previous page")

previous

jax.numpy.convolve

[](jax.numpy.copysign.html "next page")

next

jax.numpy.copysign

Contents

- [`copy()`](#jax.numpy.copy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
