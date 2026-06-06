- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.signbit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.signbit.rst "Download source file")
-  .pdf

# jax.numpy.signbit

## Contents

- [`signbit()`](#jax.numpy.signbit)

# jax.numpy.signbit[\#](#jax-numpy-signbit "Link to this heading")

jax.numpy.signbit(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2893-L2955)[\#](#jax.numpy.signbit "Link to this definition")  
Return the sign bit of array elements.

JAX implementation of [`numpy.signbit`](https://numpy.org/doc/stable/reference/generated/numpy.signbit.html#numpy.signbit "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array. Complex values are not supported.

Returns:  
A boolean array of the same shape as `x`, containing `True` where the sign of `x` is negative, and `False` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.sign()`](jax.numpy.sign.html#jax.numpy.sign "jax.numpy.sign"): return the mathematical sign of array elements, i.e. `-1`, `0`, or `+1`.

Examples

[`signbit()`](#jax.numpy.signbit "jax.numpy.signbit") on boolean values is always `False`:

    >>> x = jnp.array([True, False])
    >>> jnp.signbit(x)
    Array([False, False], dtype=bool)

[`signbit()`](#jax.numpy.signbit "jax.numpy.signbit") on integer values is equivalent to `x`` ``<`` ``0`:

    >>> x = jnp.array([-2, -1, 0, 1, 2])
    >>> jnp.signbit(x)
    Array([ True,  True, False, False, False], dtype=bool)

[`signbit()`](#jax.numpy.signbit "jax.numpy.signbit") on floating point values returns the value of the actual sign bit from the float representation, including signed zero:

    >>> x = jnp.array([-1.5, -0.0, 0.0, 1.5])
    >>> jnp.signbit(x)
    Array([ True, True, False, False], dtype=bool)

This also returns the sign bit for special values such as signed NaN and signed infinity:

    >>> x = jnp.array([jnp.nan, -jnp.nan, jnp.inf, -jnp.inf])
    >>> jnp.signbit(x)
    Array([False,  True, False,  True], dtype=bool)

[](jax.numpy.sign.html "previous page")

previous

jax.numpy.sign

[](jax.numpy.signedinteger.html "next page")

next

jax.numpy.signedinteger

Contents

- [`signbit()`](#jax.numpy.signbit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
