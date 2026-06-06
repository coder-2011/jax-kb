- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.frexp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.frexp.rst "Download source file")
-  .pdf

# jax.numpy.frexp

## Contents

- [`frexp()`](#jax.numpy.frexp)

# jax.numpy.frexp[\#](#jax-numpy-frexp "Link to this heading")

jax.numpy.frexp(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3025-L3063)[\#](#jax.numpy.frexp "Link to this definition")  
Split floating point values into mantissa and twos exponent.

JAX implementation of `numpy.frexp()`.

Parameters:  
**x** (*ArrayLike*) – real-valued array

Returns:  
A tuple `(mantissa,`` ``exponent)` where `mantissa` is a floating point value between -1 and 1, and `exponent` is an integer such that `x`` ``==`` ``mantissa`` ``*`` ``2`` ``**`` ``exponent`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.ldexp()`](jax.numpy.ldexp.html#jax.numpy.ldexp "jax.numpy.ldexp"): compute the inverse of `frexp`.

Examples

Split values into mantissa and exponent:

    >>> x = jnp.array([1., 2., 3., 4., 5.])
    >>> m, e = jnp.frexp(x)
    >>> m
    Array([0.5  , 0.5  , 0.75 , 0.5  , 0.625], dtype=float32)
    >>> e
    Array([1, 2, 2, 3, 3], dtype=int32)

Reconstruct the original array:

    >>> m * 2 ** e
    Array([1., 2., 3., 4., 5.], dtype=float32)

[](jax.numpy.fmod.html "previous page")

previous

jax.numpy.fmod

[](jax.numpy.frombuffer.html "next page")

next

jax.numpy.frombuffer

Contents

- [`frexp()`](#jax.numpy.frexp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
