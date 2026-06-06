- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logical_not

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logical_not.rst "Download source file")
-  .pdf

# jax.numpy.logical_not

## Contents

- [`logical_not()`](#jax.numpy.logical_not)

# jax.numpy.logical_not[\#](#jax-numpy-logical-not "Link to this heading")

jax.numpy.logical_not(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1934-L1970)[\#](#jax.numpy.logical_not "Link to this definition")  
Compute NOT bool(x) element-wise.

JAX implementation of `numpy.logical_not()`.

Parameters:  
**x** (*ArrayLike*) – input array of any dtype.

Returns:  
A boolean array that computes NOT bool(x) element-wise

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.invert()`](jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert") or [`jax.numpy.bitwise_invert()`](jax.numpy.bitwise_invert.html#jax.numpy.bitwise_invert "jax.numpy.bitwise_invert"): bitwise NOT operation

Examples

Compute NOT x element-wise on a boolean array:

    >>> x = jnp.array([True, False, True])
    >>> jnp.logical_not(x)
    Array([False,  True, False], dtype=bool)

For boolean input, this is equivalent to [`invert()`](jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert"), which implements the unary `~` operator:

    >>> ~x
    Array([False,  True, False], dtype=bool)

For non-boolean input, the input of [`logical_not()`](#jax.numpy.logical_not "jax.numpy.logical_not") is implicitly cast to boolean:

    >>> x = jnp.array([-1, 0, 1])
    >>> jnp.logical_not(x)
    Array([False,  True, False], dtype=bool)

[](jax.numpy.logical_and.html "previous page")

previous

jax.numpy.logical_and

[](jax.numpy.logical_or.html "next page")

next

jax.numpy.logical_or

Contents

- [`logical_not()`](#jax.numpy.logical_not)

By The JAX authors

© Copyright 2024, The JAX Authors.\
