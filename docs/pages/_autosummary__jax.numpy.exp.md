- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.exp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.exp.rst "Download source file")
-  .pdf

# jax.numpy.exp

## Contents

- [`exp()`](#jax.numpy.exp)

# jax.numpy.exp[\#](#jax-numpy-exp "Link to this heading")

jax.numpy.exp(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L414-L454)[\#](#jax.numpy.exp "Link to this definition")  
Calculate element-wise exponential of the input.

JAX implementation of [`numpy.exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html#numpy.exp "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar

Returns:  
An array containing the exponential of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.log()`](jax.numpy.log.html#jax.numpy.log "jax.numpy.log"): Calculates element-wise logarithm of the input.

- [`jax.numpy.expm1()`](jax.numpy.expm1.html#jax.numpy.expm1 "jax.numpy.expm1"): Calculates \\e^x-1\\ of each element of the input.

- [`jax.numpy.exp2()`](jax.numpy.exp2.html#jax.numpy.exp2 "jax.numpy.exp2"): Calculates base-2 exponential of each element of the input.

Examples

`jnp.exp` follows the properties of exponential such as \\e^{(a+b)} = e^a \* e^b\\.

    >>> x1 = jnp.array([2, 4, 3, 1])
    >>> x2 = jnp.array([1, 3, 2, 3])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.exp(x1+x2))
    [  20.09 1096.63  148.41   54.6 ]
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.exp(x1)*jnp.exp(x2))
    [  20.09 1096.63  148.41   54.6 ]

This property holds for complex input also:

    >>> jnp.allclose(jnp.exp(3-4j), jnp.exp(3)*jnp.exp(-4j))
    Array(True, dtype=bool)

[](jax.numpy.equal.html "previous page")

previous

jax.numpy.equal

[](jax.numpy.exp2.html "next page")

next

jax.numpy.exp2

Contents

- [`exp()`](#jax.numpy.exp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
