- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.exp2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.exp2.rst "Download source file")
-  .pdf

# jax.numpy.exp2

## Contents

- [`exp2()`](#jax.numpy.exp2)

# jax.numpy.exp2[\#](#jax-numpy-exp2 "Link to this heading")

jax.numpy.exp2(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2858-L2891)[\#](#jax.numpy.exp2 "Link to this definition")  
Calculate element-wise base-2 exponential of input.

JAX implementation of [`numpy.exp2`](https://numpy.org/doc/stable/reference/generated/numpy.exp2.html#numpy.exp2 "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar

Returns:  
An array containing the base-2 exponential of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.log2()`](jax.numpy.log2.html#jax.numpy.log2 "jax.numpy.log2"): Calculates base-2 logarithm of each element of input.

- [`jax.numpy.exp()`](jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp"): Calculates exponential of each element of the input.

- [`jax.numpy.expm1()`](jax.numpy.expm1.html#jax.numpy.expm1 "jax.numpy.expm1"): Calculates \\e^x-1\\ of each element of the input.

Examples

`jnp.exp2` follows the properties of the exponential such as \\2^{a+b} = 2^a \* 2^b\\.

    >>> x1 = jnp.array([2, -4, 3, -1])
    >>> x2 = jnp.array([-1, 3, -2, 3])
    >>> jnp.exp2(x1+x2)
    Array([2. , 0.5, 2. , 4. ], dtype=float32)
    >>> jnp.exp2(x1)*jnp.exp2(x2)
    Array([2. , 0.5, 2. , 4. ], dtype=float32)

[](jax.numpy.exp.html "previous page")

previous

jax.numpy.exp

[](jax.numpy.expand_dims.html "next page")

next

jax.numpy.expand_dims

Contents

- [`exp2()`](#jax.numpy.exp2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
