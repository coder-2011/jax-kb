- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.expm1

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.expm1.rst "Download source file")
-  .pdf

# jax.numpy.expm1

## Contents

- [`expm1()`](#jax.numpy.expm1)

# jax.numpy.expm1[\#](#jax-numpy-expm1 "Link to this heading")

jax.numpy.expm1(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L496-L539)[\#](#jax.numpy.expm1 "Link to this definition")  
Calculate `exp(x)-1` of each element of the input.

JAX implementation of [`numpy.expm1`](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html#numpy.expm1 "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing `exp(x)-1` of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.expm1` has much higher precision than the naive computation of `exp(x)-1` for small values of `x`.

See also

- [`jax.numpy.log1p()`](jax.numpy.log1p.html#jax.numpy.log1p "jax.numpy.log1p"): Calculates element-wise logarithm of one plus input.

- [`jax.numpy.exp()`](jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp"): Calculates element-wise exponential of the input.

- [`jax.numpy.exp2()`](jax.numpy.exp2.html#jax.numpy.exp2 "jax.numpy.exp2"): Calculates base-2 exponential of each element of the input.

Examples

    >>> x = jnp.array([2, -4, 3, -1])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.expm1(x))
    [ 6.39 -0.98 19.09 -0.63]
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.exp(x)-1)
    [ 6.39 -0.98 19.09 -0.63]

For values very close to 0, `jnp.expm1(x)` is much more accurate than `jnp.exp(x)-1`:

    >>> x1 = jnp.array([1e-4, 1e-6, 2e-10])
    >>> jnp.expm1(x1)
    Array([1.0000500e-04, 1.0000005e-06, 2.0000000e-10], dtype=float32)
    >>> jnp.exp(x1)-1
    Array([1.00016594e-04, 9.53674316e-07, 0.00000000e+00], dtype=float32)

[](jax.numpy.expand_dims.html "previous page")

previous

jax.numpy.expand_dims

[](jax.numpy.extract.html "next page")

next

jax.numpy.extract

Contents

- [`expm1()`](#jax.numpy.expm1)

By The JAX authors

© Copyright 2024, The JAX Authors.\
