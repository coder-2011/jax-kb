- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.real

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.real.rst "Download source file")
-  .pdf

# jax.numpy.real

## Contents

- [`real()`](#jax.numpy.real)

# jax.numpy.real[\#](#jax-numpy-real "Link to this heading")

jax.numpy.real(*val*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3394-L3424)[\#](#jax.numpy.real "Link to this definition")  
Return element-wise real part of the complex argument.

JAX implementation of [`numpy.real`](https://numpy.org/doc/stable/reference/generated/numpy.real.html#numpy.real "(in NumPy v2.4)").

Parameters:  
**val** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the real part of the elements of `val`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.conjugate()`](jax.numpy.conjugate.html#jax.numpy.conjugate "jax.numpy.conjugate") and [`jax.numpy.conj()`](jax.numpy.conj.html#jax.numpy.conj "jax.numpy.conj"): Returns the element-wise complex-conjugate of the input.

- [`jax.numpy.imag()`](jax.numpy.imag.html#jax.numpy.imag "jax.numpy.imag"): Returns the element-wise imaginary part of the complex argument.

Examples

    >>> jnp.real(5)
    Array(5, dtype=int32, weak_type=True)
    >>> jnp.real(2j)
    Array(0., dtype=float32, weak_type=True)
    >>> x = jnp.array([3-2j, 4+7j, -2j])
    >>> jnp.real(x)
    Array([ 3.,  4., -0.], dtype=float32)

[](jax.numpy.ravel_multi_index.html "previous page")

previous

jax.numpy.ravel_multi_index

[](jax.numpy.reciprocal.html "next page")

next

jax.numpy.reciprocal

Contents

- [`real()`](#jax.numpy.real)

By The JAX authors

© Copyright 2024, The JAX Authors.\
