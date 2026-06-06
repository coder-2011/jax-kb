- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.imag

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.imag.rst "Download source file")
-  .pdf

# jax.numpy.imag

## Contents

- [`imag()`](#jax.numpy.imag)

# jax.numpy.imag[\#](#jax-numpy-imag "Link to this heading")

jax.numpy.imag(*val*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3362-L3392)[\#](#jax.numpy.imag "Link to this definition")  
Return element-wise imaginary of part of the complex argument.

JAX implementation of [`numpy.imag`](https://numpy.org/doc/stable/reference/generated/numpy.imag.html#numpy.imag "(in NumPy v2.4)").

Parameters:  
**val** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the imaginary part of the elements of `val`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.conjugate()`](jax.numpy.conjugate.html#jax.numpy.conjugate "jax.numpy.conjugate") and [`jax.numpy.conj()`](jax.numpy.conj.html#jax.numpy.conj "jax.numpy.conj"): Returns the element-wise complex-conjugate of the input.

- [`jax.numpy.real()`](jax.numpy.real.html#jax.numpy.real "jax.numpy.real"): Returns the element-wise real part of the complex argument.

Examples

    >>> jnp.imag(4)
    Array(0, dtype=int32, weak_type=True)
    >>> jnp.imag(5j)
    Array(5., dtype=float32, weak_type=True)
    >>> x = jnp.array([2+3j, 5-1j, -3])
    >>> jnp.imag(x)
    Array([ 3., -1.,  0.], dtype=float32)

[](jax.numpy.iinfo.html "previous page")

previous

jax.numpy.iinfo

[](jax.numpy.index_exp.html "next page")

next

jax.numpy.index_exp

Contents

- [`imag()`](#jax.numpy.imag)

By The JAX authors

© Copyright 2024, The JAX Authors.\
