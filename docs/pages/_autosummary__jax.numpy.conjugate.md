- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.conjugate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.conjugate.rst "Download source file")
-  .pdf

# jax.numpy.conjugate

## Contents

- [`conjugate()`](#jax.numpy.conjugate)

# jax.numpy.conjugate[\#](#jax-numpy-conjugate "Link to this heading")

jax.numpy.conjugate(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3326-L3354)[\#](#jax.numpy.conjugate "Link to this definition")  
Return element-wise complex-conjugate of the input.

JAX implementation of [`numpy.conjugate`](https://numpy.org/doc/stable/reference/generated/numpy.conjugate.html#numpy.conjugate "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – inpuat array or scalar.

Returns:  
An array containing the complex-conjugate of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.real()`](jax.numpy.real.html#jax.numpy.real "jax.numpy.real"): Returns the element-wise real part of the complex argument.

- [`jax.numpy.imag()`](jax.numpy.imag.html#jax.numpy.imag "jax.numpy.imag"): Returns the element-wise imaginary part of the complex argument.

Examples

    >>> jnp.conjugate(3)
    Array(3, dtype=int32, weak_type=True)
    >>> x = jnp.array([2-1j, 3+5j, 7])
    >>> jnp.conjugate(x)
    Array([2.+1.j, 3.-5.j, 7.-0.j], dtype=complex64)

[](jax.numpy.conj.html "previous page")

previous

jax.numpy.conj

[](jax.numpy.convolve.html "next page")

next

jax.numpy.convolve

Contents

- [`conjugate()`](#jax.numpy.conjugate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
