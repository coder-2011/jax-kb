- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.fft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.fft.rst "Download source file")
-  .pdf

# jax.numpy.fft.fft

## Contents

- [`fft()`](#jax.numpy.fft.fft)

# jax.numpy.fft.fft[\#](#jax-numpy-fft-fft "Link to this heading")

jax.numpy.fft.fft(*a*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L459-L520)[\#](#jax.numpy.fft.fft "Link to this definition")  
Compute a one-dimensional discrete Fourier transform along a given axis.

JAX implementation of [`numpy.fft.fft()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – int. Specifies the dimension of the result along `axis`. If not specified, it will default to the dimension of `a` along `axis`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=-1. Specifies the axis along which the transform is computed. If not specified, the transform is computed along axis -1.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
An array containing the one-dimensional discrete Fourier transform of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.ifft()`](jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft"): Computes a one-dimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.fftn()`](jax.numpy.fft.fftn.html#jax.numpy.fft.fftn "jax.numpy.fft.fftn"): Computes a multidimensional discrete Fourier transform.

- [`jax.numpy.fft.ifftn()`](jax.numpy.fft.ifftn.html#jax.numpy.fft.ifftn "jax.numpy.fft.ifftn"): Computes a multidimensional inverse discrete Fourier transform.

Examples

`jnp.fft.fft` computes the transform along `axis`` ``-1` by default.

    >>> x = jnp.array([[1, 2, 4, 7],
    ...                [5, 3, 1, 9]])
    >>> jnp.fft.fft(x)
    Array([[14.+0.j, -3.+5.j, -4.+0.j, -3.-5.j],
           [18.+0.j,  4.+6.j, -6.+0.j,  4.-6.j]], dtype=complex64)

When `n=3`, dimension of the transform along axis -1 will be `3` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.fft(x, n=3))
    [[ 7.+0.j   -2.+1.73j -2.-1.73j]
     [ 9.+0.j    3.-1.73j  3.+1.73j]]

When `n=3` and `axis=0`, dimension of the transform along `axis`` ``0` will be `3` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.fft(x, n=3, axis=0))
    [[ 6. +0.j    5. +0.j    5. +0.j   16. +0.j  ]
     [-1.5-4.33j  0.5-2.6j   3.5-0.87j  2.5-7.79j]
     [-1.5+4.33j  0.5+2.6j   3.5+0.87j  2.5+7.79j]]

`jnp.fft.ifft` can be used to reconstruct `x` from the result of `jnp.fft.fft`.

    >>> x_fft = jnp.fft.fft(x)
    >>> jnp.allclose(x, jnp.fft.ifft(x_fft))
    Array(True, dtype=bool)

[](jax.numpy.zeros_like.html "previous page")

previous

jax.numpy.zeros_like

[](jax.numpy.fft.fft2.html "next page")

next

jax.numpy.fft.fft2

Contents

- [`fft()`](#jax.numpy.fft.fft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
