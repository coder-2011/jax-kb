- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.ifft2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.ifft2.rst "Download source file")
-  .pdf

# jax.numpy.fft.ifft2

## Contents

- [`ifft2()`](#jax.numpy.fft.ifft2)

# jax.numpy.fft.ifft2[\#](#jax-numpy-fft-ifft2 "Link to this heading")

jax.numpy.fft.ifft2(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L931-L1001)[\#](#jax.numpy.fft.ifft2 "Link to this definition")  
Compute a two-dimensional inverse discrete Fourier transform.

JAX implementation of [`numpy.fft.ifft2()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html#numpy.fft.ifft2 "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array. Must have `a.ndim`` ``>=`` ``2`.

- **s** (*Shape* *\|* *None*) – optional length-2 sequence of integers. Specifies the size of the output in each specified axis. If not specified, it will default to the size of `a` along the specified `axes`.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – optional length-2 sequence of integers, default=(-2,-1). Specifies the axes along which the transform is computed.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string, default=”backward”. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
An array containing the two-dimensional inverse discrete Fourier transform of `a` along given `axes`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.ifft()`](jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft"): Computes a one-dimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.ifftn()`](jax.numpy.fft.ifftn.html#jax.numpy.fft.ifftn "jax.numpy.fft.ifftn"): Computes a multidimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.fft2()`](jax.numpy.fft.fft2.html#jax.numpy.fft.fft2 "jax.numpy.fft.fft2"): Computes a two-dimensional discrete Fourier transform.

Examples

`jnp.fft.ifft2` computes the transform along the last two axes by default.

    >>> x = jnp.array([[[1, 3],
    ...                 [2, 4]],
    ...                [[5, 7],
    ...                 [6, 8]]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.ifft2(x)
    Array([[[ 2.5+0.j, -1. +0.j],
            [-0.5+0.j,  0. +0.j]],

           [[ 6.5+0.j, -1. +0.j],
            [-0.5+0.j,  0. +0.j]]], dtype=complex64)

When `s=[2,`` ``3]`, dimension of the transform along `axes`` ``(-2,`` ``-1)` will be `(2,`` ``3)` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.ifft2(x, s=[2, 3])
    Array([[[ 1.67+0.j  , -0.08+1.01j, -0.08-1.01j],
            [-0.33+0.j  , -0.08-0.14j, -0.08+0.14j]],

           [[ 4.33+0.j  ,  0.58+2.17j,  0.58-2.17j],
            [-0.33+0.j  , -0.08-0.14j, -0.08+0.14j]]], dtype=complex64)

When `s=[2,`` ``3]` and `axes=(0,`` ``1)`, shape of the transform along `axes`` ``(0,`` ``1)` will be `(2,`` ``3)` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.ifft2(x, s=[2, 3], axes=(0, 1))
    Array([[[ 2.33+0.j  ,  3.67+0.j  ],
            [ 0.33+1.15j,  0.67+1.73j],
            [ 0.33-1.15j,  0.67-1.73j]],

           [[-1.33+0.j  , -1.33+0.j  ],
            [-0.33-0.58j, -0.33-0.58j],
            [-0.33+0.58j, -0.33+0.58j]]], dtype=complex64)

[](jax.numpy.fft.ifft.html "previous page")

previous

jax.numpy.fft.ifft

[](jax.numpy.fft.ifftn.html "next page")

next

jax.numpy.fft.ifftn

Contents

- [`ifft2()`](#jax.numpy.fft.ifft2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
