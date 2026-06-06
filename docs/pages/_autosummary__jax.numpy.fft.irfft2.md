- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.irfft2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.irfft2.rst "Download source file")
-  .pdf

# jax.numpy.fft.irfft2

## Contents

- [`irfft2()`](#jax.numpy.fft.irfft2)

# jax.numpy.fft.irfft2[\#](#jax-numpy-fft-irfft2 "Link to this heading")

jax.numpy.fft.irfft2(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L1082-L1155)[\#](#jax.numpy.fft.irfft2 "Link to this definition")  
Compute a real-valued two-dimensional inverse discrete Fourier transform.

JAX implementation of [`numpy.fft.irfft2()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html#numpy.fft.irfft2 "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array. Must have `a.ndim`` ``>=`` ``2`.

- **s** (*Shape* *\|* *None*) – optional length-2 sequence of integers. Specifies the size of the output in each specified axis. If not specified, the dimension of output along axis `axes[1]` is `2*(m-1)`, `m` is the size of input along axis `axes[1]` and the dimension along other axes will be the same as that of input.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – optional length-2 sequence of integers, default=(-2,-1). Specifies the axes along which the transform is computed.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string, default=”backward”. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
A real-valued array containing the two-dimensional inverse discrete Fourier transform of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.rfft2()`](jax.numpy.fft.rfft2.html#jax.numpy.fft.rfft2 "jax.numpy.fft.rfft2"): Computes a two-dimensional discrete Fourier transform of a real-valued array.

- [`jax.numpy.fft.irfft()`](jax.numpy.fft.irfft.html#jax.numpy.fft.irfft "jax.numpy.fft.irfft"): Computes a real-valued one-dimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.irfftn()`](jax.numpy.fft.irfftn.html#jax.numpy.fft.irfftn "jax.numpy.fft.irfftn"): Computes a real-valued multidimensional inverse discrete Fourier transform.

Examples

`jnp.fft.irfft2` computes the transform along the last two axes by default.

    >>> x = jnp.array([[[1, 3, 5],
    ...                 [2, 4, 6]],
    ...                [[7, 9, 11],
    ...                 [8, 10, 12]]])
    >>> jnp.fft.irfft2(x)
    Array([[[ 3.5, -1. ,  0. , -1. ],
            [-0.5,  0. ,  0. ,  0. ]],

           [[ 9.5, -1. ,  0. , -1. ],
            [-0.5,  0. ,  0. ,  0. ]]], dtype=float32)

When `s=[3,`` ``3]`, dimension of the transform along `axes`` ``(-2,`` ``-1)` will be `(3,`` ``3)` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfft2(x, s=[3, 3])
    Array([[[ 1.89, -0.44, -0.44],
            [ 0.22, -0.78,  0.56],
            [ 0.22,  0.56, -0.78]],

           [[ 5.89, -0.44, -0.44],
            [ 1.22, -1.78,  1.56],
            [ 1.22,  1.56, -1.78]]], dtype=float32)

When `s=[2,`` ``3]` and `axes=(0,`` ``1)`, shape of the transform along `axes`` ``(0,`` ``1)` will be `(2,`` ``3)` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfft2(x, s=[2, 3], axes=(0, 1))
    Array([[[ 4.67,  6.67,  8.67],
            [-0.33, -0.33, -0.33],
            [-0.33, -0.33, -0.33]],

           [[-3.  , -3.  , -3.  ],
            [ 0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ]]], dtype=float32)

[](jax.numpy.fft.irfft.html "previous page")

previous

jax.numpy.fft.irfft

[](jax.numpy.fft.irfftn.html "next page")

next

jax.numpy.fft.irfftn

Contents

- [`irfft2()`](#jax.numpy.fft.irfft2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
