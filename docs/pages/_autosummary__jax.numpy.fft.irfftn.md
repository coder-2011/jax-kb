- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.irfftn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.irfftn.rst "Download source file")
-  .pdf

# jax.numpy.fft.irfftn

## Contents

- [`irfftn()`](#jax.numpy.fft.irfftn)

# jax.numpy.fft.irfftn[\#](#jax-numpy-fft-irfftn "Link to this heading")

jax.numpy.fft.irfftn(*a*, *s=None*, *axes=None*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L365-L440)[\#](#jax.numpy.fft.irfftn "Link to this definition")  
Compute a real-valued multidimensional inverse discrete Fourier transform.

JAX implementation of [`numpy.fft.irfftn()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html#numpy.fft.irfftn "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **s** (*Shape* *\|* *None*) – optional sequence of integers. Specifies the size of the output in each specified axis. If not specified, the dimension of output along axis `axes[-1]` is `2*(m-1)`, `m` is the size of input along axis `axes[-1]` and the dimension along other axes will be the same as that of input.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of integers, default=None. Specifies the axes along which the transform is computed. If not specified, the transform is computed along the last `len(s)` axes. If neither `axes` nor `s` is specified, the transform is computed along all the axes.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string, default=”backward”. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
A real-valued array containing the multidimensional inverse discrete Fourier transform of `a` with size `s` along specified `axes`, and the same as the input along other axes.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.rfftn()`](jax.numpy.fft.rfftn.html#jax.numpy.fft.rfftn "jax.numpy.fft.rfftn"): Computes a multidimensional discrete Fourier transform of a real-valued array.

- [`jax.numpy.fft.irfft()`](jax.numpy.fft.irfft.html#jax.numpy.fft.irfft "jax.numpy.fft.irfft"): Computes a real-valued one-dimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.irfft2()`](jax.numpy.fft.irfft2.html#jax.numpy.fft.irfft2 "jax.numpy.fft.irfft2"): Computes a real-valued two-dimensional inverse discrete Fourier transform.

Examples

`jnp.fft.irfftn` computes the transform along all the axes by default.

    >>> x = jnp.array([[[1, 3, 5],
    ...                 [2, 4, 6]],
    ...                [[7, 9, 11],
    ...                 [8, 10, 12]]])
    >>> jnp.fft.irfftn(x)
    Array([[[ 6.5, -1. ,  0. , -1. ],
            [-0.5,  0. ,  0. ,  0. ]],

           [[-3. ,  0. ,  0. ,  0. ],
            [ 0. ,  0. ,  0. ,  0. ]]], dtype=float32)

When `s=[3,`` ``4]`, size of the transform along `axes`` ``(-2,`` ``-1)` will be `(3,`` ``4)` and size along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfftn(x, s=[3, 4])
    Array([[[ 2.33, -0.67,  0.  , -0.67],
            [ 0.33, -0.74,  0.  ,  0.41],
            [ 0.33,  0.41,  0.  , -0.74]],

           [[ 6.33, -0.67,  0.  , -0.67],
            [ 1.33, -1.61,  0.  ,  1.28],
            [ 1.33,  1.28,  0.  , -1.61]]], dtype=float32)

When `s=[3]` and `axes=[0]`, size of the transform along `axes`` ``0` will be `3` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfftn(x, s=[3], axes=[0])
    Array([[[ 5.,  7.,  9.],
            [ 6.,  8., 10.]],

           [[-2., -2., -2.],
            [-2., -2., -2.]],

           [[-2., -2., -2.],
            [-2., -2., -2.]]], dtype=float32)

[](jax.numpy.fft.irfft2.html "previous page")

previous

jax.numpy.fft.irfft2

[](jax.numpy.fft.rfft.html "next page")

next

jax.numpy.fft.rfft

Contents

- [`irfftn()`](#jax.numpy.fft.irfftn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
