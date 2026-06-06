- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.fft2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.fft2.rst "Download source file")
-  .pdf

# jax.numpy.fft.fft2

## Contents

- [`fft2()`](#jax.numpy.fft.fft2)

# jax.numpy.fft.fft2[\#](#jax-numpy-fft-fft2 "Link to this heading")

jax.numpy.fft.fft2(*a*, *s=None*, *axes=(-2, -1)*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L852-L929)[\#](#jax.numpy.fft.fft2 "Link to this definition")  
Compute a two-dimensional discrete Fourier transform along given axes.

JAX implementation of [`numpy.fft.fft2()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2 "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array. Must have `a.ndim`` ``>=`` ``2`.

- **s** (*Shape* *\|* *None*) – optional length-2 sequence of integers. Specifies the size of the output along each specified axis. If not specified, it will default to the size of `a` along the specified `axes`.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – optional length-2 sequence of integers, default=(-2,-1). Specifies the axes along which the transform is computed.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string, default=”backward”. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
An array containing the two-dimensional discrete Fourier transform of `a` along given `axes`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.fftn()`](jax.numpy.fft.fftn.html#jax.numpy.fft.fftn "jax.numpy.fft.fftn"): Computes a multidimensional discrete Fourier transform.

- [`jax.numpy.fft.ifft2()`](jax.numpy.fft.ifft2.html#jax.numpy.fft.ifft2 "jax.numpy.fft.ifft2"): Computes a two-dimensional inverse discrete Fourier transform.

Examples

`jnp.fft.fft2` computes the transform along the last two axes by default.

    >>> x = jnp.array([[[1, 3],
    ...                 [2, 4]],
    ...                [[5, 7],
    ...                 [6, 8]]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fft2(x)
    Array([[[10.+0.j, -4.+0.j],
            [-2.+0.j,  0.+0.j]],

           [[26.+0.j, -4.+0.j],
            [-2.+0.j,  0.+0.j]]], dtype=complex64)

When `s=[2,`` ``3]`, dimension of the transform along `axes`` ``(-2,`` ``-1)` will be `(2,`` ``3)` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fft2(x, s=[2, 3])
    Array([[[10.  +0.j  , -0.5 -6.06j, -0.5 +6.06j],
            [-2.  +0.j  , -0.5 +0.87j, -0.5 -0.87j]],

           [[26.  +0.j  ,  3.5-12.99j,  3.5+12.99j],
            [-2.  +0.j  , -0.5 +0.87j, -0.5 -0.87j]]], dtype=complex64)

When `s=[2,`` ``3]` and `axes=(0,`` ``1)`, shape of the transform along `axes`` ``(0,`` ``1)` will be `(2,`` ``3)` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fft2(x, s=[2, 3], axes=(0, 1))
    Array([[[14. +0.j  , 22. +0.j  ],
            [ 2. -6.93j,  4.-10.39j],
            [ 2. +6.93j,  4.+10.39j]],

           [[-8. +0.j  , -8. +0.j  ],
            [-2. +3.46j, -2. +3.46j],
            [-2. -3.46j, -2. -3.46j]]], dtype=complex64)

`jnp.fft.ifft2` can be used to reconstruct `x` from the result of `jnp.fft.fft2`.

    >>> x_fft2 = jnp.fft.fft2(x)
    >>> jnp.allclose(x, jnp.fft.ifft2(x_fft2))
    Array(True, dtype=bool)

[](jax.numpy.fft.fft.html "previous page")

previous

jax.numpy.fft.fft

[](jax.numpy.fft.fftfreq.html "next page")

next

jax.numpy.fft.fftfreq

Contents

- [`fft2()`](#jax.numpy.fft.fft2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
