- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.fftn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.fftn.rst "Download source file")
-  .pdf

# jax.numpy.fft.fftn

## Contents

- [`fftn()`](#jax.numpy.fft.fftn)

# jax.numpy.fft.fftn[\#](#jax-numpy-fft-fftn "Link to this heading")

jax.numpy.fft.fftn(*a*, *s=None*, *axes=None*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L132-L204)[\#](#jax.numpy.fft.fftn "Link to this definition")  
Compute a multidimensional discrete Fourier transform along given axes.

JAX implementation of [`numpy.fft.fftn()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html#numpy.fft.fftn "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **s** (*Shape* *\|* *None*) – sequence of integers. Specifies the shape of the result. If not specified, it will default to the shape of `a` along the specified `axes`.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – sequence of integers, default=None. Specifies the axes along which the transform is computed.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
An array containing the multidimensional discrete Fourier transform of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.ifft()`](jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft"): Computes a one-dimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.ifftn()`](jax.numpy.fft.ifftn.html#jax.numpy.fft.ifftn "jax.numpy.fft.ifftn"): Computes a multidimensional inverse discrete Fourier transform.

Examples

`jnp.fft.fftn` computes the transform along all the axes by default when `axes` argument is `None`.

    >>> x = jnp.array([[1, 2, 5, 6],
    ...                [4, 1, 3, 7],
    ...                [5, 9, 2, 1]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.fftn(x)
    Array([[ 46.  +0.j  ,   0.  +2.j  ,  -6.  +0.j  ,   0.  -2.j  ],
           [ -2.  +1.73j,   6.12+6.73j,   0.  -1.73j, -18.12-3.27j],
           [ -2.  -1.73j, -18.12+3.27j,   0.  +1.73j,   6.12-6.73j]],      dtype=complex64)

When `s=[2]`, dimension of the transform along `axis`` ``-1` will be `2` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.numpy.fft.fftn(x, s=[2]))
    [[ 3.+0.j -1.+0.j]
     [ 5.+0.j  3.+0.j]
     [14.+0.j -4.+0.j]]

When `s=[2]` and `axes=[0]`, dimension of the transform along `axis`` ``0` will be `2` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.numpy.fft.fftn(x, s=[2], axes=[0]))
    [[ 5.+0.j  3.+0.j  8.+0.j 13.+0.j]
     [-3.+0.j  1.+0.j  2.+0.j -1.+0.j]]

When `s=[2,`` ``3]`, shape of the transform will be `(2,`` ``3)`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.numpy.fft.fftn(x, s=[2, 3]))
    [[16. +0.j   -0.5+4.33j -0.5-4.33j]
     [ 0. +0.j   -4.5+0.87j -4.5-0.87j]]

`jnp.fft.ifftn` can be used to reconstruct `x` from the result of `jnp.fft.fftn`.

    >>> x_fftn = jnp.fft.fftn(x)
    >>> jnp.allclose(x, jnp.fft.ifftn(x_fftn))
    Array(True, dtype=bool)

[](jax.numpy.fft.fftfreq.html "previous page")

previous

jax.numpy.fft.fftfreq

[](jax.numpy.fft.fftshift.html "next page")

next

jax.numpy.fft.fftshift

Contents

- [`fftn()`](#jax.numpy.fft.fftn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
