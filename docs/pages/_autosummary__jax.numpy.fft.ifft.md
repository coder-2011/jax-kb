- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.ifft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.ifft.rst "Download source file")
-  .pdf

# jax.numpy.fft.ifft

## Contents

- [`ifft()`](#jax.numpy.fft.ifft)

# jax.numpy.fft.ifft[\#](#jax-numpy-fft-ifft "Link to this heading")

jax.numpy.fft.ifft(*a*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L522-L576)[\#](#jax.numpy.fft.ifft "Link to this definition")  
Compute a one-dimensional inverse discrete Fourier transform.

JAX implementation of [`numpy.fft.ifft()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft "(in NumPy v2.4)").

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

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.fftn()`](jax.numpy.fft.fftn.html#jax.numpy.fft.fftn "jax.numpy.fft.fftn"): Computes a multidimensional discrete Fourier transform.

- [`jax.numpy.fft.ifftn()`](jax.numpy.fft.ifftn.html#jax.numpy.fft.ifftn "jax.numpy.fft.ifftn"): Computes a multidimensional inverse of discrete Fourier transform.

Examples

`jnp.fft.ifft` computes the transform along `axis`` ``-1` by default.

    >>> x = jnp.array([[3, 1, 4, 6],
    ...                [2, 5, 7, 1]])
    >>> jnp.fft.ifft(x)
    Array([[ 3.5 +0.j  , -0.25-1.25j,  0.  +0.j  , -0.25+1.25j],
          [ 3.75+0.j  , -1.25+1.j  ,  0.75+0.j  , -1.25-1.j  ]],      dtype=complex64)

When `n=5`, dimension of the transform along axis -1 will be `5` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifft(x, n=5))
    [[ 2.8 +0.j   -0.96-0.04j  1.06+0.5j   1.06-0.5j  -0.96+0.04j]
     [ 3.  +0.j   -0.59+1.66j  0.09-0.55j  0.09+0.55j -0.59-1.66j]]

When `n=3` and `axis=0`, dimension of the transform along `axis`` ``0` will be `3` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifft(x, n=3, axis=0))
    [[ 1.67+0.j    2.  +0.j    3.67+0.j    2.33+0.j  ]
     [ 0.67+0.58j -0.5 +1.44j  0.17+2.02j  1.83+0.29j]
     [ 0.67-0.58j -0.5 -1.44j  0.17-2.02j  1.83-0.29j]]

[](jax.numpy.fft.hfft.html "previous page")

previous

jax.numpy.fft.hfft

[](jax.numpy.fft.ifft2.html "next page")

next

jax.numpy.fft.ifft2

Contents

- [`ifft()`](#jax.numpy.fft.ifft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
