- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.rfft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.rfft.rst "Download source file")
-  .pdf

# jax.numpy.fft.rfft

## Contents

- [`rfft()`](#jax.numpy.fft.rfft)

# jax.numpy.fft.rfft[\#](#jax-numpy-fft-rfft "Link to this heading")

jax.numpy.fft.rfft(*a*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L578-L637)[\#](#jax.numpy.fft.rfft "Link to this definition")  
Compute a one-dimensional discrete Fourier transform of a real-valued array.

JAX implementation of [`numpy.fft.rfft()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html#numpy.fft.rfft "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – real-valued input array.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – int. Specifies the effective dimension of the input along `axis`. If not specified, it will default to the dimension of input along `axis`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=-1. Specifies the axis along which the transform is computed. If not specified, the transform is computed along axis -1.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
An array containing the one-dimensional discrete Fourier transform of `a`. The dimension of the array along `axis` is `(n/2)+1`, if `n` is even and `(n+1)/2`, if `n` is odd.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.irfft()`](jax.numpy.fft.irfft.html#jax.numpy.fft.irfft "jax.numpy.fft.irfft"): Computes a one-dimensional inverse discrete Fourier transform for real input.

- [`jax.numpy.fft.rfftn()`](jax.numpy.fft.rfftn.html#jax.numpy.fft.rfftn "jax.numpy.fft.rfftn"): Computes a multidimensional discrete Fourier transform for real input.

- [`jax.numpy.fft.irfftn()`](jax.numpy.fft.irfftn.html#jax.numpy.fft.irfftn "jax.numpy.fft.irfftn"): Computes a multidimensional inverse discrete Fourier transform for real input.

Examples

`jnp.fft.rfft` computes the transform along `axis`` ``-1` by default.

    >>> x = jnp.array([[1, 3, 5],
    ...                [2, 4, 6]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.rfft(x)
    Array([[ 9.+0.j  , -3.+1.73j],
           [12.+0.j  , -3.+1.73j]], dtype=complex64)

When `n=5`, dimension of the transform along axis -1 will be `(5+1)/2`` ``=3` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.rfft(x, n=5)
    Array([[ 9.  +0.j  , -2.12-5.79j,  0.12+2.99j],
           [12.  +0.j  , -1.62-7.33j,  0.62+3.36j]], dtype=complex64)

When `n=4` and `axis=0`, dimension of the transform along `axis`` ``0` will be `(4/2)+1`` ``=3` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.rfft(x, n=4, axis=0)
    Array([[ 3.+0.j,  7.+0.j, 11.+0.j],
           [ 1.-2.j,  3.-4.j,  5.-6.j],
           [-1.+0.j, -1.+0.j, -1.+0.j]], dtype=complex64)

[](jax.numpy.fft.irfftn.html "previous page")

previous

jax.numpy.fft.irfftn

[](jax.numpy.fft.rfft2.html "next page")

next

jax.numpy.fft.rfft2

Contents

- [`rfft()`](#jax.numpy.fft.rfft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
