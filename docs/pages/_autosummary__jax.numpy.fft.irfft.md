- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.irfft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.irfft.rst "Download source file")
-  .pdf

# jax.numpy.fft.irfft

## Contents

- [`irfft()`](#jax.numpy.fft.irfft)

# jax.numpy.fft.irfft[\#](#jax-numpy-fft-irfft "Link to this heading")

jax.numpy.fft.irfft(*a*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L639-L697)[\#](#jax.numpy.fft.irfft "Link to this definition")  
Compute a real-valued one-dimensional inverse discrete Fourier transform.

JAX implementation of [`numpy.fft.irfft()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html#numpy.fft.irfft "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – int. Specifies the dimension of the result along `axis`. If not specified, `n`` ``=`` ``2*(m-1)`, where `m` is the dimension of `a` along `axis`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=-1. Specifies the axis along which the transform is computed. If not specified, the transform is computed along axis -1.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
A real-valued array containing the one-dimensional inverse discrete Fourier transform of `a`, with a dimension of `n` along `axis`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.ifft()`](jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft"): Computes a one-dimensional inverse discrete Fourier transform.

- [`jax.numpy.fft.irfft()`](#jax.numpy.fft.irfft "jax.numpy.fft.irfft"): Computes a one-dimensional inverse discrete Fourier transform for real input.

- [`jax.numpy.fft.rfftn()`](jax.numpy.fft.rfftn.html#jax.numpy.fft.rfftn "jax.numpy.fft.rfftn"): Computes a multidimensional discrete Fourier transform for real input.

- [`jax.numpy.fft.irfftn()`](jax.numpy.fft.irfftn.html#jax.numpy.fft.irfftn "jax.numpy.fft.irfftn"): Computes a multidimensional inverse discrete Fourier transform for real input.

Examples

`jnp.fft.rfft` computes the transform along `axis`` ``-1` by default.

    >>> x = jnp.array([[1, 3, 5],
    ...                [2, 4, 6]])
    >>> jnp.fft.irfft(x)
    Array([[ 3., -1.,  0., -1.],
           [ 4., -1.,  0., -1.]], dtype=float32)

When `n=3`, dimension of the transform along axis -1 will be `3` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfft(x, n=3)
    Array([[ 2.33, -0.67, -0.67],
           [ 3.33, -0.67, -0.67]], dtype=float32)

When `n=4` and `axis=0`, dimension of the transform along `axis`` ``0` will be `4` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jnp.fft.irfft(x, n=4, axis=0)
    Array([[ 1.25,  2.75,  4.25],
           [ 0.25,  0.75,  1.25],
           [-0.75, -1.25, -1.75],
           [ 0.25,  0.75,  1.25]], dtype=float32)

[](jax.numpy.fft.ihfft.html "previous page")

previous

jax.numpy.fft.ihfft

[](jax.numpy.fft.irfft2.html "next page")

next

jax.numpy.fft.irfft2

Contents

- [`irfft()`](#jax.numpy.fft.irfft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
