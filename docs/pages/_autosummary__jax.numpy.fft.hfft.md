- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.hfft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.hfft.rst "Download source file")
-  .pdf

# jax.numpy.fft.hfft

## Contents

- [`hfft()`](#jax.numpy.fft.hfft)

# jax.numpy.fft.hfft[\#](#jax-numpy-fft-hfft "Link to this heading")

jax.numpy.fft.hfft(*a*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L699-L787)[\#](#jax.numpy.fft.hfft "Link to this definition")  
Compute a 1-D FFT of an array whose spectrum has Hermitian symmetry.

JAX implementation of [`numpy.fft.hfft()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html#numpy.fft.hfft "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional, int. Specifies the dimension of the result along `axis`. If not specified, `n`` ``=`` ``2*(m-1)`, where `m` is the dimension of `a` along `axis`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, int, default=-1. Specifies the axis along which the transform is computed. If not specified, the transform is computed along axis -1.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional, string. The normalization mode. “backward”, “ortho” and “forward” are supported. Default is “backward”.

Returns:  
A real-valued array containing the one-dimensional discrete Fourier transform of `a` by exploiting its inherent Hermitian-symmetry, having a dimension of `n` along `axis`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.ihfft()`](jax.numpy.fft.ihfft.html#jax.numpy.fft.ihfft "jax.numpy.fft.ihfft"): Computes a one-dimensional inverse FFT of an array whose spectrum has Hermitian symmetry.

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.rfft()`](jax.numpy.fft.rfft.html#jax.numpy.fft.rfft "jax.numpy.fft.rfft"): Computes a one-dimensional discrete Fourier transform of a real-valued input.

Examples

    >>> x = jnp.array([[1, 3, 5, 7],
    ...                [2, 4, 6, 8]])
    >>> jnp.fft.hfft(x)
    Array([[24., -8.,  0., -2.,  0., -8.],
           [30., -8.,  0., -2.,  0., -8.]], dtype=float32)

This value is equal to the real component of the discrete Fourier transform of the following array `x1` computed using `jnp.fft.fft`.

    >>> x1 = jnp.array([[1, 3, 5, 7, 5, 3],
    ...                 [2, 4, 6, 8, 6, 4]])
    >>> jnp.fft.fft(x1)
    Array([[24.+0.j, -8.+0.j,  0.+0.j, -2.+0.j,  0.+0.j, -8.+0.j],
           [30.+0.j, -8.+0.j,  0.+0.j, -2.+0.j,  0.+0.j, -8.+0.j]],      dtype=complex64)
    >>> jnp.allclose(jnp.fft.hfft(x), jnp.fft.fft(x1))
    Array(True, dtype=bool)

To obtain an odd-length output from `jnp.fft.hfft`, `n` must be specified with an odd value, as the default behavior produces an even-length result along the specified `axis`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.hfft(x, n=5))
    [[17.   -5.24 -0.76 -0.76 -5.24]
     [22.   -5.24 -0.76 -0.76 -5.24]]

When `n=3` and `axis=0`, dimension of the transform along `axis`` ``0` will be `3` and dimension along other axes will be same as that of input.

    >>> jnp.fft.hfft(x, n=3, axis=0)
    Array([[ 5., 11., 17., 23.],
           [-1., -1., -1., -1.],
           [-1., -1., -1., -1.]], dtype=float32)

`x` can be reconstructed (but of complex datatype) using `jnp.fft.ihfft` from the result of `jnp.fft.hfft`, only when `n` is specified as `2*(m-1)` if m is even or `2*m-1` if `m` is odd, where `m` is the dimension of input along `axis`.

    >>> jnp.fft.ihfft(jnp.fft.hfft(x, 2*(x.shape[-1]-1)))
    Array([[1.+0.j, 3.+0.j, 5.+0.j, 7.+0.j],
           [2.+0.j, 4.+0.j, 6.+0.j, 8.+0.j]], dtype=complex64)
    >>> jnp.allclose(x, jnp.fft.ihfft(jnp.fft.hfft(x, 2*(x.shape[-1]-1))))
    Array(True, dtype=bool)

For complex-valued inputs:

    >>> x2 = jnp.array([[1+2j, 3-4j, 5+6j],
    ...                 [2-3j, 4+5j, 6-7j]])
    >>> jnp.fft.hfft(x2)
    Array([[ 12., -12.,   0.,   4.],
           [ 16.,   6.,   0., -14.]], dtype=float32)

[](jax.numpy.fft.fftshift.html "previous page")

previous

jax.numpy.fft.fftshift

[](jax.numpy.fft.ifft.html "next page")

next

jax.numpy.fft.ifft

Contents

- [`hfft()`](#jax.numpy.fft.hfft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
