- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.ihfft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.ihfft.rst "Download source file")
-  .pdf

# jax.numpy.fft.ihfft

## Contents

- [`ihfft()`](#jax.numpy.fft.ihfft)

# jax.numpy.fft.ihfft[\#](#jax-numpy-fft-ihfft "Link to this heading")

jax.numpy.fft.ihfft(*a*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L789-L838)[\#](#jax.numpy.fft.ihfft "Link to this definition")  
Compute a 1-D inverse FFT of an array whose spectrum has Hermitian-symmetry.

JAX implementation of [`numpy.fft.ihfft()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html#numpy.fft.ihfft "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional, int. Specifies the effective dimension of the input along `axis`. If not specified, it will default to the dimension of input along `axis`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, int, default=-1. Specifies the axis along which the transform is computed. If not specified, the transform is computed along axis -1.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional, string. The normalization mode. “backward”, “ortho” and “forward” are supported. Default is “backward”.

Returns:  
An array containing one-dimensional discrete Fourier transform of `a` by exploiting its inherent Hermitian symmetry. The dimension of the array along `axis` is `(n/2)+1`, if `n` is even and `(n+1)/2`, if `n` is odd.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.hfft()`](jax.numpy.fft.hfft.html#jax.numpy.fft.hfft "jax.numpy.fft.hfft"): Computes a one-dimensional FFT of an array whose spectrum has Hermitian symmetry.

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.rfft()`](jax.numpy.fft.rfft.html#jax.numpy.fft.rfft "jax.numpy.fft.rfft"): Computes a one-dimensional discrete Fourier transform of a real-valued input.

Examples

    >>> x = jnp.array([[1, 3, 5, 7],
    ...                [2, 4, 6, 8]])
    >>> jnp.fft.ihfft(x)
    Array([[ 4.+0.j, -1.-1.j, -1.-0.j],
           [ 5.+0.j, -1.-1.j, -1.-0.j]], dtype=complex64)

When `n=4` and `axis=0`, dimension of the transform along `axis`` ``0` will be `(4/2)+1`` ``=3` and dimension along other axes will be same as that of input.

    >>> jnp.fft.ihfft(x, n=4, axis=0)
    Array([[ 0.75+0.j ,  1.75+0.j ,  2.75+0.j ,  3.75+0.j ],
           [ 0.25+0.5j,  0.75+1.j ,  1.25+1.5j,  1.75+2.j ],
           [-0.25-0.j , -0.25-0.j , -0.25-0.j , -0.25-0.j ]], dtype=complex64)

[](jax.numpy.fft.ifftshift.html "previous page")

previous

jax.numpy.fft.ifftshift

[](jax.numpy.fft.irfft.html "next page")

next

jax.numpy.fft.irfft

Contents

- [`ihfft()`](#jax.numpy.fft.ihfft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
