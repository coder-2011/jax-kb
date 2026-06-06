- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.fftshift

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.fftshift.rst "Download source file")
-  .pdf

# jax.numpy.fft.fftshift

## Contents

- [`fftshift()`](#jax.numpy.fft.fftshift)

# jax.numpy.fft.fftshift[\#](#jax-numpy-fft-fftshift "Link to this heading")

jax.numpy.fft.fftshift(*x*, *axes=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L1247-L1293)[\#](#jax.numpy.fft.fftshift "Link to this definition")  
Shift zero-frequency fft component to the center of the spectrum.

JAX implementation of [`numpy.fft.fftshift()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array array of frequencies.

- **axes** (*None* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – optional integer or sequence of integers specifying which axes to shift. If None (default), then shift all axes.

Returns:  
A shifted copy of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.ifftshift()`](jax.numpy.fft.ifftshift.html#jax.numpy.fft.ifftshift "jax.numpy.fft.ifftshift"): inverse of `fftshift`.

- [`jax.numpy.fft.fftfreq()`](jax.numpy.fft.fftfreq.html#jax.numpy.fft.fftfreq "jax.numpy.fft.fftfreq"): generate FFT frequencies.

Examples

Generate FFT frequencies with [`fftfreq()`](jax.numpy.fft.fftfreq.html#jax.numpy.fft.fftfreq "jax.numpy.fft.fftfreq"):

    >>> freq = jnp.fft.fftfreq(5)
    >>> freq
    Array([ 0. ,  0.2,  0.4, -0.4, -0.2], dtype=float32)

Use `fftshift` to shift the zero-frequency entry to the middle of the array:

    >>> shifted_freq = jnp.fft.fftshift(freq)
    >>> shifted_freq
    Array([-0.4, -0.2,  0. ,  0.2,  0.4], dtype=float32)

Unshift with [`ifftshift()`](jax.numpy.fft.ifftshift.html#jax.numpy.fft.ifftshift "jax.numpy.fft.ifftshift") to recover the original frequencies:

    >>> jnp.fft.ifftshift(shifted_freq)
    Array([ 0. ,  0.2,  0.4, -0.4, -0.2], dtype=float32)

[](jax.numpy.fft.fftn.html "previous page")

previous

jax.numpy.fft.fftn

[](jax.numpy.fft.hfft.html "next page")

next

jax.numpy.fft.hfft

Contents

- [`fftshift()`](#jax.numpy.fft.fftshift)

By The JAX authors

© Copyright 2024, The JAX Authors.\
