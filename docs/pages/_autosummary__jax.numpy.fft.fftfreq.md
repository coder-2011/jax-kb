- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.fftfreq

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.fftfreq.rst "Download source file")
-  .pdf

# jax.numpy.fft.fftfreq

## Contents

- [`fftfreq()`](#jax.numpy.fft.fftfreq)

# jax.numpy.fft.fftfreq[\#](#jax-numpy-fft-fftfreq "Link to this heading")

jax.numpy.fft.fftfreq(*n*, *d=1.0*, *\**, *dtype=None*, *device=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L1157-L1198)[\#](#jax.numpy.fft.fftfreq "Link to this definition")  
Return sample frequencies for the discrete Fourier transform.

JAX implementation of [`numpy.fft.fftfreq()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq "(in NumPy v2.4)"). Returns frequencies appropriate for use with the outputs of [`fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft") and [`ifft()`](jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft").

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – length of the FFT window

- **d** (*ArrayLike*) – optional scalar sample spacing (default: 1.0)

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype of returned frequencies. If not specified, JAX’s default floating point dtype will be used.

- **device** (*xla_client.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – optional [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

Returns:  
Array of sample frequencies, length `n`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.rfftfreq()`](jax.numpy.fft.rfftfreq.html#jax.numpy.fft.rfftfreq "jax.numpy.fft.rfftfreq"): frequencies for use with [`rfft()`](jax.numpy.fft.rfft.html#jax.numpy.fft.rfft "jax.numpy.fft.rfft") and [`irfft()`](jax.numpy.fft.irfft.html#jax.numpy.fft.irfft "jax.numpy.fft.irfft").

[](jax.numpy.fft.fft2.html "previous page")

previous

jax.numpy.fft.fft2

[](jax.numpy.fft.fftn.html "next page")

next

jax.numpy.fft.fftn

Contents

- [`fftfreq()`](#jax.numpy.fft.fftfreq)

By The JAX authors

© Copyright 2024, The JAX Authors.\
