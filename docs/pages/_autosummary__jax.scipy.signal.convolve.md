- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.convolve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.convolve.rst "Download source file")
-  .pdf

# jax.scipy.signal.convolve

## Contents

- [`convolve()`](#jax.scipy.signal.convolve)

# jax.scipy.signal.convolve[\#](#jax-scipy-signal-convolve "Link to this heading")

jax.scipy.signal.convolve(*in1*, *in2*, *mode='full'*, *method='auto'*, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L198-L261)[\#](#jax.scipy.signal.convolve "Link to this definition")  
Convolution of two N-dimensional arrays.

JAX implementation of [`scipy.signal.convolve()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve "(in SciPy v1.19.0.dev)").

Parameters:  
- **in1** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – left-hand input to the convolution.

- **in2** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – right-hand input to the convolution. Must have `in1.ndim`` ``==`` ``in2.ndim`.

- **mode** (*ModeString*) –

  controls the size of the output. Available operations are:

  - `"full"`: (default) output the full convolution of the inputs.

  - `"same"`: return a centered portion of the `"full"` output which is the same size as `in1`.

  - `"valid"`: return the portion of the `"full"` output which do not depend on padding at the array edges.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  controls the computation method. Options are

  - `"auto"`: (default) always uses the `"direct"` method.

  - `"direct"`: lower to [`jax.lax.conv_general_dilated()`](jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated "jax.lax.conv_general_dilated").

  - `"fft"`: compute the result via a fast Fourier transform.

- **precision** (*PrecisionLike*) – Specify the precision of the computation. Refer to [`jax.lax.Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") for a description of available values.

Returns:  
Array containing the convolved result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.convolve()`](jax.numpy.convolve.html#jax.numpy.convolve "jax.numpy.convolve"): 1D convolution

- [`jax.scipy.signal.convolve2d()`](jax.scipy.signal.convolve2d.html#jax.scipy.signal.convolve2d "jax.scipy.signal.convolve2d"): 2D convolution

- [`jax.scipy.signal.correlate()`](jax.scipy.signal.correlate.html#jax.scipy.signal.correlate "jax.scipy.signal.correlate"): ND correlation

Examples

A few 1D convolution examples:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([1, 1, 1])

Full convolution uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.convolve(x, y, mode='full')
    Array([1., 3., 6., 7., 6., 3., 1.], dtype=float32)

Specifying `mode`` ``=`` ``'same'` returns a centered convolution the same size as the first input:

    >>> jax.scipy.signal.convolve(x, y, mode='same')
    Array([3., 6., 7., 6., 3.], dtype=float32)

Specifying `mode`` ``=`` ``'valid'` returns only the portion where the two arrays fully overlap:

    >>> jax.scipy.signal.convolve(x, y, mode='valid')
    Array([6., 7., 6.], dtype=float32)

[](jax.scipy.signal.fftconvolve.html "previous page")

previous

jax.scipy.signal.fftconvolve

[](jax.scipy.signal.convolve2d.html "next page")

next

jax.scipy.signal.convolve2d

Contents

- [`convolve()`](#jax.scipy.signal.convolve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
