- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.correlate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.correlate.rst "Download source file")
-  .pdf

# jax.scipy.signal.correlate

## Contents

- [`correlate()`](#jax.scipy.signal.correlate)

# jax.scipy.signal.correlate[\#](#jax-scipy-signal-correlate "Link to this heading")

jax.scipy.signal.correlate(*in1*, *in2*, *mode='full'*, *method='auto'*, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L337-L395)[\#](#jax.scipy.signal.correlate "Link to this definition")  
Cross-correlation of two N-dimensional arrays.

JAX implementation of [`scipy.signal.correlate()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate "(in SciPy v1.19.0.dev)").

Parameters:  
- **in1** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – left-hand input to the cross-correlation.

- **in2** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – right-hand input to the cross-correlation. Must have `in1.ndim`` ``==`` ``in2.ndim`.

- **mode** (*ModeString*) –

  controls the size of the output. Available operations are:

  - `"full"`: (default) output the full cross-correlation of the inputs.

  - `"same"`: return a centered portion of the `"full"` output which is the same size as `in1`.

  - `"valid"`: return the portion of the `"full"` output which do not depend on padding at the array edges.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  controls the computation method. Options are

  - `"auto"`: (default) always uses the `"direct"` method.

  - `"direct"`: lower to [`jax.lax.conv_general_dilated()`](jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated "jax.lax.conv_general_dilated").

  - `"fft"`: compute the result via a fast Fourier transform.

- **precision** (*PrecisionLike*) – Specify the precision of the computation. Refer to [`jax.lax.Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") for a description of available values.

Returns:  
Array containing the cross-correlation result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.correlate()`](jax.numpy.correlate.html#jax.numpy.correlate "jax.numpy.correlate"): 1D cross-correlation

- [`jax.scipy.signal.correlate2d()`](jax.scipy.signal.correlate2d.html#jax.scipy.signal.correlate2d "jax.scipy.signal.correlate2d"): 2D cross-correlation

- [`jax.scipy.signal.convolve()`](jax.scipy.signal.convolve.html#jax.scipy.signal.convolve "jax.scipy.signal.convolve"): ND convolution

Examples

A few 1D correlation examples:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([1, 3, 2])

Full 1D correlation uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.correlate(x, y, mode='full')
    Array([ 2.,  7., 13., 15., 11.,  5.,  1.], dtype=float32)

Specifying `mode`` ``=`` ``'same'` returns a centered 1D correlation of the same size as the first input:

    >>> jax.scipy.signal.correlate(x, y, mode='same')
    Array([ 7., 13., 15., 11.,  5.], dtype=float32)

Specifying `mode`` ``=`` ``'valid'` returns only the portion of 1D correlation where the two arrays fully overlap:

    >>> jax.scipy.signal.correlate(x, y, mode='valid')
    Array([13., 15., 11.], dtype=float32)

[](jax.scipy.signal.convolve2d.html "previous page")

previous

jax.scipy.signal.convolve2d

[](jax.scipy.signal.correlate2d.html "next page")

next

jax.scipy.signal.correlate2d

Contents

- [`correlate()`](#jax.scipy.signal.correlate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
