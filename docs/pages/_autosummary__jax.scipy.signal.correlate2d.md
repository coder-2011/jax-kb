- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.correlate2d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.correlate2d.rst "Download source file")
-  .pdf

# jax.scipy.signal.correlate2d

## Contents

- [`correlate2d()`](#jax.scipy.signal.correlate2d)

# jax.scipy.signal.correlate2d[\#](#jax-scipy-signal-correlate2d "Link to this heading")

jax.scipy.signal.correlate2d(*in1*, *in2*, *mode='full'*, *boundary='fill'*, *fillvalue=0*, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L397-L491)[\#](#jax.scipy.signal.correlate2d "Link to this definition")  
Cross-correlation of two 2-dimensional arrays.

JAX implementation of [`scipy.signal.correlate2d()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.correlate2d.html#scipy.signal.correlate2d "(in SciPy v1.19.0.dev)").

Parameters:  
- **in1** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – left-hand input to the cross-correlation. Must have `in1.ndim`` ``==`` ``2`.

- **in2** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – right-hand input to the cross-correlation. Must have `in2.ndim`` ``==`` ``2`.

- **mode** (*ModeString*) –

  controls the size of the output. Available operations are:

  - `"full"`: (default) output the full cross-correlation of the inputs.

  - `"same"`: return a centered portion of the `"full"` output which is the same size as `in1`.

  - `"valid"`: return the portion of the `"full"` output which do not depend on padding at the array edges.

- **boundary** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – only `"fill"` is supported.

- **fillvalue** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – only `0` is supported.

- **method** –

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

- [`jax.scipy.signal.correlate()`](jax.scipy.signal.correlate.html#jax.scipy.signal.correlate "jax.scipy.signal.correlate"): ND cross-correlation

- [`jax.scipy.signal.convolve()`](jax.scipy.signal.convolve.html#jax.scipy.signal.convolve "jax.scipy.signal.convolve"): ND convolution

Examples

A few 2D correlation examples:

    >>> x = jnp.array([[2, 1, 3],
    ...                [1, 3, 1],
    ...                [4, 1, 2]])
    >>> y = jnp.array([[1, 3],
    ...                [4, 2]])

Full 2D correlation uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.correlate2d(x, y, mode='full')
    Array([[ 4., 10., 10., 12.],
           [ 8., 15., 24.,  7.],
           [11., 28., 14.,  9.],
           [12.,  7.,  7.,  2.]], dtype=float32)

Specifying `mode`` ``=`` ``'same'` returns a centered 2D correlation of the same size as the first input:

    >>> jax.scipy.signal.correlate2d(x, y, mode='same')
    Array([[15., 24.,  7.],
           [28., 14.,  9.],
           [ 7.,  7.,  2.]], dtype=float32)

Specifying `mode`` ``=`` ``'valid'` returns only the portion of 2D correlation where the two arrays fully overlap:

    >>> jax.scipy.signal.correlate2d(x, y, mode='valid')
    Array([[15., 24.],
           [28., 14.]], dtype=float32)

[](jax.scipy.signal.correlate.html "previous page")

previous

jax.scipy.signal.correlate

[](jax.scipy.signal.csd.html "next page")

next

jax.scipy.signal.csd

Contents

- [`correlate2d()`](#jax.scipy.signal.correlate2d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
