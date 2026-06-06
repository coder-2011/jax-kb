- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.convolve2d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.convolve2d.rst "Download source file")
-  .pdf

# jax.scipy.signal.convolve2d

## Contents

- [`convolve2d()`](#jax.scipy.signal.convolve2d)

# jax.scipy.signal.convolve2d[\#](#jax-scipy-signal-convolve2d "Link to this heading")

jax.scipy.signal.convolve2d(*in1*, *in2*, *mode='full'*, *boundary='fill'*, *fillvalue=0*, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L263-L335)[\#](#jax.scipy.signal.convolve2d "Link to this definition")  
Convolution of two 2-dimensional arrays.

JAX implementation of [`scipy.signal.convolve2d()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.convolve2d.html#scipy.signal.convolve2d "(in SciPy v1.19.0.dev)").

Parameters:  
- **in1** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – left-hand input to the convolution. Must have `in1.ndim`` ``==`` ``2`.

- **in2** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – right-hand input to the convolution. Must have `in2.ndim`` ``==`` ``2`.

- **mode** (*ModeString*) –

  controls the size of the output. Available operations are:

  - `"full"`: (default) output the full convolution of the inputs.

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
Array containing the convolved result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.convolve()`](jax.numpy.convolve.html#jax.numpy.convolve "jax.numpy.convolve"): 1D convolution

- [`jax.scipy.signal.convolve()`](jax.scipy.signal.convolve.html#jax.scipy.signal.convolve "jax.scipy.signal.convolve"): ND convolution

- [`jax.scipy.signal.correlate()`](jax.scipy.signal.correlate.html#jax.scipy.signal.correlate "jax.scipy.signal.correlate"): ND correlation

Examples

A few 2D convolution examples:

    >>> x = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> y = jnp.array([[2, 1, 1],
    ...                [4, 3, 4],
    ...                [1, 3, 2]])

Full 2D convolution uses implicit zero-padding at the edges:

    >>> jax.scipy.signal.convolve2d(x, y, mode='full')
    Array([[ 2.,  5.,  3.,  2.],
           [10., 22., 17., 12.],
           [13., 30., 32., 20.],
           [ 3., 13., 18.,  8.]], dtype=float32)

Specifying `mode`` ``=`` ``'same'` returns a centered 2D convolution of the same size as the first input:

    >>> jax.scipy.signal.convolve2d(x, y, mode='same')
    Array([[22., 17.],
           [30., 32.]], dtype=float32)

Specifying `mode`` ``=`` ``'valid'` returns only the portion of 2D convolution where the two arrays fully overlap:

    >>> jax.scipy.signal.convolve2d(x, y, mode='valid')
    Array([[22., 17.],
           [30., 32.]], dtype=float32)

[](jax.scipy.signal.convolve.html "previous page")

previous

jax.scipy.signal.convolve

[](jax.scipy.signal.correlate.html "next page")

next

jax.scipy.signal.correlate

Contents

- [`convolve2d()`](#jax.scipy.signal.convolve2d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
