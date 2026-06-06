- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.fftconvolve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.fftconvolve.rst "Download source file")
-  .pdf

# jax.scipy.signal.fftconvolve

## Contents

- [`fftconvolve()`](#jax.scipy.signal.fftconvolve)

# jax.scipy.signal.fftconvolve[\#](#jax-scipy-signal-fftconvolve "Link to this heading")

jax.scipy.signal.fftconvolve(*in1*, *in2*, *mode='full'*, *axes=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L47-L118)[\#](#jax.scipy.signal.fftconvolve "Link to this definition")  
Convolve two N-dimensional arrays using Fast Fourier Transform (FFT).

JAX implementation of [`scipy.signal.fftconvolve()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.fftconvolve.html#scipy.signal.fftconvolve "(in SciPy v1.19.0.dev)").

Parameters:  
- **in1** (*ArrayLike*) – left-hand input to the convolution.

- **in2** (*ArrayLike*) – right-hand input to the convolution. Must have `in1.ndim`` ``==`` ``in2.ndim`.

- **mode** (*ModeString*) –

  controls the size of the output. Available operations are:

  - `"full"`: (default) output the full convolution of the inputs.

  - `"same"`: return a centered portion of the `"full"` output which is the same size as `in1`.

  - `"valid"`: return the portion of the `"full"` output which do not depend on padding at the array edges.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of axes along which to apply the convolution.

Returns:  
Array containing the convolved result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.convolve()`](jax.numpy.convolve.html#jax.numpy.convolve "jax.numpy.convolve"): 1D convolution

- [`jax.scipy.signal.convolve()`](jax.scipy.signal.convolve.html#jax.scipy.signal.convolve "jax.scipy.signal.convolve"): direct convolution

Examples

A few 1D convolution examples. Because FFT-based convolution is approximate, We use [`jax.numpy.printoptions()`](jax.numpy.printoptions.html#jax.numpy.printoptions "jax.numpy.printoptions") below to adjust the printing precision:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([1, 1, 1])

Full convolution uses implicit zero-padding at the edges:

    >>> with jax.numpy.printoptions(precision=3):
    ...   print(jax.scipy.signal.fftconvolve(x, y, mode='full'))
    [1. 3. 6. 7. 6. 3. 1.]

Specifying `mode`` ``=`` ``'same'` returns a centered convolution the same size as the first input:

    >>> with jax.numpy.printoptions(precision=3):
    ...   print(jax.scipy.signal.fftconvolve(x, y, mode='same'))
    [3. 6. 7. 6. 3.]

Specifying `mode`` ``=`` ``'valid'` returns only the portion where the two arrays fully overlap:

    >>> with jax.numpy.printoptions(precision=3):
    ...   print(jax.scipy.signal.fftconvolve(x, y, mode='valid'))
    [6. 7. 6.]

[](jax.scipy.optimize.OptimizeResults.html "previous page")

previous

jax.scipy.optimize.OptimizeResults

[](jax.scipy.signal.convolve.html "next page")

next

jax.scipy.signal.convolve

Contents

- [`fftconvolve()`](#jax.scipy.signal.fftconvolve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
