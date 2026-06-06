- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.istft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.istft.rst "Download source file")
-  .pdf

# jax.scipy.signal.istft

## Contents

- [`istft()`](#jax.scipy.signal.istft)

# jax.scipy.signal.istft[\#](#jax-scipy-signal-istft "Link to this heading")

jax.scipy.signal.istft(*Zxx*, *fs=1.0*, *window='hann'*, *nperseg=None*, *noverlap=None*, *nfft=None*, *input_onesided=True*, *boundary=True*, *time_axis=-1*, *freq_axis=-2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L1062-L1199)[\#](#jax.scipy.signal.istft "Link to this definition")  
Perform the inverse short-time Fourier transform (ISTFT).

JAX implementation of [`scipy.signal.istft()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.istft.html#scipy.signal.istft "(in SciPy v1.19.0.dev)"); computes the inverse of [`jax.scipy.signal.stft()`](jax.scipy.signal.stft.html#jax.scipy.signal.stft "jax.scipy.signal.stft").

Parameters:  
- **Zxx** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – STFT of the signal to be reconstructed.

- **fs** (*ArrayLike*) – Sampling frequency of the time series (default: 1.0)

- **window** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Data tapering window to apply to each segment. Can be a window function name, a tuple specifying a window length and function, or an array (default: `'hann'`).

- **nperseg** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of data points per segment in the STFT. If `None` (default), the value is determined from the size of `Zxx`.

- **noverlap** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of points to overlap between segments (default: `nperseg`` ``//`` ``2`).

- **nfft** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of FFT points used in the STFT. If `None` (default), the value is determined from the size of `Zxx`.

- **input_onesided** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True (default), interpret the input as a one-sided STFT (positive frequencies only). If False, interpret the input as a two-sided STFT.

- **boundary** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True (default), it is assumed that the input signal was extended at its boundaries by `stft`. If False, the input signal is assumed to have been truncated at the boundaries by stft.

- **time_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Axis in Zxx corresponding to time segments (default: -1).

- **freq_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Axis in Zxx corresponding to frequency bins (default: -2).

Returns:  
A length-2 tuple of arrays `(t,`` ``x)`. `t` is the Array of signal times, and `x` is the reconstructed time series.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

[`jax.scipy.signal.stft()`](jax.scipy.signal.stft.html#jax.scipy.signal.stft "jax.scipy.signal.stft"): short-time Fourier transform.

Examples

Demonstrate that this gives the inverse of [`stft()`](jax.scipy.signal.stft.html#jax.scipy.signal.stft "jax.scipy.signal.stft"):

    >>> x = jnp.array([1., 2., 3., 2., 1., 0., 1., 2.])
    >>> f, t, Zxx = jax.scipy.signal.stft(x, nperseg=4)
    >>> print(Zxx)  
    [[ 1. +0.j   2.5+0.j   1. +0.j   1. +0.j   0.5+0.j ]
     [-0.5+0.5j -1.5+0.j  -0.5-0.5j -0.5+0.5j  0. -0.5j]
     [ 0. +0.j   0.5+0.j   0. +0.j   0. +0.j  -0.5+0.j ]]
    >>> t, x_reconstructed = jax.scipy.signal.istft(Zxx)
    >>> print(x_reconstructed)
    [1. 2. 3. 2. 1. 0. 1. 2.]

[](jax.scipy.signal.detrend.html "previous page")

previous

jax.scipy.signal.detrend

[](jax.scipy.signal.stft.html "next page")

next

jax.scipy.signal.stft

Contents

- [`istft()`](#jax.scipy.signal.istft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
