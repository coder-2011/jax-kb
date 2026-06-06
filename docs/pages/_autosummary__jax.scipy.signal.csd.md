- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.csd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.csd.rst "Download source file")
-  .pdf

# jax.scipy.signal.csd

## Contents

- [`csd()`](#jax.scipy.signal.csd)

# jax.scipy.signal.csd[\#](#jax-scipy-signal-csd "Link to this heading")

jax.scipy.signal.csd(*x*, *y*, *fs=1.0*, *window='hann'*, *nperseg=None*, *noverlap=None*, *nfft=None*, *detrend='constant'*, *return_onesided=True*, *scaling='density'*, *axis=-1*, *average='mean'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L889-L964)[\#](#jax.scipy.signal.csd "Link to this definition")  
Estimate cross power spectral density (CSD) using Welch’s method.

This is a JAX implementation of [`scipy.signal.csd()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.csd.html#scipy.signal.csd "(in SciPy v1.19.0.dev)"). It is similar to [`jax.scipy.signal.welch()`](jax.scipy.signal.welch.html#jax.scipy.signal.welch "jax.scipy.signal.welch"), but it operates on two input signals and estimates their cross-spectral density instead of the power spectral density (PSD).

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Array representing a time series of input values.

- **y** (*ArrayLike* *\|* *None*) – Array representing the second time series of input values, the same length as `x` along the specified `axis`. If not specified, then assume `y`` ``=`` ``x` and compute the PSD `Pxx` of `x` via Welch’s method.

- **fs** (*ArrayLike*) – Sampling frequency of the inputs (default: 1.0).

- **window** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Data tapering window to apply to each segment. Can be a window function name, a tuple specifying a window length and function, or an array (default: `'hann'`).

- **nperseg** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Length of each segment (default: 256).

- **noverlap** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of points to overlap between segments (default: `nperseg`` ``//`` ``2`).

- **nfft** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Length of the FFT used, if a zero-padded FFT is desired. If `None` (default), the FFT length is `nperseg`.

- **detrend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Specifies how to detrend each segment. Can be `False` (default: no detrending), `'constant'` (remove mean), `'linear'` (remove linear trend), or a callable accepting a segment and returning a detrended segment.

- **return_onesided** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True (default), return a one-sided spectrum for real inputs. If False, return a two-sided spectrum.

- **scaling** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Selects between computing the power spectral density (`'density'`, default) or the power spectrum (`'spectrum'`)

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Axis along which the CSD is computed (default: -1).

- **average** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The type of averaging to use on the periodograms; one of `'mean'` (default) or `'median'`.

Returns:  
A length-2 tuple of arrays `(f,`` ``Pxy)`. `f` is the array of sample frequencies, and `Pxy` is the cross spectral density of x and y

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Notes

The original SciPy function exhibits slightly different behavior between `csd(x,`` ``x)` and `csd(x,`` ``x.copy())`. The LAX-backend version is designed to follow the latter behavior. To replicate the former, call this function function as `csd(x,`` ``None)`.

See also

- [`jax.scipy.signal.welch()`](jax.scipy.signal.welch.html#jax.scipy.signal.welch "jax.scipy.signal.welch"): Power spectral density.

- [`jax.scipy.signal.stft()`](jax.scipy.signal.stft.html#jax.scipy.signal.stft "jax.scipy.signal.stft"): Short-time Fourier transform.

[](jax.scipy.signal.correlate2d.html "previous page")

previous

jax.scipy.signal.correlate2d

[](jax.scipy.signal.detrend.html "next page")

next

jax.scipy.signal.detrend

Contents

- [`csd()`](#jax.scipy.signal.csd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
