- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.welch

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.welch.rst "Download source file")
-  .pdf

# jax.scipy.signal.welch

## Contents

- [`welch()`](#jax.scipy.signal.welch)

# jax.scipy.signal.welch[\#](#jax-scipy-signal-welch "Link to this heading")

jax.scipy.signal.welch(*x*, *fs=1.0*, *window='hann'*, *nperseg=None*, *noverlap=None*, *nfft=None*, *detrend='constant'*, *return_onesided=True*, *scaling='density'*, *axis=-1*, *average='mean'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L966-L1012)[\#](#jax.scipy.signal.welch "Link to this definition")  
Estimate power spectral density (PSD) using Welch’s method.

This is a JAX implementation of [`scipy.signal.welch()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.welch.html#scipy.signal.welch "(in SciPy v1.19.0.dev)"). It divides the input signal into overlapping segments, computes the modified periodogram for each segment, and averages the results to obtain a smoother estimate of the PSD.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Array representing a time series of input values.

- **fs** (*ArrayLike*) – Sampling frequency of the inputs (default: 1.0).

- **window** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Data tapering window to apply to each segment. Can be a window function name, a tuple specifying a window length and function, or an array (default: `'hann'`).

- **nperseg** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Length of each segment (default: 256).

- **noverlap** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of points to overlap between segments (default: `nperseg`` ``//`` ``2`).

- **nfft** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Length of the FFT used, if a zero-padded FFT is desired. If `None` (default), the FFT length is `nperseg`.

- **detrend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Specifies how to detrend each segment. Can be `False` (default: no detrending), `'constant'` (remove mean), `'linear'` (remove linear trend), or a callable accepting a segment and returning a detrended segment.

- **return_onesided** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True (default), return a one-sided spectrum for real inputs. If False, return a two-sided spectrum.

- **scaling** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Selects between computing the power spectral density (`'density'`, default) or the power spectrum (`'spectrum'`)

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Axis along which the PSD is computed (default: -1).

- **average** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The type of averaging to use on the periodograms; one of `'mean'` (default) or `'median'`.

Returns:  
A length-2 tuple of arrays `(f,`` ``Pxx)`. `f` is the array of sample frequencies, and `Pxx` is the power spectral density of `x`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.scipy.signal.csd()`](jax.scipy.signal.csd.html#jax.scipy.signal.csd "jax.scipy.signal.csd"): Cross power spectral density.

- [`jax.scipy.signal.stft()`](jax.scipy.signal.stft.html#jax.scipy.signal.stft "jax.scipy.signal.stft"): Short-time Fourier transform.

[](jax.scipy.signal.stft.html "previous page")

previous

jax.scipy.signal.stft

[](jax.scipy.spatial.transform.Rotation.html "next page")

next

jax.scipy.spatial.transform.Rotation

Contents

- [`welch()`](#jax.scipy.signal.welch)

By The JAX authors

© Copyright 2024, The JAX Authors.\
