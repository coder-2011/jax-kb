- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.stft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.stft.rst "Download source file")
-  .pdf

# jax.scipy.signal.stft

## Contents

- [`stft()`](#jax.scipy.signal.stft)

# jax.scipy.signal.stft[\#](#jax-scipy-signal-stft "Link to this heading")

jax.scipy.signal.stft(*x*, *fs=1.0*, *window='hann'*, *nperseg=256*, *noverlap=None*, *nfft=None*, *detrend=False*, *return_onesided=True*, *boundary='zeros'*, *padded=True*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L844-L887)[\#](#jax.scipy.signal.stft "Link to this definition")  
Compute the short-time Fourier transform (STFT).

JAX implementation of [`scipy.signal.stft()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.stft.html#scipy.signal.stft "(in SciPy v1.19.0.dev)").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Array representing a time series of input values.

- **fs** (*ArrayLike*) – Sampling frequency of the time series (default: 1.0).

- **window** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Data tapering window to apply to each segment. Can be a window function name, a tuple specifying a window length and function, or an array (default: `'hann'`).

- **nperseg** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Length of each segment (default: 256).

- **noverlap** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of points to overlap between segments (default: `nperseg`` ``//`` ``2`).

- **nfft** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Length of the FFT used, if a zero-padded FFT is desired. If `None` (default), the FFT length is `nperseg`.

- **detrend** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Specifies how to detrend each segment. Can be `False` (default: no detrending), `'constant'` (remove mean), `'linear'` (remove linear trend), or a callable accepting a segment and returning a detrended segment.

- **return_onesided** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True (default), return a one-sided spectrum for real inputs. If False, return a two-sided spectrum.

- **boundary** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Specifies whether the input signal is extended at both ends, and how. Options are `None` (no extension), `'zeros'` (default), `'even'`, `'odd'`, or `'constant'`.

- **padded** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Specifies whether the input signal is zero-padded at the end to make its length a multiple of nperseg. If True (default), the padded signal length is the next multiple of `nperseg`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Axis along which the STFT is computed; the default is over the last axis (-1).

Returns:  
A length-3 tuple of arrays `(f,`` ``t,`` ``Zxx)`. `f` is the Array of sample frequencies. `t` is the Array of segment times, and `Zxx` is the STFT of `x`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

[`jax.scipy.signal.istft()`](jax.scipy.signal.istft.html#jax.scipy.signal.istft "jax.scipy.signal.istft"): inverse short-time Fourier transform.

[](jax.scipy.signal.istft.html "previous page")

previous

jax.scipy.signal.istft

[](jax.scipy.signal.welch.html "next page")

next

jax.scipy.signal.welch

Contents

- [`stft()`](#jax.scipy.signal.stft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
