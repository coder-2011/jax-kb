- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.signal.detrend

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.signal.detrend.rst "Download source file")
-  .pdf

# jax.scipy.signal.detrend

## Contents

- [`detrend()`](#jax.scipy.signal.detrend)

# jax.scipy.signal.detrend[\#](#jax-scipy-signal-detrend "Link to this heading")

jax.scipy.signal.detrend(*data*, *axis=-1*, *type='linear'*, *bp=0*, *overwrite_data=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/signal.py#L493-L565)[\#](#jax.scipy.signal.detrend "Link to this definition")  
Remove linear or piecewise linear trends from data.

JAX implementation of [`scipy.signal.detrend()`](https://scipy.github.io/devdocs/reference/generated/scipy.signal.detrend.html#scipy.signal.detrend "(in SciPy v1.19.0.dev)").

Parameters:  
- **data** (*ArrayLike*) – The input array containing the data to detrend.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The axis along which to detrend. Default is -1 (the last axis).

- **type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  The type of detrending. Can be:

  - `'linear'`: Fit a single linear trend for the entire data.

  - `'constant'`: Remove the mean value of the data.

- **bp** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *np.ndarray*) – A sequence of breakpoints. If given, piecewise linear trends are fit between these breakpoints.

- **overwrite_data** (*None*) – This argument is not supported by JAX’s implementation.

Returns:  
The detrended data array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

A simple detrend operation in one dimension:

    >>> data = jnp.array([1., 4., 8., 8., 9.])

Removing a linear trend from the data:

    >>> detrended = jax.scipy.signal.detrend(data)
    >>> with jnp.printoptions(precision=3, suppress=True):  # suppress float error
    ...   print("Detrended:", detrended)
    ...   print("Underlying trend:", data - detrended)
    Detrended: [-1. -0.  2. -0. -1.]
    Underlying trend: [ 2.  4.  6.  8. 10.]

Removing a constant trend from the data:

    >>> detrended = jax.scipy.signal.detrend(data, type='constant')
    >>> with jnp.printoptions(precision=3):  # suppress float error
    ...   print("Detrended:", detrended)
    ...   print("Underlying trend:", data - detrended)
    Detrended: [-5. -2.  2.  2.  3.]
    Underlying trend: [6. 6. 6. 6. 6.]

[](jax.scipy.signal.csd.html "previous page")

previous

jax.scipy.signal.csd

[](jax.scipy.signal.istft.html "next page")

next

jax.scipy.signal.istft

Contents

- [`detrend()`](#jax.scipy.signal.detrend)

By The JAX authors

© Copyright 2024, The JAX Authors.\
