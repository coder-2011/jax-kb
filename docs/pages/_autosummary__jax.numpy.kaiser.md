- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.kaiser

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.kaiser.rst "Download source file")
-  .pdf

# jax.numpy.kaiser

## Contents

- [`kaiser()`](#jax.numpy.kaiser)

# jax.numpy.kaiser[\#](#jax-numpy-kaiser "Link to this heading")

jax.numpy.kaiser(*M*, *beta*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/window_functions.py#L152-L183)[\#](#jax.numpy.kaiser "Link to this definition")  
Return a Kaiser window of size M.

JAX implementation of [`numpy.kaiser()`](https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html#numpy.kaiser "(in NumPy v2.4)").

Parameters:  
- **M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The window size.

- **beta** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – The Kaiser window parameter.

Returns:  
An array of size M containing the Kaiser window.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.kaiser(4, 1.5))
    [0.61 0.95 0.95 0.61]

See also

- [`jax.numpy.bartlett()`](jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett"): return a Bartlett window of size M.

- [`jax.numpy.blackman()`](jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman"): return a Blackman window of size M.

- [`jax.numpy.hamming()`](jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming"): return a Hamming window of size M.

- [`jax.numpy.hanning()`](jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning"): return a Hanning window of size M.

[](jax.numpy.ix_.html "previous page")

previous

jax.numpy.ix\_

[](jax.numpy.kron.html "next page")

next

jax.numpy.kron

Contents

- [`kaiser()`](#jax.numpy.kaiser)

By The JAX authors

© Copyright 2024, The JAX Authors.\
