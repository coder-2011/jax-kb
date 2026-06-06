- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.hanning

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.hanning.rst "Download source file")
-  .pdf

# jax.numpy.hanning

## Contents

- [`hanning()`](#jax.numpy.hanning)

# jax.numpy.hanning[\#](#jax-numpy-hanning "Link to this heading")

jax.numpy.hanning(*M*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/window_functions.py#L121-L150)[\#](#jax.numpy.hanning "Link to this definition")  
Return a Hanning window of size M.

JAX implementation of [`numpy.hanning()`](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html#numpy.hanning "(in NumPy v2.4)").

Parameters:  
**M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The window size.

Returns:  
An array of size M containing the Hanning window.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.hanning(4))
    [0.   0.75 0.75 0.  ]

See also

- [`jax.numpy.bartlett()`](jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett"): return a Bartlett window of size M.

- [`jax.numpy.blackman()`](jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman"): return a Blackman window of size M.

- [`jax.numpy.hamming()`](jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming"): return a Hamming window of size M.

- [`jax.numpy.kaiser()`](jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser"): return a Kaiser window of size M.

[](jax.numpy.hamming.html "previous page")

previous

jax.numpy.hamming

[](jax.numpy.heaviside.html "next page")

next

jax.numpy.heaviside

Contents

- [`hanning()`](#jax.numpy.hanning)

By The JAX authors

© Copyright 2024, The JAX Authors.\
