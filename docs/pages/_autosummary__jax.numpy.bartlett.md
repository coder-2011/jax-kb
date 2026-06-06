- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.bartlett

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.bartlett.rst "Download source file")
-  .pdf

# jax.numpy.bartlett

## Contents

- [`bartlett()`](#jax.numpy.bartlett)

# jax.numpy.bartlett[\#](#jax-numpy-bartlett "Link to this heading")

jax.numpy.bartlett(*M*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/window_functions.py#L59-L88)[\#](#jax.numpy.bartlett "Link to this definition")  
Return a Bartlett window of size M.

JAX implementation of [`numpy.bartlett()`](https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html#numpy.bartlett "(in NumPy v2.4)").

Parameters:  
**M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The window size.

Returns:  
An array of size M containing the Bartlett window.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.bartlett(4))
    [0.   0.67 0.67 0.  ]

See also

- [`jax.numpy.blackman()`](jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman"): return a Blackman window of size M.

- [`jax.numpy.hamming()`](jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming"): return a Hamming window of size M.

- [`jax.numpy.hanning()`](jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning"): return a Hanning window of size M.

- [`jax.numpy.kaiser()`](jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser"): return a Kaiser window of size M.

[](jax.numpy.average.html "previous page")

previous

jax.numpy.average

[](jax.numpy.bincount.html "next page")

next

jax.numpy.bincount

Contents

- [`bartlett()`](#jax.numpy.bartlett)

By The JAX authors

© Copyright 2024, The JAX Authors.\
