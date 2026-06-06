- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.hamming

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.hamming.rst "Download source file")
-  .pdf

# jax.numpy.hamming

## Contents

- [`hamming()`](#jax.numpy.hamming)

# jax.numpy.hamming[\#](#jax-numpy-hamming "Link to this heading")

jax.numpy.hamming(*M*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/window_functions.py#L90-L119)[\#](#jax.numpy.hamming "Link to this definition")  
Return a Hamming window of size M.

JAX implementation of [`numpy.hamming()`](https://numpy.org/doc/stable/reference/generated/numpy.hamming.html#numpy.hamming "(in NumPy v2.4)").

Parameters:  
**M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The window size.

Returns:  
An array of size M containing the Hamming window.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.hamming(4))
    [0.08 0.77 0.77 0.08]

See also

- [`jax.numpy.bartlett()`](jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett"): return a Bartlett window of size M.

- [`jax.numpy.blackman()`](jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman"): return a Blackman window of size M.

- [`jax.numpy.hanning()`](jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning"): return a Hanning window of size M.

- [`jax.numpy.kaiser()`](jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser"): return a Kaiser window of size M.

[](jax.numpy.greater_equal.html "previous page")

previous

jax.numpy.greater_equal

[](jax.numpy.hanning.html "next page")

next

jax.numpy.hanning

Contents

- [`hamming()`](#jax.numpy.hamming)

By The JAX authors

© Copyright 2024, The JAX Authors.\
