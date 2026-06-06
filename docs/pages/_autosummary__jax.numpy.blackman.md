- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.blackman

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.blackman.rst "Download source file")
-  .pdf

# jax.numpy.blackman

## Contents

- [`blackman()`](#jax.numpy.blackman)

# jax.numpy.blackman[\#](#jax-numpy-blackman "Link to this heading")

jax.numpy.blackman(*M*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/window_functions.py#L28-L57)[\#](#jax.numpy.blackman "Link to this definition")  
Return a Blackman window of size M.

JAX implementation of [`numpy.blackman()`](https://numpy.org/doc/stable/reference/generated/numpy.blackman.html#numpy.blackman "(in NumPy v2.4)").

Parameters:  
**M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The window size.

Returns:  
An array of size M containing the Blackman window.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.blackman(4))
    [-0.    0.63  0.63 -0.  ]

See also

- [`jax.numpy.bartlett()`](jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett"): return a Bartlett window of size M.

- [`jax.numpy.hamming()`](jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming"): return a Hamming window of size M.

- [`jax.numpy.hanning()`](jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning"): return a Hanning window of size M.

- [`jax.numpy.kaiser()`](jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser"): return a Kaiser window of size M.

[](jax.numpy.bitwise_xor.html "previous page")

previous

jax.numpy.bitwise_xor

[](jax.numpy.block.html "next page")

next

jax.numpy.block

Contents

- [`blackman()`](#jax.numpy.blackman)

By The JAX authors

© Copyright 2024, The JAX Authors.\
