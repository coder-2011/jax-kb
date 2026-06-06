- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.array_str

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.array_str.rst "Download source file")
-  .pdf

# jax.numpy.array_str

## Contents

- [`array_str()`](#jax.numpy.array_str)

# jax.numpy.array_str[\#](#jax-numpy-array-str "Link to this heading")

jax.numpy.array_str(*a*, *max_line_width=None*, *precision=None*, *suppress_small=None*)[\#](#jax.numpy.array_str "Link to this definition")  
Return a string representation of the data in an array.

The data in the array is returned as a single string. This function is similar to array_repr, the difference being that array_repr also returns information on the kind of array and its data type.

Parameters:  
- **a** (*ndarray*) – Input array.

- **max_line_width** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) – Inserts newlines if text is longer than max_line_width. Defaults to `numpy.get_printoptions()['linewidth']`.

- **precision** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) – Floating point precision. Defaults to `numpy.get_printoptions()['precision']`.

- **suppress_small** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – Represent numbers “very close” to zero as zero; default is False. Very close is defined by precision: if the precision is 8, e.g., numbers smaller (in absolute value) than 5e-9 are represented as zero. Defaults to `numpy.get_printoptions()['suppress']`.

See also

`array2string`, [`array_repr`](jax.numpy.array_repr.html#jax.numpy.array_repr "jax.numpy.array_repr"), [`set_printoptions`](jax.numpy.set_printoptions.html#jax.numpy.set_printoptions "jax.numpy.set_printoptions")

Examples

    >>> import numpy as np
    >>> np.array_str(np.arange(3))
    '[0 1 2]'

[](jax.numpy.array_split.html "previous page")

previous

jax.numpy.array_split

[](jax.numpy.asarray.html "next page")

next

jax.numpy.asarray

Contents

- [`array_str()`](#jax.numpy.array_str)

By The JAX authors

© Copyright 2024, The JAX Authors.\
