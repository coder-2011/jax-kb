- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.array_repr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.array_repr.rst "Download source file")
-  .pdf

# jax.numpy.array_repr

## Contents

- [`array_repr()`](#jax.numpy.array_repr)

# jax.numpy.array_repr[\#](#jax-numpy-array-repr "Link to this heading")

jax.numpy.array_repr(*arr*, *max_line_width=None*, *precision=None*, *suppress_small=None*)[\#](#jax.numpy.array_repr "Link to this definition")  
Return the string representation of an array.

Parameters:  
- **arr** (*ndarray*) – Input array.

- **max_line_width** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) – Inserts newlines if text is longer than max_line_width. Defaults to `numpy.get_printoptions()['linewidth']`.

- **precision** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *optional*) – Floating point precision. Defaults to `numpy.get_printoptions()['precision']`.

- **suppress_small** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – Represent numbers “very close” to zero as zero; default is False. Very close is defined by precision: if the precision is 8, e.g., numbers smaller (in absolute value) than 5e-9 are represented as zero. Defaults to `numpy.get_printoptions()['suppress']`.

Returns:  
**string** – The string representation of an array.

Return type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

See also

[`array_str`](jax.numpy.array_str.html#jax.numpy.array_str "jax.numpy.array_str"), `array2string`, [`set_printoptions`](jax.numpy.set_printoptions.html#jax.numpy.set_printoptions "jax.numpy.set_printoptions")

Examples

    >>> import numpy as np
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> np.array_repr(x, precision=6, suppress_small=True)
    'array([0.000001,  0.      ,  2.      ,  3.      ])'

[](jax.numpy.array_equiv.html "previous page")

previous

jax.numpy.array_equiv

[](jax.numpy.array_split.html "next page")

next

jax.numpy.array_split

Contents

- [`array_repr()`](#jax.numpy.array_repr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
