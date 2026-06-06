- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isin.rst "Download source file")
-  .pdf

# jax.numpy.isin

## Contents

- [`isin()`](#jax.numpy.isin)

# jax.numpy.isin[\#](#jax-numpy-isin "Link to this heading")

jax.numpy.isin(*element*, *test_elements*, *assume_unique=False*, *invert=False*, *\**, *method='auto'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L538-L571)[\#](#jax.numpy.isin "Link to this definition")  
Determine whether elements in `element` appear in `test_elements`.

JAX implementation of [`numpy.isin()`](https://numpy.org/doc/stable/reference/generated/numpy.isin.html#numpy.isin "(in NumPy v2.4)").

Parameters:  
- **element** (*ArrayLike*) – input array of elements for which membership will be checked.

- **test_elements** (*ArrayLike*) – N-dimensional array of test values to check for the presence of each element.

- **invert** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, return `~isin(element,`` ``test_elements)`. Default is False.

- **assume_unique** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if true, input arrays are assumed to be unique, which can lead to more efficient computation. If the input arrays are not unique and assume_unique is set to True, the results are undefined.

- **method** – string specifying the method used to compute the result. Supported options are ‘compare_all’, ‘binary_search’, ‘sort’, and ‘auto’ (default).

Returns:  
A boolean array of shape `element.shape` that specifies whether each element appears in `test_elements`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> elements = jnp.array([1, 2, 3, 4])
    >>> test_elements = jnp.array([[1, 5, 6, 3, 7, 1]])
    >>> jnp.isin(elements, test_elements)
    Array([ True, False,  True, False], dtype=bool)

[](jax.numpy.isfinite.html "previous page")

previous

jax.numpy.isfinite

[](jax.numpy.isinf.html "next page")

next

jax.numpy.isinf

Contents

- [`isin()`](#jax.numpy.isin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
