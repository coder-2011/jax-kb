- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.median

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.median.rst "Download source file")
-  .pdf

# jax.numpy.median

## Contents

- [`median()`](#jax.numpy.median)

# jax.numpy.median[\#](#jax-numpy-median "Link to this heading")

jax.numpy.median(*a*, *axis=None*, *out=None*, *overwrite_input=False*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2836-L2886)[\#](#jax.numpy.median "Link to this definition")  
Return the median of array elements along a given axis.

JAX implementation of [`numpy.median()`](https://numpy.org/doc/stable/reference/generated/numpy.median.html#numpy.median "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional, int or sequence of ints, default=None. Axis along which the median to be computed. If None, median is computed for the flattened array.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **out** (*None*) – Unused by JAX.

- **overwrite_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Unused by JAX.

Returns:  
An array of the median along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.mean()`](jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean"): Compute the mean of array elements over a given axis.

- [`jax.numpy.max()`](jax.numpy.max.html#jax.numpy.max "jax.numpy.max"): Compute the maximum of array elements over given axis.

- [`jax.numpy.min()`](jax.numpy.min.html#jax.numpy.min "jax.numpy.min"): Compute the minimum of array elements over given axis.

Examples

By default, the median is computed for the flattened array.

    >>> x = jnp.array([[2, 4, 7, 1],
    ...                [3, 5, 9, 2],
    ...                [6, 1, 8, 3]])
    >>> jnp.median(x)
    Array(3.5, dtype=float32)

If `axis=1`, the median is computed along axis 1.

    >>> jnp.median(x, axis=1)
    Array([3. , 4. , 4.5], dtype=float32)

If `keepdims=True`, `ndim` of the output is equal to that of the input.

    >>> jnp.median(x, axis=1, keepdims=True)
    Array([[3. ],
           [4. ],
           [4.5]], dtype=float32)

[](jax.numpy.mean.html "previous page")

previous

jax.numpy.mean

[](jax.numpy.meshgrid.html "next page")

next

jax.numpy.meshgrid

Contents

- [`median()`](#jax.numpy.median)

By The JAX authors

© Copyright 2024, The JAX Authors.\
