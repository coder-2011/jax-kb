- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.correlate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.correlate.rst "Download source file")
-  .pdf

# jax.numpy.correlate

## Contents

- [`correlate()`](#jax.numpy.correlate)

# jax.numpy.correlate[\#](#jax-numpy-correlate "Link to this heading")

jax.numpy.correlate(*a*, *v*, *mode='valid'*, *\**, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L669-L753)[\#](#jax.numpy.correlate "Link to this definition")  
Correlation of two one dimensional arrays.

JAX implementation of [`numpy.correlate()`](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html#numpy.correlate "(in NumPy v2.4)").

Correlation of one dimensional arrays is defined as:

\\c_k = \sum_j a\_{k + j} \overline{v_j}\\

where \\\overline{v_j}\\ is the complex conjugate of \\v_j\\.

Parameters:  
- **a** (*ArrayLike*) – left-hand input to the correlation. Must have `a.ndim`` ``==`` ``1`.

- **v** (*ArrayLike*) – right-hand input to the correlation. Must have `v.ndim`` ``==`` ``1`.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  controls the size of the output. Available operations are:

  - `"full"`: output the full correlation of the inputs.

  - `"same"`: return a centered portion of the `"full"` output which is the same size as `a`.

  - `"valid"`: (default) return the portion of the `"full"` output which do not depend on padding at the array edges.

- **precision** (*lax.PrecisionLike*) – Specify the precision of the computation. Refer to [`jax.lax.Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") for a description of available values.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – A datatype, indicating to accumulate results to and return a result with that datatype. Default is `None`, which means the default accumulation type for the input types.

Returns:  
Array containing the cross-correlation result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.signal.correlate()`](jax.scipy.signal.correlate.html#jax.scipy.signal.correlate "jax.scipy.signal.correlate"): ND correlation

- [`jax.numpy.convolve()`](jax.numpy.convolve.html#jax.numpy.convolve "jax.numpy.convolve"): 1D convolution

Examples

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([4, 5, 6])

Since default `mode`` ``=`` ``'valid'`, `jax.numpy.correlate` returns only the portion of correlation where the two arrays fully overlap:

    >>> jnp.correlate(x, y)
    Array([32., 35., 28.], dtype=float32)

Specifying `mode`` ``=`` ``'full'` returns full correlation using implicit zero-padding at the edges.

    >>> jnp.correlate(x, y, mode='full')
    Array([ 6., 17., 32., 35., 28., 13.,  4.], dtype=float32)

Specifying `mode`` ``=`` ``'same'` returns a centered correlation the same size as the first input:

    >>> jnp.correlate(x, y, mode='same')
    Array([17., 32., 35., 28., 13.], dtype=float32)

If both the inputs arrays are real-valued and symmetric then the result will also be symmetric and will be equal to the result of `jax.numpy.convolve`.

    >>> x1 = jnp.array([1, 2, 3, 2, 1])
    >>> y1 = jnp.array([4, 5, 4])
    >>> jnp.correlate(x1, y1, mode='full')
    Array([ 4., 13., 26., 31., 26., 13.,  4.], dtype=float32)
    >>> jnp.convolve(x1, y1, mode='full')
    Array([ 4., 13., 26., 31., 26., 13.,  4.], dtype=float32)

For complex-valued inputs:

    >>> x2 = jnp.array([3+1j, 2, 2-3j])
    >>> y2 = jnp.array([4, 2-5j, 1])
    >>> jnp.correlate(x2, y2, mode='full')
    Array([ 3. +1.j,  3.+17.j, 18.+11.j, 27. +4.j,  8.-12.j], dtype=complex64)

[](jax.numpy.corrcoef.html "previous page")

previous

jax.numpy.corrcoef

[](jax.numpy.cos.html "next page")

next

jax.numpy.cos

Contents

- [`correlate()`](#jax.numpy.correlate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
