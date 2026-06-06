- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.convolve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.convolve.rst "Download source file")
-  .pdf

# jax.numpy.convolve

## Contents

- [`convolve()`](#jax.numpy.convolve)

# jax.numpy.convolve[\#](#jax-numpy-convolve "Link to this heading")

jax.numpy.convolve(*a*, *v*, *mode='full'*, *\**, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L593-L667)[\#](#jax.numpy.convolve "Link to this definition")  
Convolution of two one dimensional arrays.

JAX implementation of [`numpy.convolve()`](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html#numpy.convolve "(in NumPy v2.4)").

Convolution of one dimensional arrays is defined as:

\\c_k = \sum_j a\_{k - j} v_j\\

Parameters:  
- **a** (*ArrayLike*) – left-hand input to the convolution. Must have `a.ndim`` ``==`` ``1`.

- **v** (*ArrayLike*) – right-hand input to the convolution. Must have `v.ndim`` ``==`` ``1`.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  controls the size of the output. Available operations are:

  - `"full"`: (default) output the full convolution of the inputs.

  - `"same"`: return a centered portion of the `"full"` output which is the same size as `a`.

  - `"valid"`: return the portion of the `"full"` output which do not depend on padding at the array edges.

- **precision** (*lax.PrecisionLike*) – Specify the precision of the computation. Refer to [`jax.lax.Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") for a description of available values.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – A datatype, indicating to accumulate results to and return a result with that datatype. Default is `None`, which means the default accumulation type for the input types.

Returns:  
Array containing the convolved result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.signal.convolve()`](jax.scipy.signal.convolve.html#jax.scipy.signal.convolve "jax.scipy.signal.convolve"): ND convolution

- [`jax.numpy.correlate()`](jax.numpy.correlate.html#jax.numpy.correlate "jax.numpy.correlate"): 1D correlation

Examples

A few 1D convolution examples:

    >>> x = jnp.array([1, 2, 3, 2, 1])
    >>> y = jnp.array([4, 1, 2])

`jax.numpy.convolve`, by default, returns full convolution using implicit zero-padding at the edges:

    >>> jnp.convolve(x, y)
    Array([ 4.,  9., 16., 15., 12.,  5.,  2.], dtype=float32)

Specifying `mode`` ``=`` ``'same'` returns a centered convolution the same size as the first input:

    >>> jnp.convolve(x, y, mode='same')
    Array([ 9., 16., 15., 12.,  5.], dtype=float32)

Specifying `mode`` ``=`` ``'valid'` returns only the portion where the two arrays fully overlap:

    >>> jnp.convolve(x, y, mode='valid')
    Array([16., 15., 12.], dtype=float32)

For complex-valued inputs:

    >>> x1 = jnp.array([3+1j, 2, 4-3j])
    >>> y1 = jnp.array([1, 2-3j, 4+5j])
    >>> jnp.convolve(x1, y1)
    Array([ 3. +1.j, 11. -7.j, 15.+10.j,  7. -8.j, 31. +8.j], dtype=complex64)

[](jax.numpy.conjugate.html "previous page")

previous

jax.numpy.conjugate

[](jax.numpy.copy.html "next page")

next

jax.numpy.copy

Contents

- [`convolve()`](#jax.numpy.convolve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
