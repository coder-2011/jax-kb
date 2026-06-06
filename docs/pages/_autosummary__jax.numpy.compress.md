- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.compress

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.compress.rst "Download source file")
-  .pdf

# jax.numpy.compress

## Contents

- [`compress()`](#jax.numpy.compress)

# jax.numpy.compress[\#](#jax-numpy-compress "Link to this heading")

jax.numpy.compress(*condition*, *a*, *axis=None*, *\**, *size=None*, *fill_value=0*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8959-L9052)[\#](#jax.numpy.compress "Link to this definition")  
Compress an array along a given axis using a boolean condition.

JAX implementation of [`numpy.compress()`](https://numpy.org/doc/stable/reference/generated/numpy.compress.html#numpy.compress "(in NumPy v2.4)").

Parameters:  
- **condition** (*ArrayLike*) – 1-dimensional array of conditions. Will be converted to boolean.

- **a** (*ArrayLike*) – N-dimensional array of values.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – axis along which to compress. If None (default) then `a` will be flattened, and axis will be set to 0.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional static size for output. Must be specified in order for `compress` to be compatible with JAX transformations like [`jit()`](jax.jit.html#jax.jit "jax.jit") or [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap").

- **fill_value** (*ArrayLike*) – if `size` is specified, fill padded entries with this value (default: 0).

- **out** (*None*) – not implemented by JAX.

Returns:  
An array of dimension `a.ndim`, compressed along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.extract()`](jax.numpy.extract.html#jax.numpy.extract "jax.numpy.extract"): 1D version of `compress`.

- [`jax.Array.compress()`](jax.Array.compress.html#jax.Array.compress "jax.Array.compress"): equivalent functionality as an array method.

Notes

This function does not require strict shape agreement between `condition` and `a`. If `condition.size`` ``>`` ``a.shape[axis]`, then `condition` will be truncated, and if `a.shape[axis]`` ``>`` ``condition.size`, then `a` will be truncated.

Examples

Compressing along the rows of a 2D array:

    >>> a = jnp.array([[1,  2,  3,  4],
    ...                [5,  6,  7,  8],
    ...                [9,  10, 11, 12]])
    >>> condition = jnp.array([True, False, True])
    >>> jnp.compress(condition, a, axis=0)
    Array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]], dtype=int32)

For convenience, you can equivalently use the [`compress()`](jax.Array.compress.html#jax.Array.compress "jax.Array.compress") method of JAX arrays:

    >>> a.compress(condition, axis=0)
    Array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]], dtype=int32)

Note that the condition need not match the shape of the specified axis; here we compress the columns with the length-3 condition. Values beyond the size of the condition are ignored:

    >>> jnp.compress(condition, a, axis=1)
    Array([[ 1,  3],
           [ 5,  7],
           [ 9, 11]], dtype=int32)

The optional `size` argument lets you specify a static output size so that the output is statically-shaped, and so this function can be used with transformations like [`jit()`](jax.jit.html#jax.jit "jax.jit") and [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap"):

    >>> f = lambda c, a: jnp.extract(c, a, size=len(a), fill_value=0)
    >>> mask = (a % 3 == 0)
    >>> jax.vmap(f)(mask, a)
    Array([[ 3,  0,  0,  0],
           [ 6,  0,  0,  0],
           [ 9, 12,  0,  0]], dtype=int32)

[](jax.numpy.ComplexWarning.html "previous page")

previous

jax.numpy.ComplexWarning

[](jax.numpy.concat.html "next page")

next

jax.numpy.concat

Contents

- [`compress()`](#jax.numpy.compress)

By The JAX authors

© Copyright 2024, The JAX Authors.\
