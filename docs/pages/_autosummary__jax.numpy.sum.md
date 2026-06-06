- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sum.rst "Download source file")
-  .pdf

# jax.numpy.sum

## Contents

- [`sum()`](#jax.numpy.sum)

# jax.numpy.sum[\#](#jax-numpy-sum "Link to this heading")

jax.numpy.sum(*a*, *axis=None*, *dtype=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*, *promote_integers=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L237-L310)[\#](#jax.numpy.sum "Link to this definition")  
Sum of the elements of the array over a given axis.

JAX implementation of [`numpy.sum()`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or array, default=None. Axis along which the sum to be computed. If None, the sum is computed along all the axes.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **out** (*None*) – Unused by JAX

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, Default=None. Initial value for the sum.

- **where** (*ArrayLike* *\|* *None*) – int or array, default=None. The elements to be used in the sum. Array should be broadcast compatible to the input.

- **promote_integers** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=True. If True, then integer inputs will be promoted to the widest available integer dtype, following numpy’s behavior. If False, the result will have the same dtype as the input. `promote_integers` is ignored if `dtype` is specified.

Returns:  
An array of the sum along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.prod()`](jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod"): Compute the product of array elements over a given axis.

- [`jax.numpy.max()`](jax.numpy.max.html#jax.numpy.max "jax.numpy.max"): Compute the maximum of array elements over given axis.

- [`jax.numpy.min()`](jax.numpy.min.html#jax.numpy.min "jax.numpy.min"): Compute the minimum of array elements over given axis.

Examples

By default, the sum is computed along all the axes.

    >>> x = jnp.array([[1, 3, 4, 2],
    ...                [5, 2, 6, 3],
    ...                [8, 1, 3, 9]])
    >>> jnp.sum(x)
    Array(47, dtype=int32)

If `axis=1`, the sum is computed along axis 1.

    >>> jnp.sum(x, axis=1)
    Array([10, 16, 21], dtype=int32)

If `keepdims=True`, `ndim` of the output is equal to that of the input.

    >>> jnp.sum(x, axis=1, keepdims=True)
    Array([[10],
           [16],
           [21]], dtype=int32)

To include only specific elements in the sum, you can use `where`.

    >>> where=jnp.array([[0, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.sum(x, axis=1, keepdims=True, where=where)
    Array([[ 4],
           [ 9],
           [12]], dtype=int32)
    >>> where=jnp.array([[False],
    ...                  [False],
    ...                  [False]])
    >>> jnp.sum(x, axis=0, keepdims=True, where=where)
    Array([[0, 0, 0, 0]], dtype=int32)

[](jax.numpy.subtract.html "previous page")

previous

jax.numpy.subtract

[](jax.numpy.swapaxes.html "next page")

next

jax.numpy.swapaxes

Contents

- [`sum()`](#jax.numpy.sum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
