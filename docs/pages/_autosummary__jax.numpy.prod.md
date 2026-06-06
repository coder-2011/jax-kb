- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.prod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.prod.rst "Download source file")
-  .pdf

# jax.numpy.prod

## Contents

- [`prod()`](#jax.numpy.prod)

# jax.numpy.prod[\#](#jax-numpy-prod "Link to this heading")

jax.numpy.prod(*a*, *axis=None*, *dtype=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*, *promote_integers=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L323-L397)[\#](#jax.numpy.prod "Link to this definition")  
Return product of the array elements over a given axis.

JAX implementation of [`numpy.prod()`](https://numpy.org/doc/stable/reference/generated/numpy.prod.html#numpy.prod "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or array, default=None. Axis along which the product to be computed. If None, the product is computed along all the axes.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, Default=None. Initial value for the product.

- **where** (*ArrayLike* *\|* *None*) – int or array, default=None. The elements to be used in the product. Array should be broadcast compatible to the input.

- **promote_integers** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=True. If True, then integer inputs will be promoted to the widest available integer dtype, following numpy’s behavior. If False, the result will have the same dtype as the input. `promote_integers` is ignored if `dtype` is specified.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of the product along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): Compute the sum of array elements over a given axis.

- [`jax.numpy.max()`](jax.numpy.max.html#jax.numpy.max "jax.numpy.max"): Compute the maximum of array elements over given axis.

- [`jax.numpy.min()`](jax.numpy.min.html#jax.numpy.min "jax.numpy.min"): Compute the minimum of array elements over given axis.

Examples

By default, `jnp.prod` computes along all the axes.

    >>> x = jnp.array([[1, 3, 4, 2],
    ...                [5, 2, 1, 3],
    ...                [2, 1, 3, 1]])
    >>> jnp.prod(x)
    Array(4320, dtype=int32)

If `axis=1`, product is computed along axis 1.

    >>> jnp.prod(x, axis=1)
    Array([24, 30,  6], dtype=int32)

If `keepdims=True`, `ndim` of the output is equal to that of the input.

    >>> jnp.prod(x, axis=1, keepdims=True)
    Array([[24],
           [30],
           [ 6]], dtype=int32)

To include only specific elements in the sum, you can use a\`\`where\`\`.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.prod(x, axis=1, keepdims=True, where=where)
    Array([[4],
           [3],
           [6]], dtype=int32)
    >>> where = jnp.array([[False],
    ...                    [False],
    ...                    [False]])
    >>> jnp.prod(x, axis=1, keepdims=True, where=where)
    Array([[1],
           [1],
           [1]], dtype=int32)

[](jax.numpy.printoptions.html "previous page")

previous

jax.numpy.printoptions

[](jax.numpy.promote_types.html "next page")

next

jax.numpy.promote_types

Contents

- [`prod()`](#jax.numpy.prod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
