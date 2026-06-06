- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nonzero

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nonzero.rst "Download source file")
-  .pdf

# jax.numpy.nonzero

## Contents

- [`nonzero()`](#jax.numpy.nonzero)

# jax.numpy.nonzero[\#](#jax-numpy-nonzero "Link to this heading")

jax.numpy.nonzero(*a*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3610-L3717)[\#](#jax.numpy.nonzero "Link to this definition")  
Return indices of nonzero elements of an array.

JAX implementation of [`numpy.nonzero()`](https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero "(in NumPy v2.4)").

Because the size of the output of `nonzero` is data-dependent, the function is not compatible with JIT and other transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.nonzero` to be used within JAX’s transformations.

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional static integer specifying the number of nonzero entries to return. If there are more nonzero elements than the specified `size`, then indices will be truncated at the end. If there are fewer nonzero elements than the specified size, then indices will be padded with `fill_value`, which defaults to zero.

- **fill_value** (*None* *\|* *ArrayLike* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[ArrayLike,* *...\]*) – optional padding value when `size` is specified. Defaults to 0.

Returns:  
Tuple of JAX Arrays of length `a.ndim`, containing the indices of each nonzero value.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

- [`jax.numpy.flatnonzero()`](jax.numpy.flatnonzero.html#jax.numpy.flatnonzero "jax.numpy.flatnonzero")

- [`jax.numpy.where()`](jax.numpy.where.html#jax.numpy.where "jax.numpy.where")

Examples

One-dimensional array returns a length-1 tuple of indices:

    >>> x = jnp.array([0, 5, 0, 6, 0, 7])
    >>> jnp.nonzero(x)
    (Array([1, 3, 5], dtype=int32),)

Two-dimensional array returns a length-2 tuple of indices:

    >>> x = jnp.array([[0, 5, 0],
    ...                [6, 0, 7]])
    >>> jnp.nonzero(x)
    (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))

In either case, the resulting tuple of indices can be used directly to extract the nonzero values:

    >>> indices = jnp.nonzero(x)
    >>> x[indices]
    Array([5, 6, 7], dtype=int32)

The output of `nonzero` has a dynamic shape, because the number of returned indices depends on the contents of the input array. As such, it is incompatible with JIT and other JAX transformations:

    >>> x = jnp.array([0, 5, 0, 6, 0, 7])
    >>> jax.jit(jnp.nonzero)(x)  
    Traceback (most recent call last):
      ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[].
    The size argument of jnp.nonzero must be statically specified to use jnp.nonzero within JAX transformations.

This can be addressed by passing a static `size` parameter to specify the desired output shape:

    >>> nonzero_jit = jax.jit(jnp.nonzero, static_argnames='size')
    >>> nonzero_jit(x, size=3)
    (Array([1, 3, 5], dtype=int32),)

If `size` does not match the true size, the result will be either truncated or padded:

    >>> nonzero_jit(x, size=2)  # size < 3: indices are truncated
    (Array([1, 3], dtype=int32),)
    >>> nonzero_jit(x, size=5)  # size > 3: indices are padded with zeros.
    (Array([1, 3, 5, 0, 0], dtype=int32),)

You can specify a custom fill value for the padding using the `fill_value` argument:

    >>> nonzero_jit(x, size=5, fill_value=len(x))
    (Array([1, 3, 5, 6, 6], dtype=int32),)

[](jax.numpy.nextafter.html "previous page")

previous

jax.numpy.nextafter

[](jax.numpy.not_equal.html "next page")

next

jax.numpy.not_equal

Contents

- [`nonzero()`](#jax.numpy.nonzero)

By The JAX authors

© Copyright 2024, The JAX Authors.\
