- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.expand_dims

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.expand_dims.rst "Download source file")
-  .pdf

# jax.numpy.expand_dims

## Contents

- [`expand_dims()`](#jax.numpy.expand_dims)

# jax.numpy.expand_dims[\#](#jax-numpy-expand-dims "Link to this heading")

jax.numpy.expand_dims(*a*, *axis*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2340-L2409)[\#](#jax.numpy.expand_dims "Link to this definition")  
Insert dimensions of length 1 into array

JAX implementation of [`numpy.expand_dims()`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy.expand_dims "(in NumPy v2.4)"), implemented via [`jax.lax.expand_dims()`](jax.lax.expand_dims.html#jax.lax.expand_dims "jax.lax.expand_dims").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – integer or sequence of integers specifying positions of axes to add.

Returns:  
Copy of `a` with added dimensions.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.expand_dims()`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy.expand_dims "(in NumPy v2.4)"), [`jax.numpy.expand_dims()`](#jax.numpy.expand_dims "jax.numpy.expand_dims") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.numpy.squeeze()`](jax.numpy.squeeze.html#jax.numpy.squeeze "jax.numpy.squeeze"): inverse of this operation, i.e. remove length-1 dimensions.

- [`jax.lax.expand_dims()`](jax.lax.expand_dims.html#jax.lax.expand_dims "jax.lax.expand_dims"): XLA version of this functionality.

Examples

    >>> x = jnp.array([1, 2, 3])
    >>> x.shape
    (3,)

Expand the leading dimension:

    >>> jnp.expand_dims(x, 0)
    Array([[1, 2, 3]], dtype=int32)
    >>> _.shape
    (1, 3)

Expand the trailing dimension:

    >>> jnp.expand_dims(x, 1)
    Array([[1],
           [2],
           [3]], dtype=int32)
    >>> _.shape
    (3, 1)

Expand multiple dimensions:

    >>> jnp.expand_dims(x, (0, 1, 3))
    Array([[[[1],
             [2],
             [3]]]], dtype=int32)
    >>> _.shape
    (1, 1, 3, 1)

Dimensions can also be expanded more succinctly by indexing with `None`:

    >>> x[None]  # equivalent to jnp.expand_dims(x, 0)
    Array([[1, 2, 3]], dtype=int32)
    >>> x[:, None]  # equivalent to jnp.expand_dims(x, 1)
    Array([[1],
           [2],
           [3]], dtype=int32)
    >>> x[None, None, :, None]  # equivalent to jnp.expand_dims(x, (0, 1, 3))
    Array([[[[1],
             [2],
             [3]]]], dtype=int32)

[](jax.numpy.exp2.html "previous page")

previous

jax.numpy.exp2

[](jax.numpy.expm1.html "next page")

next

jax.numpy.expm1

Contents

- [`expand_dims()`](#jax.numpy.expand_dims)

By The JAX authors

© Copyright 2024, The JAX Authors.\
