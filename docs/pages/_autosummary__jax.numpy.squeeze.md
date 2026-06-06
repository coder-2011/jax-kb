- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.squeeze

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.squeeze.rst "Download source file")
-  .pdf

# jax.numpy.squeeze

## Contents

- [`squeeze()`](#jax.numpy.squeeze)

# jax.numpy.squeeze[\#](#jax-numpy-squeeze "Link to this heading")

jax.numpy.squeeze(*a*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2269-L2328)[\#](#jax.numpy.squeeze "Link to this definition")  
Remove one or more length-1 axes from array

JAX implementation of `numpy.sqeeze()`, implemented via [`jax.lax.squeeze()`](jax.lax.squeeze.html#jax.lax.squeeze "jax.lax.squeeze").

Parameters:  
- **a** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or sequence of integers specifying axes to remove. If any specified axis does not have a length of 1, an error is raised. If not specified, squeeze all length-1 axes in `a`.

Returns:  
copy of `a` with length-1 axes removed.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`numpy.squeeze()`](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html#numpy.squeeze "(in NumPy v2.4)"), [`jax.numpy.squeeze()`](#jax.numpy.squeeze "jax.numpy.squeeze") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize-away such copies when possible, so this doesn’t have performance impacts in practice.

See also

- [`jax.numpy.expand_dims()`](jax.numpy.expand_dims.html#jax.numpy.expand_dims "jax.numpy.expand_dims"): the inverse of `squeeze`: add dimensions of length 1.

- [`jax.Array.squeeze()`](jax.Array.squeeze.html#jax.Array.squeeze "jax.Array.squeeze"): equivalent functionality via an array method.

- [`jax.lax.squeeze()`](jax.lax.squeeze.html#jax.lax.squeeze "jax.lax.squeeze"): equivalent XLA API.

- [`jax.numpy.ravel()`](jax.numpy.ravel.html#jax.numpy.ravel "jax.numpy.ravel"): flatten an array into a 1D shape.

- [`jax.numpy.reshape()`](jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape"): general array reshape.

Examples

    >>> x = jnp.array([[[0]], [[1]], [[2]]])
    >>> x.shape
    (3, 1, 1)

Squeeze all length-1 dimensions:

    >>> jnp.squeeze(x)
    Array([0, 1, 2], dtype=int32)
    >>> _.shape
    (3,)

Equivalent while specifying the axes explicitly:

    >>> jnp.squeeze(x, axis=(1, 2))
    Array([0, 1, 2], dtype=int32)

Attempting to squeeze a non-unit axis results in an error:

    >>> jnp.squeeze(x, axis=0)  
    Traceback (most recent call last):
      ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one, got shape=(3, 1, 1) and dimensions=(0,)

For convenience, this functionality is also available via the [`jax.Array.squeeze()`](jax.Array.squeeze.html#jax.Array.squeeze "jax.Array.squeeze") method:

    >>> x.squeeze()
    Array([0, 1, 2], dtype=int32)

[](jax.numpy.square.html "previous page")

previous

jax.numpy.square

[](jax.numpy.stack.html "next page")

next

jax.numpy.stack

Contents

- [`squeeze()`](#jax.numpy.squeeze)

By The JAX authors

© Copyright 2024, The JAX Authors.\
