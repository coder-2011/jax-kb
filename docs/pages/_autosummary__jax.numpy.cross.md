- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cross

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cross.rst "Download source file")
-  .pdf

# jax.numpy.cross

## Contents

- [`cross()`](#jax.numpy.cross)

# jax.numpy.cross[\#](#jax-numpy-cross "Link to this heading")

jax.numpy.cross(*a*, *b*, *axisa=-1*, *axisb=-1*, *axisc=-1*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7941-L8040)[\#](#jax.numpy.cross "Link to this definition")  
Compute the (batched) cross product of two arrays.

JAX implementation of [`numpy.cross()`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html#numpy.cross "(in NumPy v2.4)").

This computes the 2-dimensional or 3-dimensional cross product,

\\c = a \times b\\

In 3 dimensions, `c` is a length-3 array. In 2 dimensions, `c` is a scalar.

Parameters:  
- **a** – N-dimensional array. `a.shape[axisa]` indicates the dimension of the cross product, and must be 2 or 3.

- **b** – N-dimensional array. Must have `b.shape[axisb]`` ``==`` ``a.shape[axisb]`, and other dimensions of `a` and `b` must be broadcast compatible.

- **axisa** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – specicy the axis of `a` along which to compute the cross product.

- **axisb** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – specicy the axis of `b` along which to compute the cross product.

- **axisc** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – specicy the axis of `c` along which the cross product result will be stored.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, this overrides `axisa`, `axisb`, and `axisc` with a single value.

Returns:  
The array `c` containing the (batched) cross product of `a` and `b` along the specified axes.

See also

- [`jax.numpy.linalg.cross()`](jax.numpy.linalg.cross.html#jax.numpy.linalg.cross "jax.numpy.linalg.cross"): an array API compatible function for computing cross products over 3-vectors.

Examples

A 2-dimensional cross product returns a scalar:

    >>> a = jnp.array([1, 2])
    >>> b = jnp.array([3, 4])
    >>> jnp.cross(a, b)
    Array(-2, dtype=int32)

A 3-dimensional cross product returns a length-3 vector:

    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.cross(a, b)
    Array([-3,  6, -3], dtype=int32)

With multi-dimensional inputs, the cross-product is computed along the last axis by default. Here’s a batched 3-dimensional cross product, operating on the rows of the inputs:

    >>> a = jnp.array([[1, 2, 3],
    ...                [3, 4, 3]])
    >>> b = jnp.array([[2, 3, 2],
    ...                [4, 5, 6]])
    >>> jnp.cross(a, b)
    Array([[-5,  4, -1],
           [ 9, -6, -1]], dtype=int32)

Specifying axis=0 makes this a batched 2-dimensional cross product, operating on the columns of the inputs:

    >>> jnp.cross(a, b, axis=0)
    Array([-2, -2, 12], dtype=int32)

Equivalently, we can independently specify the axis of the inputs `a` and `b` and the output `c`:

    >>> jnp.cross(a, b, axisa=0, axisb=0, axisc=0)
    Array([-2, -2, 12], dtype=int32)

[](jax.numpy.cov.html "previous page")

previous

jax.numpy.cov

[](jax.numpy.csingle.html "next page")

next

jax.numpy.csingle

Contents

- [`cross()`](#jax.numpy.cross)

By The JAX authors

© Copyright 2024, The JAX Authors.\
