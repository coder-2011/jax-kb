- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fromfunction

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fromfunction.rst "Download source file")
-  .pdf

# jax.numpy.fromfunction

## Contents

- [`fromfunction()`](#jax.numpy.fromfunction)

# jax.numpy.fromfunction[\#](#jax-numpy-fromfunction "Link to this heading")

jax.numpy.fromfunction(*function*, *shape*, *\**, *dtype=\<class 'float'\>*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5601-L5686)[\#](#jax.numpy.fromfunction "Link to this definition")  
Create an array from a function applied over indices.

JAX implementation of [`numpy.fromfunction()`](https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html#numpy.fromfunction "(in NumPy v2.4)"). The JAX implementation differs in that it dispatches via [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"), and so unlike in NumPy the function logically operates on scalar inputs, and need not explicitly handle broadcasted inputs (See *Examples* below).

Parameters:  
- **function** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – a function that takes *N* dynamic scalars and outputs a scalar.

- **shape** (*Any*) – a length-*N* tuple of integers specifying the output shape.

- **dtype** (*DTypeLike*) – optionally specify the dtype of the inputs. Defaults to floating-point.

- **kwargs** – additional keyword arguments are passed statically to `function`.

Returns:  
An array of shape `shape` if `function` returns a scalar, or in general a pytree of arrays with leading dimensions `shape`, as determined by the output of `function`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"): the core transformation that the [`fromfunction()`](#jax.numpy.fromfunction "jax.numpy.fromfunction") API is built on.

Examples

Generate a multiplication table of a given shape:

    >>> jnp.fromfunction(jnp.multiply, shape=(3, 6), dtype=int)
    Array([[ 0,  0,  0,  0,  0,  0],
           [ 0,  1,  2,  3,  4,  5],
           [ 0,  2,  4,  6,  8, 10]], dtype=int32)

When `function` returns a non-scalar the output will have leading dimension of `shape`:

    >>> def f(x):
    ...   return (x + 1) * jnp.arange(3)
    >>> jnp.fromfunction(f, shape=(2,))
    Array([[0., 1., 2.],
           [0., 2., 4.]], dtype=float32)

`function` may return multiple results, in which case each is mapped independently:

    >>> def f(x, y):
    ...   return x + y, x * y
    >>> x_plus_y, x_times_y = jnp.fromfunction(f, shape=(3, 5))
    >>> print(x_plus_y)
    [[0. 1. 2. 3. 4.]
     [1. 2. 3. 4. 5.]
     [2. 3. 4. 5. 6.]]
    >>> print(x_times_y)
    [[0. 0. 0. 0. 0.]
     [0. 1. 2. 3. 4.]
     [0. 2. 4. 6. 8.]]

The JAX implementation differs slightly from NumPy’s implementation. In [`numpy.fromfunction()`](https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html#numpy.fromfunction "(in NumPy v2.4)"), the function is expected to explicitly operate element-wise on the full grid of input values:

    >>> def f(x, y):
    ...   print(f"{x.shape = }\n{y.shape = }")
    ...   return x + y
    ...
    >>> np.fromfunction(f, (2, 3))
    x.shape = (2, 3)
    y.shape = (2, 3)
    array([[0., 1., 2.],
           [1., 2., 3.]])

In [`jax.numpy.fromfunction()`](#jax.numpy.fromfunction "jax.numpy.fromfunction"), the function is vectorized via [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"), and so is expected to operate on scalar values:

    >>> jnp.fromfunction(f, (2, 3))
    x.shape = ()
    y.shape = ()
    Array([[0., 1., 2.],
           [1., 2., 3.]], dtype=float32)

[](jax.numpy.fromfile.html "previous page")

previous

jax.numpy.fromfile

[](jax.numpy.fromiter.html "next page")

next

jax.numpy.fromiter

Contents

- [`fromfunction()`](#jax.numpy.fromfunction)

By The JAX authors

© Copyright 2024, The JAX Authors.\
