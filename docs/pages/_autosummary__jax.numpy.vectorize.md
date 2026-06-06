- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vectorize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vectorize.rst "Download source file")
-  .pdf

# jax.numpy.vectorize

## Contents

- [`vectorize()`](#jax.numpy.vectorize)

# jax.numpy.vectorize[\#](#jax-numpy-vectorize "Link to this heading")

jax.numpy.vectorize(*pyfunc*, *\**, *excluded=frozenset({})*, *signature=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/vectorize.py#L189-L359)[\#](#jax.numpy.vectorize "Link to this definition")  
Define a vectorized function with broadcasting.

[`vectorize()`](#jax.numpy.vectorize "jax.numpy.vectorize") is a convenience wrapper for defining vectorized functions with broadcasting, in the style of NumPy’s [generalized universal functions](https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html). It allows for defining functions that are automatically repeated across any leading dimensions, without the implementation of the function needing to be concerned about how to handle higher dimensional inputs.

[`jax.numpy.vectorize()`](#jax.numpy.vectorize "jax.numpy.vectorize") has the same interface as [`numpy.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html#numpy.vectorize "(in NumPy v2.4)"), but it is syntactic sugar for an auto-batching transformation (`vmap()`) rather than a Python loop. This should be considerably more efficient, but the implementation must be written in terms of functions that act on JAX arrays.

Parameters:  
- **pyfunc** – function to vectorize.

- **excluded** – optional set of integers representing positional arguments for which the function will not be vectorized. These will be passed directly to `pyfunc` unmodified.

- **signature** – optional generalized universal function signature, e.g., `(m,n),(n)->(m)` for vectorized matrix-vector multiplication. If provided, `pyfunc` will be called with (and expected to return) arrays with shapes given by the size of corresponding core dimensions. By default, pyfunc is assumed to take scalar arrays as input, and if `signature` is `None`, `pyfunc` can produce outputs of any shape.

Returns:  
Vectorized version of the given function.

Examples

Here are a few examples of how one could write vectorized linear algebra routines using [`vectorize()`](#jax.numpy.vectorize "jax.numpy.vectorize"):

    >>> from functools import partial

    >>> @partial(jnp.vectorize, signature='(k),(k)->(k)')
    ... def cross_product(a, b):
    ...   assert a.shape == b.shape and a.ndim == b.ndim == 1
    ...   return jnp.array([a[1] * b[2] - a[2] * b[1],
    ...                     a[2] * b[0] - a[0] * b[2],
    ...                     a[0] * b[1] - a[1] * b[0]])

    >>> @partial(jnp.vectorize, signature='(n,m),(m)->(n)')
    ... def matrix_vector_product(matrix, vector):
    ...   assert matrix.ndim == 2 and matrix.shape[1:] == vector.shape
    ...   return matrix @ vector

These functions are only written to handle 1D or 2D arrays (the `assert` statements will never be violated), but with vectorize they support arbitrary dimensional inputs with NumPy style broadcasting, e.g.,

    >>> cross_product(jnp.ones(3), jnp.ones(3)).shape
    (3,)
    >>> cross_product(jnp.ones((2, 3)), jnp.ones(3)).shape
    (2, 3)
    >>> cross_product(jnp.ones((1, 2, 3)), jnp.ones((2, 1, 3))).shape
    (2, 2, 3)
    >>> matrix_vector_product(jnp.ones(3), jnp.ones(3))  
    Traceback (most recent call last):
    ValueError: input with shape (3,) does not have enough dimensions for all
    core dimensions ('n', 'k') on vectorized function with excluded=frozenset()
    and signature='(n,k),(k)->(k)'
    >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones(3)).shape
    (2,)
    >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones((4, 3))).shape
    (4, 2)

Note that this has different semantics than jnp.matmul:

    >>> jnp.matmul(jnp.ones((2, 3)), jnp.ones((4, 3)))  
    Traceback (most recent call last):
    TypeError: dot_general requires contracting dimensions to have the same shape, got [3] and [4].

[](jax.numpy.vecmat.html "previous page")

previous

jax.numpy.vecmat

[](jax.numpy.vsplit.html "next page")

next

jax.numpy.vsplit

Contents

- [`vectorize()`](#jax.numpy.vectorize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
