- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.multi_dot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.multi_dot.rst "Download source file")
-  .pdf

# jax.numpy.linalg.multi_dot

## Contents

- [`multi_dot()`](#jax.numpy.linalg.multi_dot)

# jax.numpy.linalg.multi_dot[\#](#jax-numpy-linalg-multi-dot "Link to this heading")

jax.numpy.linalg.multi_dot(*arrays*, *\**, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L2112-L2202)[\#](#jax.numpy.linalg.multi_dot "Link to this definition")  
Efficiently compute matrix products between a sequence of arrays.

JAX implementation of [`numpy.linalg.multi_dot()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot "(in NumPy v2.4)").

JAX internally uses the opt_einsum library to compute the most efficient operation order.

Parameters:  
- **arrays** (*Sequence\[ArrayLike\]*) – sequence of arrays. All must be two-dimensional, except the first and last which may be one-dimensional.

- **precision** (*lax.PrecisionLike*) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`).

Returns:  
an array representing the equivalent of `reduce(jnp.matmul,`` ``arrays)`, but evaluated in the optimal order.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

This function exists because the cost of computing sequences of matmul operations can differ vastly depending on the order in which the operations are evaluated. For a single matmul, the number of floating point operations (flops) required to compute a matrix product can be approximated this way:

    >>> def approx_flops(x, y):
    ...   # for 2D x and y, with x.shape[1] == y.shape[0]
    ...   return 2 * x.shape[0] * x.shape[1] * y.shape[1]

Suppose we have three matrices that we’d like to multiply in sequence:

    >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    >>> x = jax.random.normal(key1, shape=(200, 5))
    >>> y = jax.random.normal(key2, shape=(5, 100))
    >>> z = jax.random.normal(key3, shape=(100, 10))

Because of associativity of matrix products, there are two orders in which we might evaluate the product `x`` ``@`` ``y`` ``@`` ``z`, and both produce equivalent outputs up to floating point precision:

    >>> result1 = (x @ y) @ z
    >>> result2 = x @ (y @ z)
    >>> jnp.allclose(result1, result2, atol=1E-4)
    Array(True, dtype=bool)

But the computational cost of these differ greatly:

    >>> print("(x @ y) @ z flops:", approx_flops(x, y) + approx_flops(x @ y, z))
    (x @ y) @ z flops: 600000
    >>> print("x @ (y @ z) flops:", approx_flops(y, z) + approx_flops(x, y @ z))
    x @ (y @ z) flops: 30000

The second approach is about 20x more efficient in terms of estimated flops!

`multi_dot` is a function that will automatically choose the fastest computational path for such problems:

    >>> result3 = jnp.linalg.multi_dot([x, y, z])
    >>> jnp.allclose(result1, result3, atol=1E-4)
    Array(True, dtype=bool)

We can use JAX’s [Ahead-of-time lowering and compilation](../aot.html#ahead-of-time-lowering) tools to estimate the total flops of each approach, and confirm that `multi_dot` is choosing the more efficient option:

    >>> jax.jit(lambda x, y, z: (x @ y) @ z).lower(x, y, z).cost_analysis()['flops']
    600000.0
    >>> jax.jit(lambda x, y, z: x @ (y @ z)).lower(x, y, z).cost_analysis()['flops']
    30000.0
    >>> jax.jit(jnp.linalg.multi_dot).lower([x, y, z]).cost_analysis()['flops']
    30000.0

[](jax.numpy.linalg.matrix_transpose.html "previous page")

previous

jax.numpy.linalg.matrix_transpose

[](jax.numpy.linalg.norm.html "next page")

next

jax.numpy.linalg.norm

Contents

- [`multi_dot()`](#jax.numpy.linalg.multi_dot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
