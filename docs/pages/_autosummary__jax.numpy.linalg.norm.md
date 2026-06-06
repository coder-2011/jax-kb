- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.norm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.norm.rst "Download source file")
-  .pdf

# jax.numpy.linalg.norm

## Contents

- [`norm()`](#jax.numpy.linalg.norm)

# jax.numpy.linalg.norm[\#](#jax-numpy-linalg-norm "Link to this heading")

jax.numpy.linalg.norm(*x*, *ord=None*, *axis=None*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1152-L1285)[\#](#jax.numpy.linalg.norm "Link to this definition")  
Compute the norm of a matrix or vector.

JAX implementation of [`numpy.linalg.norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array for which the norm will be computed.

- **ord** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – specify the kind of norm to take. Default is Frobenius norm for matrices, and the 2-norm for vectors. For other options, see Notes below.

- **axis** (*None* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer or sequence of integers specifying the axes over which the norm will be computed. For a single axis, compute a vector norm. For two axes, compute a matrix norm. Defaults to all axes of `x`.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, the output array will have the same number of dimensions as the input, with the size of reduced axes replaced by `1` (default: False).

Returns:  
array containing the specified norm of x.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The flavor of norm computed depends on the value of `ord` and the number of axes being reduced.

For **vector norms** (i.e. a single axis reduction):

- `ord=None` (default) computes the 2-norm

- `ord=inf` computes `max(abs(x))`

- `ord=-inf` computes min(abs(x))\`\`

- `ord=0` computes `sum(x!=0)`

- for other numerical values, computes `sum(abs(x)`` ``**`` ``ord)**(1/ord)`

For **matrix norms** (i.e. two axes reductions):

- `ord='fro'` or `ord=None` (default) computes the Frobenius norm

- `ord='nuc'` computes the nuclear norm, or the sum of the singular values

- `ord=1` computes `max(abs(x).sum(0))`

- `ord=-1` computes `min(abs(x).sum(0))`

- `ord=2` computes the 2-norm, i.e. the largest singular value

- `ord=-2` computes the smallest singular value

In the special case of `ord=None` and `axis=None`, this function accepts an array of any dimension and computes the vector 2-norm of the flattened array.

Examples

Vector norms:

    >>> x = jnp.array([3., 4., 12.])
    >>> jnp.linalg.norm(x)
    Array(13., dtype=float32)
    >>> jnp.linalg.norm(x, ord=1)
    Array(19., dtype=float32)
    >>> jnp.linalg.norm(x, ord=0)
    Array(3., dtype=float32)

Matrix norms:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 7.]])
    >>> jnp.linalg.norm(x)  # Frobenius norm
    Array(10.198039, dtype=float32)
    >>> jnp.linalg.norm(x, ord='nuc')  # nuclear norm
    Array(10.762535, dtype=float32)
    >>> jnp.linalg.norm(x, ord=1)  # 1-norm
    Array(10., dtype=float32)

Batched vector norm:

    >>> jnp.linalg.norm(x, axis=1)
    Array([3.7416575, 9.486833 ], dtype=float32)

[](jax.numpy.linalg.multi_dot.html "previous page")

previous

jax.numpy.linalg.multi_dot

[](jax.numpy.linalg.outer.html "next page")

next

jax.numpy.linalg.outer

Contents

- [`norm()`](#jax.numpy.linalg.norm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
