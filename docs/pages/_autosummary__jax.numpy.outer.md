- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.outer

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.outer.rst "Download source file")
-  .pdf

# jax.numpy.outer

## Contents

- [`outer()`](#jax.numpy.outer)

# jax.numpy.outer[\#](#jax-numpy-outer "Link to this heading")

jax.numpy.outer(*a*, *b*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L647-L680)[\#](#jax.numpy.outer "Link to this definition")  
Compute the outer product of two arrays.

JAX implementation of [`numpy.outer()`](https://numpy.org/doc/stable/reference/generated/numpy.outer.html#numpy.outer "(in NumPy v2.4)").

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – first input array, if not 1D it will be flattened.

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – second input array, if not 1D it will be flattened.

- **out** (*None*) – unsupported by JAX.

Returns:  
The outer product of the inputs `a` and `b`. Returned array will be of shape `(a.size,`` ``b.size)`.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.inner()`](jax.numpy.inner.html#jax.numpy.inner "jax.numpy.inner"): compute the inner product of two arrays.

- [`jax.numpy.einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"): Einstein summation.

Examples

    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.outer(a, b)
    Array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]], dtype=int32)

[](jax.numpy.ones_like.html "previous page")

previous

jax.numpy.ones_like

[](jax.numpy.packbits.html "next page")

next

jax.numpy.packbits

Contents

- [`outer()`](#jax.numpy.outer)

By The JAX authors

© Copyright 2024, The JAX Authors.\
