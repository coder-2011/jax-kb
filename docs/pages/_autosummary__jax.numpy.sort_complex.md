- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sort_complex

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sort_complex.rst "Download source file")
-  .pdf

# jax.numpy.sort_complex

## Contents

- [`sort_complex()`](#jax.numpy.sort_complex)

# jax.numpy.sort_complex[\#](#jax-numpy-sort-complex "Link to this heading")

jax.numpy.sort_complex(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/sorting.py#L327-L363)[\#](#jax.numpy.sort_complex "Link to this definition")  
Return a sorted copy of complex array.

JAX implementation of [`numpy.sort_complex()`](https://numpy.org/doc/stable/reference/generated/numpy.sort_complex.html#numpy.sort_complex "(in NumPy v2.4)").

Complex numbers are sorted lexicographically, meaning by their real part first, and then by their imaginary part if real parts are equal.

Parameters:  
**a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – input array. If dtype is not complex, the array will be upcast to complex.

Returns:  
A sorted array of the same shape and complex dtype as the input. If `a` is multi-dimensional, it is sorted along the last axis.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.sort()`](jax.numpy.sort.html#jax.numpy.sort "jax.numpy.sort"): Return a sorted copy of an array.

Examples

    >>> a = jnp.array([1+2j, 2+4j, 3-1j, 2+3j])
    >>> jnp.sort_complex(a)
    Array([1.+2.j, 2.+3.j, 2.+4.j, 3.-1.j], dtype=complex64)

Multi-dimensional arrays are sorted along the last axis:

    >>> a = jnp.array([[5, 3, 4],
    ...                [6, 9, 2]])
    >>> jnp.sort_complex(a)
    Array([[3.+0.j, 4.+0.j, 5.+0.j],
           [2.+0.j, 6.+0.j, 9.+0.j]], dtype=complex64)

[](jax.numpy.sort.html "previous page")

previous

jax.numpy.sort

[](jax.numpy.spacing.html "next page")

next

jax.numpy.spacing

Contents

- [`sort_complex()`](#jax.numpy.sort_complex)

By The JAX authors

© Copyright 2024, The JAX Authors.\
