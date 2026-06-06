- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isscalar

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isscalar.rst "Download source file")
-  .pdf

# jax.numpy.isscalar

## Contents

- [`isscalar()`](#jax.numpy.isscalar)

# jax.numpy.isscalar[\#](#jax-numpy-isscalar "Link to this heading")

jax.numpy.isscalar(*element*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L373-L470)[\#](#jax.numpy.isscalar "Link to this definition")  
Return True if the input is a scalar.

JAX implementation of [`numpy.isscalar()`](https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html#numpy.isscalar "(in NumPy v2.4)"). JAX’s implementation differs from NumPy’s in that it considers zero-dimensional arrays to be scalars; see the *Note* below for more details.

Parameters:  
**element** (*Any*) – input object to check; any type is valid input.

Returns:  
True if `element` is a scalar value or an array-like object with zero dimensions, False otherwise.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

Note

JAX and NumPy differ in their representation of scalar values. NumPy has special scalar objects (e.g. `np.int32(0)`) which are distinct from zero-dimensional arrays (e.g. `np.array(0)`), and [`numpy.isscalar()`](https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html#numpy.isscalar "(in NumPy v2.4)") returns `True` for the former and `False` for the latter.

JAX does not define special scalar objects, but rather represents scalars as zero-dimensional arrays. As such, [`jax.numpy.isscalar()`](#jax.numpy.isscalar "jax.numpy.isscalar") returns `True` for both scalar objects (e.g. `0.0` or `np.float32(0.0)`) and array-like objects with zero dimensions (e.g. `jnp.array(0.0)`, `np.array(0.0)`).

One reason for the different conventions in `isscalar` is to maintain JIT-invariance: i.e. the property that the result of a function should not change when it is JIT-compiled. Because scalar inputs are cast to zero-dimensional JAX arrays at JIT boundaries, the semantics of [`numpy.isscalar()`](https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html#numpy.isscalar "(in NumPy v2.4)") are such that the result changes under JIT:

    >>> np.isscalar(1.0)
    True
    >>> jax.jit(np.isscalar)(1.0)
    Array(False, dtype=bool)

By treating zero-dimensional arrays as scalars, [`jax.numpy.isscalar()`](#jax.numpy.isscalar "jax.numpy.isscalar") avoids this issue:

    >>> jnp.isscalar(1.0)
    True
    >>> jax.jit(jnp.isscalar)(1.0)
    Array(True, dtype=bool)

Examples

In JAX, both scalars and zero-dimensional array-like objects are considered scalars:

    >>> jnp.isscalar(1.0)
    True
    >>> jnp.isscalar(1 + 1j)
    True
    >>> jnp.isscalar(jnp.array(1))  # zero-dimensional JAX array
    True
    >>> jnp.isscalar(jnp.int32(1))  # JAX scalar constructor
    True
    >>> jnp.isscalar(np.array(1.0))  # zero-dimensional NumPy array
    True
    >>> jnp.isscalar(np.int32(1))  # NumPy scalar type
    True

Arrays with one or more dimension are not considered scalars:

    >>> jnp.isscalar(jnp.array([1]))
    False
    >>> jnp.isscalar(np.array([1]))
    False

Compare this to [`numpy.isscalar()`](https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html#numpy.isscalar "(in NumPy v2.4)"), which returns `True` for scalar-typed objects, and `False` for *all* arrays, even those with zero dimensions:

    >>> np.isscalar(np.int32(1))  # scalar object
    True
    >>> np.isscalar(np.array(1))  # zero-dimensional array
    False

In JAX, as in NumPy, objects which are not array-like are not considered scalars:

    >>> jnp.isscalar(None)
    False
    >>> jnp.isscalar([1])
    False
    >>> jnp.isscalar(())
    False
    >>> jnp.isscalar(slice(10))
    False

[](jax.numpy.isrealobj.html "previous page")

previous

jax.numpy.isrealobj

[](jax.numpy.issubdtype.html "next page")

next

jax.numpy.issubdtype

Contents

- [`isscalar()`](#jax.numpy.isscalar)

By The JAX authors

© Copyright 2024, The JAX Authors.\
