- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.trim_zeros

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.trim_zeros.rst "Download source file")
-  .pdf

# jax.numpy.trim_zeros

## Contents

- [`trim_zeros()`](#jax.numpy.trim_zeros)

# jax.numpy.trim_zeros[\#](#jax-numpy-trim-zeros "Link to this heading")

jax.numpy.trim_zeros(*filt*, *trim='fb'*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7436-L7509)[\#](#jax.numpy.trim_zeros "Link to this definition")  
Trim leading and/or trailing zeros of the input array.

JAX implementation of [`numpy.trim_zeros()`](https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros "(in NumPy v2.4)").

Parameters:  
- **filt** (*ArrayLike*) – N-dimensional input array.

- **trim** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  string, optional, default = `fb`. Specifies from which end the input is trimmed.

  - `f` - trims only the leading zeros.

  - `b` - trims only the trailing zeros.

  - `fb` - trims both leading and trailing zeros.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional axis or axes along which to trim. If not specified, trim along all axes of the array.

Returns:  
An array containing the trimmed input with same dtype as `filt`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

One-dimensional input:

    >>> x = jnp.array([0, 0, 2, 0, 1, 4, 3, 0, 0, 0])
    >>> jnp.trim_zeros(x)
    Array([2, 0, 1, 4, 3], dtype=int32)
    >>> jnp.trim_zeros(x, trim='f')
    Array([2, 0, 1, 4, 3, 0, 0, 0], dtype=int32)
    >>> jnp.trim_zeros(x, trim='b')
    Array([0, 0, 2, 0, 1, 4, 3], dtype=int32)

Two-dimensional input:

    >>> x = jnp.zeros((4, 5)).at[1:3, 1:4].set(1)
    >>> x
    Array([[0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0.]], dtype=float32)
    >>> jnp.trim_zeros(x)
    Array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)
    >>> jnp.trim_zeros(x, trim='f')
    Array([[1., 1., 1., 0.],
           [1., 1., 1., 0.],
           [0., 0., 0., 0.]], dtype=float32)
    >>> jnp.trim_zeros(x, axis=0)
    Array([[0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.]], dtype=float32)
    >>> jnp.trim_zeros(x, axis=1)
    Array([[0., 0., 0.],
           [1., 1., 1.],
           [1., 1., 1.],
           [0., 0., 0.]], dtype=float32)

[](jax.numpy.tril_indices_from.html "previous page")

previous

jax.numpy.tril_indices_from

[](jax.numpy.triu.html "next page")

next

jax.numpy.triu

Contents

- [`trim_zeros()`](#jax.numpy.trim_zeros)

By The JAX authors

© Copyright 2024, The JAX Authors.\
