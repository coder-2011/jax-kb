- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ceil

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ceil.rst "Download source file")
-  .pdf

# jax.numpy.ceil

## Contents

- [`ceil()`](#jax.numpy.ceil)

# jax.numpy.ceil[\#](#jax-numpy-ceil "Link to this heading")

jax.numpy.ceil(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L375-L412)[\#](#jax.numpy.ceil "Link to this definition")  
Round input to the nearest integer upwards.

JAX implementation of [`numpy.ceil`](https://numpy.org/doc/stable/reference/generated/numpy.ceil.html#numpy.ceil "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar. Must not have complex dtype.

Returns:  
An array with same shape and dtype as `x` containing the values rounded to the nearest integer that is greater than or equal to the value itself.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- `jax.numpy.fix()`: Rounds the input to the nearest integer towards zero.

- [`jax.numpy.trunc()`](jax.numpy.trunc.html#jax.numpy.trunc "jax.numpy.trunc"): Rounds the input to the nearest integer towards zero.

- [`jax.numpy.floor()`](jax.numpy.floor.html#jax.numpy.floor "jax.numpy.floor"): Rounds the input down to the nearest integer.

Examples

    >>> key = jax.random.key(1)
    >>> x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(x)
    [[-0.61  0.34 -0.54]
     [-0.62  3.97  0.59]
     [ 4.84  3.42 -1.14]]
    >>> jnp.ceil(x)
    Array([[-0.,  1., -0.],
           [-0.,  4.,  1.],
           [ 5.,  4., -1.]], dtype=float32)

[](jax.numpy.cdouble.html "previous page")

previous

jax.numpy.cdouble

[](jax.numpy.character.html "next page")

next

jax.numpy.character

Contents

- [`ceil()`](#jax.numpy.ceil)

By The JAX authors

© Copyright 2024, The JAX Authors.\
