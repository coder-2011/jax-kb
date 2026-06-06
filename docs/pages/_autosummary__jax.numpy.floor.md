- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.floor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.floor.rst "Download source file")
-  .pdf

# jax.numpy.floor

## Contents

- [`floor()`](#jax.numpy.floor)

# jax.numpy.floor[\#](#jax-numpy-floor "Link to this heading")

jax.numpy.floor(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L336-L373)[\#](#jax.numpy.floor "Link to this definition")  
Round input to the nearest integer downwards.

JAX implementation of [`numpy.floor`](https://numpy.org/doc/stable/reference/generated/numpy.floor.html#numpy.floor "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar. Must not have complex dtype.

Returns:  
An array with same shape and dtype as `x` containing the values rounded to the nearest integer that is less than or equal to the value itself.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- `jax.numpy.fix()`: Rounds the input to the nearest integer towards zero.

- [`jax.numpy.trunc()`](jax.numpy.trunc.html#jax.numpy.trunc "jax.numpy.trunc"): Rounds the input to the nearest integer towards zero.

- [`jax.numpy.ceil()`](jax.numpy.ceil.html#jax.numpy.ceil "jax.numpy.ceil"): Rounds the input up to the nearest integer.

Examples

    >>> key = jax.random.key(42)
    >>> x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(x)
    [[-0.11  1.8   1.16]
     [ 0.61 -0.49  0.86]
     [-4.25  2.75  1.99]]
    >>> jnp.floor(x)
    Array([[-1.,  1.,  1.],
           [ 0., -1.,  0.],
           [-5.,  2.,  1.]], dtype=float32)

[](jax.numpy.floating.html "previous page")

previous

jax.numpy.floating

[](jax.numpy.floor_divide.html "next page")

next

jax.numpy.floor_divide

Contents

- [`floor()`](#jax.numpy.floor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
