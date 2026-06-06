- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.trunc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.trunc.rst "Download source file")
-  .pdf

# jax.numpy.trunc

## Contents

- [`trunc()`](#jax.numpy.trunc)

# jax.numpy.trunc[\#](#jax-numpy-trunc "Link to this heading")

jax.numpy.trunc(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L516-L551)[\#](#jax.numpy.trunc "Link to this definition")  
Round input to the nearest integer towards zero.

JAX implementation of `numpy.trunc()`.

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array with same shape and dtype as `x` containing the rounded values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- `jax.numpy.fix()`: Rounds the input to the nearest integer towards zero.

- [`jax.numpy.ceil()`](jax.numpy.ceil.html#jax.numpy.ceil "jax.numpy.ceil"): Rounds the input up to the nearest integer.

- [`jax.numpy.floor()`](jax.numpy.floor.html#jax.numpy.floor "jax.numpy.floor"): Rounds the input down to the nearest integer.

Examples

    >>> key = jax.random.key(42)
    >>> x = jax.random.uniform(key, (3, 3), minval=-10, maxval=10)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...     print(x)
    [[-0.23  3.6   2.33]
     [ 1.22 -0.99  1.72]
     [-8.5   5.5   3.98]]
    >>> jnp.trunc(x)
    Array([[-0.,  3.,  2.],
           [ 1., -0.,  1.],
           [-8.,  5.,  3.]], dtype=float32)

[](jax.numpy.true_divide.html "previous page")

previous

jax.numpy.true_divide

[](jax.numpy.ufunc.html "next page")

next

jax.numpy.ufunc

Contents

- [`trunc()`](#jax.numpy.trunc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
