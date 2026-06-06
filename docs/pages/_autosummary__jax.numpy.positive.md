- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.positive

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.positive.rst "Download source file")
-  .pdf

# jax.numpy.positive

## Contents

- [`positive()`](#jax.numpy.positive)

# jax.numpy.positive[\#](#jax-numpy-positive "Link to this heading")

jax.numpy.positive(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L236-L283)[\#](#jax.numpy.positive "Link to this definition")  
Return element-wise positive values of the input.

JAX implementation of [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html#numpy.positive "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar

Returns:  
An array of same shape and dtype as `x` containing `+x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.positive` is equivalent to `x.copy()` and is defined only for the types that support arithmetic operations.

See also

- [`jax.numpy.negative()`](jax.numpy.negative.html#jax.numpy.negative "jax.numpy.negative"): Returns element-wise negative values of the input.

- [`jax.numpy.sign()`](jax.numpy.sign.html#jax.numpy.sign "jax.numpy.sign"): Returns element-wise indication of sign of the input.

Examples

For real-valued inputs:

    >>> x = jnp.array([-5, 4, 7., -9.5])
    >>> jnp.positive(x)
    Array([-5. ,  4. ,  7. , -9.5], dtype=float32)
    >>> x.copy()
    Array([-5. ,  4. ,  7. , -9.5], dtype=float32)

For complex inputs:

    >>> x1 = jnp.array([1-2j, -3+4j, 5-6j])
    >>> jnp.positive(x1)
    Array([ 1.-2.j, -3.+4.j,  5.-6.j], dtype=complex64)
    >>> x1.copy()
    Array([ 1.-2.j, -3.+4.j,  5.-6.j], dtype=complex64)

For uint32:

    >>> x2 = jnp.array([6, 0, -4]).astype(jnp.uint32)
    >>> x2
    Array([         6,          0, 4294967292], dtype=uint32)
    >>> jnp.positive(x2)
    Array([         6,          0, 4294967292], dtype=uint32)

[](jax.numpy.polyval.html "previous page")

previous

jax.numpy.polyval

[](jax.numpy.pow.html "next page")

next

jax.numpy.pow

Contents

- [`positive()`](#jax.numpy.positive)

By The JAX authors

© Copyright 2024, The JAX Authors.\
