- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.negative

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.negative.rst "Download source file")
-  .pdf

# jax.numpy.negative

## Contents

- [`negative`](#jax.numpy.negative)

# jax.numpy.negative[\#](#jax-numpy-negative "Link to this heading")

jax.numpy.negative *= \<jnp.ufunc 'negative'\>*[\#](#jax.numpy.negative "Link to this definition")  
Return element-wise negative values of the input.

JAX implementation of [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html#numpy.negative "(in NumPy v2.4)").

Parameters:  
- **x** – input array or scalar.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
An array with same shape and dtype as `x` containing `-x`.

Return type:  
Any

See also

- [`jax.numpy.positive()`](jax.numpy.positive.html#jax.numpy.positive "jax.numpy.positive"): Returns element-wise positive values of the input.

- [`jax.numpy.sign()`](jax.numpy.sign.html#jax.numpy.sign "jax.numpy.sign"): Returns element-wise indication of sign of the input.

Note

`jnp.negative`, when applied over `unsigned`` ``integer`, produces the result of their two’s complement negation, which typically results in unexpected large positive values due to integer underflow.

Examples

For real-valued inputs:

    >>> x = jnp.array([0., -3., 7])
    >>> jnp.negative(x)
    Array([-0.,  3., -7.], dtype=float32)

For complex inputs:

    >>> x1 = jnp.array([1-2j, -3+4j, 5-6j])
    >>> jnp.negative(x1)
    Array([-1.+2.j,  3.-4.j, -5.+6.j], dtype=complex64)

For unit32:

    >>> x2 = jnp.array([5, 0, -7]).astype(jnp.uint32)
    >>> x2
    Array([         5,          0, 4294967289], dtype=uint32)
    >>> jnp.negative(x2)
    Array([4294967291,          0,          7], dtype=uint32)

[](jax.numpy.ndim.html "previous page")

previous

jax.numpy.ndim

[](jax.numpy.nextafter.html "next page")

next

jax.numpy.nextafter

Contents

- [`negative`](#jax.numpy.negative)

By The JAX authors

© Copyright 2024, The JAX Authors.\
