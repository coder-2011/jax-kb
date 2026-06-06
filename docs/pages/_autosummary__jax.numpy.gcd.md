- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.gcd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.gcd.rst "Download source file")
-  .pdf

# jax.numpy.gcd

## Contents

- [`gcd()`](#jax.numpy.gcd)

# jax.numpy.gcd[\#](#jax-numpy-gcd "Link to this heading")

jax.numpy.gcd(*x1*, *x2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8802-L8847)[\#](#jax.numpy.gcd "Link to this definition")  
Compute the greatest common divisor of two arrays.

JAX implementation of `numpy.gcd()`.

Parameters:  
- **x1** (*ArrayLike*) – First input array. The elements must have integer dtype.

- **x2** (*ArrayLike*) – Second input array. The elements must have integer dtype.

Returns:  
An array containing the greatest common divisors of the corresponding elements from the absolute values of x1 and x2.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.lcm()`](jax.numpy.lcm.html#jax.numpy.lcm "jax.numpy.lcm"): compute the least common multiple of two arrays.

Examples

Scalar inputs:

    >>> jnp.gcd(12, 18)
    Array(6, dtype=int32, weak_type=True)

Array inputs:

    >>> x1 = jnp.array([12, 18, 24])
    >>> x2 = jnp.array([5, 10, 15])
    >>> jnp.gcd(x1, x2)
    Array([1, 2, 3], dtype=int32)

Broadcasting:

    >>> x1 = jnp.array([12])
    >>> x2 = jnp.array([6, 9, 12])
    >>> jnp.gcd(x1, x2)
    Array([ 6,  3, 12], dtype=int32)

[](jax.numpy.full_like.html "previous page")

previous

jax.numpy.full_like

[](jax.numpy.generic.html "next page")

next

jax.numpy.generic

Contents

- [`gcd()`](#jax.numpy.gcd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
