- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.lcm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.lcm.rst "Download source file")
-  .pdf

# jax.numpy.lcm

## Contents

- [`lcm()`](#jax.numpy.lcm)

# jax.numpy.lcm[\#](#jax-numpy-lcm "Link to this heading")

jax.numpy.lcm(*x1*, *x2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8849-L8895)[\#](#jax.numpy.lcm "Link to this definition")  
Compute the least common multiple of two arrays.

JAX implementation of `numpy.lcm()`.

Parameters:  
- **x1** (*ArrayLike*) – First input array. The elements must have integer dtype.

- **x2** (*ArrayLike*) – Second input array. The elements must have integer dtype.

Returns:  
An array containing the least common multiple of the corresponding elements from the absolute values of x1 and x2.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.gcd()`](jax.numpy.gcd.html#jax.numpy.gcd "jax.numpy.gcd"): compute the greatest common divisor of two arrays.

Examples

Scalar inputs:

    >>> jnp.lcm(12, 18)
    Array(36, dtype=int32, weak_type=True)

Array inputs:

    >>> x1 = jnp.array([12, 18, 24])
    >>> x2 = jnp.array([5, 10, 15])
    >>> jnp.lcm(x1, x2)
    Array([ 60,  90, 120], dtype=int32)

Broadcasting:

    >>> x1 = jnp.array([12])
    >>> x2 = jnp.array([6, 9, 12])
    >>> jnp.lcm(x1, x2)
    Array([12, 36, 12], dtype=int32)

[](jax.numpy.kron.html "previous page")

previous

jax.numpy.kron

[](jax.numpy.ldexp.html "next page")

next

jax.numpy.ldexp

Contents

- [`lcm()`](#jax.numpy.lcm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
