- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logaddexp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logaddexp.rst "Download source file")
-  .pdf

# jax.numpy.logaddexp

## Contents

- [`logaddexp`](#jax.numpy.logaddexp)

# jax.numpy.logaddexp[\#](#jax-numpy-logaddexp "Link to this heading")

jax.numpy.logaddexp *= \<jnp.ufunc 'logaddexp'\>*[\#](#jax.numpy.logaddexp "Link to this definition")  
Compute `log(exp(x1)`` ``+`` ``exp(x2))` avoiding overflow.

JAX implementation of [`numpy.logaddexp`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html#numpy.logaddexp "(in NumPy v2.4)")

Parameters:  
- **x1** – input array

- **x2** – input array

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
array containing the result.

Return type:  
Any

Examples:

    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> result1 = jnp.logaddexp(x1, x2)
    >>> result2 = jnp.log(jnp.exp(x1) + jnp.exp(x2))
    >>> print(jnp.allclose(result1, result2))
    True

[](jax.numpy.log2.html "previous page")

previous

jax.numpy.log2

[](jax.numpy.logaddexp2.html "next page")

next

jax.numpy.logaddexp2

Contents

- [`logaddexp`](#jax.numpy.logaddexp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
