- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logaddexp2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logaddexp2.rst "Download source file")
-  .pdf

# jax.numpy.logaddexp2

## Contents

- [`logaddexp2`](#jax.numpy.logaddexp2)

# jax.numpy.logaddexp2[\#](#jax-numpy-logaddexp2 "Link to this heading")

jax.numpy.logaddexp2 *= \<jnp.ufunc 'logaddexp2'\>*[\#](#jax.numpy.logaddexp2 "Link to this definition")  
Logarithm of the sum of exponentials of inputs in base-2 avoiding overflow.

JAX implementation of [`numpy.logaddexp2`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html#numpy.logaddexp2 "(in NumPy v2.4)").

Parameters:  
- **x1** – input array or scalar.

- **x2** – input array or scalar. `x1` and `x2` should either have same shape or be broadcast compatible.

- **args** (*ArrayLike*)

- **out** (*None*)

- **where** (*None*)

Returns:  
An array containing the result, \\log_2(2^{x1}+2^{x2})\\, element-wise.

Return type:  
Any

See also

- [`jax.numpy.logaddexp()`](jax.numpy.logaddexp.html#jax.numpy.logaddexp "jax.numpy.logaddexp"): Computes `log(exp(x1)`` ``+`` ``exp(x2))`, element-wise.

- [`jax.numpy.log2()`](jax.numpy.log2.html#jax.numpy.log2 "jax.numpy.log2"): Calculates the base-2 logarithm of `x` element-wise.

Examples

    >>> x1 = jnp.array([[3, -1, 4],
    ...                 [8, 5, -2]])
    >>> x2 = jnp.array([2, 3, -5])
    >>> result1 = jnp.logaddexp2(x1, x2)
    >>> result2 = jnp.log2(jnp.exp2(x1) + jnp.exp2(x2))
    >>> jnp.allclose(result1, result2)
    Array(True, dtype=bool)

[](jax.numpy.logaddexp.html "previous page")

previous

jax.numpy.logaddexp

[](jax.numpy.logical_and.html "next page")

next

jax.numpy.logical_and

Contents

- [`logaddexp2`](#jax.numpy.logaddexp2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
