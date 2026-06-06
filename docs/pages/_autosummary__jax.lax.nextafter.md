- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.nextafter

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.nextafter.rst "Download source file")
-  .pdf

# jax.lax.nextafter

## Contents

- [`nextafter()`](#jax.lax.nextafter)

# jax.lax.nextafter[\#](#jax-lax-nextafter "Link to this heading")

jax.lax.nextafter(*x1*, *x2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L308-L338)[\#](#jax.lax.nextafter "Link to this definition")  
Returns the next representable value after `x1` in the direction of `x2`.

This function lowers directly to the `chlo.next_after` operation.

Parameters:  
- **x1** (*ArrayLike*) – input arrays. Must have a matching floating-point dtypes. If neither is a scalar, must have the same number of dimensions and be broadcast-compatible.

- **x2** (*ArrayLike*) – input arrays. Must have a matching floating-point dtypes. If neither is a scalar, must have the same number of dimensions and be broadcast-compatible.

Returns:  
Array of the same dtype and broadcasted shape of the inputs, containing the next representable floating-point value after `x1` in the direction of `x2`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

In some environments flush-denormal-to-zero semantics is used. This means that, around zero, this function returns strictly non-zero values which appear as zero in any operations. Consider this example:

    >>> from jax import lax
    >>> lax.nextafter(0.0, 1.0)  # denormal numbers are representable
    Array(1.e-45, dtype=float32, weak_type=True)
    >>> lax.nextafter(0.0, 1.0) * 1  # but are flushed to zero
    Array(0., dtype=float32, weak_type=True)

For the smallest usable (i.e. normal) float, use `tiny` of `jnp.finfo`.

[](jax.lax.neg.html "previous page")

previous

jax.lax.neg

[](jax.lax.optimization_barrier.html "next page")

next

jax.lax.optimization_barrier

Contents

- [`nextafter()`](#jax.lax.nextafter)

By The JAX authors

© Copyright 2024, The JAX Authors.\
