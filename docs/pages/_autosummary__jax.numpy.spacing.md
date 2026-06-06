- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.spacing

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.spacing.rst "Download source file")
-  .pdf

# jax.numpy.spacing

## Contents

- [`spacing()`](#jax.numpy.spacing)

# jax.numpy.spacing[\#](#jax-numpy-spacing "Link to this heading")

jax.numpy.spacing(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1826-L1868)[\#](#jax.numpy.spacing "Link to this definition")  
Return the spacing between `x` and the next adjacent number.

JAX implementation of `numpy.spacing()`.

Parameters:  
**x** (*ArrayLike*) – real-valued array. Integer or boolean types will be cast to float.

Returns:  
Array of same shape as `x` containing spacing between each entry of `x` and its closest adjacent value.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nextafter()`](jax.numpy.nextafter.html#jax.numpy.nextafter "jax.numpy.nextafter"): find the next representable value.

Examples

    >>> x = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype='float32')
    >>> jnp.spacing(x)
    Array([1.4012985e-45, 2.9802322e-08, 5.9604645e-08, 5.9604645e-08,
          1.1920929e-07], dtype=float32)

For `x`` ``=`` ``1`, the spacing is equal to the `eps` value given by [`jax.numpy.finfo`](jax.numpy.finfo.html#jax.numpy.finfo "jax.numpy.finfo"):

    >>> x = jnp.float32(1)
    >>> jnp.spacing(x) == jnp.finfo(x.dtype).eps
    Array(True, dtype=bool)

[](jax.numpy.sort_complex.html "previous page")

previous

jax.numpy.sort_complex

[](jax.numpy.split.html "next page")

next

jax.numpy.split

Contents

- [`spacing()`](#jax.numpy.spacing)

By The JAX authors

© Copyright 2024, The JAX Authors.\
