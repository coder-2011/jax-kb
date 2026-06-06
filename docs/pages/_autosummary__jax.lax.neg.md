- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.neg

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.neg.rst "Download source file")
-  .pdf

# jax.lax.neg

## Contents

- [`neg()`](#jax.lax.neg)

# jax.lax.neg[\#](#jax-lax-neg "Link to this heading")

jax.lax.neg(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L245-L264)[\#](#jax.lax.neg "Link to this definition")  
Elementwise negation: \\-x\\.

This function lowers directly to the [stablehlo.negate](https://openxla.org/stablehlo/spec#negate) operation.

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
Array of same shape and dtype as `x`, containing the element-wise negative.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

For unsigned integer inputs, this function returns `2`` ``**`` ``nbits`` ``-`` ``x`, where `nbits` is the number of bits in the integer representation.

[](jax.lax.ne.html "previous page")

previous

jax.lax.ne

[](jax.lax.nextafter.html "next page")

next

jax.lax.nextafter

Contents

- [`neg()`](#jax.lax.neg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
