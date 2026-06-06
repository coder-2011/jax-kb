- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.shift_left

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.shift_left.rst "Download source file")
-  .pdf

# jax.lax.shift_left

## Contents

- [`shift_left()`](#jax.lax.shift_left)

# jax.lax.shift_left[\#](#jax-lax-shift-left "Link to this heading")

jax.lax.shift_left(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1352-L1377)[\#](#jax.lax.shift_left "Link to this definition")  
Elementwise left shift: \\x \ll y\\.

This function lowers directly to the [stablehlo.shift_left](https://openxla.org/stablehlo/spec#shift_left) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the element-wise left shift of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.left_shift()`](jax.numpy.left_shift.html#jax.numpy.left_shift "jax.numpy.left_shift"): NumPy wrapper for this API, also accessible via the `x`` ``<<`` ``y` operator on JAX arrays.

- [`jax.lax.shift_right_arithmetic()`](jax.lax.shift_right_arithmetic.html#jax.lax.shift_right_arithmetic "jax.lax.shift_right_arithmetic"): Elementwise arithmetic right shift.

- [`jax.lax.shift_right_logical()`](jax.lax.shift_right_logical.html#jax.lax.shift_right_logical "jax.lax.shift_right_logical"): Elementwise logical right shift.

[](jax.lax.scatter_sub.html "previous page")

previous

jax.lax.scatter_sub

[](jax.lax.shift_right_arithmetic.html "next page")

next

jax.lax.shift_right_arithmetic

Contents

- [`shift_left()`](#jax.lax.shift_left)

By The JAX authors

© Copyright 2024, The JAX Authors.\
