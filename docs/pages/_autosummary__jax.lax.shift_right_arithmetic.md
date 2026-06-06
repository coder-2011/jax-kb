- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.shift_right_arithmetic

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.shift_right_arithmetic.rst "Download source file")
-  .pdf

# jax.lax.shift_right_arithmetic

## Contents

- [`shift_right_arithmetic()`](#jax.lax.shift_right_arithmetic)

# jax.lax.shift_right_arithmetic[\#](#jax-lax-shift-right-arithmetic "Link to this heading")

jax.lax.shift_right_arithmetic(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1378-L1404)[\#](#jax.lax.shift_right_arithmetic "Link to this definition")  
Elementwise arithmetic right shift: \\x \gg y\\.

This function lowers directly to the [stablehlo.shift_right_arithmetic](https://openxla.org/stablehlo/spec#shift_right_arithmetic) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the element-wise arithmetic right shift of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.right_shift()`](jax.numpy.right_shift.html#jax.numpy.right_shift "jax.numpy.right_shift"): NumPy wrapper for this API when applied to signed integers, also accessible via the `x`` ``>>`` ``y` operator on JAX arrays with signed integer dtype.

- [`jax.lax.shift_left()`](jax.lax.shift_left.html#jax.lax.shift_left "jax.lax.shift_left"): Elementwise left shift.

- [`jax.lax.shift_right_logical()`](jax.lax.shift_right_logical.html#jax.lax.shift_right_logical "jax.lax.shift_right_logical"): Elementwise logical right shift.

[](jax.lax.shift_left.html "previous page")

previous

jax.lax.shift_left

[](jax.lax.shift_right_logical.html "next page")

next

jax.lax.shift_right_logical

Contents

- [`shift_right_arithmetic()`](#jax.lax.shift_right_arithmetic)

By The JAX authors

© Copyright 2024, The JAX Authors.\
