- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.shift_right_logical

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.shift_right_logical.rst "Download source file")
-  .pdf

# jax.lax.shift_right_logical

## Contents

- [`shift_right_logical()`](#jax.lax.shift_right_logical)

# jax.lax.shift_right_logical[\#](#jax-lax-shift-right-logical "Link to this heading")

jax.lax.shift_right_logical(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1405-L1431)[\#](#jax.lax.shift_right_logical "Link to this definition")  
Elementwise logical right shift: \\x \gg y\\.

This function lowers directly to the [stablehlo.shift_right_logical](https://openxla.org/stablehlo/spec#shift_right_logical) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the element-wise logical right shift of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.right_shift()`](jax.numpy.right_shift.html#jax.numpy.right_shift "jax.numpy.right_shift"): NumPy wrapper for this API when applied to unsigned integers, also accessible via the `x`` ``>>`` ``y` operator on JAX arrays with unsigned integer dtype.

- [`jax.lax.shift_left()`](jax.lax.shift_left.html#jax.lax.shift_left "jax.lax.shift_left"): Elementwise left shift.

- [`jax.lax.shift_right_arithmetic()`](jax.lax.shift_right_arithmetic.html#jax.lax.shift_right_arithmetic "jax.lax.shift_right_arithmetic"): Elementwise arithmetic right shift.

[](jax.lax.shift_right_arithmetic.html "previous page")

previous

jax.lax.shift_right_arithmetic

[](jax.lax.sign.html "next page")

next

jax.lax.sign

Contents

- [`shift_right_logical()`](#jax.lax.shift_right_logical)

By The JAX authors

© Copyright 2024, The JAX Authors.\
