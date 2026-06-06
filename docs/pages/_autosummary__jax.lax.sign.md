- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sign

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sign.rst "Download source file")
-  .pdf

# jax.lax.sign

## Contents

- [`sign()`](#jax.lax.sign)

# jax.lax.sign[\#](#jax-lax-sign "Link to this heading")

jax.lax.sign(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L265-L307)[\#](#jax.lax.sign "Link to this definition")  
Elementwise sign.

This function lowers directly to the [stablehlo.sign](https://openxla.org/stablehlo/spec#sign) operation.

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
Array of same shape and dtype as `x`, containing the sign of the value, as defined in Notes below.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

For floating-point inputs, returns

\\\begin{split} \mathrm{sign}(x) = \begin{cases} -1 & x \< 0\\ -0 & x = -0\\ \mathit{NaN} & x = \mathit{NaN}\\ +0 & x = +0\\ 1 & x \> 0 \end{cases}\end{split}\\

For signed integer inputs, returns

\\\begin{split}\mathrm{sign}(x) = \begin{cases} -1 & x \< 0\\ 0 & x = 0\\ 1 & x \> 0 \end{cases}\end{split}\\

For complex inputs, returns the complex phase, i.e. \\\mathrm{sign}(x) = x / \|x\|\\.

[](jax.lax.shift_right_logical.html "previous page")

previous

jax.lax.shift_right_logical

[](jax.lax.sin.html "next page")

next

jax.lax.sin

Contents

- [`sign()`](#jax.lax.sign)

By The JAX authors

© Copyright 2024, The JAX Authors.\
