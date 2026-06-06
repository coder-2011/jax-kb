- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.boxcox1p

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.boxcox1p.rst "Download source file")
-  .pdf

# jax.scipy.special.boxcox1p

## Contents

- [`boxcox1p()`](#jax.scipy.special.boxcox1p)

# jax.scipy.special.boxcox1p[\#](#jax-scipy-special-boxcox1p "Link to this heading")

jax.scipy.special.boxcox1p(*x*, *lmbda*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1077-L1107)[\#](#jax.scipy.special.boxcox1p "Link to this definition")  
Box-Cox transformation of `1`` ``+`` ``x`.

JAX implementation of [`scipy.special.boxcox1p`](https://scipy.github.io/devdocs/reference/generated/scipy.special.boxcox1p.html#scipy.special.boxcox1p "(in SciPy v1.19.0.dev)").

\\\begin{split}\mathrm{boxcox1p}(x, \lambda) = \begin{cases} ((1 + x)^\lambda - 1) / \lambda & \lambda \ne 0 \\ \log(1 + x) & \lambda = 0 \end{cases}\end{split}\\

Defined for \\x \> -1\\; returns `nan` for `x`` ``<=`` ``-1`.

Parameters:  
- **x** (*ArrayLike*) – arraylike, real-valued, greater than `-1`.

- **lmbda** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of the shifted Box-Cox transform.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.special.boxcox()`](jax.scipy.special.boxcox.html#jax.scipy.special.boxcox "jax.scipy.special.boxcox")

[](jax.scipy.special.boxcox.html "previous page")

previous

jax.scipy.special.boxcox

[](jax.scipy.special.comb.html "next page")

next

jax.scipy.special.comb

Contents

- [`boxcox1p()`](#jax.scipy.special.boxcox1p)

By The JAX authors

© Copyright 2024, The JAX Authors.\
