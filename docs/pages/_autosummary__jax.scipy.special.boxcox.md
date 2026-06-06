- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.boxcox

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.boxcox.rst "Download source file")
-  .pdf

# jax.scipy.special.boxcox

## Contents

- [`boxcox()`](#jax.scipy.special.boxcox)

# jax.scipy.special.boxcox[\#](#jax-scipy-special-boxcox "Link to this heading")

jax.scipy.special.boxcox(*x*, *lmbda*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1045-L1075)[\#](#jax.scipy.special.boxcox "Link to this definition")  
Box-Cox power transformation.

JAX implementation of [`scipy.special.boxcox`](https://scipy.github.io/devdocs/reference/generated/scipy.special.boxcox.html#scipy.special.boxcox "(in SciPy v1.19.0.dev)").

\\\begin{split}\mathrm{boxcox}(x, \lambda) = \begin{cases} (x^\lambda - 1) / \lambda & \lambda \ne 0 \\ \log(x) & \lambda = 0 \end{cases}\end{split}\\

Defined for \\x \> 0\\; returns `nan` for non-positive `x`.

Parameters:  
- **x** (*ArrayLike*) – arraylike, positive real-valued.

- **lmbda** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing Box-Cox transformed values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.special.boxcox1p()`](jax.scipy.special.boxcox1p.html#jax.scipy.special.boxcox1p "jax.scipy.special.boxcox1p")

[](jax.scipy.special.betaln.html "previous page")

previous

jax.scipy.special.betaln

[](jax.scipy.special.boxcox1p.html "next page")

next

jax.scipy.special.boxcox1p

Contents

- [`boxcox()`](#jax.scipy.special.boxcox)

By The JAX authors

© Copyright 2024, The JAX Authors.\
