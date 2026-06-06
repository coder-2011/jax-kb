- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.hyp1f1

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.hyp1f1.rst "Download source file")
-  .pdf

# jax.scipy.special.hyp1f1

## Contents

- [`hyp1f1`](#jax.scipy.special.hyp1f1)

# jax.scipy.special.hyp1f1[\#](#jax-scipy-special-hyp1f1 "Link to this heading")

jax.scipy.special.hyp1f1 *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3338-L3381)[\#](#jax.scipy.special.hyp1f1 "Link to this definition")  
The 1F1 hypergeometric function.

JAX implementation of [`scipy.special.hyp1f1`](https://scipy.github.io/devdocs/reference/generated/scipy.special.hyp1f1.html#scipy.special.hyp1f1 "(in SciPy v1.19.0.dev)").

\\\mathrm{hyp1f1}(a, b, x) = {}\_1F_1(x;a, b) = \sum\_{k=0}^\infty \frac{(a)\_k}{(b)\_kk!}x^k\\

where \\(\cdot)\_k\\ is the Pochammer symbol (refer to [`poch()`](jax.scipy.special.poch.html#jax.scipy.special.poch "jax.scipy.special.poch")).

The JAX version only accepts positive and real inputs. Values of `a`, `b`, and `x`, leading to high values of 1F1 may lead to erroneous results; consider enabling double precision in this case. The convention for `a`` ``=`` ``b`` ``=`` ``0` is `1`, unlike in scipy’s implementation.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued

- **b** (*ArrayLike*) – arraylike, real-valued

- **x** (*ArrayLike*) – arraylike, real-valued

Returns:  
array of 1F1 values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.gammasgn.html "previous page")

previous

jax.scipy.special.gammasgn

[](jax.scipy.special.hyp2f1.html "next page")

next

jax.scipy.special.hyp2f1

Contents

- [`hyp1f1`](#jax.scipy.special.hyp1f1)

By The JAX authors

© Copyright 2024, The JAX Authors.\
