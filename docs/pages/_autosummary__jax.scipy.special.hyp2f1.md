- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.hyp2f1

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.hyp2f1.rst "Download source file")
-  .pdf

# jax.scipy.special.hyp2f1

## Contents

- [`hyp2f1`](#jax.scipy.special.hyp2f1)

# jax.scipy.special.hyp2f1[\#](#jax-scipy-special-hyp2f1 "Link to this heading")

jax.scipy.special.hyp2f1 *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3676-L3731)[\#](#jax.scipy.special.hyp2f1 "Link to this definition")  
The 2F1 hypergeometric function.

JAX implementation of [`scipy.special.hyp2f1`](https://scipy.github.io/devdocs/reference/generated/scipy.special.hyp2f1.html#scipy.special.hyp2f1 "(in SciPy v1.19.0.dev)").

\\\mathrm{hyp2f1}(a, b, c, x) = {}\_2F_1(a; b; c; x) = \sum\_{k=0}^\infty \frac{(a)\_k(b)\_k}{(c)\_k}\frac{x^k}{k!}\\

where \\(\cdot)\_k\\ is the Pochammer symbol.

The JAX version only accepts positive and real inputs. Values of `a`, `b`, `c`, and `x` leading to high values of 2F1 may lead to erroneous results; consider enabling double precision in this case.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued

- **b** (*ArrayLike*) – arraylike, real-valued

- **c** (*ArrayLike*) – arraylike, real-valued

- **x** (*ArrayLike*) – arraylike, real-valued

Returns:  
array of 2F1 values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.hyp1f1.html "previous page")

previous

jax.scipy.special.hyp1f1

[](jax.scipy.special.i0.html "next page")

next

jax.scipy.special.i0

Contents

- [`hyp2f1`](#jax.scipy.special.hyp2f1)

By The JAX authors

© Copyright 2024, The JAX Authors.\
