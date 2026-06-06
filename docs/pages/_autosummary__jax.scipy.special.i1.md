- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.i1

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.i1.rst "Download source file")
-  .pdf

# jax.scipy.special.i1

## Contents

- [`i1()`](#jax.scipy.special.i1)

# jax.scipy.special.i1[\#](#jax-scipy-special-i1 "Link to this heading")

jax.scipy.special.i1(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1847-L1869)[\#](#jax.scipy.special.i1 "Link to this definition")  
Modified bessel function of first order.

JAX implementation of [`scipy.special.i1`](https://scipy.github.io/devdocs/reference/generated/scipy.special.i1.html#scipy.special.i1 "(in SciPy v1.19.0.dev)").

\\\mathrm{i1}(x) = I_1(x) = \frac{1}{2}x\sum\_{k=0}^\infty\frac{(x^2/4)^k}{k!(k+1)!}\\

Parameters:  
**x** (*ArrayLike*) – array, real-valued

Returns:  
array of bessel function values

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.i0()`](jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0")

- [`jax.scipy.special.i0e()`](jax.scipy.special.i0e.html#jax.scipy.special.i0e "jax.scipy.special.i0e")

- [`jax.scipy.special.i1e()`](jax.scipy.special.i1e.html#jax.scipy.special.i1e "jax.scipy.special.i1e")

[](jax.scipy.special.i0e.html "previous page")

previous

jax.scipy.special.i0e

[](jax.scipy.special.i1e.html "next page")

next

jax.scipy.special.i1e

Contents

- [`i1()`](#jax.scipy.special.i1)

By The JAX authors

© Copyright 2024, The JAX Authors.\
