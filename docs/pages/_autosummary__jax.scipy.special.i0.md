- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.i0

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.i0.rst "Download source file")
-  .pdf

# jax.scipy.special.i0

## Contents

- [`i0()`](#jax.scipy.special.i0)

# jax.scipy.special.i0[\#](#jax-scipy-special-i0 "Link to this heading")

jax.scipy.special.i0(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1797-L1819)[\#](#jax.scipy.special.i0 "Link to this definition")  
Modified bessel function of zeroth order.

JAX implementation of [`scipy.special.i0`](https://scipy.github.io/devdocs/reference/generated/scipy.special.i0.html#scipy.special.i0 "(in SciPy v1.19.0.dev)").

\\\mathrm{i0}(x) = I_0(x) = \sum\_{k=0}^\infty \frac{(x^2/4)^k}{(k!)^2}\\

Parameters:  
**x** (*ArrayLike*) – array, real-valued

Returns:  
array of bessel function values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.i0e()`](jax.scipy.special.i0e.html#jax.scipy.special.i0e "jax.scipy.special.i0e")

- [`jax.scipy.special.i1()`](jax.scipy.special.i1.html#jax.scipy.special.i1 "jax.scipy.special.i1")

- [`jax.scipy.special.i1e()`](jax.scipy.special.i1e.html#jax.scipy.special.i1e "jax.scipy.special.i1e")

[](jax.scipy.special.hyp2f1.html "previous page")

previous

jax.scipy.special.hyp2f1

[](jax.scipy.special.i0e.html "next page")

next

jax.scipy.special.i0e

Contents

- [`i0()`](#jax.scipy.special.i0)

By The JAX authors

© Copyright 2024, The JAX Authors.\
