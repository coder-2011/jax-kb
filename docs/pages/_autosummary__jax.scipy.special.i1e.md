- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.i1e

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.i1e.rst "Download source file")
-  .pdf

# jax.scipy.special.i1e

## Contents

- [`i1e()`](#jax.scipy.special.i1e)

# jax.scipy.special.i1e[\#](#jax-scipy-special-i1e "Link to this heading")

jax.scipy.special.i1e(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1821-L1845)[\#](#jax.scipy.special.i1e "Link to this definition")  
Exponentially scaled modified bessel function of first order.

JAX implementation of [`scipy.special.i1e`](https://scipy.github.io/devdocs/reference/generated/scipy.special.i1e.html#scipy.special.i1e "(in SciPy v1.19.0.dev)").

\\\mathrm{i1e}(x) = e^{-\|x\|} I_1(x)\\

where \\I_1(x)\\ is the modified Bessel function [`i1()`](jax.scipy.special.i1.html#jax.scipy.special.i1 "jax.scipy.special.i1").

Parameters:  
**x** (*ArrayLike*) – array, real-valued

Returns:  
array of bessel function values

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.i0()`](jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0")

- [`jax.scipy.special.i0e()`](jax.scipy.special.i0e.html#jax.scipy.special.i0e "jax.scipy.special.i0e")

- [`jax.scipy.special.i1()`](jax.scipy.special.i1.html#jax.scipy.special.i1 "jax.scipy.special.i1")

[](jax.scipy.special.i1.html "previous page")

previous

jax.scipy.special.i1

[](jax.scipy.special.kl_div.html "next page")

next

jax.scipy.special.kl_div

Contents

- [`i1e()`](#jax.scipy.special.i1e)

By The JAX authors

© Copyright 2024, The JAX Authors.\
