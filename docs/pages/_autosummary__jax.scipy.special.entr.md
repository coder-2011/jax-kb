- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.entr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.entr.rst "Download source file")
-  .pdf

# jax.scipy.special.entr

## Contents

- [`entr()`](#jax.scipy.special.entr)

# jax.scipy.special.entr[\#](#jax-scipy-special-entr "Link to this heading")

jax.scipy.special.entr(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1014-L1043)[\#](#jax.scipy.special.entr "Link to this definition")  
The entropy function

JAX implementation of [`scipy.special.entr`](https://scipy.github.io/devdocs/reference/generated/scipy.special.entr.html#scipy.special.entr "(in SciPy v1.19.0.dev)").

\\\begin{split}\mathrm{entr}(x) = \begin{cases} -x\log(x) & x \> 0 \\ 0 & x = 0\\ -\infty & \mathrm{otherwise} \end{cases}\end{split}\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing entropy values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.kl_div()`](jax.scipy.special.kl_div.html#jax.scipy.special.kl_div "jax.scipy.special.kl_div")

- [`jax.scipy.special.rel_entr()`](jax.scipy.special.rel_entr.html#jax.scipy.special.rel_entr "jax.scipy.special.rel_entr")

[](jax.scipy.special.digamma.html "previous page")

previous

jax.scipy.special.digamma

[](jax.scipy.special.erf.html "next page")

next

jax.scipy.special.erf

Contents

- [`entr()`](#jax.scipy.special.entr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
