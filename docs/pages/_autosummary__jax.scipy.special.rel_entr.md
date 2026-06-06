- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.rel_entr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.rel_entr.rst "Download source file")
-  .pdf

# jax.scipy.special.rel_entr

## Contents

- [`rel_entr()`](#jax.scipy.special.rel_entr)

# jax.scipy.special.rel_entr[\#](#jax-scipy-special-rel-entr "Link to this heading")

jax.scipy.special.rel_entr(*p*, *q*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1182-L1223)[\#](#jax.scipy.special.rel_entr "Link to this definition")  
The relative entropy function.

JAX implementation of [`scipy.special.rel_entr`](https://scipy.github.io/devdocs/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr "(in SciPy v1.19.0.dev)").

\\\begin{split} \mathrm{rel\\entr}(p, q) = \begin{cases} p\log(p/q) & p\>0,q\>0\\ 0 & p=0,q\ge 0\\ \infty & \mathrm{otherwise} \end{cases}\end{split}\\

Parameters:  
- **p** (*ArrayLike*) – arraylike, real-valued.

- **q** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array of relative entropy values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.entr()`](jax.scipy.special.entr.html#jax.scipy.special.entr "jax.scipy.special.entr")

- [`jax.scipy.special.kl_div()`](jax.scipy.special.kl_div.html#jax.scipy.special.kl_div "jax.scipy.special.kl_div")

[](jax.scipy.special.polygamma.html "previous page")

previous

jax.scipy.special.polygamma

[](jax.scipy.special.sici.html "next page")

next

jax.scipy.special.sici

Contents

- [`rel_entr()`](#jax.scipy.special.rel_entr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
