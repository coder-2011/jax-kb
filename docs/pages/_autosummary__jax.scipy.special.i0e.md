- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.i0e

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.i0e.rst "Download source file")
-  .pdf

# jax.scipy.special.i0e

## Contents

- [`i0e()`](#jax.scipy.special.i0e)

# jax.scipy.special.i0e[\#](#jax-scipy-special-i0e "Link to this heading")

jax.scipy.special.i0e(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1771-L1795)[\#](#jax.scipy.special.i0e "Link to this definition")  
Exponentially scaled modified bessel function of zeroth order.

JAX implementation of [`scipy.special.i0e`](https://scipy.github.io/devdocs/reference/generated/scipy.special.i0e.html#scipy.special.i0e "(in SciPy v1.19.0.dev)").

\\\mathrm{i0e}(x) = e^{-\|x\|} I_0(x)\\

where \\I_0(x)\\ is the modified Bessel function [`i0()`](jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0").

Parameters:  
**x** (*ArrayLike*) – array, real-valued

Returns:  
array of bessel function values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.i0()`](jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0")

- [`jax.scipy.special.i1()`](jax.scipy.special.i1.html#jax.scipy.special.i1 "jax.scipy.special.i1")

- [`jax.scipy.special.i1e()`](jax.scipy.special.i1e.html#jax.scipy.special.i1e "jax.scipy.special.i1e")

[](jax.scipy.special.i0.html "previous page")

previous

jax.scipy.special.i0

[](jax.scipy.special.i1.html "next page")

next

jax.scipy.special.i1

Contents

- [`i0e()`](#jax.scipy.special.i0e)

By The JAX authors

© Copyright 2024, The JAX Authors.\
