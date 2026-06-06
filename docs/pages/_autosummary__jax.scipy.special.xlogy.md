- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.xlogy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.xlogy.rst "Download source file")
-  .pdf

# jax.scipy.special.xlogy

## Contents

- [`xlogy`](#jax.scipy.special.xlogy)

# jax.scipy.special.xlogy[\#](#jax-scipy-special-xlogy "Link to this heading")

jax.scipy.special.xlogy *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L937-L961)[\#](#jax.scipy.special.xlogy "Link to this definition")  
Compute x\*log(y), returning 0 for x=0.

JAX implementation of [`scipy.special.xlogy`](https://scipy.github.io/devdocs/reference/generated/scipy.special.xlogy.html#scipy.special.xlogy "(in SciPy v1.19.0.dev)").

This is defined to return zero when \\(x, y) = (0, 0)\\, with a custom derivative rule so that automatic differentiation is well-defined at this point.

Parameters:  
- **x** (*ArrayLike*) – arraylike, real-valued.

- **y** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing xlogy values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.special.xlog1py()`](jax.scipy.special.xlog1py.html#jax.scipy.special.xlog1py "jax.scipy.special.xlog1py")

[](jax.scipy.special.xlog1py.html "previous page")

previous

jax.scipy.special.xlog1py

[](jax.scipy.special.zeta.html "next page")

next

jax.scipy.special.zeta

Contents

- [`xlogy`](#jax.scipy.special.xlogy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
