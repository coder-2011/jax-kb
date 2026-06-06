- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.xlog1py

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.xlog1py.rst "Download source file")
-  .pdf

# jax.scipy.special.xlog1py

## Contents

- [`xlog1py`](#jax.scipy.special.xlog1py)

# jax.scipy.special.xlog1py[\#](#jax-scipy-special-xlog1py "Link to this heading")

jax.scipy.special.xlog1py *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L970-L994)[\#](#jax.scipy.special.xlog1py "Link to this definition")  
Compute x\*log(1 + y), returning 0 for x=0.

JAX implementation of [`scipy.special.xlog1py`](https://scipy.github.io/devdocs/reference/generated/scipy.special.xlog1py.html#scipy.special.xlog1py "(in SciPy v1.19.0.dev)").

This is defined to return 0 when \\(x, y) = (0, -1)\\, with a custom derivative rule so that automatic differentiation is well-defined at this point.

Parameters:  
- **x** (*ArrayLike*) – arraylike, real-valued.

- **y** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing xlog1py values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.special.xlogy()`](jax.scipy.special.xlogy.html#jax.scipy.special.xlogy "jax.scipy.special.xlogy")

[](jax.scipy.special.wofz.html "previous page")

previous

jax.scipy.special.wofz

[](jax.scipy.special.xlogy.html "next page")

next

jax.scipy.special.xlogy

Contents

- [`xlog1py`](#jax.scipy.special.xlog1py)

By The JAX authors

© Copyright 2024, The JAX Authors.\
