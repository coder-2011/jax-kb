- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.fresnel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.fresnel.rst "Download source file")
-  .pdf

# jax.scipy.special.fresnel

## Contents

- [`fresnel`](#jax.scipy.special.fresnel)

# jax.scipy.special.fresnel[\#](#jax-scipy-special-fresnel "Link to this heading")

jax.scipy.special.fresnel *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/third_party/scipy/special.py#L51-L318)[\#](#jax.scipy.special.fresnel "Link to this definition")  
The Fresnel integrals

JAX implementation of [`scipy.special.fresnel`](https://scipy.github.io/devdocs/reference/generated/scipy.special.fresnel.html#scipy.special.fresnel "(in SciPy v1.19.0.dev)").

The Fresnel integrals are defined as  
\\\begin{split}S(x) &= \int_0^x \sin(\pi t^2 /2) dt \\ C(x) &= \int_0^x \cos(\pi t^2 /2) dt.\end{split}\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
Arrays containing the values of the Fresnel integrals.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Notes

The JAX version only supports real-valued inputs, and is based on the SciPy C++ implementation, see [here](https://github.com/scipy/scipy/blob/v1.14.0/scipy/special/special/cephes/fresnl.h). For `float32` dtypes, the implementation is directly based on the Cephes implementation `fresnlf`.

As for the original Cephes implementation, the accuracy is only guaranteed in the domain \[-10, 10\]. Outside of that domain, one could observe divergence between the theoretical derivatives and the custom JVP implementation, especially for large input values.

Finally, for half-precision data types, `float16` and `bfloat16`, the array elements are upcasted to `float32` as the Cephes coefficients used in series expansions would otherwise lead to poor results. Other data types, like `float8`, are not supported.

[](jax.scipy.special.factorial.html "previous page")

previous

jax.scipy.special.factorial

[](jax.scipy.special.gamma.html "next page")

next

jax.scipy.special.gamma

Contents

- [`fresnel`](#jax.scipy.special.fresnel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
