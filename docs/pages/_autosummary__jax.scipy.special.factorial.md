- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.factorial

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.factorial.rst "Download source file")
-  .pdf

# jax.scipy.special.factorial

## Contents

- [`factorial()`](#jax.scipy.special.factorial)

# jax.scipy.special.factorial[\#](#jax-scipy-special-factorial "Link to this heading")

jax.scipy.special.factorial(*n*, *exact=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L333-L362)[\#](#jax.scipy.special.factorial "Link to this definition")  
Factorial function

JAX implementation of [`scipy.special.factorial`](https://scipy.github.io/devdocs/reference/generated/scipy.special.factorial.html#scipy.special.factorial "(in SciPy v1.19.0.dev)")

\\\mathrm{factorial}(n) = n! = \prod\_{k=1}^n k\\

Parameters:  
- **n** (*ArrayLike*) – arraylike, values for which factorial will be computed elementwise

- **exact** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, only `exact=False` is supported.

Returns:  
array containing values of the factorial.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

This computes the float-valued factorial via the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function. JAX does not support exact factorials, because it is not particularly useful: above `n=20`, the exact result cannot be represented by 64-bit integers, which are the largest integers available to JAX.

See also

[`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

[](jax.scipy.special.expn.html "previous page")

previous

jax.scipy.special.expn

[](jax.scipy.special.fresnel.html "next page")

next

jax.scipy.special.fresnel

Contents

- [`factorial()`](#jax.scipy.special.factorial)

By The JAX authors

© Copyright 2024, The JAX Authors.\
