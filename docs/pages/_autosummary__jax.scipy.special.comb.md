- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.comb

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.comb.rst "Download source file")
-  .pdf

# jax.scipy.special.comb

## Contents

- [`comb()`](#jax.scipy.special.comb)

# jax.scipy.special.comb[\#](#jax-scipy-special-comb "Link to this heading")

jax.scipy.special.comb(*N*, *k*, *\**, *repetition=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L364-L404)[\#](#jax.scipy.special.comb "Link to this definition")  
The number of combinations of N things taken k at a time (“N choose k”).

JAX implementation of [`scipy.special.comb()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.comb.html#scipy.special.comb "(in SciPy v1.19.0.dev)").

\\\mathrm{comb}(N, k) = \binom{N}{k} = \frac{N!}{k!\\(N - k)!}\\

Parameters:  
- **N** (*ArrayLike*) – arraylike, number of things.

- **k** (*ArrayLike*) – arraylike, number of elements taken.

- **repetition** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, compute the number of combinations with repetition.

Returns:  
array containing the total number of combinations.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

This computes the float-valued binomial coefficient via the [`gammaln()`](jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln") function. The `exact` argument from [`scipy.special.comb()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.comb.html#scipy.special.comb "(in SciPy v1.19.0.dev)") is not supported because JAX does not support arbitrary-precision integers. If `N`` ``<`` ``0`, `k`` ``<`` ``0`, or `k`` ``>`` ``N` and `repetition=False`, then 0 is returned.

See also

- [`jax.scipy.special.factorial()`](jax.scipy.special.factorial.html#jax.scipy.special.factorial "jax.scipy.special.factorial")

- [`jax.scipy.special.gammaln()`](jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln")

[](jax.scipy.special.boxcox1p.html "previous page")

previous

jax.scipy.special.boxcox1p

[](jax.scipy.special.dawsn.html "next page")

next

jax.scipy.special.dawsn

Contents

- [`comb()`](#jax.scipy.special.comb)

By The JAX authors

© Copyright 2024, The JAX Authors.\
