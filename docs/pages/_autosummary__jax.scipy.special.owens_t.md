- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.owens_t

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.owens_t.rst "Download source file")
-  .pdf

# jax.scipy.special.owens_t

## Contents

- [`owens_t()`](#jax.scipy.special.owens_t)

# jax.scipy.special.owens_t[\#](#jax-scipy-special-owens-t "Link to this heading")

jax.scipy.special.owens_t(*h*, *a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3826-L3857)[\#](#jax.scipy.special.owens_t "Link to this definition")  
Owen’s T function.

JAX implementation of [`scipy.special.owens_t`](https://scipy.github.io/devdocs/reference/generated/scipy.special.owens_t.html#scipy.special.owens_t "(in SciPy v1.19.0.dev)").

Computes Owen’s T function:

\\T(h, a) = \frac{1}{2\pi} \int_0^a \frac{\exp\\\left(-\tfrac{1}{2}h^2(1+x^2)\right)}{1+x^2} \\ dx\\

Computed via 13-point Gauss-type quadrature on the canonical integral form (Patefield & Tandy 2000 method T5). The full 18-region dispatch from Patefield & Tandy is intentionally avoided because XLA evaluates every branch of a `where` / `select` unconditionally, which turns per-region dispatch into added cost rather than savings.

Parameters:  
- **h** (*ArrayLike*) – array_like, real-valued.

- **a** (*ArrayLike*) – array_like, real-valued.

Returns:  
Array of Owen’s T values with dtype matching the promoted inputs.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.special.ndtr()`](jax.scipy.special.ndtr.html#jax.scipy.special.ndtr "jax.scipy.special.ndtr")

[](jax.scipy.special.ndtri.html "previous page")

previous

jax.scipy.special.ndtri

[](jax.scipy.special.poch.html "next page")

next

jax.scipy.special.poch

Contents

- [`owens_t()`](#jax.scipy.special.owens_t)

By The JAX authors

© Copyright 2024, The JAX Authors.\
