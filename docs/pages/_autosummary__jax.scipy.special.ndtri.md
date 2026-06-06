- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.ndtri

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.ndtri.rst "Download source file")
-  .pdf

# jax.scipy.special.ndtri

## Contents

- [`ndtri()`](#jax.scipy.special.ndtri)

# jax.scipy.special.ndtri[\#](#jax-scipy-special-ndtri "Link to this heading")

jax.scipy.special.ndtri(*p*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1484-L1510)[\#](#jax.scipy.special.ndtri "Link to this definition")  
The inverse of the CDF of the Normal distribution function.

JAX implementation of [`scipy.special.ndtri`](https://scipy.github.io/devdocs/reference/generated/scipy.special.ndtri.html#scipy.special.ndtri "(in SciPy v1.19.0.dev)").

Returns x such that the area under the PDF from \\-\infty\\ to x is equal to p.

A piece-wise rational approximation is done for the function. This is based on the implementation in netlib.

Parameters:  
**p** (*ArrayLike*) – an array of type float32, float64.

Returns:  
an array with dtype=p.dtype.

Raises:  
[**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – if p is not floating-type.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.ndtr.html "previous page")

previous

jax.scipy.special.ndtr

[](jax.scipy.special.owens_t.html "next page")

next

jax.scipy.special.owens_t

Contents

- [`ndtri()`](#jax.scipy.special.ndtri)

By The JAX authors

© Copyright 2024, The JAX Authors.\
