- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.spence

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.spence.rst "Download source file")
-  .pdf

# jax.scipy.special.spence

## Contents

- [`spence()`](#jax.scipy.special.spence)

# jax.scipy.special.spence[\#](#jax-scipy-special-spence "Link to this heading")

jax.scipy.special.spence(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3090-L3132)[\#](#jax.scipy.special.spence "Link to this definition")  
Spence’s function, also known as the dilogarithm for real values.

JAX implementation of [`scipy.special.spence`](https://scipy.github.io/devdocs/reference/generated/scipy.special.spence.html#scipy.special.spence "(in SciPy v1.19.0.dev)").

It is defined to be:

\\\mathrm{spence}(x) = \begin{equation} \int_1^x \frac{\log(t)}{1 - t}dt \end{equation}\\

Unlike the SciPy implementation, this is only defined for positive real values of z. For negative values, NaN is returned.

Parameters:  
- **z** – An array of type float32, float64.

- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

Returns:  
An array with dtype=z.dtype. computed values of Spence’s function.

Raises:  
[**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – if elements of array z are not in (float32, float64).

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes: There is a different convention which defines Spence’s function by the integral:

\\\begin{equation} -\int_0^z \frac{\log(1 - t)}{t}dt \end{equation}\\

This is our spence(1 - z).

[](jax.scipy.special.softmax.html "previous page")

previous

jax.scipy.special.softmax

[](jax.scipy.special.sph_harm_y.html "next page")

next

jax.scipy.special.sph_harm_y

Contents

- [`spence()`](#jax.scipy.special.spence)

By The JAX authors

© Copyright 2024, The JAX Authors.\
