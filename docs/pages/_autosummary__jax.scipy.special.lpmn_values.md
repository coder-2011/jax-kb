- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.lpmn_values

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.lpmn_values.rst "Download source file")
-  .pdf

# jax.scipy.special.lpmn_values

## Contents

- [`lpmn_values()`](#jax.scipy.special.lpmn_values)

# jax.scipy.special.lpmn_values[\#](#jax-scipy-special-lpmn-values "Link to this heading")

jax.scipy.special.lpmn_values(*m*, *n*, *z*, *is_normalized*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L2241-L2293)[\#](#jax.scipy.special.lpmn_values "Link to this definition")  
The associated Legendre functions (ALFs) of the first kind.

Unlike lpmn, this function only computes the values of ALFs. The ALFs of the first kind can be used in spherical harmonics. The spherical harmonic of degree l and order m can be written as \\Y_l^m(\theta, \phi) = N_l^m \* P_l^m(\cos \theta) \* \exp(i m \phi)\\, where \\N_l^m\\ is the normalization factor and θ and φ are the colatitude and longitude, respectively. \\N_l^m\\ is chosen in the way that the spherical harmonics form a set of orthonormal basis function of \\L^2(S^2)\\. Normalizing \\P_l^m\\ avoids overflow/underflow and achieves better numerical stability.

Parameters:  
- **m** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum order of the associated Legendre functions.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum degree of the associated Legendre function, often called l in describing ALFs. Both the degrees and orders are \[0, 1, 2, …, l_max\], where l_max denotes the maximum degree.

- **z** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A vector of type float32 or float64 containing the sampling points at which the ALFs are computed.

- **is_normalized** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – True if the associated Legendre functions are normalized. With normalization, \\N_l^m\\ is applied such that the spherical harmonics form a set of orthonormal basis functions of \\L^2(S^2)\\.

Returns:  
A 3D array of shape (l_max + 1, l_max + 1, len(z)) containing the values of the associated Legendre functions of the first kind. The return type matches the type of z.

Raises:  
- **TypeError if elements** **of** **array z are not in** **(**[**float32**](jax.numpy.float32.html#jax.numpy.float32 "jax.numpy.float32")**,** [**float64**](jax.numpy.float64.html#jax.numpy.float64 "jax.numpy.float64")**).** –

- **ValueError if array z is not 1D.** –

- **NotImplementedError if m!=n.** –

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.lpmn.html "previous page")

previous

jax.scipy.special.lpmn

[](jax.scipy.special.multigammaln.html "next page")

next

jax.scipy.special.multigammaln

Contents

- [`lpmn_values()`](#jax.scipy.special.lpmn_values)

By The JAX authors

© Copyright 2024, The JAX Authors.\
