- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.lpmn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.lpmn.rst "Download source file")
-  .pdf

# jax.scipy.special.lpmn

## Contents

- [`lpmn()`](#jax.scipy.special.lpmn)

# jax.scipy.special.lpmn[\#](#jax-scipy-special-lpmn "Link to this heading")

jax.scipy.special.lpmn(*m*, *n*, *z*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L2197-L2239)[\#](#jax.scipy.special.lpmn "Link to this definition")  
The associated Legendre functions (ALFs) of the first kind.

Parameters:  
- **m** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum order of the associated Legendre functions.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum degree of the associated Legendre function, often called l in describing ALFs. Both the degrees and orders are \[0, 1, 2, …, l_max\], where l_max denotes the maximum degree.

- **z** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A vector of type float32 or float64 containing the sampling points at which the ALFs are computed.

Returns:  
A 2-tuple of 3D arrays of shape (l_max + 1, l_max + 1, len(z)) containing the values and derivatives of the associated Legendre functions of the first kind. The return type matches the type of z.

Raises:  
- **TypeError if elements** **of** **array z are not in** **(**[**float32**](jax.numpy.float32.html#jax.numpy.float32 "jax.numpy.float32")**,** [**float64**](jax.numpy.float64.html#jax.numpy.float64 "jax.numpy.float64")**).** –

- **ValueError if array z is not 1D.** –

- **NotImplementedError if m!=n.** –

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.scipy.special.logsumexp.html "previous page")

previous

jax.scipy.special.logsumexp

[](jax.scipy.special.lpmn_values.html "next page")

next

jax.scipy.special.lpmn_values

Contents

- [`lpmn()`](#jax.scipy.special.lpmn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
