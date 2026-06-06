- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.glu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.glu.rst "Download source file")
-  .pdf

# jax.nn.glu

## Contents

- [`glu()`](#jax.nn.glu)

# jax.nn.glu[\#](#jax-nn-glu "Link to this heading")

jax.nn.glu(*x*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L471-L500)[\#](#jax.nn.glu "Link to this definition")  
Gated linear unit activation function.

Computes the function:

\\\mathrm{glu}(x) = x\left\[\ldots, 0:\frac{n}{2}, \ldots\right\] \cdot \mathrm{sigmoid} \left( x\left\[\ldots, \frac{n}{2}:n, \ldots\right\] \right)\\

where the array is split into two along `axis`. The size of the `axis` dimension must be divisible by two.

Parameters:  
- **x** (*ArrayLike*) – input array

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which the split should be computed (default: -1)

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`sigmoid()`](jax.nn.sigmoid.html#jax.nn.sigmoid "jax.nn.sigmoid")

[](jax.nn.gelu.html "previous page")

previous

jax.nn.gelu

[](jax.nn.squareplus.html "next page")

next

jax.nn.squareplus

Contents

- [`glu()`](#jax.nn.glu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
