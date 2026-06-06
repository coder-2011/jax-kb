- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.relu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.relu.rst "Download source file")
-  .pdf

# jax.nn.relu

## Contents

- [`relu`](#jax.nn.relu)

# jax.nn.relu[\#](#jax-nn-relu "Link to this heading")

jax.nn.relu *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L71-L105)[\#](#jax.nn.relu "Link to this definition")  
Rectified linear unit activation function.

Computes the element-wise function:

\\\mathrm{relu}(x) = \max(x, 0)\\

except under differentiation, we take:

\\\nabla \mathrm{relu}(0) = 0\\

For more information see [Numerical influence of ReLU’(0) on backpropagation](https://dl.acm.org/doi/10.5555/3540261.3540297).

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.nn.relu(jax.numpy.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
    Array([0. , 0. , 0. , 0. , 0.5, 1. , 2. ], dtype=float32)

See also

[`relu6()`](jax.nn.relu6.html#jax.nn.relu6 "jax.nn.relu6")

[](jax.nn.initializers.Initializer.html "previous page")

previous

jax.nn.initializers.Initializer

[](jax.nn.relu6.html "next page")

next

jax.nn.relu6

Contents

- [`relu`](#jax.nn.relu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
