- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.logistic

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.logistic.rst "Download source file")
-  .pdf

# jax.lax.logistic

## Contents

- [`logistic()`](#jax.lax.logistic)

# jax.lax.logistic[\#](#jax-lax-logistic "Link to this heading")

jax.lax.logistic(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L640-L664)[\#](#jax.lax.logistic "Link to this definition")  
Elementwise logistic (sigmoid) function: \\\frac{1}{1 + e^{-x}}\\.

There is no HLO logistic/sigmoid primitive, so this lowers to a sequence of HLO arithmetic operations.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating point or complex dtype.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise logistic/sigmoid function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.nn.sigmoid()`](jax.nn.sigmoid.html#jax.nn.sigmoid "jax.nn.sigmoid"): an alternative API for this functionality.

[](jax.lax.log1p.html "previous page")

previous

jax.lax.log1p

[](jax.lax.lt.html "next page")

next

jax.lax.lt

Contents

- [`logistic()`](#jax.lax.logistic)

By The JAX authors

© Copyright 2024, The JAX Authors.\
