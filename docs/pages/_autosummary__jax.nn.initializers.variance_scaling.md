- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.variance_scaling

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.variance_scaling.rst "Download source file")
-  .pdf

# jax.nn.initializers.variance_scaling

## Contents

- [`variance_scaling()`](#jax.nn.initializers.variance_scaling)

# jax.nn.initializers.variance_scaling[\#](#jax-nn-initializers-variance-scaling "Link to this heading")

jax.nn.initializers.variance_scaling(*scale*, *mode*, *distribution*, *in_axis=-2*, *out_axis=-1*, *batch_axis=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L281-L373)[\#](#jax.nn.initializers.variance_scaling "Link to this definition")  
Initializer that adapts its scale to the shape of the weights tensor.

With `distribution="truncated_normal"` or `distribution="normal"`, samples are drawn from a (truncated) normal distribution with a mean of zero and a standard deviation (after truncation, if applicable) of \\\sqrt{\frac{scale}{n}}\\, where n is, for each `mode`:

- `"fan_in"`: the number of inputs

- `"fan_out"`: the number of outputs

- `"fan_avg"`: the arithmetic average of the numbers of inputs and outputs

- `"fan_geo_avg"`: the geometric average of the numbers of inputs and outputs

This initializer can be configured with `in_axis`, `out_axis`, and `batch_axis` to work with general convolutional or dense layers; axes that are not in any of those arguments are assumed to be the “receptive field” (convolution kernel spatial axes).

With `distribution="truncated_normal"`, the absolute values of the samples are truncated at 2 standard deviations before scaling.

With `distribution="uniform"`, samples are drawn from:

- a uniform interval, if dtype is real, or

- a uniform disk, if dtype is complex,

with a mean of zero and a standard deviation of \\\sqrt{\frac{scale}{n}}\\ where n is defined above.

Parameters:  
- **scale** (*RealNumeric*) – scaling factor (positive float).

- **mode** ([*Literal*](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")*\['fan_in'\]* *\|* *Literal\['fan_out'\]* *\|* *Literal\['fan_avg'\]* *\|* *Literal\['fan_geo_avg'\]*) – one of `"fan_in"`, `"fan_out"`, `"fan_avg"`, and `"fan_geo_avg"`.

- **distribution** ([*Literal*](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")*\['truncated_normal'\]* *\|* *Literal\['normal'\]* *\|* *Literal\['uniform'\]*) – random distribution to use. One of `"truncated_normal"`, `"normal"` and `"uniform"`.

- **in_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – axis or sequence of axes of the input dimension in the weights array.

- **out_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – axis or sequence of axes of the output dimension in the weights array.

- **batch_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – axis or sequence of axes in the weight array that should be ignored.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – the dtype of the weights.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

[](jax.nn.initializers.uniform.html "previous page")

previous

jax.nn.initializers.uniform

[](jax.nn.initializers.xavier_normal.html "next page")

next

jax.nn.initializers.xavier_normal

Contents

- [`variance_scaling()`](#jax.nn.initializers.variance_scaling)

By The JAX authors

© Copyright 2024, The JAX Authors.\
