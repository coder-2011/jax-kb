- [](index.html)
- [API Reference](jax.html)
- `jax.nn` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.nn.rst "Download source file")
-  .pdf

# jax.nn module

## Contents

- [Activation functions](#activation-functions)
- [Other functions](#other-functions)

# `jax.nn` module[\#](#jax-nn-module "Link to this heading")

- [`jax.nn.initializers` module](jax.nn.initializers.html)

Common functions for neural network libraries.

## Activation functions[\#](#activation-functions "Link to this heading")

|  |  |
|----|----|
| [`relu`](_autosummary/jax.nn.relu.html#jax.nn.relu "jax.nn.relu") | Rectified linear unit activation function. |
| [`relu6`](_autosummary/jax.nn.relu6.html#jax.nn.relu6 "jax.nn.relu6") | Rectified Linear Unit 6 activation function. |
| [`sigmoid`](_autosummary/jax.nn.sigmoid.html#jax.nn.sigmoid "jax.nn.sigmoid")(x) | Sigmoid activation function. |
| [`softplus`](_autosummary/jax.nn.softplus.html#jax.nn.softplus "jax.nn.softplus")(x) | Softplus activation function. |
| [`sparse_plus`](_autosummary/jax.nn.sparse_plus.html#jax.nn.sparse_plus "jax.nn.sparse_plus")(x) | Sparse plus function. |
| [`sparse_sigmoid`](_autosummary/jax.nn.sparse_sigmoid.html#jax.nn.sparse_sigmoid "jax.nn.sparse_sigmoid")(x) | Sparse sigmoid activation function. |
| [`soft_sign`](_autosummary/jax.nn.soft_sign.html#jax.nn.soft_sign "jax.nn.soft_sign")(x) | Soft-sign activation function. |
| [`silu`](_autosummary/jax.nn.silu.html#jax.nn.silu "jax.nn.silu")(x) | SiLU (aka swish) activation function. |
| [`swish`](_autosummary/jax.nn.swish.html#jax.nn.swish "jax.nn.swish")(x) | SiLU (aka swish) activation function. |
| [`log_sigmoid`](_autosummary/jax.nn.log_sigmoid.html#jax.nn.log_sigmoid "jax.nn.log_sigmoid")(x) | Log-sigmoid activation function. |
| [`leaky_relu`](_autosummary/jax.nn.leaky_relu.html#jax.nn.leaky_relu "jax.nn.leaky_relu")(x\[, negative_slope\]) | Leaky rectified linear unit activation function. |
| [`hard_sigmoid`](_autosummary/jax.nn.hard_sigmoid.html#jax.nn.hard_sigmoid "jax.nn.hard_sigmoid")(x) | Hard Sigmoid activation function. |
| [`hard_silu`](_autosummary/jax.nn.hard_silu.html#jax.nn.hard_silu "jax.nn.hard_silu")(x) | Hard SiLU (swish) activation function |
| [`hard_swish`](_autosummary/jax.nn.hard_swish.html#jax.nn.hard_swish "jax.nn.hard_swish")(x) | Hard SiLU (swish) activation function |
| [`hard_tanh`](_autosummary/jax.nn.hard_tanh.html#jax.nn.hard_tanh "jax.nn.hard_tanh")(x) | Hard \\\mathrm{tanh}\\ activation function. |
| [`tanh`](_autosummary/jax.nn.tanh.html#jax.nn.tanh "jax.nn.tanh")(x, /) | Calculate element-wise hyperbolic tangent of input. |
| [`elu`](_autosummary/jax.nn.elu.html#jax.nn.elu "jax.nn.elu")(x\[, alpha\]) | Exponential linear unit activation function. |
| [`celu`](_autosummary/jax.nn.celu.html#jax.nn.celu "jax.nn.celu")(x\[, alpha\]) | Continuously-differentiable exponential linear unit activation. |
| [`selu`](_autosummary/jax.nn.selu.html#jax.nn.selu "jax.nn.selu")(x) | Scaled exponential linear unit activation. |
| [`gelu`](_autosummary/jax.nn.gelu.html#jax.nn.gelu "jax.nn.gelu")(x\[, approximate\]) | Gaussian error linear unit activation function. |
| [`glu`](_autosummary/jax.nn.glu.html#jax.nn.glu "jax.nn.glu")(x\[, axis\]) | Gated linear unit activation function. |
| [`squareplus`](_autosummary/jax.nn.squareplus.html#jax.nn.squareplus "jax.nn.squareplus")(x\[, b\]) | Squareplus activation function. |
| [`mish`](_autosummary/jax.nn.mish.html#jax.nn.mish "jax.nn.mish")(x) | Mish activation function. |
| [`identity`](_autosummary/jax.nn.identity.html#jax.nn.identity "jax.nn.identity")(x) | Identity activation function. |

## Other functions[\#](#other-functions "Link to this heading")

|  |  |
|----|----|
| [`softmax`](_autosummary/jax.nn.softmax.html#jax.nn.softmax "jax.nn.softmax")(x\[, axis, where\]) | Softmax function. |
| [`log_softmax`](_autosummary/jax.nn.log_softmax.html#jax.nn.log_softmax "jax.nn.log_softmax")(x\[, axis, where\]) | Log-Softmax function. |
| [`logmeanexp`](_autosummary/jax.nn.logmeanexp.html#jax.nn.logmeanexp "jax.nn.logmeanexp")(x\[, axis, where, keepdims\]) | Log mean exp. |
| [`logsumexp`](_autosummary/jax.nn.logsumexp.html#jax.nn.logsumexp "jax.nn.logsumexp")() | Log-sum-exp reduction. |
| [`standardize`](_autosummary/jax.nn.standardize.html#jax.nn.standardize "jax.nn.standardize")(x\[, axis, mean, variance, ...\]) | Standardizes input to zero mean and unit variance. |
| [`one_hot`](_autosummary/jax.nn.one_hot.html#jax.nn.one_hot "jax.nn.one_hot")(x, num_classes, \*\[, dtype, axis\]) | One-hot encodes the given indices. |
| [`dot_product_attention`](_autosummary/jax.nn.dot_product_attention.html#jax.nn.dot_product_attention "jax.nn.dot_product_attention")() | Scaled dot product attention function. |
| [`scaled_matmul`](_autosummary/jax.nn.scaled_matmul.html#jax.nn.scaled_matmul "jax.nn.scaled_matmul")(lhs, rhs, lhs_scales, rhs_scales) | Scaled matrix multiplication function. |
| [`get_scaled_dot_general_config`](_autosummary/jax.nn.get_scaled_dot_general_config.html#jax.nn.get_scaled_dot_general_config "jax.nn.get_scaled_dot_general_config")(mode\[, ...\]) | Get quantization configs for scaled_dot_general. |
| [`scaled_dot_general`](_autosummary/jax.nn.scaled_dot_general.html#jax.nn.scaled_dot_general "jax.nn.scaled_dot_general")(lhs, rhs, dimension_numbers) | Scaled dot general operation. |
| [`log1mexp`](_autosummary/jax.nn.log1mexp.html#jax.nn.log1mexp "jax.nn.log1mexp") | Numerically stable calculation of \\\log(1 - \exp(-x))\\. |

[](_autosummary/jax.image.scale_and_translate.html "previous page")

previous

jax.image.scale_and_translate

[](jax.nn.initializers.html "next page")

next

`jax.nn.initializers` module

Contents

- [Activation functions](#activation-functions)
- [Other functions](#other-functions)

By The JAX authors

© Copyright 2024, The JAX Authors.\
