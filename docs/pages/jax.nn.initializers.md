- [](index.html)
- [API Reference](jax.html)
- [`jax.nn` module](jax.nn.html)
- `jax.nn.initializers` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.nn.initializers.rst "Download source file")
-  .pdf

# jax.nn.initializers module

## Contents

- [Initializers](#initializers)

# `jax.nn.initializers` module[\#](#module-jax.nn.initializers "Link to this heading")

Common neural network layer initializers, consistent with definitions used in Keras and Sonnet.

## Initializers[\#](#initializers "Link to this heading")

This module provides common neural network layer initializers, consistent with definitions used in Keras and Sonnet.

An initializer is a function that takes three arguments: `(key,`` ``shape,`` ``dtype)` and returns an array with dimensions `shape` and data type `dtype`. Argument `key` is a PRNG key (e.g. from [`jax.random.key()`](_autosummary/jax.random.key.html#jax.random.key "jax.random.key")), used to generate random numbers to initialize the array.

|  |  |
|----|----|
| [`constant`](_autosummary/jax.nn.initializers.constant.html#jax.nn.initializers.constant "jax.nn.initializers.constant")(value\[, dtype\]) | Builds an initializer that returns arrays full of a constant `value`. |
| [`delta_orthogonal`](_autosummary/jax.nn.initializers.delta_orthogonal.html#jax.nn.initializers.delta_orthogonal "jax.nn.initializers.delta_orthogonal")(\[scale, column_axis, dtype\]) | Builds an initializer for delta orthogonal kernels. |
| [`glorot_normal`](_autosummary/jax.nn.initializers.glorot_normal.html#jax.nn.initializers.glorot_normal "jax.nn.initializers.glorot_normal")(\[in_axis, out_axis, ...\]) | Builds a Glorot normal initializer (aka Xavier normal initializer). |
| [`glorot_uniform`](_autosummary/jax.nn.initializers.glorot_uniform.html#jax.nn.initializers.glorot_uniform "jax.nn.initializers.glorot_uniform")(\[in_axis, out_axis, ...\]) | Builds a Glorot uniform initializer (aka Xavier uniform initializer). |
| [`he_normal`](_autosummary/jax.nn.initializers.he_normal.html#jax.nn.initializers.he_normal "jax.nn.initializers.he_normal")(\[in_axis, out_axis, batch_axis, dtype\]) | Builds a He normal initializer (aka Kaiming normal initializer). |
| [`he_uniform`](_autosummary/jax.nn.initializers.he_uniform.html#jax.nn.initializers.he_uniform "jax.nn.initializers.he_uniform")(\[in_axis, out_axis, batch_axis, ...\]) | Builds a He uniform initializer (aka Kaiming uniform initializer). |
| [`kaiming_normal`](_autosummary/jax.nn.initializers.kaiming_normal.html#jax.nn.initializers.kaiming_normal "jax.nn.initializers.kaiming_normal")(\[in_axis, out_axis, ...\]) | Builds a He normal initializer (aka Kaiming normal initializer). |
| [`kaiming_uniform`](_autosummary/jax.nn.initializers.kaiming_uniform.html#jax.nn.initializers.kaiming_uniform "jax.nn.initializers.kaiming_uniform")(\[in_axis, out_axis, ...\]) | Builds a He uniform initializer (aka Kaiming uniform initializer). |
| [`lecun_normal`](_autosummary/jax.nn.initializers.lecun_normal.html#jax.nn.initializers.lecun_normal "jax.nn.initializers.lecun_normal")(\[in_axis, out_axis, ...\]) | Builds a Lecun normal initializer. |
| [`lecun_uniform`](_autosummary/jax.nn.initializers.lecun_uniform.html#jax.nn.initializers.lecun_uniform "jax.nn.initializers.lecun_uniform")(\[in_axis, out_axis, ...\]) | Builds a Lecun uniform initializer. |
| [`normal`](_autosummary/jax.nn.initializers.normal.html#jax.nn.initializers.normal "jax.nn.initializers.normal")(\[stddev, dtype\]) | Builds an initializer that returns real normally-distributed random arrays. |
| [`ones`](_autosummary/jax.nn.initializers.ones.html#jax.nn.initializers.ones "jax.nn.initializers.ones")(key, shape\[, dtype, out_sharding\]) | An initializer that returns a constant array full of ones. |
| [`orthogonal`](_autosummary/jax.nn.initializers.orthogonal.html#jax.nn.initializers.orthogonal "jax.nn.initializers.orthogonal")(\[scale, column_axis, dtype\]) | Builds an initializer that returns uniformly distributed orthogonal matrices. |
| [`truncated_normal`](_autosummary/jax.nn.initializers.truncated_normal.html#jax.nn.initializers.truncated_normal "jax.nn.initializers.truncated_normal")(\[stddev, dtype, lower, upper\]) | Builds an initializer that returns truncated-normal random arrays. |
| [`uniform`](_autosummary/jax.nn.initializers.uniform.html#jax.nn.initializers.uniform "jax.nn.initializers.uniform")(\[scale, dtype\]) | Builds an initializer that returns real uniformly-distributed random arrays. |
| [`variance_scaling`](_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling "jax.nn.initializers.variance_scaling")(scale, mode, distribution) | Initializer that adapts its scale to the shape of the weights tensor. |
| [`xavier_normal`](_autosummary/jax.nn.initializers.xavier_normal.html#jax.nn.initializers.xavier_normal "jax.nn.initializers.xavier_normal")(\[in_axis, out_axis, ...\]) | Builds a Glorot normal initializer (aka Xavier normal initializer). |
| [`xavier_uniform`](_autosummary/jax.nn.initializers.xavier_uniform.html#jax.nn.initializers.xavier_uniform "jax.nn.initializers.xavier_uniform")(\[in_axis, out_axis, ...\]) | Builds a Glorot uniform initializer (aka Xavier uniform initializer). |
| [`zeros`](_autosummary/jax.nn.initializers.zeros.html#jax.nn.initializers.zeros "jax.nn.initializers.zeros")(key, shape\[, dtype, out_sharding\]) | An initializer that returns a constant array full of zeros. |
| [`Initializer`](_autosummary/jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")(\*args, \*\*kwargs) | Protocol for initializers returned by [`jax.nn.initializers`](#module-jax.nn.initializers "jax.nn.initializers") APIs. |

[](jax.nn.html "previous page")

previous

`jax.nn` module

[](_autosummary/jax.nn.initializers.constant.html "next page")

next

jax.nn.initializers.constant

Contents

- [Initializers](#initializers)

By The JAX authors

© Copyright 2024, The JAX Authors.\
