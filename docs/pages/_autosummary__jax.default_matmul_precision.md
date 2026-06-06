- [](../index.html)
- [API Reference](../jax.html)
- jax.default_matmul_precision

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.default_matmul_precision.rst "Download source file")
-  .pdf

# jax.default_matmul_precision

## Contents

- [`default_matmul_precision`](#jax.default_matmul_precision)

# jax.default_matmul_precision[\#](#jax-default-matmul-precision "Link to this heading")

jax.default_matmul_precision *= \<jax.\_src.config.State object\>*[\#](#jax.default_matmul_precision "Link to this definition")  
Context manager for jax_default_matmul_precision config option.

Control the default matmul and conv precision for 32bit inputs.

Some platforms, like TPU, offer configurable precision levels for matrix multiplication and convolution computations, trading off accuracy for speed. The precision can be controlled for each operation; for example, see the [`jax.lax.conv_general_dilated()`](jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated "jax.lax.conv_general_dilated") and [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot") docstrings. But it can be useful to control the default behavior obtained when an operation is not given a specific precision.

This option can be used to control the default precision level for computations involved in matrix multiplication and convolution on 32bit inputs. The levels roughly describe the precision at which scalar products are computed. The ‘bfloat16’ option is the fastest and least precise; ‘float32’ is similar to full float32 precision; ‘tensorfloat32’ is intermediate.

This parameter can also be used to specify an accumulation “algorithm” for functions that perform matrix multiplications, like [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot"). To specify an algorithm, set this option to the name of a [`DotAlgorithmPreset`](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax.lax.DotAlgorithmPreset").

Parameters:  
**new_val** (*Any*)

[](jax.default_device.html "previous page")

previous

jax.default_device

[](jax.default_prng_impl.html "next page")

next

jax.default_prng_impl

Contents

- [`default_matmul_precision`](#jax.default_matmul_precision)

By The JAX authors

© Copyright 2024, The JAX Authors.\
