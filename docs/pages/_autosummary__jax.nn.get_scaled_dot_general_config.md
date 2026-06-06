- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.get_scaled_dot_general_config

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.get_scaled_dot_general_config.rst "Download source file")
-  .pdf

# jax.nn.get_scaled_dot_general_config

## Contents

- [`get_scaled_dot_general_config()`](#jax.nn.get_scaled_dot_general_config)

# jax.nn.get_scaled_dot_general_config[\#](#jax-nn-get-scaled-dot-general-config "Link to this heading")

jax.nn.get_scaled_dot_general_config(*mode*, *global_scale=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L1376-L1407)[\#](#jax.nn.get_scaled_dot_general_config "Link to this definition")  
Get quantization configs for scaled_dot_general.

Create quantization configs for the jax.nn.scaled_dot_general.

See also

- [`jax.nn.scaled_dot_general()`](jax.nn.scaled_dot_general.html#jax.nn.scaled_dot_general "jax.nn.scaled_dot_general"): Scaled dot general function.

Parameters:  
- **mode** ([*Literal*](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")*\['nvfp4',* *'mxfp8'\]*)

- **global_scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

[](jax.nn.scaled_matmul.html "previous page")

previous

jax.nn.scaled_matmul

[](jax.nn.scaled_dot_general.html "next page")

next

jax.nn.scaled_dot_general

Contents

- [`get_scaled_dot_general_config()`](#jax.nn.get_scaled_dot_general_config)

By The JAX authors

© Copyright 2024, The JAX Authors.\
