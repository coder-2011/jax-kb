- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.ChipVersion

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.ChipVersion.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.ChipVersion

## Contents

- [`ChipVersion`](#jax.experimental.pallas.tpu.ChipVersion)
  - [`ChipVersion.__init__()`](#jax.experimental.pallas.tpu.ChipVersion.__init__)

# jax.experimental.pallas.tpu.ChipVersion[\#](#jax-experimental-pallas-tpu-chipversion "Link to this heading")

*class* jax.experimental.pallas.tpu.ChipVersion(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py#L34-L109)[\#](#jax.experimental.pallas.tpu.ChipVersion "Link to this definition")  
TPU chip version.

The following table summarizes the differences between TPU versions:

| Version | Physical TensorCores per chip | Lite chip | Megacore support |
|---------|-------------------------------|-----------|------------------|
| v2      | 2                             | No        | No               |
| v3      | 2                             | No        | No               |
| v4i     | 1                             | Yes       | No               |
| v4      | 2                             | No        | Yes              |
| v5e     | 1                             | Yes       | No               |
| v5p     | 2                             | No        | Yes              |
| v6e     | 1                             | Yes       | No               |
| 7       | 2                             | No        | No               |
| 7x      | 2                             | No        | No               |
| 8i      | 2                             | No        | No               |

\_\_init\_\_(*\*args*, *\*\*kwds*)[\#](#jax.experimental.pallas.tpu.ChipVersion.__init__ "Link to this definition")  

Attributes

|                                      |     |
|--------------------------------------|-----|
| `num_physical_tensor_cores_per_chip` |     |
| `supports_megacore`                  |     |
| `is_lite`                            |     |
| `TPU_V2`                             |     |
| `TPU_V3`                             |     |
| `TPU_V4I`                            |     |
| `TPU_V4`                             |     |
| `TPU_V5E`                            |     |
| `TPU_V5P`                            |     |
| `TPU_V6E`                            |     |
| `TPU_7`                              |     |
| `TPU_7X`                             |     |
| `TPU_8I`                             |     |

[](../jax.experimental.pallas.tpu.html "previous page")

previous

`jax.experimental.pallas.tpu` module

[](jax.experimental.pallas.tpu.CompilerParams.html "next page")

next

jax.experimental.pallas.tpu.CompilerParams

Contents

- [`ChipVersion`](#jax.experimental.pallas.tpu.ChipVersion)
  - [`ChipVersion.__init__()`](#jax.experimental.pallas.tpu.ChipVersion.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
