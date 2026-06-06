- [](../index.html)
- [API Reference](../jax.html)
- jax.ShapeDtypeStruct

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ShapeDtypeStruct.rst "Download source file")
-  .pdf

# jax.ShapeDtypeStruct

## Contents

- [`ShapeDtypeStruct`](#jax.ShapeDtypeStruct)
  - [`ShapeDtypeStruct.__init__()`](#jax.ShapeDtypeStruct.__init__)

# jax.ShapeDtypeStruct[\#](#jax-shapedtypestruct "Link to this heading")

*class* jax.ShapeDtypeStruct(*shape*, *dtype*, *\**, *sharding=None*, *weak_type=False*, *manual_axis_type=None*, *is_ref=False*)[\#](#jax.ShapeDtypeStruct "Link to this definition")  
A container for the shape, dtype, and other static attributes of an array.

`ShapeDtypeStruct` is often used in conjunction with [`jax.eval_shape()`](jax.eval_shape.html#jax.eval_shape "jax.eval_shape").

Parameters:  
- **shape** (*Any*) – a sequence of integers representing an array shape

- **dtype** (*Any*) – a dtype-like object

- **sharding** – (optional) a `jax.Sharding` object

- **weak_type** (*Any*)

- **manual_axis_type** (*Any*)

- **is_ref** (*Any*)

\_\_init\_\_(*shape*, *dtype*, *\**, *sharding=None*, *weak_type=False*, *manual_axis_type=None*, *is_ref=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L3787-L3819)[\#](#jax.ShapeDtypeStruct.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.ShapeDtypeStruct.__init__ "jax.ShapeDtypeStruct.__init__")(shape, dtype, \*\[, sharding, ...\]) |  |
| `like`(x) |  |
| `update`(\*\*kwargs) |  |

Attributes

|                    |     |
|--------------------|-----|
| `shape`            |     |
| `dtype`            |     |
| `weak_type`        |     |
| `manual_axis_type` |     |
| `is_ref`           |     |
| `format`           |     |
| `ndim`             |     |
| `sharding`         |     |
| `size`             |     |

[](jax.eval_shape.html "previous page")

previous

jax.eval_shape

[](jax.device_put.html "next page")

next

jax.device_put

Contents

- [`ShapeDtypeStruct`](#jax.ShapeDtypeStruct)
  - [`ShapeDtypeStruct.__init__()`](#jax.ShapeDtypeStruct.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
