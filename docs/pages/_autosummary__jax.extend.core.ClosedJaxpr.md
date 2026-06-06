- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.ClosedJaxpr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.ClosedJaxpr.rst "Download source file")
-  .pdf

# jax.extend.core.ClosedJaxpr

## Contents

- [`ClosedJaxpr`](#jax.extend.core.ClosedJaxpr)
  - [`ClosedJaxpr.__init__()`](#jax.extend.core.ClosedJaxpr.__init__)

# jax.extend.core.ClosedJaxpr[\#](#jax-extend-core-closedjaxpr "Link to this heading")

*class* jax.extend.core.ClosedJaxpr(*jaxpr*, *consts*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L246-L312)[\#](#jax.extend.core.ClosedJaxpr "Link to this definition")  
Parameters:  
- **jaxpr** ([*Jaxpr*](jax.extend.core.Jaxpr.html#jax.extend.core.Jaxpr "jax.extend.core.Jaxpr"))

- **consts** (*Sequence*)

\_\_init\_\_(*jaxpr*, *consts*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L264-L269)[\#](#jax.extend.core.ClosedJaxpr.__init__ "Link to this definition")  
Parameters:  
- **jaxpr** ([*Jaxpr*](jax.extend.core.Jaxpr.html#jax.extend.core.Jaxpr "jax.extend.core.Jaxpr"))

- **consts** (*Sequence*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.core.ClosedJaxpr.__init__ "jax.extend.core.ClosedJaxpr.__init__")(jaxpr, consts) |  |
| `map_jaxpr`(f) |  |
| `pretty_print`(\*\[, source_info, print_shapes, ...\]) |  |
| `replace`(\*\[, jaxpr, consts\]) |  |

Attributes

|                   |     |
|-------------------|-----|
| `consts`          |     |
| `constvars`       |     |
| `debug_info`      |     |
| `effects`         |     |
| `eqns`            |     |
| `final_aval_qdds` |     |
| `in_aval_qdds`    |     |
| `in_avals`        |     |
| `invars`          |     |
| `is_high`         |     |
| `jaxpr`           |     |
| `literals`        |     |
| `out_avals`       |     |
| `outvars`         |     |

[](jax.extend.core.CallPrimitive.html "previous page")

previous

jax.extend.core.CallPrimitive

[](jax.extend.core.DebugInfo.html "next page")

next

jax.extend.core.DebugInfo

Contents

- [`ClosedJaxpr`](#jax.extend.core.ClosedJaxpr)
  - [`ClosedJaxpr.__init__()`](#jax.extend.core.ClosedJaxpr.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
