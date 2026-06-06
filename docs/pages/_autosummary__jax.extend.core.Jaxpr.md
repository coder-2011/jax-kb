- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.Jaxpr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.Jaxpr.rst "Download source file")
-  .pdf

# jax.extend.core.Jaxpr

## Contents

- [`Jaxpr`](#jax.extend.core.Jaxpr)
  - [`Jaxpr.__init__()`](#jax.extend.core.Jaxpr.__init__)

# jax.extend.core.Jaxpr[\#](#jax-extend-core-jaxpr "Link to this heading")

*class* jax.extend.core.Jaxpr(*constvars*, *invars*, *outvars*, *eqns*, *effects=frozenset({})*, *debug_info=None*, *is_high=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L97-L222)[\#](#jax.extend.core.Jaxpr "Link to this definition")  
Parameters:  
- **constvars** (*Sequence\[*[*Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*\]*)

- **invars** (*Sequence\[*[*Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*\]*)

- **outvars** (*Sequence\[Atom\]*)

- **eqns** (*Sequence\[*[*JaxprEqn*](jax.extend.core.JaxprEqn.html#jax.extend.core.JaxprEqn "jax.extend.core.JaxprEqn")*\]*)

- **effects** (*Effects*)

- **debug_info** ([*DebugInfo*](jax.extend.core.DebugInfo.html#jax.extend.core.DebugInfo "jax.extend.core.DebugInfo"))

- **is_high** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

\_\_init\_\_(*constvars*, *invars*, *outvars*, *eqns*, *effects=frozenset({})*, *debug_info=None*, *is_high=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L155-L187)[\#](#jax.extend.core.Jaxpr.__init__ "Link to this definition")  
Parameters:  
- **constvars** (*Sequence\[*[*Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*\]*) – list of variables introduced for constants. Array constants are replaced with such variables while scalar constants are kept inline.

- **invars** (*Sequence\[*[*Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*\]*) – list of input variables. Together, constvars and invars are the inputs to the Jaxpr.

- **outvars** (*Sequence\[Atom\]*) – list of output atoms.

- **eqns** (*Sequence\[*[*JaxprEqn*](jax.extend.core.JaxprEqn.html#jax.extend.core.JaxprEqn "jax.extend.core.JaxprEqn")*\]*) – list of equations.

- **effects** (*Effects*) – set of effects. The effects on a jaxpr are a superset of the union of the effects for each equation.

- **debug_info** ([*DebugInfo*](jax.extend.core.DebugInfo.html#jax.extend.core.DebugInfo "jax.extend.core.DebugInfo")) – debugging information.

- **is_high** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.core.Jaxpr.__init__ "jax.extend.core.Jaxpr.__init__")(constvars, invars, outvars, eqns\[, ...\]) |  |
| `pretty_print`(\*\[, source_info, print_shapes, ...\]) |  |
| `replace`(\*\*kwargs) |  |

Attributes

|                   |     |
|-------------------|-----|
| `constvars`       |     |
| `debug_info`      |     |
| `effects`         |     |
| `eqns`            |     |
| `final_aval_qdds` |     |
| `in_aval_qdds`    |     |
| `in_avals`        |     |
| `invars`          |     |
| `is_high`         |     |
| `out_avals`       |     |
| `outvars`         |     |

[](jax.extend.core.InconclusiveDimensionOperation.html "previous page")

previous

jax.extend.core.InconclusiveDimensionOperation

[](jax.extend.core.JaxprEqn.html "next page")

next

jax.extend.core.JaxprEqn

Contents

- [`Jaxpr`](#jax.extend.core.Jaxpr)
  - [`Jaxpr.__init__()`](#jax.extend.core.Jaxpr.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
