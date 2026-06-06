- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.JaxprEqn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.JaxprEqn.rst "Download source file")
-  .pdf

# jax.extend.core.JaxprEqn

## Contents

- [`JaxprEqn`](#jax.extend.core.JaxprEqn)
  - [`JaxprEqn.__init__()`](#jax.extend.core.JaxprEqn.__init__)

# jax.extend.core.JaxprEqn[\#](#jax-extend-core-jaxpreqn "Link to this heading")

*class* jax.extend.core.JaxprEqn(*invars*, *outvars*, *primitive*, *params*, *effs*, *source_info*, *ctx*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L422-L473)[\#](#jax.extend.core.JaxprEqn "Link to this definition")  
Parameters:  
- **invars** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[Atom\]*)

- **outvars** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*\]*)

- **primitive** ([*Primitive*](jax.extend.core.Primitive.html#jax.extend.core.Primitive "jax.extend.core.Primitive"))

- **params** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Any\]*)

- **source_info** (*source_info_util.SourceInfo*)

- **ctx** (*JaxprEqnContext*)

\_\_init\_\_(*invars*, *outvars*, *primitive*, *params*, *effs*, *source_info*, *ctx*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L441-L450)[\#](#jax.extend.core.JaxprEqn.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.core.JaxprEqn.__init__ "jax.extend.core.JaxprEqn.__init__")(invars, outvars, primitive, params, ...) |  |
| `replace`(\[invars, outvars, primitive, ...\]) |  |

Attributes

|               |     |
|---------------|-----|
| `invars`      |     |
| `outvars`     |     |
| `primitive`   |     |
| `params`      |     |
| `effects`     |     |
| `source_info` |     |
| `ctx`         |     |

[](jax.extend.core.Jaxpr.html "previous page")

previous

jax.extend.core.Jaxpr

[](jax.extend.core.JaxprTypeError.html "next page")

next

jax.extend.core.JaxprTypeError

Contents

- [`JaxprEqn`](#jax.extend.core.JaxprEqn)
  - [`JaxprEqn.__init__()`](#jax.extend.core.JaxprEqn.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
