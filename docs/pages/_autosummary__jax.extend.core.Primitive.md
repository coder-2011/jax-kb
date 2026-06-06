- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.Primitive

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.Primitive.rst "Download source file")
-  .pdf

# jax.extend.core.Primitive

## Contents

- [`Primitive`](#jax.extend.core.Primitive)
  - [`Primitive.__init__()`](#jax.extend.core.Primitive.__init__)

# jax.extend.core.Primitive[\#](#jax-extend-core-primitive "Link to this heading")

*class* jax.extend.core.Primitive(*name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L622-L740)[\#](#jax.extend.core.Primitive "Link to this definition")  
Parameters:  
**name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

\_\_init\_\_(*name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L638-L640)[\#](#jax.extend.core.Primitive.__init__ "Link to this definition")  
Parameters:  
**name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.core.Primitive.__init__ "jax.extend.core.Primitive.__init__")(name) |  |
| `abstract_eval`(\*args, \*\*params) |  |
| `bind`(\*args, \*\*params) |  |
| `bind_with_trace`(trace, args, avals, params, /) |  |
| `def_abstract_eval`(abstract_eval) |  |
| `def_bind_with_trace`(bind_with_trace) |  |
| `def_effectful_abstract_eval`(...) |  |
| `def_effectful_abstract_eval2`(abstract_eval) |  |
| `def_impl`(impl) |  |
| `get_bind_params`(params) |  |
| `impl`(\*args, \*\*params) |  |
| `is_high`(\*avals, \*\*params) |  |
| `to_lojax`(\*args, \*\*params) |  |

Attributes

|                         |     |
|-------------------------|-----|
| `call_primitive`        |     |
| `is_effectful`          |     |
| `multiple_results`      |     |
| `ref_allocating`        |     |
| `ref_primitive`         |     |
| `skip_canonicalization` |     |
| `name`                  |     |

[](jax.extend.core.Literal.html "previous page")

previous

jax.extend.core.Literal

[](jax.extend.core.Token.html "next page")

next

jax.extend.core.Token

Contents

- [`Primitive`](#jax.extend.core.Primitive)
  - [`Primitive.__init__()`](#jax.extend.core.Primitive.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
