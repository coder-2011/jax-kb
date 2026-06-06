- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.lowering` module](../jax.extend.lowering.html)
- jax.extend.lowering.LoweringRuleContext

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.lowering.LoweringRuleContext.rst "Download source file")
-  .pdf

# jax.extend.lowering.LoweringRuleContext

## Contents

- [`LoweringRuleContext`](#jax.extend.lowering.LoweringRuleContext)
  - [`LoweringRuleContext.__init__()`](#jax.extend.lowering.LoweringRuleContext.__init__)

# jax.extend.lowering.LoweringRuleContext[\#](#jax-extend-lowering-loweringrulecontext "Link to this heading")

*class* jax.extend.lowering.LoweringRuleContext(*module_context*, *name_stack*, *traceback*, *primitive*, *avals_in*, *avals_out*, *tokens_in*, *tokens_out*, *const_lowering*, *axis_size_env=None*, *dim_var_values=()*, *jaxpr_eqn_ctx=None*, *platforms=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/interpreters/mlir.py#L898-L949)[\#](#jax.extend.lowering.LoweringRuleContext "Link to this definition")  
Per-rule context information for MLIR lowering.

Parameters:  
- **module_context** (*ModuleContext*)

- **name_stack** (*source_info_util.NameStack*)

- **traceback** (*xc.Traceback* *\|* *None*)

- **primitive** ([*core.Primitive*](jax.extend.core.Primitive.html#jax.extend.core.Primitive "jax.extend.core.Primitive") *\|* *None*)

- **avals_in** (*Sequence\[core.AbstractValue\]*)

- **avals_out** (*Any*)

- **tokens_in** (*TokenSet*)

- **tokens_out** (*TokenSet* *\|* *None*)

- **const_lowering** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *core.AbstractValue\],* *IrValues\]*)

- **axis_size_env** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*core.Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*,* *ir.Value\]* *\|* *None*)

- **dim_var_values** (*Sequence\[ir.Value\]*)

- **jaxpr_eqn_ctx** (*core.JaxprEqnContext* *\|* *None*)

- **platforms** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*)

\_\_init\_\_(*module_context*, *name_stack*, *traceback*, *primitive*, *avals_in*, *avals_out*, *tokens_in*, *tokens_out*, *const_lowering*, *axis_size_env=None*, *dim_var_values=()*, *jaxpr_eqn_ctx=None*, *platforms=None*)[\#](#jax.extend.lowering.LoweringRuleContext.__init__ "Link to this definition")  
Parameters:  
- **module_context** (*ModuleContext*)

- **name_stack** (*source_info_util.NameStack*)

- **traceback** (*xc.Traceback* *\|* *None*)

- **primitive** ([*core.Primitive*](jax.extend.core.Primitive.html#jax.extend.core.Primitive "jax.extend.core.Primitive") *\|* *None*)

- **avals_in** (*Sequence\[core.AbstractValue\]*)

- **avals_out** (*Any*)

- **tokens_in** (*TokenSet*)

- **tokens_out** (*TokenSet* *\|* *None*)

- **const_lowering** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *core.AbstractValue\],* *IrValues\]*)

- **axis_size_env** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*core.Var*](jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")*,* *ir.Value\]* *\|* *None*)

- **dim_var_values** (*Sequence\[ir.Value\]*)

- **jaxpr_eqn_ctx** (*core.JaxprEqnContext* *\|* *None*)

- **platforms** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.lowering.LoweringRuleContext.__init__ "jax.extend.lowering.LoweringRuleContext.__init__")(module_context, name_stack, ...\[, ...\]) |  |
| `is_forward_compat`() | Returns true if the lowering parameters are in forward compatibility mode. |
| `replace`(\*\*kw) |  |
| `set_tokens_out`(tokens_out) |  |

Attributes

|                  |     |
|------------------|-----|
| `module_context` |     |
| `name_stack`     |     |
| `traceback`      |     |
| `primitive`      |     |
| `avals_in`       |     |
| `avals_out`      |     |
| `tokens_in`      |     |
| `tokens_out`     |     |
| `const_lowering` |     |
| `axis_size_env`  |     |
| `dim_var_values` |     |
| `jaxpr_eqn_ctx`  |     |
| `platforms`      |     |

[](jax.extend.lowering.JaxIrContext.html "previous page")

previous

jax.extend.lowering.JaxIrContext

[](jax.extend.lowering.upstream_dialects.html "next page")

next

jax.extend.lowering.upstream_dialects

Contents

- [`LoweringRuleContext`](#jax.extend.lowering.LoweringRuleContext)
  - [`LoweringRuleContext.__init__()`](#jax.extend.lowering.LoweringRuleContext.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
