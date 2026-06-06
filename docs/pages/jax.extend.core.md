- [](index.html)
- [API Reference](jax.html)
- [`jax.extend` module](jax.extend.html)
- `jax.extend.core` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.extend.core.rst "Download source file")
-  .pdf

# jax.extend.core module

# `jax.extend.core` module[\#](#module-jax.extend.core "Link to this heading")

|  |  |
|----|----|
| [`AbstractToken`](_autosummary/jax.extend.core.AbstractToken.html#jax.extend.core.AbstractToken "jax.extend.core.AbstractToken")() |  |
| [`CallPrimitive`](_autosummary/jax.extend.core.CallPrimitive.html#jax.extend.core.CallPrimitive "jax.extend.core.CallPrimitive")(name) |  |
| [`ClosedJaxpr`](_autosummary/jax.extend.core.ClosedJaxpr.html#jax.extend.core.ClosedJaxpr "jax.extend.core.ClosedJaxpr")(jaxpr, consts) |  |
| [`DebugInfo`](_autosummary/jax.extend.core.DebugInfo.html#jax.extend.core.DebugInfo "jax.extend.core.DebugInfo")(traced_for, func_src_info, ...) | Debugging info about a func, its arguments, and results. |
| [`DropVar`](_autosummary/jax.extend.core.DropVar.html#jax.extend.core.DropVar "jax.extend.core.DropVar")(aval) |  |
| [`Effect`](_autosummary/jax.extend.core.Effect.html#jax.extend.core.Effect "jax.extend.core.Effect")() | A generic side-effect. |
| [`Effects`](_autosummary/jax.extend.core.Effects.html#jax.extend.core.Effects "jax.extend.core.Effects") | A set is a finite, iterable container. |
| [`InconclusiveDimensionOperation`](_autosummary/jax.extend.core.InconclusiveDimensionOperation.html#jax.extend.core.InconclusiveDimensionOperation "jax.extend.core.InconclusiveDimensionOperation") | Raised when we cannot conclusively compute with symbolic dimensions. |
| [`Jaxpr`](_autosummary/jax.extend.core.Jaxpr.html#jax.extend.core.Jaxpr "jax.extend.core.Jaxpr")(constvars, invars, outvars, eqns\[, ...\]) |  |
| [`JaxprEqn`](_autosummary/jax.extend.core.JaxprEqn.html#jax.extend.core.JaxprEqn "jax.extend.core.JaxprEqn")(invars, outvars, primitive, params, ...) |  |
| [`JaxprTypeError`](_autosummary/jax.extend.core.JaxprTypeError.html#jax.extend.core.JaxprTypeError "jax.extend.core.JaxprTypeError") |  |
| [`Literal`](_autosummary/jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")(val, aval) |  |
| [`Primitive`](_autosummary/jax.extend.core.Primitive.html#jax.extend.core.Primitive "jax.extend.core.Primitive")(name) |  |
| [`Token`](_autosummary/jax.extend.core.Token.html#jax.extend.core.Token "jax.extend.core.Token")(buf) |  |
| [`TraceTag`](_autosummary/jax.extend.core.TraceTag.html#jax.extend.core.TraceTag "jax.extend.core.TraceTag")() |  |
| [`Var`](_autosummary/jax.extend.core.Var.html#jax.extend.core.Var "jax.extend.core.Var")(aval\[, initial_qdd, final_qdd\]) |  |
| [`array_types`](_autosummary/jax.extend.core.array_types.html#jax.extend.core.array_types "jax.extend.core.array_types") | set() -\> new empty set object set(iterable) -\> new set object |
| [`call_impl`](_autosummary/jax.extend.core.call_impl.html#jax.extend.core.call_impl "jax.extend.core.call_impl")(f, \*args, \*\*params) |  |
| [`check_jaxpr`](_autosummary/jax.extend.core.check_jaxpr.html#jax.extend.core.check_jaxpr "jax.extend.core.check_jaxpr")(jaxpr) | Checks well-formedness of a jaxpr. |
| [`concrete_or_error`](_autosummary/jax.extend.core.concrete_or_error.html#jax.extend.core.concrete_or_error "jax.extend.core.concrete_or_error")(force, val\[, context\]) | Like force(val), but gives the context in the error message. |
| [`find_top_trace`](_autosummary/jax.extend.core.find_top_trace.html#jax.extend.core.find_top_trace "jax.extend.core.find_top_trace")(\_) |  |
| [`gensym`](_autosummary/jax.extend.core.gensym.html#jax.extend.core.gensym "jax.extend.core.gensym")() |  |
| [`get_opaque_trace_state`](_autosummary/jax.extend.core.get_opaque_trace_state.html#jax.extend.core.get_opaque_trace_state "jax.extend.core.get_opaque_trace_state")(\[convention\]) |  |
| [`jaxpr_as_fun`](_autosummary/jax.extend.core.jaxpr_as_fun.html#jax.extend.core.jaxpr_as_fun "jax.extend.core.jaxpr_as_fun") |  |
| [`jaxprs_in_params`](_autosummary/jax.extend.core.jaxprs_in_params.html#jax.extend.core.jaxprs_in_params "jax.extend.core.jaxprs_in_params")(params) |  |
| [`mapped_aval`](_autosummary/jax.extend.core.mapped_aval.html#jax.extend.core.mapped_aval "jax.extend.core.mapped_aval")(size, axis, aval) |  |
| [`new_jaxpr_eqn`](_autosummary/jax.extend.core.new_jaxpr_eqn.html#jax.extend.core.new_jaxpr_eqn "jax.extend.core.new_jaxpr_eqn")(invars, outvars, primitive, ...) |  |
| [`no_effects`](_autosummary/jax.extend.core.no_effects.html#jax.extend.core.no_effects "jax.extend.core.no_effects") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`nonempty_axis_env_DO_NOT_USE`](_autosummary/jax.extend.core.nonempty_axis_env_DO_NOT_USE.html#jax.extend.core.nonempty_axis_env_DO_NOT_USE "jax.extend.core.nonempty_axis_env_DO_NOT_USE")() |  |
| [`primal_dtype_to_tangent_dtype`](_autosummary/jax.extend.core.primal_dtype_to_tangent_dtype.html#jax.extend.core.primal_dtype_to_tangent_dtype "jax.extend.core.primal_dtype_to_tangent_dtype")(primal_dtype) |  |
| [`primitives`](_autosummary/jax.extend.core.primitives.html#module-jax.extend.core.primitives "jax.extend.core.primitives") |  |
| [`set_current_trace`](_autosummary/jax.extend.core.set_current_trace.html#jax.extend.core.set_current_trace "jax.extend.core.set_current_trace") | alias of `SetCurrentTraceContextManager` |
| [`subjaxprs`](_autosummary/jax.extend.core.subjaxprs.html#jax.extend.core.subjaxprs "jax.extend.core.subjaxprs")(jaxpr) | Generator for all subjaxprs found in the params of jaxpr.eqns. |
| [`take_current_trace`](_autosummary/jax.extend.core.take_current_trace.html#jax.extend.core.take_current_trace "jax.extend.core.take_current_trace") | alias of `TakeCurrentTraceContextManager` |
| [`unmapped_aval`](_autosummary/jax.extend.core.unmapped_aval.html#jax.extend.core.unmapped_aval "jax.extend.core.unmapped_aval")(size, axis, aval\[, ...\]) |  |
| [`unsafe_am_i_under_a_jit_DO_NOT_USE`](_autosummary/jax.extend.core.unsafe_am_i_under_a_jit_DO_NOT_USE.html#jax.extend.core.unsafe_am_i_under_a_jit_DO_NOT_USE "jax.extend.core.unsafe_am_i_under_a_jit_DO_NOT_USE")() |  |
| [`unsafe_am_i_under_a_vmap_DO_NOT_USE`](_autosummary/jax.extend.core.unsafe_am_i_under_a_vmap_DO_NOT_USE.html#jax.extend.core.unsafe_am_i_under_a_vmap_DO_NOT_USE "jax.extend.core.unsafe_am_i_under_a_vmap_DO_NOT_USE")() |  |
| [`unsafe_get_axis_names_DO_NOT_USE`](_autosummary/jax.extend.core.unsafe_get_axis_names_DO_NOT_USE.html#jax.extend.core.unsafe_get_axis_names_DO_NOT_USE "jax.extend.core.unsafe_get_axis_names_DO_NOT_USE")() |  |
| [`valid_jaxtype`](_autosummary/jax.extend.core.valid_jaxtype.html#jax.extend.core.valid_jaxtype "jax.extend.core.valid_jaxtype")(x) |  |

[](_autosummary/jax.extend.backend.register_backend_factory.html "previous page")

previous

jax.extend.backend.register_backend_factory

[](_autosummary/jax.extend.core.AbstractToken.html "next page")

next

jax.extend.core.AbstractToken

By The JAX authors

© Copyright 2024, The JAX Authors.\
