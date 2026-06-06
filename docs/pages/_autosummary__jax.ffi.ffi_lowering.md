- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ffi` module](../jax.ffi.html)
- jax.ffi.ffi_lowering

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ffi.ffi_lowering.rst "Download source file")
-  .pdf

# jax.ffi.ffi_lowering

## Contents

- [`ffi_lowering()`](#jax.ffi.ffi_lowering)

# jax.ffi.ffi_lowering[\#](#jax-ffi-ffi-lowering "Link to this heading")

jax.ffi.ffi_lowering(*call_target_name*, *\**, *operand_layouts=None*, *result_layouts=None*, *backend_config=None*, *skip_ffi_layout_processing=False*, *\*\*lowering_args*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ffi.py#L286-L337)[\#](#jax.ffi.ffi_lowering "Link to this definition")  
Build a lowering rule for an foreign function interface (FFI) target.

By default, this lowering rule can use the input and output abstract values to compute the input and output types and shapes for the custom call, assuming row-major layouts.

Note that layouts passed to this function as tuples should be in minor-to-major order (as expected by XLA) rather than major-to-minor as used by [`ffi_call()`](jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") and `Layout`.

If keyword arguments are passed to the lowering rule, these are treated as attributes, and added to backend_config.

Parameters:  
- **call_target_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The name of the custom call target.

- **operand_layouts** (*Sequence\[FfiLayoutOptions\]* *\|* *None*) – A sequence of layouts (dimension orders) for each operand. By default, the operands are assumed to be row-major.

- **result_layouts** (*Sequence\[FfiLayoutOptions\]* *\|* *None*) – A sequence of layouts (dimension orders) for each result. By default, the results are assumed to be row-major.

- **backend_config** (*Mapping\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *ir.Attribute\]* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Configuration data for the custom call. Any keyword arguments passed to the lowering rule will added to this dictionary.

- **lowering_args** (*Any*) – Any other arguments to `mlir.custom_call()` will also be passed through if provided as extra arguments to this function.

- **skip_ffi_layout_processing** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If true, skip processing of operand and result layout arguments passed to the lowering rule.

Return type:  
mlir.LoweringRule

[](jax.ffi.ffi_call.html "previous page")

previous

jax.ffi.ffi_call

[](jax.ffi.pycapsule.html "next page")

next

jax.ffi.pycapsule

Contents

- [`ffi_lowering()`](#jax.ffi.ffi_lowering)

By The JAX authors

© Copyright 2024, The JAX Authors.\
