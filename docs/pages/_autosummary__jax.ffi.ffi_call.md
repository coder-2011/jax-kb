- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ffi` module](../jax.ffi.html)
- jax.ffi.ffi_call

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ffi.ffi_call.rst "Download source file")
-  .pdf

# jax.ffi.ffi_call

## Contents

- [`ffi_call()`](#jax.ffi.ffi_call)

# jax.ffi.ffi_call[\#](#jax-ffi-ffi-call "Link to this heading")

jax.ffi.ffi_call(*target_name: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*, *result_shape_dtypes: ResultMetadata*, *\**, *has_side_effect: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *vmap_method: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *input_layouts: Sequence\[FfiLayoutOptions\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *output_layouts: FfiLayoutOptions \| Sequence\[FfiLayoutOptions\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *input_output_aliases: [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *custom_call_api_version: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 4*, *legacy_backend_config: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[..., [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ffi.py#L415-L582)[\#](#jax.ffi.ffi_call "Link to this definition")\
jax.ffi.ffi_call(*target_name: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*, *result_shape_dtypes: Sequence\[ResultMetadata\]*, *\**, *has_side_effect: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *vmap_method: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *input_layouts: Sequence\[FfiLayoutOptions\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *output_layouts: FfiLayoutOptions \| Sequence\[FfiLayoutOptions\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *input_output_aliases: [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *custom_call_api_version: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 4*, *legacy_backend_config: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[..., Sequence\[[Array](jax.Array.html#jax.Array "jax.Array")\]\]  
Call a foreign function interface (FFI) target.

See the [Foreign function interface (FFI)](../ffi.html#ffi-tutorial) tutorial for more information.

Like [`pure_callback()`](jax.pure_callback.html#jax.pure_callback "jax.pure_callback"), the behavior of `ffi_call` under [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") depends on the value of `vmap_method`. See the [`pure_callback()`](jax.pure_callback.html#jax.pure_callback "jax.pure_callback") documentation for more details about the allowed values and examples of their behavior.

The current default behavior is to use `vmap_method="sequential"` when not specified, but this behavior is deprecated, and in the future, the default will be to raise a `NotImplementedError` unless `vmap_method` is explicitly specified.

Parameters:  
- **target_name** – the name of the XLA FFI custom call target that was registered using [`register_ffi_target()`](jax.ffi.register_ffi_target.html#jax.ffi.register_ffi_target "jax.ffi.register_ffi_target").

- **result_shape_dtypes** – an object, or sequence of objects, with `shape` and `dtype` attributes which are expected to match the shape and dtype of the custom call output or outputs. [`ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") is often used to define the elements of `result_shape_dtypes`. `jax.core.abstract_token` may be used to represent a token-typed output.

- **has_side_effect** – boolean specifying whether the custom call has side effects. When `True`, the FFI call will be executed even when the outputs are not used.

- **vmap_method** – string specifying how the FFI call transforms under [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") as described above.

- **input_layouts** – a sequence of layouts for each input argument. In each case, the layout can be (a) `None` indicating that this input is in default row-major order, (b) a `Layout` specifying the axis order, or (c) a sequence of integers specifying the major-to-minor axis ordering. Users who are familiar with XLA layouts should note that this function expects layouts in major-to-minor order instead of the minor-to-major order that XLA uses. For example, a batch of row-major matrices could be specified using the layout `[0,`` ``1,`` ``2]`, whereas a batch of column-major matrices would have layout `[0,`` ``2,`` ``1]`. In both of these examples, the leading/batch dimension is the “slowest” axis. The `input_layouts` parameter should be used to request the memory layout expected by the FFI call target, and XLA will ensure that the buffers have the correct layouts before the handler is executed.

- **output_layouts** – like `input_layouts`, but specifying the required layouts for the output arrays.

- **input_output_aliases** – a dictionary where the keys are input indices and the values are output indices. This mapping indicates which output arrays alias specific input arrays.

- **custom_call_api_version** – the version number of the custom call API implemented by the FFI target `target_name`. The only formally supported version is the typed FFI API with `custom_call_api_version=4`, but earlier unsupported custom calls can be executed using this argument.

- **legacy_backend_config** – for legacy targets implemented using `custom_call_api_version<4`, attributes are passed using the opaque string representation provided by this argument. This parameter cannot be used with `custom_call_api_version>=4`.

Returns:  
A function that can be called with the input arrays as positional arguments to execute the FFI handler. Any keyword arguments are passed as named attributes to the FFI handler using XLA’s FFI interface.

[](../jax.ffi.html "previous page")

previous

`jax.ffi` module

[](jax.ffi.ffi_lowering.html "next page")

next

jax.ffi.ffi_lowering

Contents

- [`ffi_call()`](#jax.ffi.ffi_call)

By The JAX authors

© Copyright 2024, The JAX Authors.\
