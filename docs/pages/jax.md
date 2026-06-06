- [](index.html)
- API Reference

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.rst "Download source file")
-  .pdf

# API Reference

## Contents

- [Subpackages](#subpackages)
- [Configuration](#configuration)
- [Just-in-time compilation (`jit`)](#just-in-time-compilation-jit)
- [Automatic differentiation](#automatic-differentiation)
- [Vectorization](#vectorization)
- [Parallelization](#parallelization)
- [Customization](#customization)
  - [`custom_jvp`](#custom-jvp)
  - [`custom_vjp`](#custom-vjp)
  - [`custom_batching`](#custom-batching)
- [jax.Array (`jax.Array`)](#jax-array-jax-array)
  - [Array properties and methods](#array-properties-and-methods)
- [Callbacks](#callbacks)
- [Miscellaneous](#miscellaneous)
- [Checkpoint policies](#checkpoint-policies)

# API Reference[\#](#api-reference "Link to this heading")

## Subpackages[\#](#subpackages "Link to this heading")

- [`jax.numpy` module](jax.numpy.html)
- [`jax.scipy` module](jax.scipy.html)
- [`jax.lax` module](jax.lax.html)
- [`jax.random` module](jax.random.html)
- [`jax.sharding` module](jax.sharding.html)
- [`jax.ad_checkpoint` module](jax.ad_checkpoint.html)
- [`jax.debug` module](jax.debug.html)
- [`jax.dlpack` module](jax.dlpack.html)
- [`jax.distributed` module](jax.distributed.html)
- [`jax.dtypes` module](jax.dtypes.html)
- [`jax.ffi` module](jax.ffi.html)
- [`jax.flatten_util` module](jax.flatten_util.html)
- [`jax.image` module](jax.image.html)
- [`jax.nn` module](jax.nn.html)
- [`jax.ops` module](jax.ops.html)
- [`jax.profiler` module](jax.profiler.html)
- [`jax.ref` module](jax.ref.html)
- [`jax.stages` module](jax.stages.html)
- [`jax.test_util` module](jax.test_util.html)
- [`jax.tree` module](jax.tree.html)
- [`jax.tree_util` module](jax.tree_util.html)
- [`jax.typing` module](jax.typing.html)
- [`jax.export` module](jax.export.html)
- [`jax.extend` module](jax.extend.html)
- [`jax.example_libraries` module](jax.example_libraries.html)
- [`jax.experimental` module](jax.experimental.html)

## Configuration[\#](#configuration "Link to this heading")

|  |  |
|----|----|
| [`array_garbage_collection_guard`](_autosummary/jax.array_garbage_collection_guard.html#jax.array_garbage_collection_guard "jax.array_garbage_collection_guard") | Context manager for jax_array_garbage_collection_guard config option. |
| [`config`](_autosummary/jax.config.html#jax.config "jax.config") |  |
| [`check_tracer_leaks`](_autosummary/jax.check_tracer_leaks.html#jax.check_tracer_leaks "jax.check_tracer_leaks") | Context manager for jax_check_tracer_leaks config option. |
| [`checking_leaks`](_autosummary/jax.checking_leaks.html#jax.checking_leaks "jax.checking_leaks") | Context manager for jax_check_tracer_leaks config option. |
| [`debug_nans`](_autosummary/jax.debug_nans.html#jax.debug_nans "jax.debug_nans") | Context manager for jax_debug_nans config option. |
| [`debug_infs`](_autosummary/jax.debug_infs.html#jax.debug_infs "jax.debug_infs") | Context manager for jax_debug_infs config option. |
| [`default_device`](_autosummary/jax.default_device.html#jax.default_device "jax.default_device") | Context manager for jax_default_device config option. |
| [`default_matmul_precision`](_autosummary/jax.default_matmul_precision.html#jax.default_matmul_precision "jax.default_matmul_precision") | Context manager for jax_default_matmul_precision config option. |
| [`default_prng_impl`](_autosummary/jax.default_prng_impl.html#jax.default_prng_impl "jax.default_prng_impl") | Context manager for jax_default_prng_impl config option. |
| [`enable_checks`](_autosummary/jax.enable_checks.html#jax.enable_checks "jax.enable_checks") | Context manager for jax_enable_checks config option. |
| [`enable_custom_prng`](_autosummary/jax.enable_custom_prng.html#jax.enable_custom_prng "jax.enable_custom_prng") | Context manager for jax_enable_custom_prng config option (transient). |
| [`enable_x64`](_autosummary/jax.enable_x64.html#jax.enable_x64 "jax.enable_x64") | Context manager for jax_enable_x64 config option. |
| [`log_compiles`](_autosummary/jax.log_compiles.html#jax.log_compiles "jax.log_compiles") | Context manager for jax_log_compiles config option. |
| [`no_tracing`](_autosummary/jax.no_tracing.html#jax.no_tracing "jax.no_tracing") | Context manager for jax_no_tracing config option. |
| [`numpy_rank_promotion`](_autosummary/jax.numpy_rank_promotion.html#jax.numpy_rank_promotion "jax.numpy_rank_promotion") | Context manager for jax_numpy_rank_promotion config option. |
| [`transfer_guard`](_autosummary/jax.transfer_guard.html#jax.transfer_guard "jax.transfer_guard")(new_val) | A contextmanager to control the transfer guard level for all transfers. |

## Just-in-time compilation (`jit`)[\#](#just-in-time-compilation-jit "Link to this heading")

|  |  |
|----|----|
| [`jit`](_autosummary/jax.jit.html#jax.jit "jax.jit")() | Sets up `fun` for just-in-time compilation with XLA. |
| [`disable_jit`](_autosummary/jax.disable_jit.html#jax.disable_jit "jax.disable_jit")(\[disable\]) | Context manager that disables [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") behavior under its dynamic context. |
| [`ensure_compile_time_eval`](_autosummary/jax.ensure_compile_time_eval.html#jax.ensure_compile_time_eval "jax.ensure_compile_time_eval")() | Context manager to ensure evaluation at trace/compile time (or error). |
| [`make_jaxpr`](_autosummary/jax.make_jaxpr.html#jax.make_jaxpr "jax.make_jaxpr")(\[axis_env, return_shape\]) | Create a function that returns the jaxpr of `fun` given example args. |
| [`eval_shape`](_autosummary/jax.eval_shape.html#jax.eval_shape "jax.eval_shape")(fun, \*args, \*\*kwargs) | Compute the shape/dtype of `fun` without any FLOPs. |
| [`ShapeDtypeStruct`](_autosummary/jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct")(shape, dtype, \*\[, ...\]) | A container for the shape, dtype, and other static attributes of an array. |
| [`device_put`](_autosummary/jax.device_put.html#jax.device_put "jax.device_put")(x\[, device, src, donate, may_alias\]) | Transfers `x` to `device`. |
| [`device_get`](_autosummary/jax.device_get.html#jax.device_get "jax.device_get")(x) | Transfer `x` to host. |
| [`default_backend`](_autosummary/jax.default_backend.html#jax.default_backend "jax.default_backend")() | Returns the platform name of the default XLA backend. |
| [`named_call`](_autosummary/jax.named_call.html#jax.named_call "jax.named_call")(fun, \*\[, name\]) | Adds a user specified name to a function when staging out JAX computations. |
| [`named_scope`](_autosummary/jax.named_scope.html#jax.named_scope "jax.named_scope")(name) | A context manager that adds a user specified name to the JAX name stack. |
| [`block_until_ready`](_autosummary/jax.block_until_ready.html#jax.block_until_ready "jax.block_until_ready")(x) | Tries to call a `block_until_ready` method on pytree leaves. |
| [`copy_to_host_async`](_autosummary/jax.copy_to_host_async.html#jax.copy_to_host_async "jax.copy_to_host_async")(x) | Tries to call a `copy_to_host_async` method on pytree leaves. |
| [`make_mesh`](_autosummary/jax.make_mesh.html#jax.make_mesh "jax.make_mesh")(axis_shapes, axis_names\[, ...\]) | Creates an efficient mesh with the shape and axis names specified. |
| [`set_mesh`](_autosummary/jax.set_mesh.html#jax.set_mesh "jax.set_mesh")(mesh) | Sets a concrete mesh in a thread-local context. |

## Automatic differentiation[\#](#automatic-differentiation "Link to this heading")

|  |  |
|----|----|
| [`grad`](_autosummary/jax.grad.html#jax.grad "jax.grad")(fun\[, argnums, has_aux, holomorphic, ...\]) | Creates a function that evaluates the gradient of `fun`. |
| [`value_and_grad`](_autosummary/jax.value_and_grad.html#jax.value_and_grad "jax.value_and_grad")(fun\[, argnums, has_aux, ...\]) | Create a function that evaluates both `fun` and the gradient of `fun`. |
| [`jacobian`](_autosummary/jax.jacobian.html#jax.jacobian "jax.jacobian")(fun\[, argnums, has_aux, ...\]) | Alias of [`jax.jacrev()`](_autosummary/jax.jacrev.html#jax.jacrev "jax.jacrev"). |
| [`jacfwd`](_autosummary/jax.jacfwd.html#jax.jacfwd "jax.jacfwd")(fun\[, argnums, has_aux, holomorphic\]) | Jacobian of `fun` evaluated column-by-column using forward-mode AD. |
| [`jacrev`](_autosummary/jax.jacrev.html#jax.jacrev "jax.jacrev")(fun\[, argnums, has_aux, holomorphic, ...\]) | Jacobian of `fun` evaluated row-by-row using reverse-mode AD. |
| [`hessian`](_autosummary/jax.hessian.html#jax.hessian "jax.hessian")(fun\[, argnums, has_aux, holomorphic\]) | Hessian of `fun` as a dense array. |
| [`jvp`](_autosummary/jax.jvp.html#jax.jvp "jax.jvp")(fun, primals, tangents\[, has_aux\]) | Computes a (forward-mode) Jacobian-vector product of `fun`. |
| [`linearize`](_autosummary/jax.linearize.html#jax.linearize "jax.linearize")() | Produces a linear approximation to `fun` using [`jvp()`](_autosummary/jax.jvp.html#jax.jvp "jax.jvp") and partial eval. |
| [`linear_transpose`](_autosummary/jax.linear_transpose.html#jax.linear_transpose "jax.linear_transpose")(fun, \*primals\[, reduce_axes\]) | Transpose a function that is promised to be linear. |
| [`vjp`](_autosummary/jax.vjp.html#jax.vjp "jax.vjp")() )) | Compute a (reverse-mode) vector-Jacobian product of `fun`. |
| [`custom_gradient`](_autosummary/jax.custom_gradient.html#jax.custom_gradient "jax.custom_gradient")(fun) | Convenience function for defining custom VJP rules (aka custom gradients). |
| [`closure_convert`](_autosummary/jax.closure_convert.html#jax.closure_convert "jax.closure_convert")(fun, \*example_args) | Closure conversion utility, for use with higher-order custom derivatives. |
| [`checkpoint`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint")(fun, \*\[, prevent_cse, policy, ...\]) | Make `fun` recompute internal linearization points when differentiated. |

## Vectorization[\#](#vectorization "Link to this heading")

|  |  |
|----|----|
| [`vmap`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap")(fun\[, in_axes, out_axes, axis_name, ...\]) | Vectorizing map. |
| [`numpy.vectorize`](_autosummary/jax.numpy.vectorize.html#jax.numpy.vectorize "jax.numpy.vectorize")(pyfunc, \*\[, excluded, signature\]) | Define a vectorized function with broadcasting. |

## Parallelization[\#](#parallelization "Link to this heading")

|  |  |
|----|----|
| [`shard_map`](_autosummary/jax.shard_map.html#jax.shard_map "jax.shard_map")(\[check_vma\]) | Map a function over shards of data using a mesh of devices. |
| [`smap`](_autosummary/jax.smap.html#jax.smap "jax.smap")() | Single axis shard_map that maps a function f one axis at a time. |
| [`pmap`](_autosummary/jax.pmap.html#jax.pmap "jax.pmap")(fun\[, axis_name, in_axes, out_axes, ...\]) | Old way of doing parallel map. |
| [`devices`](_autosummary/jax.devices.html#jax.devices "jax.devices")(\[backend\]) | Returns a list of all devices for a given backend. |
| [`local_devices`](_autosummary/jax.local_devices.html#jax.local_devices "jax.local_devices")(\[process_index, backend, host_id\]) | Like [`jax.devices()`](_autosummary/jax.devices.html#jax.devices "jax.devices"), but only returns devices local to a given process. |
| [`process_index`](_autosummary/jax.process_index.html#jax.process_index "jax.process_index")(\[backend\]) | Returns the integer process index of this process. |
| [`device_count`](_autosummary/jax.device_count.html#jax.device_count "jax.device_count")(\[backend\]) | Returns the total number of devices. |
| [`local_device_count`](_autosummary/jax.local_device_count.html#jax.local_device_count "jax.local_device_count")(\[backend\]) | Returns the number of devices addressable by this process. |
| [`process_count`](_autosummary/jax.process_count.html#jax.process_count "jax.process_count")(\[backend\]) | Returns the number of JAX processes associated with the backend. |
| [`process_indices`](_autosummary/jax.process_indices.html#jax.process_indices "jax.process_indices")(\[backend\]) | Returns the list of all JAX process indices associated with the backend. |

## Customization[\#](#customization "Link to this heading")

### `custom_jvp`[\#](#custom-jvp "Link to this heading")

|  |  |
|----|----|
| [`custom_jvp`](_autosummary/jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp")(fun\[, nondiff_argnums, ...\]) | Set up a JAX-transformable function for a custom JVP rule definition. |
| [`custom_jvp.defjvp`](_autosummary/jax.custom_jvp.defjvp.html#jax.custom_jvp.defjvp "jax.custom_jvp.defjvp")(jvp\[, symbolic_zeros\]) | Define a custom JVP rule for the function represented by this instance. |
| [`custom_jvp.defjvps`](_autosummary/jax.custom_jvp.defjvps.html#jax.custom_jvp.defjvps "jax.custom_jvp.defjvps")(\*jvps) | Convenience wrapper for defining JVPs for each argument separately. |

### `custom_vjp`[\#](#custom-vjp "Link to this heading")

|  |  |
|----|----|
| [`custom_vjp`](_autosummary/jax.custom_vjp.html#jax.custom_vjp "jax.custom_vjp")(fun\[, nondiff_argnums, ...\]) | Set up a JAX-transformable function for a custom VJP rule definition. |
| [`custom_vjp.defvjp`](_autosummary/jax.custom_vjp.defvjp.html#jax.custom_vjp.defvjp "jax.custom_vjp.defvjp")(fwd, bwd\[, ...\]) | Define a custom VJP rule for the function represented by this instance. |

### `custom_batching`[\#](#custom-batching "Link to this heading")

|  |  |
|----|----|
| [`custom_batching.custom_vmap`](_autosummary/jax.custom_batching.custom_vmap.html#jax.custom_batching.custom_vmap "jax.custom_batching.custom_vmap")(fun) | Customize the vmap behavior of a JAX-transformable function. |
| [`custom_batching.custom_vmap.def_vmap`](_autosummary/jax.custom_batching.custom_vmap.def_vmap.html#jax.custom_batching.custom_vmap.def_vmap "jax.custom_batching.custom_vmap.def_vmap")(vmap_rule) | Define the vmap rule for this custom_vmap function. |
| [`custom_batching.sequential_vmap`](_autosummary/jax.custom_batching.sequential_vmap.html#jax.custom_batching.sequential_vmap "jax.custom_batching.sequential_vmap")(f) | A special case of `custom_vmap` that uses a loop. |

## jax.Array (`jax.Array`)[\#](#jax-array-jax-array "Link to this heading")

|  |  |
|----|----|
| [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array")() | Array base class for JAX |
| [`make_array_from_callback`](_autosummary/jax.make_array_from_callback.html#jax.make_array_from_callback "jax.make_array_from_callback")(shape, sharding, ...) | Returns a `jax.Array` via data fetched from `data_callback`. |
| [`make_array_from_single_device_arrays`](_autosummary/jax.make_array_from_single_device_arrays.html#jax.make_array_from_single_device_arrays "jax.make_array_from_single_device_arrays")(shape, ...) | Returns a `jax.Array` from a sequence of `jax.Array`s each on a single device. |
| [`make_array_from_process_local_data`](_autosummary/jax.make_array_from_process_local_data.html#jax.make_array_from_process_local_data "jax.make_array_from_process_local_data")(sharding, ...) | Creates distributed tensor using the data available in process. |

### Array properties and methods[\#](#array-properties-and-methods "Link to this heading")

|  |  |
|----|----|
| [`Array.addressable_shards`](_autosummary/jax.Array.addressable_shards.html#jax.Array.addressable_shards "jax.Array.addressable_shards") | List of addressable shards. |
| [`Array.all`](_autosummary/jax.Array.all.html#jax.Array.all "jax.Array.all")(\[axis, out, keepdims, where\]) | Test whether all array elements along a given axis evaluate to True. |
| [`Array.any`](_autosummary/jax.Array.any.html#jax.Array.any "jax.Array.any")(\[axis, out, keepdims, where\]) | Test whether any array elements along a given axis evaluate to True. |
| [`Array.argmax`](_autosummary/jax.Array.argmax.html#jax.Array.argmax "jax.Array.argmax")(\[axis, out, keepdims\]) | Return the index of the maximum value. |
| [`Array.argmin`](_autosummary/jax.Array.argmin.html#jax.Array.argmin "jax.Array.argmin")(\[axis, out, keepdims\]) | Return the index of the minimum value. |
| [`Array.argpartition`](_autosummary/jax.Array.argpartition.html#jax.Array.argpartition "jax.Array.argpartition")(kth\[, axis\]) | Return the indices that partially sort the array. |
| [`Array.argsort`](_autosummary/jax.Array.argsort.html#jax.Array.argsort "jax.Array.argsort")(\[axis, kind, order, stable, ...\]) | Return the indices that sort the array. |
| [`Array.astype`](_autosummary/jax.Array.astype.html#jax.Array.astype "jax.Array.astype")(dtype\[, copy, device\]) | Copy the array and cast to a specified dtype. |
| [`Array.at`](_autosummary/jax.Array.at.html#jax.Array.at "jax.Array.at") | Helper property for index update functionality. |
| [`Array.byteswap`](_autosummary/jax.Array.byteswap.html#jax.Array.byteswap "jax.Array.byteswap")() | Swap the bytes of the array elements. |
| [`Array.choose`](_autosummary/jax.Array.choose.html#jax.Array.choose "jax.Array.choose")(choices\[, out, mode\]) | Construct an array choosing from elements of multiple arrays. |
| [`Array.clip`](_autosummary/jax.Array.clip.html#jax.Array.clip "jax.Array.clip")(\[min, max\]) | Return an array whose values are limited to a specified range. |
| [`Array.compress`](_autosummary/jax.Array.compress.html#jax.Array.compress "jax.Array.compress")(condition\[, axis, out, size, ...\]) | Return selected slices of this array along given axis. |
| [`Array.committed`](_autosummary/jax.Array.committed.html#jax.Array.committed "jax.Array.committed") | Whether the array is committed or not. |
| [`Array.conj`](_autosummary/jax.Array.conj.html#jax.Array.conj "jax.Array.conj")() | Return the complex conjugate of the array. |
| [`Array.conjugate`](_autosummary/jax.Array.conjugate.html#jax.Array.conjugate "jax.Array.conjugate")() | Return the complex conjugate of the array. |
| [`Array.copy`](_autosummary/jax.Array.copy.html#jax.Array.copy "jax.Array.copy")() | Return a copy of the array. |
| [`Array.copy_to_host_async`](_autosummary/jax.Array.copy_to_host_async.html#jax.Array.copy_to_host_async "jax.Array.copy_to_host_async")() | Copies an `Array` to the host asynchronously. |
| [`Array.cumprod`](_autosummary/jax.Array.cumprod.html#jax.Array.cumprod "jax.Array.cumprod")(\[axis, dtype, out\]) | Return the cumulative product of the array. |
| [`Array.cumsum`](_autosummary/jax.Array.cumsum.html#jax.Array.cumsum "jax.Array.cumsum")(\[axis, dtype, out\]) | Return the cumulative sum of the array. |
| [`Array.device`](_autosummary/jax.Array.device.html#jax.Array.device "jax.Array.device") | Array API-compatible device attribute. |
| [`Array.diagonal`](_autosummary/jax.Array.diagonal.html#jax.Array.diagonal "jax.Array.diagonal")(\[offset, axis1, axis2\]) | Return the specified diagonal from the array. |
| [`Array.dot`](_autosummary/jax.Array.dot.html#jax.Array.dot "jax.Array.dot")(b, \*\[, precision, ...\]) | Compute the dot product of two arrays. |
| [`Array.dtype`](_autosummary/jax.Array.dtype.html#jax.Array.dtype "jax.Array.dtype") | The data type ([`numpy.dtype`](_autosummary/jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype")) of the array. |
| [`Array.flat`](_autosummary/jax.Array.flat.html#jax.Array.flat "jax.Array.flat") | Use [`flatten()`](_autosummary/jax.Array.flatten.html#jax.Array.flatten "jax.Array.flatten") instead. |
| [`Array.flatten`](_autosummary/jax.Array.flatten.html#jax.Array.flatten "jax.Array.flatten")(\[order, out_sharding\]) | Flatten array into a 1-dimensional shape. |
| [`Array.global_shards`](_autosummary/jax.Array.global_shards.html#jax.Array.global_shards "jax.Array.global_shards") | List of global shards. |
| [`Array.imag`](_autosummary/jax.Array.imag.html#jax.Array.imag "jax.Array.imag") | Return the imaginary part of the array. |
| [`Array.is_fully_addressable`](_autosummary/jax.Array.is_fully_addressable.html#jax.Array.is_fully_addressable "jax.Array.is_fully_addressable") | Is this Array fully addressable? |
| [`Array.is_fully_replicated`](_autosummary/jax.Array.is_fully_replicated.html#jax.Array.is_fully_replicated "jax.Array.is_fully_replicated") | Is this Array fully replicated? |
| [`Array.item`](_autosummary/jax.Array.item.html#jax.Array.item "jax.Array.item")(\*args) | Copy an element of an array to a standard Python scalar and return it. |
| [`Array.itemsize`](_autosummary/jax.Array.itemsize.html#jax.Array.itemsize "jax.Array.itemsize") | Length of one array element in bytes. |
| [`Array.max`](_autosummary/jax.Array.max.html#jax.Array.max "jax.Array.max")(\[axis, out, keepdims, initial, where\]) | Return the maximum of array elements along a given axis. |
| [`Array.mean`](_autosummary/jax.Array.mean.html#jax.Array.mean "jax.Array.mean")(\[axis, dtype, out, keepdims, where\]) | Return the mean of array elements along a given axis. |
| [`Array.min`](_autosummary/jax.Array.min.html#jax.Array.min "jax.Array.min")(\[axis, out, keepdims, initial, where\]) | Return the minimum of array elements along a given axis. |
| [`Array.nbytes`](_autosummary/jax.Array.nbytes.html#jax.Array.nbytes "jax.Array.nbytes") | Total bytes consumed by the elements of the array. |
| [`Array.ndim`](_autosummary/jax.Array.ndim.html#jax.Array.ndim "jax.Array.ndim") | The number of dimensions in the array. |
| [`Array.nonzero`](_autosummary/jax.Array.nonzero.html#jax.Array.nonzero "jax.Array.nonzero")(\*\[, fill_value, size\]) | Return indices of nonzero elements of an array. |
| [`Array.prod`](_autosummary/jax.Array.prod.html#jax.Array.prod "jax.Array.prod")(\[axis, dtype, out, keepdims, ...\]) | Return product of the array elements over a given axis. |
| [`Array.ptp`](_autosummary/jax.Array.ptp.html#jax.Array.ptp "jax.Array.ptp")(\[axis, out, keepdims\]) | Return the peak-to-peak range along a given axis. |
| [`Array.ravel`](_autosummary/jax.Array.ravel.html#jax.Array.ravel "jax.Array.ravel")(\[order, out_sharding\]) | Flatten array into a 1-dimensional shape. |
| [`Array.real`](_autosummary/jax.Array.real.html#jax.Array.real "jax.Array.real") | Return the real part of the array. |
| [`Array.repeat`](_autosummary/jax.Array.repeat.html#jax.Array.repeat "jax.Array.repeat")(repeats\[, axis, ...\]) | Construct an array from repeated elements. |
| [`Array.reshape`](_autosummary/jax.Array.reshape.html#jax.Array.reshape "jax.Array.reshape")(\*args\[, order, out_sharding\]) | Returns an array containing the same data with a new shape. |
| [`Array.round`](_autosummary/jax.Array.round.html#jax.Array.round "jax.Array.round")(\[decimals, out\]) | Round array elements to a given decimal. |
| [`Array.searchsorted`](_autosummary/jax.Array.searchsorted.html#jax.Array.searchsorted "jax.Array.searchsorted")(v\[, side, sorter, method\]) | Perform a binary search within a sorted array. |
| [`Array.shape`](_autosummary/jax.Array.shape.html#jax.Array.shape "jax.Array.shape") | The shape of the array. |
| [`Array.sharding`](_autosummary/jax.Array.sharding.html#jax.Array.sharding "jax.Array.sharding") | The sharding for the array. |
| [`Array.size`](_autosummary/jax.Array.size.html#jax.Array.size "jax.Array.size") | The total number of elements in the array. |
| [`Array.sort`](_autosummary/jax.Array.sort.html#jax.Array.sort "jax.Array.sort")(\[axis, kind, order, stable, ...\]) | Return a sorted copy of an array. |
| [`Array.squeeze`](_autosummary/jax.Array.squeeze.html#jax.Array.squeeze "jax.Array.squeeze")(\[axis\]) | Remove one or more length-1 axes from array. |
| [`Array.std`](_autosummary/jax.Array.std.html#jax.Array.std "jax.Array.std")(\[axis, dtype, out, ddof, ...\]) | Compute the standard deviation along a given axis. |
| [`Array.sum`](_autosummary/jax.Array.sum.html#jax.Array.sum "jax.Array.sum")(\[axis, dtype, out, keepdims, ...\]) | Sum of the elements of the array over a given axis. |
| [`Array.swapaxes`](_autosummary/jax.Array.swapaxes.html#jax.Array.swapaxes "jax.Array.swapaxes")(axis1, axis2) | Swap two axes of an array. |
| [`Array.take`](_autosummary/jax.Array.take.html#jax.Array.take "jax.Array.take")(indices\[, axis, out, mode, ...\]) | Take elements from an array. |
| [`Array.to_device`](_autosummary/jax.Array.to_device.html#jax.Array.to_device "jax.Array.to_device")(device, \*\[, stream\]) | Return a copy of the array on the specified device |
| [`Array.trace`](_autosummary/jax.Array.trace.html#jax.Array.trace "jax.Array.trace")(\[offset, axis1, axis2, dtype, out\]) | Return the sum along the diagonal. |
| [`Array.transpose`](_autosummary/jax.Array.transpose.html#jax.Array.transpose "jax.Array.transpose")(\*args) | Returns a copy of the array with axes transposed. |
| [`Array.var`](_autosummary/jax.Array.var.html#jax.Array.var "jax.Array.var")(\[axis, dtype, out, ddof, ...\]) | Compute the variance along a given axis. |
| [`Array.view`](_autosummary/jax.Array.view.html#jax.Array.view "jax.Array.view")(\[dtype, type\]) | Return a bitwise copy of the array, viewed as a new dtype. |
| [`Array.T`](_autosummary/jax.Array.T.html#jax.Array.T "jax.Array.T") | Compute the all-axis array transpose. |
| [`Array.mT`](_autosummary/jax.Array.mT.html#jax.Array.mT "jax.Array.mT") | Compute the (batched) matrix transpose. |

## Callbacks[\#](#callbacks "Link to this heading")

|  |  |
|----|----|
| [`pure_callback`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback")(callback, result_shape_dtypes, ...) | Calls a pure Python callback. |
| [`experimental.io_callback`](_autosummary/jax.experimental.io_callback.html#jax.experimental.io_callback "jax.experimental.io_callback")(callback, ...\[, ...\]) | Calls an impure Python callback. |
| [`debug.callback`](_autosummary/jax.debug.callback.html#jax.debug.callback "jax.debug.callback")(\[callback, ordered, partitioned\]) | Calls a stageable Python callback. |
| [`debug.print`](_autosummary/jax.debug.print.html#jax.debug.print "jax.debug.print")(\[fmt, ordered, partitioned, ...\]) | Prints values and works in staged out JAX functions. |

## Miscellaneous[\#](#miscellaneous "Link to this heading")

|  |  |
|----|----|
| [`Device`](_autosummary/jax.Device.html#jax.Device "jax.Device") | A descriptor of an available device. |
| [`print_environment_info`](_autosummary/jax.print_environment_info.html#jax.print_environment_info "jax.print_environment_info")(\[return_string\]) | Returns a string containing local environment & JAX installation information. |
| [`live_arrays`](_autosummary/jax.live_arrays.html#jax.live_arrays "jax.live_arrays")(\[platform\]) | Return all live arrays in the backend for platform. |
| [`clear_caches`](_autosummary/jax.clear_caches.html#jax.clear_caches "jax.clear_caches")() | Clear all compilation and staging caches. |
| [`typeof`](_autosummary/jax.typeof.html#jax.typeof "jax.typeof")(x) | Return the JAX type (i.e. `AbstractValue`) of the input. |
| [`ds`](_autosummary/jax.ds.html#jax.ds "jax.ds")(start\[, size, stride\]) | Constructs a `Slice` from a start index and a size. |

## Checkpoint policies[\#](#checkpoint-policies "Link to this heading")

|  |  |
|----|----|
| [`checkpoint_policies.everything_saveable`](_autosummary/jax.checkpoint_policies.everything_saveable.html#jax.checkpoint_policies.everything_saveable "jax.checkpoint_policies.everything_saveable")(\*\*\_\_) | The default strategy, as if `jax.checkpoint` were not being used at all. |
| [`checkpoint_policies.nothing_saveable`](_autosummary/jax.checkpoint_policies.nothing_saveable.html#jax.checkpoint_policies.nothing_saveable "jax.checkpoint_policies.nothing_saveable")(\*\*\_\_) | Rematerialize everything, as if a custom policy were not being used at all. |
| [`checkpoint_policies.dots_saveable`](_autosummary/jax.checkpoint_policies.dots_saveable.html#jax.checkpoint_policies.dots_saveable "jax.checkpoint_policies.dots_saveable")(\*\_, \*\*\_\_) |  |
| [`checkpoint_policies.checkpoint_dots`](_autosummary/jax.checkpoint_policies.checkpoint_dots.html#jax.checkpoint_policies.checkpoint_dots "jax.checkpoint_policies.checkpoint_dots")(\*\_, \*\*\_\_) |  |
| [`checkpoint_policies.dots_with_no_batch_dims_saveable`](_autosummary/jax.checkpoint_policies.dots_with_no_batch_dims_saveable.html#jax.checkpoint_policies.dots_with_no_batch_dims_saveable "jax.checkpoint_policies.dots_with_no_batch_dims_saveable")(...) | This is a useful heuristic for transformers. |
| [`checkpoint_policies.checkpoint_dots_with_no_batch_dims`](_autosummary/jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims.html#jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims "jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims")(...) | This is a useful heuristic for transformers. |
| [`checkpoint_policies.save_any_names_but_these`](_autosummary/jax.checkpoint_policies.save_any_names_but_these.html#jax.checkpoint_policies.save_any_names_but_these "jax.checkpoint_policies.save_any_names_but_these")() | Save only named values, i.e. any outputs of checkpoint_name, excluding the names given. |
| [`checkpoint_policies.save_only_these_names`](_autosummary/jax.checkpoint_policies.save_only_these_names.html#jax.checkpoint_policies.save_only_these_names "jax.checkpoint_policies.save_only_these_names")() | Save only named values, and only among the names given. |
| [`checkpoint_policies.offload_dot_with_no_batch_dims`](_autosummary/jax.checkpoint_policies.offload_dot_with_no_batch_dims.html#jax.checkpoint_policies.offload_dot_with_no_batch_dims "jax.checkpoint_policies.offload_dot_with_no_batch_dims")(...) | Same as `dots_with_no_batch_dims_saveable`, but offload to CPU memory instead of recomputing. |
| [`checkpoint_policies.save_and_offload_only_these_names`](_autosummary/jax.checkpoint_policies.save_and_offload_only_these_names.html#jax.checkpoint_policies.save_and_offload_only_these_names "jax.checkpoint_policies.save_and_offload_only_these_names")(\*, ...) | Same as `save_only_these_names`, but offload to CPU memory instead of recomputing. |
| [`checkpoint_policies.save_from_both_policies`](_autosummary/jax.checkpoint_policies.save_from_both_policies.html#jax.checkpoint_policies.save_from_both_policies "jax.checkpoint_policies.save_from_both_policies")(...) | Logical OR of the given policies. |

[](jaxpr.html "previous page")

previous

JAX internals: The jaxpr language

[](jax.numpy.html "next page")

next

`jax.numpy` module

Contents

- [Subpackages](#subpackages)
- [Configuration](#configuration)
- [Just-in-time compilation (`jit`)](#just-in-time-compilation-jit)
- [Automatic differentiation](#automatic-differentiation)
- [Vectorization](#vectorization)
- [Parallelization](#parallelization)
- [Customization](#customization)
  - [`custom_jvp`](#custom-jvp)
  - [`custom_vjp`](#custom-vjp)
  - [`custom_batching`](#custom-batching)
- [jax.Array (`jax.Array`)](#jax-array-jax-array)
  - [Array properties and methods](#array-properties-and-methods)
- [Callbacks](#callbacks)
- [Miscellaneous](#miscellaneous)
- [Checkpoint policies](#checkpoint-policies)

By The JAX authors

© Copyright 2024, The JAX Authors.\
