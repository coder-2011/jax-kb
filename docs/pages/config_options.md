- [](index.html)
- Configuration Options

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/config_options.rst "Download source file")
-  .pdf

# Configuration Options

## Contents

- [How to Use Configuration Options](#how-to-use-configuration-options)
- [Common Configuration Options](#common-configuration-options)
- [All Configuration Options](#all-configuration-options)
  - [Check Vma](#check_vma)
  - [Eager Constant Folding](#eager_constant_folding)
  - [Jax2Tf Associative Scan Reductions](#jax2tf_associative_scan_reductions)
  - [Jax2Tf Default Native Serialization](#jax2tf_default_native_serialization)
  - [Allow F16 Reductions](#jax_allow_f16_reductions)
  - [Array Garbage Collection Guard](#jax_array_garbage_collection_guard)
  - [Auto Pcast](#jax_auto_pcast)
  - [Backend Target](#jax_backend_target)
  - [Bcoo Cusparse Lowering](#jax_bcoo_cusparse_lowering)
  - [Captured Constants Report Frames](#jax_captured_constants_report_frames)
  - [Captured Constants Warn Bytes](#jax_captured_constants_warn_bytes)
  - [Check Proxy Envs](#jax_check_proxy_envs)
  - [Check Static Indices](#jax_check_static_indices)
  - [Check Tracer Leaks](#jax_check_tracer_leaks)
  - [Compilation Cache Check Contents](#jax_compilation_cache_check_contents)
  - [Compilation Cache Dir](#jax_compilation_cache_dir)
  - [Compilation Cache Expect Pgle](#jax_compilation_cache_expect_pgle)
  - [Compilation Cache Include Metadata In Key](#jax_compilation_cache_include_metadata_in_key)
  - [Compilation Cache Max Size](#jax_compilation_cache_max_size)
  - [Compiler Detailed Logging Min Ops](#jax_compiler_detailed_logging_min_ops)
  - [Compiler Enable Remat Pass](#jax_compiler_enable_remat_pass)
  - [Cpu Collectives Implementation](#jax_cpu_collectives_implementation)
  - [Cpu Enable Async Dispatch](#jax_cpu_enable_async_dispatch)
  - [Cpu Get Global Topology Timeout Minutes](#jax_cpu_get_global_topology_timeout_minutes)
  - [Cpu Get Local Topology Timeout Minutes](#jax_cpu_get_local_topology_timeout_minutes)
  - [Cross Host Transfer Socket Address](#jax_cross_host_transfer_socket_address)
  - [Cross Host Transfer Timeout Seconds](#jax_cross_host_transfer_timeout_seconds)
  - [Cross Host Transfer Transfer Size](#jax_cross_host_transfer_transfer_size)
  - [Cross Host Transport Addresses](#jax_cross_host_transport_addresses)
  - [Cuda Visible Devices](#jax_cuda_visible_devices)
  - [Custom Vjp3](#jax_custom_vjp3)
  - [Debug Infs](#jax_debug_infs)
  - [Debug Key Reuse](#jax_debug_key_reuse)
  - [Debug Leaked Clients On Clear Backends](#jax_debug_leaked_clients_on_clear_backends)
  - [Debug Log Modules](#jax_debug_log_modules)
  - [Debug Nans](#jax_debug_nans)
  - [Default Device](#jax_default_device)
  - [Default Matmul Precision](#jax_default_matmul_precision)
  - [Default Prng Impl](#jax_default_prng_impl)
  - [Disable Bwd Checks](#jax_disable_bwd_checks)
  - [Disable Jit](#jax_disable_jit)
  - [Disable Most Optimizations](#jax_disable_most_optimizations)
  - [Disable Vmap Shmap Error](#jax_disable_vmap_shmap_error)
  - [Disallow Mesh Context Manager](#jax_disallow_mesh_context_manager)
  - [Distributed Debug](#jax_distributed_debug)
  - [Dump Ir Modes](#jax_dump_ir_modes)
  - [Dump Ir To](#jax_dump_ir_to)
  - [Embedded Constants Max Bytes](#jax_embedded_constants_max_bytes)
  - [Enable Checks](#jax_enable_checks)
  - [Enable Compilation Cache](#jax_enable_compilation_cache)
  - [Enable Custom Prng](#jax_enable_custom_prng)
  - [Enable Pgle](#jax_enable_pgle)
  - [Enable Preemption Service](#jax_enable_preemption_service)
  - [Enable Recoverability](#jax_enable_recoverability)
  - [Enable X64](#jax_enable_x64)
  - [Error Checking Behavior Divide](#jax_error_checking_behavior_divide)
  - [Error Checking Behavior Nan](#jax_error_checking_behavior_nan)
  - [Error Checking Behavior Oob](#jax_error_checking_behavior_oob)
  - [Exec Time Optimization Effort](#jax_exec_time_optimization_effort)
  - [Experimental Unsafe Xla Runtime Errors](#jax_experimental_unsafe_xla_runtime_errors)
  - [Explain Cache Misses](#jax_explain_cache_misses)
  - [Explicit X64 Dtypes](#jax_explicit_x64_dtypes)
  - [Export Calling Convention Version](#jax_export_calling_convention_version)
  - [Export Ignore Forward Compatibility](#jax_export_ignore_forward_compatibility)
  - [Force Dcn Cross Host Transfers](#jax_force_dcn_cross_host_transfers)
  - [High Dynamic Range Gumbel](#jax_high_dynamic_range_gumbel)
  - [Hlo Source File Canonicalization Regex](#jax_hlo_source_file_canonicalization_regex)
  - [Include Debug Info In Dumps](#jax_include_debug_info_in_dumps)
  - [Include Full Tracebacks In Locations](#jax_include_full_tracebacks_in_locations)
  - [Legacy Prng Key](#jax_legacy_prng_key)
  - [Log Checkpoint Residuals](#jax_log_checkpoint_residuals)
  - [Log Compiles](#jax_log_compiles)
  - [Logging Level](#jax_logging_level)
  - [Memory Fitting Effort](#jax_memory_fitting_effort)
  - [Memory Fitting Level](#jax_memory_fitting_level)
  - [Mock Gpu Topology](#jax_mock_gpu_topology)
  - [Mosaic Allow Hlo](#jax_mosaic_allow_hlo)
  - [Mutable Array Checks](#jax_mutable_array_checks)
  - [No Execution](#jax_no_execution)
  - [No Tracing](#jax_no_tracing)
  - [Num Cpu Devices](#jax_num_cpu_devices)
  - [Numpy Dtype Promotion](#jax_numpy_dtype_promotion)
  - [Numpy Rank Promotion](#jax_numpy_rank_promotion)
  - [Oneapi Visible Devices](#jax_oneapi_visible_devices)
  - [Optimization Level](#jax_optimization_level)
  - [Pallas Enable Debug Checks](#jax_pallas_enable_debug_checks)
  - [Pallas Poison Buffers](#jax_pallas_poison_buffers)
  - [Pallas Use Mosaic Gpu](#jax_pallas_use_mosaic_gpu)
  - [Pallas Verbose Errors](#jax_pallas_verbose_errors)
  - [Persistent Cache Enable Xla Caches](#jax_persistent_cache_enable_xla_caches)
  - [Persistent Cache Min Compile Time Secs](#jax_persistent_cache_min_compile_time_secs)
  - [Persistent Cache Min Entry Size Bytes](#jax_persistent_cache_min_entry_size_bytes)
  - [Pgle Aggregation Percentile](#jax_pgle_aggregation_percentile)
  - [Pgle Profiling Runs](#jax_pgle_profiling_runs)
  - [Pjrt Client Create Options](#jax_pjrt_client_create_options)
  - [Platform Name](#jax_platform_name)
  - [Platforms](#jax_platforms)
  - [Pprint Use Color](#jax_pprint_use_color)
  - [Ragged Dot Use Gpu Pallas Triton Lowering](#jax_ragged_dot_use_gpu_pallas_triton_lowering)
  - [Ragged Dot Use Ragged Dot Instruction](#jax_ragged_dot_use_ragged_dot_instruction)
  - [Raise Persistent Cache Errors](#jax_raise_persistent_cache_errors)
  - [Random Seed Offset](#jax_random_seed_offset)
  - [Refs To Pins](#jax_refs_to_pins)
  - [Remove Custom Partitioning Ptr From Cache Key](#jax_remove_custom_partitioning_ptr_from_cache_key)
  - [Remove Size One Mesh Axis From Type](#jax_remove_size_one_mesh_axis_from_type)
  - [Rocm Visible Devices](#jax_rocm_visible_devices)
  - [Scan3](#jax_scan3)
  - [Send Traceback To Runtime](#jax_send_traceback_to_runtime)
  - [Share Binary Between Hosts](#jax_share_binary_between_hosts)
  - [Share Binary Between Hosts Timeout Ms](#jax_share_binary_between_hosts_timeout_ms)
  - [Softmax Custom Jvp](#jax_softmax_custom_jvp)
  - [Sort Devices By Process Index](#jax_sort_devices_by_process_index)
  - [Source Url Schema](#jax_source_url_schema)
  - [Thread Guard](#jax_thread_guard)
  - [Threefry Gpu Kernel Lowering](#jax_threefry_gpu_kernel_lowering)
  - [Threefry Partitionable](#jax_threefry_partitionable)
  - [Traceback Filtering](#jax_traceback_filtering)
  - [Traceback In Locations Limit](#jax_traceback_in_locations_limit)
  - [Tracer Error Num Traceback Frames](#jax_tracer_error_num_traceback_frames)
  - [Transfer Guard](#jax_transfer_guard)
  - [Transfer Guard Device To Device](#jax_transfer_guard_device_to_device)
  - [Transfer Guard Device To Host](#jax_transfer_guard_device_to_host)
  - [Transfer Guard Host To Device](#jax_transfer_guard_host_to_device)
  - [Use Direct Linearize](#jax_use_direct_linearize)
  - [Use Magma](#jax_use_magma)
  - [Use Shardy Partitioner](#jax_use_shardy_partitioner)
  - [Use Simplified Jaxpr Constants](#jax_use_simplified_jaxpr_constants)
  - [Xla Backend](#jax_xla_backend)
  - [Xla Profile Version](#jax_xla_profile_version)
  - [Mock Num Gpu Processes](#mock_num_gpu_processes)

# Configuration Options[\#](#configuration-options "Link to this heading")

JAX provides various configuration options to customize its behavior. These options control everything from numerical precision to debugging features.

## How to Use Configuration Options[\#](#how-to-use-configuration-options "Link to this heading")

JAX configuration options can be set in several ways:

1.  **Environment variables** (set before running your program):

        export JAX_ENABLE_X64=True
        python my_program.py

2.  **Runtime configuration** (in your Python code):

        import jax
        jax.config.update("jax_enable_x64", True)

3.  **Command-line flags** (using Abseil):

        # In your code:
        import jax
        jax.config.parse_flags_with_absl()

        # When running:
        python my_program.py --jax_enable_x64=True

## Common Configuration Options[\#](#common-configuration-options "Link to this heading")

Here are some of the most frequently used configuration options:

- `jax_enable_x64` – Enable 64-bit floating-point precision

- `jax_disable_jit` – Disable JIT compilation for debugging

- `jax_debug_nans` – Check for and raise errors on NaNs

- `jax_platforms` – Control which backends (CPU/GPU/TPU) JAX will initialize

- `jax_numpy_rank_promotion` – Control automatic rank promotion behavior

- `jax_default_matmul_precision` – Set default precision for matrix multiplication operations

## All Configuration Options[\#](#all-configuration-options "Link to this heading")

Below is a complete list of all available JAX configuration options:

### Check Vma[\#](#check_vma "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'check_vma'`

Environment Variable:  
`CHECK_VMA`

internal implementation detail of shard_map, DO NOT USE

### Eager Constant Folding[\#](#eager_constant_folding "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'eager_constant_folding'`

Environment Variable:  
`EAGER_CONSTANT_FOLDING`

Attempt constant folding during staging.

### Jax2Tf Associative Scan Reductions[\#](#jax2tf_associative_scan_reductions "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax2tf_associative_scan_reductions'`

Environment Variable:  
`JAX2TF_ASSOCIATIVE_SCAN_REDUCTIONS`

JAX has two separate lowering rules for the cumulative reduction primitives (cumsum, cumprod, cummax, cummin). On CPUs and GPUs it uses a lax.associative_scan, while for TPUs it uses the HLO ReduceWindow. The latter has a slow implementation on CPUs and GPUs. By default, jax2tf uses the TPU lowering. Set this flag to True to use the associative scan lowering usage, and only if it makes a difference for your application. See the jax2tf README.md for more details.

### Jax2Tf Default Native Serialization[\#](#jax2tf_default_native_serialization "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax2tf_default_native_serialization'`

Environment Variable:  
`JAX2TF_DEFAULT_NATIVE_SERIALIZATION`

Sets the default value of the native_serialization parameter to jax2tf.convert. Prefer using the parameter instead of the flag, the flag may be removed in the future. Starting with JAX 0.4.31 non-native serialization is deprecated.

### Allow F16 Reductions[\#](#jax_allow_f16_reductions "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_allow_f16_reductions'`

Environment Variable:  
`JAX_ALLOW_F16_REDUCTIONS`

If False, reduce_sum on f16 or bf16 inputs will raise an error.Defaults to True.

### Array Garbage Collection Guard[\#](#jax_array_garbage_collection_guard "Link to this heading")

Type:  
*Enum values:* `'allow'`, `'log'`, `'fatal'`

Default Value:  
`None`

Configuration String:  
`'jax_array_garbage_collection_guard'`

Environment Variable:  
`JAX_ARRAY_GARBAGE_COLLECTION_GUARD`

Select garbage collection guard level for `jax.Array` objects.

This option can be used to control what happens when a `jax.Array` object is garbage collected. It is desirable for `jax.Array` objects to be freed by Python reference counting rather than garbage collection in order to avoid device memory being held by the arrays until garbage collection occurs.

Valid values are:

- `allow`: do not log garbage collection of `jax.Array` objects.

- `log`: log an error when a `jax.Array` is garbage collected.

- `fatal`: fatal error if a `jax.Array` is garbage collected.

Default is `allow`. Note that not all cycles may be detected.

### Auto Pcast[\#](#jax_auto_pcast "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_auto_pcast'`

Environment Variable:  
`JAX_AUTO_PCAST`

If True, automatically insert pvary to match VMAs on simple ops.

### Backend Target[\#](#jax_backend_target "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_backend_target'`

Environment Variable:  
`JAX_BACKEND_TARGET`

Either “local” or “rpc:address” to connect to a remote service target.

### Bcoo Cusparse Lowering[\#](#jax_bcoo_cusparse_lowering "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_bcoo_cusparse_lowering'`

Environment Variable:  
`JAX_BCOO_CUSPARSE_LOWERING`

Enables lowering BCOO ops to cuSparse.

### Captured Constants Report Frames[\#](#jax_captured_constants_report_frames "Link to this heading")

Type:  
`int`

Default Value:  
`0`

Configuration String:  
`'jax_captured_constants_report_frames'`

Environment Variable:  
`JAX_CAPTURED_CONSTANTS_REPORT_FRAMES`

The number of stack frames reported for each captured constant indicating the file and operation where the constant was captured. Set to -1 to print the complete set of frames, or 0 to disable. N.b. the report is only generated if the total amount of captured constants exceeds jax_captured_constants_warn_bytes, as it is expensiveto generate the report.

### Captured Constants Warn Bytes[\#](#jax_captured_constants_warn_bytes "Link to this heading")

Type:  
`int`

Default Value:  
`2000000000`

Configuration String:  
`'jax_captured_constants_warn_bytes'`

Environment Variable:  
`JAX_CAPTURED_CONSTANTS_WARN_BYTES`

The number of bytes of parameters that may be captured as constants before a warning is issued. Defaults to approximately 2GB. Set to -1 to disable issuing a warning.

### Check Proxy Envs[\#](#jax_check_proxy_envs "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_check_proxy_envs'`

Environment Variable:  
`JAX_CHECK_PROXY_ENVS`

Checks proxy vars in user envs and emit warnings.

### Check Static Indices[\#](#jax_check_static_indices "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_check_static_indices'`

Environment Variable:  
`JAX_CHECK_STATIC_INDICES`

Turn on bounds checks for static indices during array indexing operations. These will only be checked when indexing mode is PROMISE_IN_BOUNDS, which is the default for gather-type operations.

### Check Tracer Leaks[\#](#jax_check_tracer_leaks "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_check_tracer_leaks'`

Environment Variable:  
`JAX_CHECK_TRACER_LEAKS`

Turn on checking for leaked tracers as soon as a trace completes. Enabling leak checking may have performance impacts: some caching is disabled, and other overheads may be added. Additionally, be aware that some Python debuggers can cause false positives, so it is recommended to disable any debuggers while leak checking is enabled.

### Compilation Cache Check Contents[\#](#jax_compilation_cache_check_contents "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_compilation_cache_check_contents'`

Environment Variable:  
`JAX_COMPILATION_CACHE_CHECK_CONTENTS`

When the compilation cache is enabled, check that the value found in the disk cache matches the result of a fresh compilation. This check is performed only the first time a key is encountered in a process.

### Compilation Cache Dir[\#](#jax_compilation_cache_dir "Link to this heading")

Type:  
`str`

Default Value:  
`None`

Configuration String:  
`'jax_compilation_cache_dir'`

Environment Variable:  
`JAX_COMPILATION_CACHE_DIR`

Path for the cache. Precedence: 1. A call to compilation_cache.set_cache_dir(). 2. The value of this flag set in the command line or by default.

### Compilation Cache Expect Pgle[\#](#jax_compilation_cache_expect_pgle "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_compilation_cache_expect_pgle'`

Environment Variable:  
`JAX_COMPILATION_CACHE_EXPECT_PGLE`

If set to True, compilation cache entries that were compiled with profile data (i.e. PGLE was enabled and the requisite number of executions were profiled) will be preferentially loaded, even if PGLE is not currently enabled. A warning will be printed when no preferred cache entry is found.

### Compilation Cache Include Metadata In Key[\#](#jax_compilation_cache_include_metadata_in_key "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_compilation_cache_include_metadata_in_key'`

Environment Variable:  
`JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY`

Include metadata, such as file names and line numbers, in the compilation cache key. If false, the cache will still get hits even if functions or files are moved, etc. However, it means that executables loaded from the cache may have stale metadata, which may show up in, e.g., profiles.

### Compilation Cache Max Size[\#](#jax_compilation_cache_max_size "Link to this heading")

Type:  
`int`

Default Value:  
`-1`

Configuration String:  
`'jax_compilation_cache_max_size'`

Environment Variable:  
`JAX_COMPILATION_CACHE_MAX_SIZE`

The maximum size (in bytes) allowed for the persistent compilation cache. When set, the least recently accessed cache entry(s) will be deleted once the total cache directory size exceeds the specified limit. Caching will be disabled if this value is set to 0. A special value of -1 indicates no limit, allowing the cache size to grow indefinitely.

### Compiler Detailed Logging Min Ops[\#](#jax_compiler_detailed_logging_min_ops "Link to this heading")

Type:  
`int`

Default Value:  
`10`

Configuration String:  
`'jax_compiler_detailed_logging_min_ops'`

Environment Variable:  
`JAX_COMPILER_DETAILED_LOGGING_MIN_OPS`

How big should a module be in MLIR operations before JAX enables detailed compiler logging? The intent of this flag is to suppress detailed logging for small/uninteresting computations.

### Compiler Enable Remat Pass[\#](#jax_compiler_enable_remat_pass "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_compiler_enable_remat_pass'`

Environment Variable:  
`JAX_COMPILER_ENABLE_REMAT_PASS`

Config to enable / disable the rematerialization HLO pass. Useful to allow XLA to automatically trade off memory and compute when encountering OOM errors. However, you are likely to get better results manually with jax.checkpoint

### Cpu Collectives Implementation[\#](#jax_cpu_collectives_implementation "Link to this heading")

Type:  
*Enum values:* `'gloo'`, `'mpi'`, `'megascale'`

Default Value:  
`'gloo'`

Configuration String:  
`'jax_cpu_collectives_implementation'`

Environment Variable:  
`JAX_CPU_COLLECTIVES_IMPLEMENTATION`

Cross-process collective implementation used on CPU. Must be one of (“gloo”, “mpi”)

### Cpu Enable Async Dispatch[\#](#jax_cpu_enable_async_dispatch "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_cpu_enable_async_dispatch'`

Environment Variable:  
`JAX_CPU_ENABLE_ASYNC_DISPATCH`

Only applies to non-parallel computations. If False, run computationsinline without async dispatch.

### Cpu Get Global Topology Timeout Minutes[\#](#jax_cpu_get_global_topology_timeout_minutes "Link to this heading")

Type:  
`int`

Default Value:  
`5`

Configuration String:  
`'jax_cpu_get_global_topology_timeout_minutes'`

Environment Variable:  
`JAX_CPU_GET_GLOBAL_TOPOLOGY_TIMEOUT_MINUTES`

Timeout in minutes for getting the global topology of CPU devices; should be strictly greater than –jax_cpu_get_local_topology_timeout_minutes.

### Cpu Get Local Topology Timeout Minutes[\#](#jax_cpu_get_local_topology_timeout_minutes "Link to this heading")

Type:  
`int`

Default Value:  
`2`

Configuration String:  
`'jax_cpu_get_local_topology_timeout_minutes'`

Environment Variable:  
`JAX_CPU_GET_LOCAL_TOPOLOGY_TIMEOUT_MINUTES`

Timeout in minutes for getting the local topology of each CPU device when building the global topology.

### Cross Host Transfer Socket Address[\#](#jax_cross_host_transfer_socket_address "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_cross_host_transfer_socket_address'`

Environment Variable:  
`JAX_CROSS_HOST_TRANSFER_SOCKET_ADDRESS`

Socket address to use for cross host device transfers via DCN. Necessary only if the PjRt plugin does not support cross host transfers.

### Cross Host Transfer Timeout Seconds[\#](#jax_cross_host_transfer_timeout_seconds "Link to this heading")

Type:  
`int`

Default Value:  
`None`

Configuration String:  
`'jax_cross_host_transfer_timeout_seconds'`

Environment Variable:  
`JAX_CROSS_HOST_TRANSFER_TIMEOUT_SECONDS`

Timeout for cross host transfer metadata exchange through KV store. Default is one minute.

### Cross Host Transfer Transfer Size[\#](#jax_cross_host_transfer_transfer_size "Link to this heading")

Type:  
`int`

Default Value:  
`None`

Configuration String:  
`'jax_cross_host_transfer_transfer_size'`

Environment Variable:  
`JAX_CROSS_HOST_TRANSFER_TRANSFER_SIZE`

Chunk size for chunked transfer requests.

### Cross Host Transport Addresses[\#](#jax_cross_host_transport_addresses "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_cross_host_transport_addresses'`

Environment Variable:  
`JAX_CROSS_HOST_TRANSPORT_ADDRESSES`

Comma-separated list of transport addresses to use for cross host device transfers via DCN. If not set, defaults to \[0.0.0.0:0\] \* 4.

### Cuda Visible Devices[\#](#jax_cuda_visible_devices "Link to this heading")

Type:  
`str`

Default Value:  
`'all'`

Configuration String:  
`'jax_cuda_visible_devices'`

Environment Variable:  
`JAX_CUDA_VISIBLE_DEVICES`

Restricts the set of CUDA devices that JAX will use. Either “all”, or a comma-separate list of integer device IDs.

### Custom Vjp3[\#](#jax_custom_vjp3 "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_custom_vjp3'`

Environment Variable:  
`JAX_CUSTOM_VJP3`

If True, embrace the future of custom autodiff rules. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Debug Infs[\#](#jax_debug_infs "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_debug_infs'`

Environment Variable:  
`JAX_DEBUG_INFS`

Add inf checks to every operation. When an inf is detected on the output of a jit-compiled computation, call into the un-compiled version in an attempt to more precisely identify the operation which produced the inf.

### Debug Key Reuse[\#](#jax_debug_key_reuse "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_debug_key_reuse'`

Environment Variable:  
`JAX_DEBUG_KEY_REUSE`

Turn on experimental key reuse checking. With this configuration enabled, typed PRNG keys (i.e. keys created with jax.random.key()) will have their usage tracked, and incorrect reuse of a previously-used key will lead to an error. Currently enabling this leads to a small Python overhead on every call to a JIT-compiled function with keys as inputs or outputs.

### Debug Leaked Clients On Clear Backends[\#](#jax_debug_leaked_clients_on_clear_backends "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_debug_leaked_clients_on_clear_backends'`

Environment Variable:  
`JAX_DEBUG_LEAKED_CLIENTS_ON_CLEAR_BACKENDS`

Check that clear_backends actually clears the backends.

### Debug Log Modules[\#](#jax_debug_log_modules "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_debug_log_modules'`

Environment Variable:  
`JAX_DEBUG_LOG_MODULES`

Comma-separated list of module names (e.g. “jax” or “jax.\_src.xla_bridge,jax.\_src.dispatch”) to enable debug logging for.

### Debug Nans[\#](#jax_debug_nans "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_debug_nans'`

Environment Variable:  
`JAX_DEBUG_NANS`

Add nan checks to every operation. When a nan is detected on the output of a jit-compiled computation, call into the un-compiled version in an attempt to more precisely identify the operation which produced the nan.

### Default Device[\#](#jax_default_device "Link to this heading")

Type:  
`str`

Default Value:  
`None`

Configuration String:  
`'jax_default_device'`

Environment Variable:  
`JAX_DEFAULT_DEVICE`

Configure the default device for JAX operations. Set to a Device object (e.g. `jax.devices("cpu")[0]`) to use that Device as the default device for JAX operations and jit’d function calls (there is no effect on multi-device computations, e.g. pmapped function calls). Set to None to use the system default device.

### Default Matmul Precision[\#](#jax_default_matmul_precision "Link to this heading")

Type:  
*Enum values:* `'default'`, `'high'`, `'highest'`, `'bfloat16'`, `'tensorfloat32'`, `'float32'`, `'ANY_F8_ANY_F8_F32'`, `'ANY_F8_ANY_F8_F32_FAST_ACCUM'`, `'ANY_F8_ANY_F8_ANY'`, `'ANY_F8_ANY_F8_ANY_FAST_ACCUM'`, `'F16_F16_F16'`, `'F16_F16_F32'`, `'BF16_BF16_BF16'`, `'BF16_BF16_F32'`, `'BF16_BF16_F32_X3'`, `'BF16_BF16_F32_X6'`, `'BF16_BF16_F32_X9'`, `'TF32_TF32_F32'`, `'TF32_TF32_F32_X3'`, `'F32_F32_F32'`, `'F64_F64_F64'`

Default Value:  
`None`

Configuration String:  
`'jax_default_matmul_precision'`

Environment Variable:  
`JAX_DEFAULT_MATMUL_PRECISION`

Control the default matmul and conv precision for 32bit inputs.

Some platforms, like TPU, offer configurable precision levels for matrix multiplication and convolution computations, trading off accuracy for speed. The precision can be controlled for each operation; for example, see the [`jax.lax.conv_general_dilated()`](_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated "jax.lax.conv_general_dilated") and [`jax.lax.dot()`](_autosummary/jax.lax.dot.html#jax.lax.dot "jax.lax.dot") docstrings. But it can be useful to control the default behavior obtained when an operation is not given a specific precision.

This option can be used to control the default precision level for computations involved in matrix multiplication and convolution on 32bit inputs. The levels roughly describe the precision at which scalar products are computed. The ‘bfloat16’ option is the fastest and least precise; ‘float32’ is similar to full float32 precision; ‘tensorfloat32’ is intermediate.

This parameter can also be used to specify an accumulation “algorithm” for functions that perform matrix multiplications, like [`jax.lax.dot()`](_autosummary/jax.lax.dot.html#jax.lax.dot "jax.lax.dot"). To specify an algorithm, set this option to the name of a [`DotAlgorithmPreset`](jax.lax.html#jax.lax.DotAlgorithmPreset "jax.lax.DotAlgorithmPreset").

### Default Prng Impl[\#](#jax_default_prng_impl "Link to this heading")

Type:  
*Enum values:* `'threefry2x32'`, `'threefry4x32'`, `'rbg'`, `'unsafe_rbg'`, `'philox2x32'`, `'philox4x32'`

Default Value:  
`'threefry2x32'`

Configuration String:  
`'jax_default_prng_impl'`

Environment Variable:  
`JAX_DEFAULT_PRNG_IMPL`

Select the default PRNG implementation, used when one is not explicitly provided at seeding time.

### Disable Bwd Checks[\#](#jax_disable_bwd_checks "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_disable_bwd_checks'`

Environment Variable:  
`JAX_DISABLE_BWD_CHECKS`

Disables all bwd pass checks This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Disable Jit[\#](#jax_disable_jit "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_disable_jit'`

Environment Variable:  
`JAX_DISABLE_JIT`

Disable JIT compilation and just call original Python.

### Disable Most Optimizations[\#](#jax_disable_most_optimizations "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_disable_most_optimizations'`

Environment Variable:  
`JAX_DISABLE_MOST_OPTIMIZATIONS`

### Disable Vmap Shmap Error[\#](#jax_disable_vmap_shmap_error "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_disable_vmap_shmap_error'`

Environment Variable:  
`JAX_DISABLE_VMAP_SHMAP_ERROR`

Temporary workaround to disable an error check in vmap-of-shmap.

### Disallow Mesh Context Manager[\#](#jax_disallow_mesh_context_manager "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_disallow_mesh_context_manager'`

Environment Variable:  
`JAX_DISALLOW_MESH_CONTEXT_MANAGER`

If set to True, trying to use a mesh as a context manager will result in a RuntimeError.

### Distributed Debug[\#](#jax_distributed_debug "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_distributed_debug'`

Environment Variable:  
`JAX_DISTRIBUTED_DEBUG`

Enable logging useful for debugging multi-process distributed computations. Logging is performed with logging at WARNING level.

### Dump Ir Modes[\#](#jax_dump_ir_modes "Link to this heading")

Type:  
`str`

Default Value:  
`'stablehlo'`

Configuration String:  
`'jax_dump_ir_modes'`

Environment Variable:  
`JAX_DUMP_IR_MODES`

Comma-delimited modes in which to dump IR. Can be ‘stablehlo’ (the default), ‘jaxpr’, ‘jaxpr_html’, or ‘eqn_count_pprof’ for jaxpr equation count pprof profile.

### Dump Ir To[\#](#jax_dump_ir_to "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_dump_ir_to'`

Environment Variable:  
`JAX_DUMP_IR_TO`

Path to which IR(s) emitted by JAX should be dumped as text files.If omitted, JAX will not dump any IR. Supports the special value ‘sponge’ to pick the path from the environment variable TEST_UNDECLARED_OUTPUTS_DIR. See jax_dump_ir_modes for options governing what is dumped.

### Embedded Constants Max Bytes[\#](#jax_embedded_constants_max_bytes "Link to this heading")

Type:  
`int`

Default Value:  
`32`

Configuration String:  
`'jax_embedded_constants_max_bytes'`

Environment Variable:  
`JAX_EMBEDDED_CONSTANTS_MAX_BYTES`

Maximum size in bytes of a constant that is allowed to be embedded in the lowered HLO. Constants larger than this are hoisted as additional arguments to the executable. See [https://docs.jax.dev/en/latest/internals/constants.html](https://docs.jax.dev/en/latest/internals/constants.html).

### Enable Checks[\#](#jax_enable_checks "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_enable_checks'`

Environment Variable:  
`JAX_ENABLE_CHECKS`

Turn on invariant checking for JAX internals. Makes things slower.

### Enable Compilation Cache[\#](#jax_enable_compilation_cache "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_enable_compilation_cache'`

Environment Variable:  
`JAX_ENABLE_COMPILATION_CACHE`

If set to False, the compilation cache will be disabled regardless of whether set_cache_dir() was called. If set to True, the path could be set to a default value or via a call to set_cache_dir().

### Enable Custom Prng[\#](#jax_enable_custom_prng "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_enable_custom_prng'`

Environment Variable:  
`JAX_ENABLE_CUSTOM_PRNG`

Enables an internal upgrade that allows one to define custom pseudo-random number generator implementations. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Enable Pgle[\#](#jax_enable_pgle "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_enable_pgle'`

Environment Variable:  
`JAX_ENABLE_PGLE`

If set to True and the property jax_pgle_profiling_runs is set to greater than 0, the modules will be recompiled after running specified number times with collected data provided to the profile guided latency estimator.

### Enable Preemption Service[\#](#jax_enable_preemption_service "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_enable_preemption_service'`

Environment Variable:  
`JAX_ENABLE_PREEMPTION_SERVICE`

Enables the preemption service. See multihost_utils.reached_preemption_sync_point for details.

### Enable Recoverability[\#](#jax_enable_recoverability "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_enable_recoverability'`

Environment Variable:  
`JAX_ENABLE_RECOVERABILITY`

Allows a multi-controller JAX job to continue running, even after some tasks have failed.

### Enable X64[\#](#jax_enable_x64 "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_enable_x64'`

Environment Variable:  
`JAX_ENABLE_X64`

Enable 64-bit types to be used

### Error Checking Behavior Divide[\#](#jax_error_checking_behavior_divide "Link to this heading")

Type:  
*Enum values:* `'ignore'`, `'raise'`

Default Value:  
`'ignore'`

Configuration String:  
`'jax_error_checking_behavior_divide'`

Environment Variable:  
`JAX_ERROR_CHECKING_BEHAVIOR_DIVIDE`

Specify the behavior when a divide by zero is encountered. Options are “ignore” or “raise”.

### Error Checking Behavior Nan[\#](#jax_error_checking_behavior_nan "Link to this heading")

Type:  
*Enum values:* `'ignore'`, `'raise'`

Default Value:  
`'ignore'`

Configuration String:  
`'jax_error_checking_behavior_nan'`

Environment Variable:  
`JAX_ERROR_CHECKING_BEHAVIOR_NAN`

Specify the behavior when a NaN is encountered. Options are “ignore” or “raise”.

### Error Checking Behavior Oob[\#](#jax_error_checking_behavior_oob "Link to this heading")

Type:  
*Enum values:* `'ignore'`, `'raise'`

Default Value:  
`'ignore'`

Configuration String:  
`'jax_error_checking_behavior_oob'`

Environment Variable:  
`JAX_ERROR_CHECKING_BEHAVIOR_OOB`

Specify the behavior when an out of bounds access is encountered. Options are “ignore” or “raise”.

### Exec Time Optimization Effort[\#](#jax_exec_time_optimization_effort "Link to this heading")

Type:  
`float`

Default Value:  
`0.0`

Configuration String:  
`'jax_exec_time_optimization_effort'`

Environment Variable:  
`JAX_EXEC_TIME_OPTIMIZATION_EFFORT`

Effort for minimizing execution time (higher means more effort), valid range \[-1.0, 1.0\].

### Experimental Unsafe Xla Runtime Errors[\#](#jax_experimental_unsafe_xla_runtime_errors "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_experimental_unsafe_xla_runtime_errors'`

Environment Variable:  
`JAX_EXPERIMENTAL_UNSAFE_XLA_RUNTIME_ERRORS`

Enable XLA runtime errors for jax.experimental.checkify.checks on CPU and GPU. These errors are async, might get lost and are not very readable. But, they crash the computation and enable you to write jittable checks without needing to checkify. Does not work under pmap/pjit.

### Explain Cache Misses[\#](#jax_explain_cache_misses "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_explain_cache_misses'`

Environment Variable:  
`JAX_EXPLAIN_CACHE_MISSES`

Each time there is a miss on one of the main caches (e.g. the tracing cache), log an explanation. Logging is performed with logging. When this option is set, the log level is WARNING; otherwise the level is DEBUG.

### Explicit X64 Dtypes[\#](#jax_explicit_x64_dtypes "Link to this heading")

Type:  
*Enum values:* `WARN`, `ERROR`, `ALLOW`

Default Value:  
`<ExplicitX64Mode.WARN:`` ``1>`

Configuration String:  
`'jax_explicit_x64_dtypes'`

Environment Variable:  
`JAX_EXPLICIT_X64_DTYPES`

If set to ALLOW, explicit specification of 64-bit types will be respected even if enable_x64 is false. If set to WARN, a warning will be issued, and if set to ERROR, an error will be raised.

### Export Calling Convention Version[\#](#jax_export_calling_convention_version "Link to this heading")

Type:  
`int`

Default Value:  
`10`

Configuration String:  
`'jax_export_calling_convention_version'`

Environment Variable:  
`JAX_EXPORT_CALLING_CONVENTION_VERSION`

The calling convention version number to use for exporting. This must be within the range of versions supported by the tf.XlaCallModule used in your deployment environment. See [https://docs.jax.dev/en/latest/export/shape_poly.html#calling-convention-versions](https://docs.jax.dev/en/latest/export/shape_poly.html#calling-convention-versions).

### Export Ignore Forward Compatibility[\#](#jax_export_ignore_forward_compatibility "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_export_ignore_forward_compatibility'`

Environment Variable:  
`JAX_EXPORT_IGNORE_FORWARD_COMPATIBILITY`

Whether to ignore the forward compatibility lowering rules. See [https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls](https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls).

### Force Dcn Cross Host Transfers[\#](#jax_force_dcn_cross_host_transfers "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_force_dcn_cross_host_transfers'`

Environment Variable:  
`JAX_FORCE_DCN_CROSS_HOST_TRANSFERS`

Force cross host transfers to use the DCN socket transfer library even when the plugin supports cross-host transfers.

### High Dynamic Range Gumbel[\#](#jax_high_dynamic_range_gumbel "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_high_dynamic_range_gumbel'`

Environment Variable:  
`JAX_HIGH_DYNAMIC_RANGE_GUMBEL`

If True, gumble noise draws two samples to cover low probability events with more precision.

### Hlo Source File Canonicalization Regex[\#](#jax_hlo_source_file_canonicalization_regex "Link to this heading")

Type:  
`str`

Default Value:  
`None`

Configuration String:  
`'jax_hlo_source_file_canonicalization_regex'`

Environment Variable:  
`JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX`

Used to canonicalize the source_path metadata of HLO instructions by removing the given regex. If set, re.sub() is called on each source_file with the given regex, and all matches are removed.

### Include Debug Info In Dumps[\#](#jax_include_debug_info_in_dumps "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_include_debug_info_in_dumps'`

Environment Variable:  
`JAX_INCLUDE_DEBUG_INFO_IN_DUMPS`

Determine whether or not to keep debug symbols and location information when dumping IR code. By default, debug information will be preserved in the IR dump. To avoid exposing source code and potentially sensitive information, set to false

### Include Full Tracebacks In Locations[\#](#jax_include_full_tracebacks_in_locations "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_include_full_tracebacks_in_locations'`

Environment Variable:  
`JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS`

Include Python tracebacks in MLIR locations in IR emitted by JAX.

### Legacy Prng Key[\#](#jax_legacy_prng_key "Link to this heading")

Type:  
*Enum values:* `ALLOW`, `WARN`, `ERROR`

Default Value:  
`<LegacyPrngKeyState.ALLOW:`` ``'allow'>`

Configuration String:  
`'jax_legacy_prng_key'`

Environment Variable:  
`JAX_LEGACY_PRNG_KEY`

Specify the behavior when raw PRNG keys are passed to jax.random APIs.

### Log Checkpoint Residuals[\#](#jax_log_checkpoint_residuals "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_log_checkpoint_residuals'`

Environment Variable:  
`JAX_LOG_CHECKPOINT_RESIDUALS`

Log a message every time jax.checkpoint (aka jax.remat) is partially evaluated (e.g. for autodiff), printing what residuals are saved.

### Log Compiles[\#](#jax_log_compiles "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_log_compiles'`

Environment Variable:  
`JAX_LOG_COMPILES`

Log a message each time jit or pmap compiles an XLA computation. Logging is performed with logging. When this option is set, the log level is WARNING; otherwise the level is DEBUG.

### Logging Level[\#](#jax_logging_level "Link to this heading")

Type:  
*Enum values:* `'NOTSET'`, `'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`, `'CRITICAL'`

Default Value:  
`'NOTSET'`

Configuration String:  
`'jax_logging_level'`

Environment Variable:  
`JAX_LOGGING_LEVEL`

Set the corresponding logging level on all jax loggers. Only string values from \[“NOTSET”, “DEBUG”, “INFO”, “WARNING”, “ERROR”, “CRITICAL”\] are accepted. If None, the logging level will not be set. Includes C++ logging.

### Memory Fitting Effort[\#](#jax_memory_fitting_effort "Link to this heading")

Type:  
`float`

Default Value:  
`0.0`

Configuration String:  
`'jax_memory_fitting_effort'`

Environment Variable:  
`JAX_MEMORY_FITTING_EFFORT`

Effort for minimizing memory usage (higher means more effort), valid range \[-1.0, 1.0\].

### Memory Fitting Level[\#](#jax_memory_fitting_level "Link to this heading")

Type:  
*Enum values:* `'UNKNOWN'`, `'O0'`, `'O1'`, `'O2'`, `'O3'`

Default Value:  
`'O2'`

Configuration String:  
`'jax_memory_fitting_level'`

Environment Variable:  
`JAX_MEMORY_FITTING_LEVEL`

The degree to which the compiler should attempt to make the program fit in memory

### Mock Gpu Topology[\#](#jax_mock_gpu_topology "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_mock_gpu_topology'`

Environment Variable:  
`JAX_MOCK_GPU_TOPOLOGY`

Mock multi-host GPU topology in GPU client. The value should be of the form “\<number-of-slices\> x \<number-of-hosts-per-slice\> x \<number-of-devices-per-host\>”. Empty string turns off mocking.

### Mosaic Allow Hlo[\#](#jax_mosaic_allow_hlo "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_mosaic_allow_hlo'`

Environment Variable:  
`JAX_MOSAIC_ALLOW_HLO`

Allow hlo dialects in Mosaic

### Mutable Array Checks[\#](#jax_mutable_array_checks "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_mutable_array_checks'`

Environment Variable:  
`JAX_MUTABLE_ARRAY_CHECKS`

Enable error checks for mutable arrays that rule out aliasing. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### No Execution[\#](#jax_no_execution "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_no_execution'`

Environment Variable:  
`JAX_NO_EXECUTION`

Disallow JAX executions.

### No Tracing[\#](#jax_no_tracing "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_no_tracing'`

Environment Variable:  
`JAX_NO_TRACING`

Disallow tracing for JIT compilation.

### Num Cpu Devices[\#](#jax_num_cpu_devices "Link to this heading")

Type:  
`int`

Default Value:  
`-1`

Configuration String:  
`'jax_num_cpu_devices'`

Environment Variable:  
`JAX_NUM_CPU_DEVICES`

Number of CPU devices to use. If not provided, the value of the XLA flag –xla_force_host_platform_device_count is used. Must be set before JAX is initialized.

### Numpy Dtype Promotion[\#](#jax_numpy_dtype_promotion "Link to this heading")

Type:  
*Enum values:* `STANDARD`, `STRICT`

Default Value:  
`<NumpyDtypePromotion.STANDARD:`` ``'standard'>`

Configuration String:  
`'jax_numpy_dtype_promotion'`

Environment Variable:  
`JAX_NUMPY_DTYPE_PROMOTION`

Specify the rules used for implicit type promotion in operations between arrays. Options are “standard” or “strict”; in strict-mode, binary operations between arrays of differing strongly-specified dtypes will result in an error.

### Numpy Rank Promotion[\#](#jax_numpy_rank_promotion "Link to this heading")

Type:  
*Enum values:* `'allow'`, `'warn'`, `'raise'`

Default Value:  
`'allow'`

Configuration String:  
`'jax_numpy_rank_promotion'`

Environment Variable:  
`JAX_NUMPY_RANK_PROMOTION`

Control NumPy-style automatic rank promotion broadcasting (“allow”, “warn”, or “raise”).

### Oneapi Visible Devices[\#](#jax_oneapi_visible_devices "Link to this heading")

Type:  
`str`

Default Value:  
`'all'`

Configuration String:  
`'jax_oneapi_visible_devices'`

Environment Variable:  
`JAX_ONEAPI_VISIBLE_DEVICES`

Restricts the set of ONEAPI devices that JAX will use. Either “all”, or a comma-separate list of integer device IDs.

### Optimization Level[\#](#jax_optimization_level "Link to this heading")

Type:  
*Enum values:* `'UNKNOWN'`, `'O0'`, `'O1'`, `'O2'`, `'O3'`

Default Value:  
`'UNKNOWN'`

Configuration String:  
`'jax_optimization_level'`

Environment Variable:  
`JAX_OPTIMIZATION_LEVEL`

The degree to which the compiler should optimize for execution time

### Pallas Enable Debug Checks[\#](#jax_pallas_enable_debug_checks "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_pallas_enable_debug_checks'`

Environment Variable:  
`JAX_PALLAS_ENABLE_DEBUG_CHECKS`

If set, `pl.debug_check` calls are checked at runtime. Otherwise, they are a noop.

### Pallas Poison Buffers[\#](#jax_pallas_poison_buffers "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_pallas_poison_buffers'`

Environment Variable:  
`JAX_PALLAS_POISON_BUFFERS`

If set, scratch buffers allocated by Pallas (e.g., in run_scoped) are initialized with poison values (NaN for floats) at allocation time.

### Pallas Use Mosaic Gpu[\#](#jax_pallas_use_mosaic_gpu "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_pallas_use_mosaic_gpu'`

Environment Variable:  
`JAX_PALLAS_USE_MOSAIC_GPU`

If True, lower Pallas kernels to the experimental Mosaic GPU dialect, instead of Triton IR.

### Pallas Verbose Errors[\#](#jax_pallas_verbose_errors "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_pallas_verbose_errors'`

Environment Variable:  
`JAX_PALLAS_VERBOSE_ERRORS`

If True, print verbose error messages for Pallas kernels.

### Persistent Cache Enable Xla Caches[\#](#jax_persistent_cache_enable_xla_caches "Link to this heading")

Type:  
`str`

Default Value:  
`'xla_gpu_per_fusion_autotune_cache_dir'`

Configuration String:  
`'jax_persistent_cache_enable_xla_caches'`

Environment Variable:  
`JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES`

When the persistent cache is enabled, additional XLA caching will also be enabled automatically. This option can be used to configurewhich XLA caching methods will be enabled.

### Persistent Cache Min Compile Time Secs[\#](#jax_persistent_cache_min_compile_time_secs "Link to this heading")

Type:  
`float`

Default Value:  
`1.0`

Configuration String:  
`'jax_persistent_cache_min_compile_time_secs'`

Environment Variable:  
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`

The minimum compile time of a computation to be written to the persistent compilation cache. This threshold can be raised to decrease the number of entries written to the cache.

### Persistent Cache Min Entry Size Bytes[\#](#jax_persistent_cache_min_entry_size_bytes "Link to this heading")

Type:  
`int`

Default Value:  
`0`

Configuration String:  
`'jax_persistent_cache_min_entry_size_bytes'`

Environment Variable:  
`JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES`

The minimum size (in bytes) of an entry that will be cached in the persistent compilation cache: \* -1: disable the size restriction and prevent overrides. \* Leave at default (0) to allow for overrides. The override will typically ensure that the minimum size is optimal for the filesystem being used for the cache. \* \> 0: the actual minimum size desired; no overrides.

### Pgle Aggregation Percentile[\#](#jax_pgle_aggregation_percentile "Link to this heading")

Type:  
`int`

Default Value:  
`90`

Configuration String:  
`'jax_pgle_aggregation_percentile'`

Environment Variable:  
`JAX_PGLE_AGGREGATION_PERCENTILE`

Percentile used to aggregate performance data between devices when PGLE is used.

### Pgle Profiling Runs[\#](#jax_pgle_profiling_runs "Link to this heading")

Type:  
`int`

Default Value:  
`3`

Configuration String:  
`'jax_pgle_profiling_runs'`

Environment Variable:  
`JAX_PGLE_PROFILING_RUNS`

Amount of times module should be profiled before recompilation when PGLE is used.

### Pjrt Client Create Options[\#](#jax_pjrt_client_create_options "Link to this heading")

Type:  
`str`

Default Value:  
`None`

Configuration String:  
`'jax_pjrt_client_create_options'`

Environment Variable:  
`JAX_PJRT_CLIENT_CREATE_OPTIONS`

A set of key-value pairs in the format of “k1:v1;k2:v2” strings provided to a device platform pjrt client as extra arguments.

### Platform Name[\#](#jax_platform_name "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_platform_name'`

Environment Variable:  
`JAX_PLATFORM_NAME`

Deprecated, please use –jax_platforms instead.

### Platforms[\#](#jax_platforms "Link to this heading")

Type:  
`str`

Default Value:  
`None`

Configuration String:  
`'jax_platforms'`

Environment Variable:  
`JAX_PLATFORMS`

Comma-separated list of platform names specifying which platforms jax should initialize. If any of the platforms in this list are not successfully initialized, an exception will be raised and the program will be aborted. The first platform in the list will be the default platform. For example, config.jax_platforms=cpu,tpu means that CPU and TPU backends will be initialized, and the CPU backend will be used unless otherwise specified. If TPU initialization fails, it will raise an exception. By default, jax will try to initialize all available platforms and will default to GPU or TPU if available, and fallback to CPU otherwise.

### Pprint Use Color[\#](#jax_pprint_use_color "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_pprint_use_color'`

Environment Variable:  
`JAX_PPRINT_USE_COLOR`

Enable jaxpr pretty-printing with colorful syntax highlighting.

### Ragged Dot Use Gpu Pallas Triton Lowering[\#](#jax_ragged_dot_use_gpu_pallas_triton_lowering "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_ragged_dot_use_gpu_pallas_triton_lowering'`

Environment Variable:  
`JAX_RAGGED_DOT_USE_GPU_PALLAS_TRITON_LOWERING`

(GPU only) If True, use Pallas Triton lowering for ragged_dot() lowering. Otherwise, rely on the default lowering for ragged_dot_general_p.

### Ragged Dot Use Ragged Dot Instruction[\#](#jax_ragged_dot_use_ragged_dot_instruction "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_ragged_dot_use_ragged_dot_instruction'`

Environment Variable:  
`JAX_RAGGED_DOT_USE_RAGGED_DOT_INSTRUCTION`

(TPU only) If True, use chlo.ragged_dot instruction for ragged_dot() lowering. Otherwise, rely on the rollout logic in lowering rule for ragged_dot_general_p.

### Raise Persistent Cache Errors[\#](#jax_raise_persistent_cache_errors "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_raise_persistent_cache_errors'`

Environment Variable:  
`JAX_RAISE_PERSISTENT_CACHE_ERRORS`

If true, exceptions raised when reading or writing to the persistent compilation cache will be allowed through, halting program execution if not manually caught. If false, exceptions are caught and raised as warnings, allowing program execution to continue. Defaults to false so cache bugs or intermittent issues are non-fatal.

### Random Seed Offset[\#](#jax_random_seed_offset "Link to this heading")

Type:  
`int`

Default Value:  
`0`

Configuration String:  
`'jax_random_seed_offset'`

Environment Variable:  
`JAX_RANDOM_SEED_OFFSET`

Offset to all random seeds (e.g. argument to jax.random.key()).

### Refs To Pins[\#](#jax_refs_to_pins "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_refs_to_pins'`

Environment Variable:  
`JAX_REFS_TO_PINS`

Lower refs to pinned buffers in HLO. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Remove Custom Partitioning Ptr From Cache Key[\#](#jax_remove_custom_partitioning_ptr_from_cache_key "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_remove_custom_partitioning_ptr_from_cache_key'`

Environment Variable:  
`JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY`

If set to True, remove the custom partitioning pointer present in the precompiled stableHLO before hashing during cache key computation. This is a potentially unsafe flag to set and only users who are sure of what they are trying to achieve should set it.

### Remove Size One Mesh Axis From Type[\#](#jax_remove_size_one_mesh_axis_from_type "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_remove_size_one_mesh_axis_from_type'`

Environment Variable:  
`JAX_REMOVE_SIZE_ONE_MESH_AXIS_FROM_TYPE`

Removes mesh axes of size 1 from ShapedArray.sharding and vma

### Rocm Visible Devices[\#](#jax_rocm_visible_devices "Link to this heading")

Type:  
`str`

Default Value:  
`'all'`

Configuration String:  
`'jax_rocm_visible_devices'`

Environment Variable:  
`JAX_ROCM_VISIBLE_DEVICES`

Restricts the set of ROCM devices that JAX will use. Either “all”, or a comma-separate list of integer device IDs.

### Scan3[\#](#jax_scan3 "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_scan3'`

Environment Variable:  
`JAX_SCAN3`

If True, embrace the future of loops. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Send Traceback To Runtime[\#](#jax_send_traceback_to_runtime "Link to this heading")

Type:  
*Enum values:* `OFF`, `ON`, `FULL`

Default Value:  
`<RuntimeTracebackMode.OFF:`` ``'off'>`

Configuration String:  
`'jax_send_traceback_to_runtime'`

Environment Variable:  
`JAX_SEND_TRACEBACK_TO_RUNTIME`

Controls the level of Python traceback information sent to the runtime at dispatch time: - “OFF”: (default) No Python traceback information is sent. - “ON”: Only the most recent user frame call location is sent. - “FULL”: The full Python traceback of the call location is sent. This has a high fixed cost on the dispatch path and should be used only for debugging.

### Share Binary Between Hosts[\#](#jax_share_binary_between_hosts "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_share_binary_between_hosts'`

Environment Variable:  
`JAX_SHARE_BINARY_BETWEEN_HOSTS`

If set to True, the compiled module will be shared between hosts directly.

### Share Binary Between Hosts Timeout Ms[\#](#jax_share_binary_between_hosts_timeout_ms "Link to this heading")

Type:  
`int`

Default Value:  
`1200000`

Configuration String:  
`'jax_share_binary_between_hosts_timeout_ms'`

Environment Variable:  
`JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS`

Timeout for the compiled module share.

### Softmax Custom Jvp[\#](#jax_softmax_custom_jvp "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_softmax_custom_jvp'`

Environment Variable:  
`JAX_SOFTMAX_CUSTOM_JVP`

Use a new custom_jvp rule for jax.nn.softmax. The new rule should improve memory usage and stability. Set True to use new behavior. See [jax-ml/jax#15677](https://github.com/jax-ml/jax/pull/15677) This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Sort Devices By Process Index[\#](#jax_sort_devices_by_process_index "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_sort_devices_by_process_index'`

Environment Variable:  
`JAX_SORT_DEVICES_BY_PROCESS_INDEX`

Sort JAX devices by process index first, then by device id. If False, sort devices only by device id, which preserves the global device ordering assigned by the PJRT client.

### Source Url Schema[\#](#jax_source_url_schema "Link to this heading")

Type:  
`str`

Default Value:  
`None`

Configuration String:  
`'jax_source_url_schema'`

Environment Variable:  
`JAX_SOURCE_URL_SCHEMA`

URL format string used to generate links to source files in the HTML jaxpr dumps. Can contain {file} and {line} placeholders.

### Thread Guard[\#](#jax_thread_guard "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_thread_guard'`

Environment Variable:  
`JAX_THREAD_GUARD`

If True, an error will be raised at runtime if a multi-process JAX operation is called from a thread other than the one in which the thread guard was set. This is useful for detecting cases where threads may schedule operations in different orders in different processes, leading to non-deterministic crashes.

### Threefry Gpu Kernel Lowering[\#](#jax_threefry_gpu_kernel_lowering "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_threefry_gpu_kernel_lowering'`

Environment Variable:  
`JAX_THREEFRY_GPU_KERNEL_LOWERING`

On GPU, lower threefry PRNG operations to a kernel implementation. This makes compile times faster at a potential runtime memory cost.

### Threefry Partitionable[\#](#jax_threefry_partitionable "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_threefry_partitionable'`

Environment Variable:  
`JAX_THREEFRY_PARTITIONABLE`

Enables internal threefry PRNG implementation changes that render it automatically partitionable in some cases. Without this flag, using the standard jax.random pseudo-random number generation may result in extraneous communication and/or redundant distributed computation. With this flag, the communication overheads disappear in some cases. This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Traceback Filtering[\#](#jax_traceback_filtering "Link to this heading")

Type:  
*Enum values:* `'off'`, `'tracebackhide'`, `'remove_frames'`, `'quiet_remove_frames'`, `'auto'`

Default Value:  
`'auto'`

Configuration String:  
`'jax_traceback_filtering'`

Environment Variable:  
`JAX_TRACEBACK_FILTERING`

Controls how JAX filters internal frames out of tracebacks. Valid values are: - `off`: disables traceback filtering. - `auto`: use `tracebackhide` if running under a sufficiently new IPython, or `remove_frames` otherwise. - `tracebackhide`: adds `__tracebackhide__` annotations to hidden stack frames, which some traceback printers support. - `remove_frames`: removes hidden frames from tracebacks, and adds the unfiltered traceback as a `__cause__` of the exception. - `quiet_remove_frames`: removes hidden frames from tracebacks, and adds a brief message (to the `__cause__` of the exception) describing that this has happened.

### Traceback In Locations Limit[\#](#jax_traceback_in_locations_limit "Link to this heading")

Type:  
`int`

Default Value:  
`100`

Configuration String:  
`'jax_traceback_in_locations_limit'`

Environment Variable:  
`JAX_TRACEBACK_IN_LOCATIONS_LIMIT`

Limit the number of frames at the Python traceback frames included in MLIR locations. If set to the negative value, traceback will not be limited.

### Tracer Error Num Traceback Frames[\#](#jax_tracer_error_num_traceback_frames "Link to this heading")

Type:  
`int`

Default Value:  
`5`

Configuration String:  
`'jax_tracer_error_num_traceback_frames'`

Environment Variable:  
`JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES`

Set the number of stack frames in JAX tracer error messages.

### Transfer Guard[\#](#jax_transfer_guard "Link to this heading")

Type:  
*Enum values:* `'allow'`, `'log'`, `'disallow'`, `'log_explicit'`, `'disallow_explicit'`

Default Value:  
`None`

Configuration String:  
`'jax_transfer_guard'`

Environment Variable:  
`JAX_TRANSFER_GUARD`

Select the transfer guard level for all transfers. This option is set-only; the transfer guard level for a specific direction should be read using the per-transfer direction option. Default is “allow”.

### Transfer Guard Device To Device[\#](#jax_transfer_guard_device_to_device "Link to this heading")

Type:  
*Enum values:* `'allow'`, `'log'`, `'disallow'`, `'log_explicit'`, `'disallow_explicit'`

Default Value:  
`None`

Configuration String:  
`'jax_transfer_guard_device_to_device'`

Environment Variable:  
`JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE`

Select the transfer guard level for device-to-device transfers. Default is “allow”.

### Transfer Guard Device To Host[\#](#jax_transfer_guard_device_to_host "Link to this heading")

Type:  
*Enum values:* `'allow'`, `'log'`, `'disallow'`, `'log_explicit'`, `'disallow_explicit'`

Default Value:  
`None`

Configuration String:  
`'jax_transfer_guard_device_to_host'`

Environment Variable:  
`JAX_TRANSFER_GUARD_DEVICE_TO_HOST`

Select the transfer guard level for device-to-host transfers. Default is “allow”.

### Transfer Guard Host To Device[\#](#jax_transfer_guard_host_to_device "Link to this heading")

Type:  
*Enum values:* `'allow'`, `'log'`, `'disallow'`, `'log_explicit'`, `'disallow_explicit'`

Default Value:  
`None`

Configuration String:  
`'jax_transfer_guard_host_to_device'`

Environment Variable:  
`JAX_TRANSFER_GUARD_HOST_TO_DEVICE`

Select the transfer guard level for host-to-device transfers. Default is “allow”.

### Use Direct Linearize[\#](#jax_use_direct_linearize "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_use_direct_linearize'`

Environment Variable:  
`JAX_USE_DIRECT_LINEARIZE`

Use direct linearization instead JVP followed by partial eval

### Use Magma[\#](#jax_use_magma "Link to this heading")

Type:  
*Enum values:* `'off'`, `'on'`, `'auto'`

Default Value:  
`'auto'`

Configuration String:  
`'jax_use_magma'`

Environment Variable:  
`JAX_USE_MAGMA`

Enable experimental support for MAGMA-backed lax.linalg.eig on GPU. See the documentation for lax.linalg.eig for more details about how to use this feature.

### Use Shardy Partitioner[\#](#jax_use_shardy_partitioner "Link to this heading")

Type:  
`bool`

Default Value:  
`True`

Configuration String:  
`'jax_use_shardy_partitioner'`

Environment Variable:  
`JAX_USE_SHARDY_PARTITIONER`

Whether to lower to Shardy. See the migration guide for more information: [https://docs.jax.dev/en/latest/shardy_jax_migration.html](https://docs.jax.dev/en/latest/shardy_jax_migration.html). This will be enabled by default in future versions of JAX, at which point all uses of the flag will be considered deprecated (following the [API compatibility policy](https://docs.jax.dev/en/latest/api_compatibility.html)).

### Use Simplified Jaxpr Constants[\#](#jax_use_simplified_jaxpr_constants "Link to this heading")

Type:  
`bool`

Default Value:  
`False`

Configuration String:  
`'jax_use_simplified_jaxpr_constants'`

Environment Variable:  
`JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS`

Enable a simplification of the handling of closed-over constants in Jaxpr. The value True enables the new behavior. This flag will exist only briefly, while we transition users. See [https://docs.jax.dev/en/latest/internals/constants.html.DO](https://docs.jax.dev/en/latest/internals/constants.html.DO) NOT RELY ON THIS FLAG.

### Xla Backend[\#](#jax_xla_backend "Link to this heading")

Type:  
`str`

Default Value:  
`''`

Configuration String:  
`'jax_xla_backend'`

Environment Variable:  
`JAX_XLA_BACKEND`

Deprecated, please use –jax_platforms instead.

### Xla Profile Version[\#](#jax_xla_profile_version "Link to this heading")

Type:  
`int`

Default Value:  
`0`

Configuration String:  
`'jax_xla_profile_version'`

Environment Variable:  
`JAX_XLA_PROFILE_VERSION`

Optional profile version for XLA compilation. This is meaningful only when XLA is configured to support the remote compilation profile feature.

### Mock Num Gpu Processes[\#](#mock_num_gpu_processes "Link to this heading")

Type:  
`int`

Default Value:  
`0`

Configuration String:  
`'mock_num_gpu_processes'`

Environment Variable:  
`MOCK_NUM_GPU_PROCESSES`

Mock number of JAX processes in GPU client. Value zero turns off mocking.

[](glossary.html "previous page")

previous

Glossary of terms

Contents

- [How to Use Configuration Options](#how-to-use-configuration-options)
- [Common Configuration Options](#common-configuration-options)
- [All Configuration Options](#all-configuration-options)
  - [Check Vma](#check_vma)
  - [Eager Constant Folding](#eager_constant_folding)
  - [Jax2Tf Associative Scan Reductions](#jax2tf_associative_scan_reductions)
  - [Jax2Tf Default Native Serialization](#jax2tf_default_native_serialization)
  - [Allow F16 Reductions](#jax_allow_f16_reductions)
  - [Array Garbage Collection Guard](#jax_array_garbage_collection_guard)
  - [Auto Pcast](#jax_auto_pcast)
  - [Backend Target](#jax_backend_target)
  - [Bcoo Cusparse Lowering](#jax_bcoo_cusparse_lowering)
  - [Captured Constants Report Frames](#jax_captured_constants_report_frames)
  - [Captured Constants Warn Bytes](#jax_captured_constants_warn_bytes)
  - [Check Proxy Envs](#jax_check_proxy_envs)
  - [Check Static Indices](#jax_check_static_indices)
  - [Check Tracer Leaks](#jax_check_tracer_leaks)
  - [Compilation Cache Check Contents](#jax_compilation_cache_check_contents)
  - [Compilation Cache Dir](#jax_compilation_cache_dir)
  - [Compilation Cache Expect Pgle](#jax_compilation_cache_expect_pgle)
  - [Compilation Cache Include Metadata In Key](#jax_compilation_cache_include_metadata_in_key)
  - [Compilation Cache Max Size](#jax_compilation_cache_max_size)
  - [Compiler Detailed Logging Min Ops](#jax_compiler_detailed_logging_min_ops)
  - [Compiler Enable Remat Pass](#jax_compiler_enable_remat_pass)
  - [Cpu Collectives Implementation](#jax_cpu_collectives_implementation)
  - [Cpu Enable Async Dispatch](#jax_cpu_enable_async_dispatch)
  - [Cpu Get Global Topology Timeout Minutes](#jax_cpu_get_global_topology_timeout_minutes)
  - [Cpu Get Local Topology Timeout Minutes](#jax_cpu_get_local_topology_timeout_minutes)
  - [Cross Host Transfer Socket Address](#jax_cross_host_transfer_socket_address)
  - [Cross Host Transfer Timeout Seconds](#jax_cross_host_transfer_timeout_seconds)
  - [Cross Host Transfer Transfer Size](#jax_cross_host_transfer_transfer_size)
  - [Cross Host Transport Addresses](#jax_cross_host_transport_addresses)
  - [Cuda Visible Devices](#jax_cuda_visible_devices)
  - [Custom Vjp3](#jax_custom_vjp3)
  - [Debug Infs](#jax_debug_infs)
  - [Debug Key Reuse](#jax_debug_key_reuse)
  - [Debug Leaked Clients On Clear Backends](#jax_debug_leaked_clients_on_clear_backends)
  - [Debug Log Modules](#jax_debug_log_modules)
  - [Debug Nans](#jax_debug_nans)
  - [Default Device](#jax_default_device)
  - [Default Matmul Precision](#jax_default_matmul_precision)
  - [Default Prng Impl](#jax_default_prng_impl)
  - [Disable Bwd Checks](#jax_disable_bwd_checks)
  - [Disable Jit](#jax_disable_jit)
  - [Disable Most Optimizations](#jax_disable_most_optimizations)
  - [Disable Vmap Shmap Error](#jax_disable_vmap_shmap_error)
  - [Disallow Mesh Context Manager](#jax_disallow_mesh_context_manager)
  - [Distributed Debug](#jax_distributed_debug)
  - [Dump Ir Modes](#jax_dump_ir_modes)
  - [Dump Ir To](#jax_dump_ir_to)
  - [Embedded Constants Max Bytes](#jax_embedded_constants_max_bytes)
  - [Enable Checks](#jax_enable_checks)
  - [Enable Compilation Cache](#jax_enable_compilation_cache)
  - [Enable Custom Prng](#jax_enable_custom_prng)
  - [Enable Pgle](#jax_enable_pgle)
  - [Enable Preemption Service](#jax_enable_preemption_service)
  - [Enable Recoverability](#jax_enable_recoverability)
  - [Enable X64](#jax_enable_x64)
  - [Error Checking Behavior Divide](#jax_error_checking_behavior_divide)
  - [Error Checking Behavior Nan](#jax_error_checking_behavior_nan)
  - [Error Checking Behavior Oob](#jax_error_checking_behavior_oob)
  - [Exec Time Optimization Effort](#jax_exec_time_optimization_effort)
  - [Experimental Unsafe Xla Runtime Errors](#jax_experimental_unsafe_xla_runtime_errors)
  - [Explain Cache Misses](#jax_explain_cache_misses)
  - [Explicit X64 Dtypes](#jax_explicit_x64_dtypes)
  - [Export Calling Convention Version](#jax_export_calling_convention_version)
  - [Export Ignore Forward Compatibility](#jax_export_ignore_forward_compatibility)
  - [Force Dcn Cross Host Transfers](#jax_force_dcn_cross_host_transfers)
  - [High Dynamic Range Gumbel](#jax_high_dynamic_range_gumbel)
  - [Hlo Source File Canonicalization Regex](#jax_hlo_source_file_canonicalization_regex)
  - [Include Debug Info In Dumps](#jax_include_debug_info_in_dumps)
  - [Include Full Tracebacks In Locations](#jax_include_full_tracebacks_in_locations)
  - [Legacy Prng Key](#jax_legacy_prng_key)
  - [Log Checkpoint Residuals](#jax_log_checkpoint_residuals)
  - [Log Compiles](#jax_log_compiles)
  - [Logging Level](#jax_logging_level)
  - [Memory Fitting Effort](#jax_memory_fitting_effort)
  - [Memory Fitting Level](#jax_memory_fitting_level)
  - [Mock Gpu Topology](#jax_mock_gpu_topology)
  - [Mosaic Allow Hlo](#jax_mosaic_allow_hlo)
  - [Mutable Array Checks](#jax_mutable_array_checks)
  - [No Execution](#jax_no_execution)
  - [No Tracing](#jax_no_tracing)
  - [Num Cpu Devices](#jax_num_cpu_devices)
  - [Numpy Dtype Promotion](#jax_numpy_dtype_promotion)
  - [Numpy Rank Promotion](#jax_numpy_rank_promotion)
  - [Oneapi Visible Devices](#jax_oneapi_visible_devices)
  - [Optimization Level](#jax_optimization_level)
  - [Pallas Enable Debug Checks](#jax_pallas_enable_debug_checks)
  - [Pallas Poison Buffers](#jax_pallas_poison_buffers)
  - [Pallas Use Mosaic Gpu](#jax_pallas_use_mosaic_gpu)
  - [Pallas Verbose Errors](#jax_pallas_verbose_errors)
  - [Persistent Cache Enable Xla Caches](#jax_persistent_cache_enable_xla_caches)
  - [Persistent Cache Min Compile Time Secs](#jax_persistent_cache_min_compile_time_secs)
  - [Persistent Cache Min Entry Size Bytes](#jax_persistent_cache_min_entry_size_bytes)
  - [Pgle Aggregation Percentile](#jax_pgle_aggregation_percentile)
  - [Pgle Profiling Runs](#jax_pgle_profiling_runs)
  - [Pjrt Client Create Options](#jax_pjrt_client_create_options)
  - [Platform Name](#jax_platform_name)
  - [Platforms](#jax_platforms)
  - [Pprint Use Color](#jax_pprint_use_color)
  - [Ragged Dot Use Gpu Pallas Triton Lowering](#jax_ragged_dot_use_gpu_pallas_triton_lowering)
  - [Ragged Dot Use Ragged Dot Instruction](#jax_ragged_dot_use_ragged_dot_instruction)
  - [Raise Persistent Cache Errors](#jax_raise_persistent_cache_errors)
  - [Random Seed Offset](#jax_random_seed_offset)
  - [Refs To Pins](#jax_refs_to_pins)
  - [Remove Custom Partitioning Ptr From Cache Key](#jax_remove_custom_partitioning_ptr_from_cache_key)
  - [Remove Size One Mesh Axis From Type](#jax_remove_size_one_mesh_axis_from_type)
  - [Rocm Visible Devices](#jax_rocm_visible_devices)
  - [Scan3](#jax_scan3)
  - [Send Traceback To Runtime](#jax_send_traceback_to_runtime)
  - [Share Binary Between Hosts](#jax_share_binary_between_hosts)
  - [Share Binary Between Hosts Timeout Ms](#jax_share_binary_between_hosts_timeout_ms)
  - [Softmax Custom Jvp](#jax_softmax_custom_jvp)
  - [Sort Devices By Process Index](#jax_sort_devices_by_process_index)
  - [Source Url Schema](#jax_source_url_schema)
  - [Thread Guard](#jax_thread_guard)
  - [Threefry Gpu Kernel Lowering](#jax_threefry_gpu_kernel_lowering)
  - [Threefry Partitionable](#jax_threefry_partitionable)
  - [Traceback Filtering](#jax_traceback_filtering)
  - [Traceback In Locations Limit](#jax_traceback_in_locations_limit)
  - [Tracer Error Num Traceback Frames](#jax_tracer_error_num_traceback_frames)
  - [Transfer Guard](#jax_transfer_guard)
  - [Transfer Guard Device To Device](#jax_transfer_guard_device_to_device)
  - [Transfer Guard Device To Host](#jax_transfer_guard_device_to_host)
  - [Transfer Guard Host To Device](#jax_transfer_guard_host_to_device)
  - [Use Direct Linearize](#jax_use_direct_linearize)
  - [Use Magma](#jax_use_magma)
  - [Use Shardy Partitioner](#jax_use_shardy_partitioner)
  - [Use Simplified Jaxpr Constants](#jax_use_simplified_jaxpr_constants)
  - [Xla Backend](#jax_xla_backend)
  - [Xla Profile Version](#jax_xla_profile_version)
  - [Mock Num Gpu Processes](#mock_num_gpu_processes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
