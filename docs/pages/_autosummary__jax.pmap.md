- [](../index.html)
- [API Reference](../jax.html)
- jax.pmap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.pmap.rst "Download source file")
-  .pdf

# jax.pmap

## Contents

- [`pmap()`](#jax.pmap)

# jax.pmap[\#](#jax-pmap "Link to this heading")

jax.pmap(*fun*, *axis_name=None*, *\**, *in_axes=0*, *out_axes=0*, *static_broadcasted_argnums=()*, *devices=None*, *backend=None*, *axis_size=None*, *donate_argnums=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pmap.py#L57-L342)[\#](#jax.pmap "Link to this definition")  
Old way of doing parallel map. Use [`jax.shard_map()`](jax.shard_map.html#jax.shard_map "jax.shard_map") instead.

Note

While [`jax.pmap()`](#jax.pmap "jax.pmap") works, you should probably use [`jax.shard_map()`](jax.shard_map.html#jax.shard_map "jax.shard_map") or `jax.smap` instead. shard_map supports more efficient autodiff, and is more composable in the multi-controller setting. See [https://docs.jax.dev/en/latest/notebooks/shard_map.html](https://docs.jax.dev/en/latest/notebooks/shard_map.html) for examples.

Note

[`pmap()`](#jax.pmap "jax.pmap") is now implemented in terms of [`jit()`](jax.jit.html#jax.jit "jax.jit") and [`shard_map()`](jax.shard_map.html#jax.shard_map "jax.shard_map"). Please see the [migration guide](https://docs.jax.dev/en/latest/migrate_pmap.html) for more information.

The purpose of [`pmap()`](#jax.pmap "jax.pmap") is to express single-program multiple-data (SPMD) programs. Applying [`pmap()`](#jax.pmap "jax.pmap") to a function will compile the function with XLA (similarly to [`jit()`](jax.jit.html#jax.jit "jax.jit")), then execute it in parallel on XLA devices, such as multiple GPUs or multiple TPU cores. Semantically it is comparable to [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") because both transformations map a function over array axes, but where [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") vectorizes functions by pushing the mapped axis down into primitive operations, [`pmap()`](#jax.pmap "jax.pmap") instead replicates the function and executes each replica on its own XLA device in parallel.

The mapped axis size must be less than or equal to the number of local XLA devices available, as returned by [`jax.local_device_count()`](jax.local_device_count.html#jax.local_device_count "jax.local_device_count") (unless `devices` is specified, see below). For nested [`pmap()`](#jax.pmap "jax.pmap") calls, the product of the mapped axis sizes must be less than or equal to the number of XLA devices.

Note

[`pmap()`](#jax.pmap "jax.pmap") compiles `fun`, so while it can be combined with [`jit()`](jax.jit.html#jax.jit "jax.jit"), it’s usually unnecessary.

[`pmap()`](#jax.pmap "jax.pmap") requires that all of the participating devices are identical. For example, it is not possible to use [`pmap()`](#jax.pmap "jax.pmap") to parallelize a computation across two different models of GPU. It is currently an error for the same device to participate twice in the same pmap.

**Multi-process platforms:** On multi-process platforms such as TPU pods, [`pmap()`](#jax.pmap "jax.pmap") is designed to be used in SPMD Python programs, where every process is running the same Python code such that all processes run the same pmapped function in the same order. Each process should still call the pmapped function with mapped axis size equal to the number of *local* devices (unless `devices` is specified, see below), and an array of the same leading axis size will be returned as usual. However, any collective operations in `fun` will be computed over *all* participating devices, including those on other processes, via device-to-device communication. Conceptually, this can be thought of as running a pmap over a single array sharded across processes, where each process “sees” only its local shard of the input and output. The SPMD model requires that the same multi-process pmaps must be run in the same order on all devices, but they can be interspersed with arbitrary operations running in a single process.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function to be mapped over argument axes. Its arguments and return value should be arrays, scalars, or (nested) standard Python containers (tuple/list/dict) thereof. Positional arguments indicated by `static_broadcasted_argnums` can be anything at all, provided they are hashable and have an equality operation defined.

- **axis_name** (*AxisName* *\|* *None*) – Optional, a hashable Python object used to identify the mapped axis so that parallel collectives can be applied.

- **in_axes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None* *\|* *Sequence\[Any\]*) – A non-negative integer, None, or nested Python container thereof that specifies which axes of positional arguments to map over. Arguments passed as keywords are always mapped over their leading axis (i.e. axis index 0). See [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") for details.

- **out_axes** (*Any*) – A non-negative integer, None, or nested Python container thereof indicating where the mapped axis should appear in the output. All outputs with a mapped axis must have a non-None `out_axes` specification (see [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap")).

- **static_broadcasted_argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Iterable\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) –

  An int or collection of ints specifying which positional arguments to treat as static (compile-time constant). Operations that only depend on static arguments will be constant-folded. Calling the pmapped function with different values for these constants will trigger recompilation. If the pmapped function is called with fewer positional arguments than indicated by `static_broadcasted_argnums` then an error is raised. Each of the static arguments will be broadcasted to all devices. Arguments that are not arrays or containers thereof must be marked as static. Defaults to ().

  Static arguments must be hashable, meaning both `__hash__` and `__eq__` are implemented, and should be immutable.

- **devices** (*Sequence\[xc.Device\]* *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a sequence of Devices to map over. (Available devices can be retrieved via jax.devices()). Must be given identically for each process in multi-process settings (and will therefore include devices across processes). If specified, the size of the mapped axis must be equal to the number of devices in the sequence local to the given process. Nested [`pmap()`](#jax.pmap "jax.pmap") s with `devices` specified in either the inner or outer [`pmap()`](#jax.pmap "jax.pmap") are not yet supported.

- **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a string representing the XLA backend. ‘cpu’, ‘gpu’, or ‘tpu’.

- **axis_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Optional; the size of the mapped axis.

- **donate_argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Iterable\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) –

  Specify which positional argument buffers are “donated” to the computation. It is safe to donate argument buffers if you no longer need them once the computation has finished. In some cases XLA can make use of donated buffers to reduce the amount of memory needed to perform a computation, for example recycling one of your input buffers to store a result. You should not reuse buffers that you donate to a computation, JAX will raise an error if you try to. Note that donate_argnums only work for positional arguments, and keyword arguments will not be donated.

  For more details on buffer donation see the [FAQ](https://docs.jax.dev/en/latest/faq.html#buffer-donation).

Returns:  
A parallelized version of `fun` with arguments that correspond to those of `fun` but with extra array axes at positions indicated by `in_axes` and with output that has an additional leading array axis (with the same size).

Return type:  
Any

For example, assuming 8 XLA devices are available, [`pmap()`](#jax.pmap "jax.pmap") can be used as a map along a leading array axis:

    >>> import jax.numpy as jnp
    >>>
    >>> out = pmap(lambda x: x ** 2)(jnp.arange(8))  
    >>> print(out)  
    [0, 1, 4, 9, 16, 25, 36, 49]

When the leading dimension is smaller than the number of available devices JAX will simply run on a subset of devices:

    >>> x = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2))
    >>> y = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2
    >>> out = pmap(jnp.dot)(x, y)  
    >>> print(out)  
    [[[    4.     9.]
      [   12.    29.]]
     [[  244.   345.]
      [  348.   493.]]
     [[ 1412.  1737.]
      [ 1740.  2141.]]]

If your leading dimension is larger than the number of available devices you will get an error:

    >>> pmap(lambda x: x ** 2)(jnp.arange(9))  
    ValueError: ... requires 9 replicas, but only 8 XLA devices are available

As with [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap"), using `None` in `in_axes` indicates that an argument doesn’t have an extra axis and should be broadcasted, rather than mapped, across the replicas:

    >>> x, y = jnp.arange(2.), 4.
    >>> out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  
    >>> print(out)  
    ([4., 5.], [8., 8.])

Note that [`pmap()`](#jax.pmap "jax.pmap") always returns values mapped over their leading axis, equivalent to using `out_axes=0` in [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap").

In addition to expressing pure maps, [`pmap()`](#jax.pmap "jax.pmap") can also be used to express parallel single-program multiple-data (SPMD) programs that communicate via collective operations. For example:

    >>> f = lambda x: x / jax.lax.psum(x, axis_name='i')
    >>> out = pmap(f, axis_name='i')(jnp.arange(4.))  
    >>> print(out)  
    [ 0.          0.16666667  0.33333334  0.5       ]
    >>> print(out.sum())  
    1.0

In this example, `axis_name` is a string, but it can be any Python object with `__hash__` and `__eq__` defined.

The argument `axis_name` to [`pmap()`](#jax.pmap "jax.pmap") names the mapped axis so that collective operations, like [`jax.lax.psum()`](jax.lax.psum.html#jax.lax.psum "jax.lax.psum"), can refer to it. Axis names are important particularly in the case of nested [`pmap()`](#jax.pmap "jax.pmap") functions, where collective operations can operate over distinct axes:

On multi-process platforms, collective operations operate over all devices, including those on other processes. For example, assuming the following code runs on two processes with 4 XLA devices each:

    >>> f = lambda x: x + jax.lax.psum(x, axis_name='i')
    >>> data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
    >>> out = pmap(f, axis_name='i')(data)  
    >>> print(out)  
    [28 29 30 31] # on process 0
    [32 33 34 35] # on process 1

Each process passes in a different length-4 array, corresponding to its 4 local devices, and the psum operates over all 8 values. Conceptually, the two length-4 arrays can be thought of as a sharded length-8 array (in this example equivalent to jnp.arange(8)) that is mapped over, with the length-8 mapped axis given name ‘i’. The pmap call on each process then returns the corresponding length-4 output shard.

The `devices` argument can be used to specify exactly which devices are used to run the parallel computation. For example, again assuming a single process with 8 devices, the following code defines two parallel computations, one which runs on the first six devices and one on the remaining two:

    >>> from functools import partial
    >>> @partial(pmap, axis_name='i', devices=jax.devices()[:6])
    ... def f1(x):
    ...   return x / jax.lax.psum(x, axis_name='i')
    >>>
    >>> @partial(pmap, axis_name='i', devices=jax.devices()[-2:])
    ... def f2(x):
    ...   return jax.lax.psum(x ** 2, axis_name='i')
    >>>
    >>> print(f1(jnp.arange(6.)))  
    [0.         0.06666667 0.13333333 0.2        0.26666667 0.33333333]
    >>> print(f2(jnp.array([2., 3.])))  
    [ 13.  13.]

[](jax.smap.html "previous page")

previous

jax.smap

[](jax.devices.html "next page")

next

jax.devices

Contents

- [`pmap()`](#jax.pmap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
