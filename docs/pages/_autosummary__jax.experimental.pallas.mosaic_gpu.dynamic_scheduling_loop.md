- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop

## Contents

- [`dynamic_scheduling_loop()`](#jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop)

# jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop[\#](#jax-experimental-pallas-mosaic-gpu-dynamic-scheduling-loop "Link to this heading")

jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop(*grid_names: Sequence\[Hashable\]*, *\**, *thread_axis: Hashable \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *cluster_axes: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), ...\], ...\] = ()*, *init_carry: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[NDLoopInfo\], [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")\]\], [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/helpers.py#L328-L432)[\#](#jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop "Link to this definition")\
jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop(*grid_names: Sequence\[Hashable\]*, *\**, *thread_axis: Hashable \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *cluster_axes: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), ...\], ...\] = ()*, *init_carry: \_T*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[NDLoopInfo, \_T\], \_T\]\], \_T\]  
A loop over program instances using dynamic work scheduling.

This loop will iterate through available program instances until all work has been scheduled. The kernel should be instantiated with a grid equal to the logical amount of work to be done (as opposed to a persistent kernel where the grid is set to the number of cores). Each core running this loop will continuously query the next available block of work and the loop will terminate when the entire grid has been scheduled.

Example usage:

    @plgpu.dynamic_scheduling_loop(grid_names)
    def body(loop_info):
      work(loop_info.index)  # do work...

Parameters:  
- **grid_names** – The names of the axes in the grid.

- **thread_axis** – The name of the thread axis. This must be passed in if the kernel uses multiple threads.

- **cluster_axes** – The name of **all** cluster axes. This must be passed in if the kernel uses a cluster size \> 1.

- **init_carry** – An optional initial carry for the loop. If passed in, the body function should expect a `carry` keyword argument and return the next carry value.

[](jax.experimental.pallas.mosaic_gpu.nd_loop.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.nd_loop

[](jax.experimental.pallas.mosaic_gpu.barrier_arrive.html "next page")

next

jax.experimental.pallas.mosaic_gpu.barrier_arrive

Contents

- [`dynamic_scheduling_loop()`](#jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\
