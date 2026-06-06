- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.nd_loop

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.nd_loop.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.nd_loop

## Contents

- [`nd_loop()`](#jax.experimental.pallas.mosaic_gpu.nd_loop)

# jax.experimental.pallas.mosaic_gpu.nd_loop[\#](#jax-experimental-pallas-mosaic-gpu-nd-loop "Link to this heading")

jax.experimental.pallas.mosaic_gpu.nd_loop(*grid: Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]*, *\**, *collective_axes: Sequence\[Hashable\] \| Hashable*, *tiling: Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *init_carry: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[NDLoopInfo\], [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")\]\], [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/helpers.py#L76-L181)[\#](#jax.experimental.pallas.mosaic_gpu.nd_loop "Link to this definition")\
jax.experimental.pallas.mosaic_gpu.nd_loop(*grid: Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]*, *\**, *collective_axes: Sequence\[Hashable\] \| Hashable*, *tiling: Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *init_carry: \_T*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[NDLoopInfo, \_T\], \_T\]\], \_T\]  
A loop over a multi-dimensional grid partitioned along the given axes.

The body of the loop a single argument `loop_info` which is an NDLoopInfo object containing index and iteration information. However if a carry is specified, the body will expect a second keyword argument carry containing the loop carry.

For example, if `collective_axes` is `"x"` with `lax.axis_size()` equal to 4 and the grid is (2, 3), the implementation would produce the following iteration order

| loop step | index  | axis index |
|-----------|--------|------------|
| 0         | (0, 0) | 0          |
| 1         | (0, 1) | 1          |
| 2         | (0, 2) | 2          |
| 3         | (1, 0) | 3          |
| 4         | (1, 1) | 0          |
| 5         | (1, 2) | 1          |

which comes from partitioning the flat iteration space into chunks in an interleaved fashion wrt the `"x"` axis index.

Note that in the example the total number of loop steps is not divisible by the axis size of `"x"`, and thus for some `"x"` axis indices the loop will do one iteration less.

| axis index | indices        |
|------------|----------------|
| 0          | (0, 0), (1, 1) |
| 1          | (0, 1), (1, 2) |
| 2          | (0, 2)         |
| 3          | (1, 0)         |

If `init_carry` is passed then `nd_loop()` will expect the body to take and return the carry. If it’s `None` then no carry argument is expected.

See also

- [`jax.experimental.pallas.loop()`](jax.experimental.pallas.loop.html#jax.experimental.pallas.loop "jax.experimental.pallas.loop"): A loop over a single dimension.

[](jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized

[](jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop.html "next page")

next

jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop

Contents

- [`nd_loop()`](#jax.experimental.pallas.mosaic_gpu.nd_loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\
