- [](../index.html)
- [API Reference](../jax.html)
- jax.smap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.smap.rst "Download source file")
-  .pdf

# jax.smap

## Contents

- [`smap()`](#jax.smap)

# jax.smap[\#](#jax-smap "Link to this heading")

jax.smap(*f: F*, */*, *\**, *in_axes: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") \| InferFromArgs \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, ...\] = jax.sharding.Infer*, *out_axes: Any*, *axis_name: AxisName*) → F[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/shard_map.py#L179-L212)[\#](#jax.smap "Link to this definition")\
jax.smap(*f: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, */*, *\**, *in_axes: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") \| InferFromArgs \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, ...\] = jax.sharding.Infer*, *out_axes: Any*, *axis_name: AxisName*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[G\], G\]  
Single axis shard_map that maps a function f one axis at a time.

Parameters:  
- **f** – Callable to be mapped. Each application of `f`, or “instance” of `f`, takes as input a shard of the mapped-over arguments and produces a shard of the output.

- **in_axes** – (optional) An integer, None, or sequence of values specifying which input array axes to map over. If not specified, smap will try to infer the axes from the arguments only under Explicit mode. An integer or `None` indicates which array axis to map over for all arguments (with `None` indicating not to map any axis), and a tuple indicates which axis to map for each corresponding positional argument. Axis integers must be in the range `[-ndim,`` ``ndim)` for each array, where `ndim` is the number of dimensions (axes) of the corresponding input array.

- **out_axes** – An integer, None, or (nested) standard Python container (tuple/list/dict) thereof indicating where the mapped axis should appear in the output.

- **axis_name** – `mesh` axis name over which the function `f` is manual.

Returns:  
A callable representing a mapped version of `f`, which accepts positional arguments corresponding to those of `f` and produces output corresponding to that of `f`.

[](jax.shard_map.html "previous page")

previous

jax.shard_map

[](jax.pmap.html "next page")

next

jax.pmap

Contents

- [`smap()`](#jax.smap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
