- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.axis_size

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.axis_size.rst "Download source file")
-  .pdf

# jax.lax.axis_size

## Contents

- [`axis_size()`](#jax.lax.axis_size)

# jax.lax.axis_size[\#](#jax-lax-axis-size "Link to this heading")

jax.lax.axis_size(*axis_name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L802-L830)[\#](#jax.lax.axis_size "Link to this definition")  
Return the size of the mapped axis `axis_name`.

Parameters:  
**axis_name** (*AxisName*) – hashable Python object used to name the mapped axis.

Returns:  
An integer representing the size.

Return type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

For example, with 8 XLA devices available:

    >>> mesh = jax.make_mesh((8,), 'i')
    >>> @jax.shard_map(mesh=mesh, in_specs=jax.P('i'), out_specs=jax.P())
    ... def f(_):
    ...   return lax.axis_size('i')
    ...
    >>> f(jax.device_put(jnp.zeros(16), jax.NamedSharding(mesh, P('i'))))
    Array(8, dtype=int32, weak_type=True)

    >>> mesh = jax.make_mesh((4, 2), ('i', 'j'))
    >>> @jax.shard_map(mesh=mesh, in_specs=jax.P('i', 'j'), out_specs=jax.P())
    ... def f(_):
    ...   return lax.axis_size(('i', 'j'))
    ...
    >>> f(jax.device_put(jnp.zeros((16, 8)), jax.NamedSharding(mesh, P('i', 'j'))))
    Array(8, dtype=int32, weak_type=True)

[](jax.lax.axis_index.html "previous page")

previous

jax.lax.axis_index

[](jax.lax.psend.html "next page")

next

jax.lax.psend

Contents

- [`axis_size()`](#jax.lax.axis_size)

By The JAX authors

© Copyright 2024, The JAX Authors.\
