- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.axis_index

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.axis_index.rst "Download source file")
-  .pdf

# jax.lax.axis_index

## Contents

- [`axis_index()`](#jax.lax.axis_index)

# jax.lax.axis_index[\#](#jax-lax-axis-index "Link to this heading")

jax.lax.axis_index(*axis_name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L761-L800)[\#](#jax.lax.axis_index "Link to this definition")  
Return the index along the mapped axis `axis_name`.

Parameters:  
**axis_name** (*AxisName*) – hashable Python object used to name the mapped axis.

Returns:  
An integer representing the index.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

For example, with 8 XLA devices available:

    >>> mesh = jax.make_mesh((8,), 'i')
    >>> @jax.shard_map(mesh=mesh, in_specs=(), out_specs=jax.P('i'))
    ... def f():
    ...   return lax.axis_index('i')[None]
    ...
    >>> f()
    Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)

    >>> mesh = jax.make_mesh((4, 2), ('i', 'j'))
    >>> @jax.shard_map(mesh=mesh, in_specs=(), out_specs=jax.P('i', 'j'))
    ... def f():
    ...   return lax.axis_index(('i', 'j'))[None, None]
    ...
    >>> f()
    Array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]], dtype=int32)

[](jax.lax.pswapaxes.html "previous page")

previous

jax.lax.pswapaxes

[](jax.lax.axis_size.html "next page")

next

jax.lax.axis_size

Contents

- [`axis_index()`](#jax.lax.axis_index)

By The JAX authors

© Copyright 2024, The JAX Authors.\
