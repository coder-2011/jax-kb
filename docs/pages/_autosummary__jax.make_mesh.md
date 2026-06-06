- [](../index.html)
- [API Reference](../jax.html)
- jax.make_mesh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.make_mesh.rst "Download source file")
-  .pdf

# jax.make_mesh

## Contents

- [`make_mesh()`](#jax.make_mesh)

# jax.make_mesh[\#](#jax-make-mesh "Link to this heading")

jax.make_mesh(*axis_shapes*, *axis_names*, *axis_types=None*, *\**, *devices=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L899-L976)[\#](#jax.make_mesh "Link to this definition")  
Creates an efficient mesh with the shape and axis names specified.

This function attempts to automatically compute a good mapping from a set of logical axes to a physical mesh. For example, on a TPU v3 with 8 devices:

    >>> mesh = jax.make_mesh((8,), ('x'))  
    >>> [d.id for d in mesh.devices.flat]  
    [0, 1, 2, 3, 6, 7, 4, 5]

The above ordering takes into account the physical topology of TPU v3. It orders the devices into a ring, which yields efficient all-reduces on a TPU v3.

Now, let’s see another example with 16 devices of TPU v3:

    >>> mesh = jax.make_mesh((2, 8), ('x', 'y'))  
    >>> [d.id for d in mesh.devices.flat]  
    [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 14, 15, 12, 13]
    >>> mesh = jax.make_mesh((4, 4), ('x', 'y'))  
    >>> [d.id for d in mesh.devices.flat]  
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

As you can see, logical axes (axis_shapes) affect the ordering of the devices.

You can use jax.experimental.mesh_utils.create_device_mesh if you want to use the extra arguments it provides like contiguous_submeshes and allow_split_physical_axes.

Parameters:  
- **axis_shapes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – Shape of the mesh. For example, axis_shape=(4, 2)

- **axis_names** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) – Names of the mesh axes. For example, axis_names=(‘x’, ‘y’)

- **axis_types** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[AxisType,* *...\]* *\|* *None*) – Optional tuple of `jax.sharding.AxisType` entries corresponding to the `axis_names`. See [Explicit Sharding](https://docs.jax.dev/en/latest/parallel.html) for more information.

- **devices** (*Sequence\[xc.Device\]* *\|* *None*) – Optional keyword only argument, that allows you to specify the devices you want to create a mesh with.

Returns:  
A [`jax.sharding.Mesh`](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") object.

Return type:  
[Mesh](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh")

[](jax.copy_to_host_async.html "previous page")

previous

jax.copy_to_host_async

[](jax.set_mesh.html "next page")

next

jax.set_mesh

Contents

- [`make_mesh()`](#jax.make_mesh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
