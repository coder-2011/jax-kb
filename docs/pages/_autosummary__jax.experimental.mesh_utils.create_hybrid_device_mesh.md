- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.mesh_utils` module](../jax.experimental.mesh_utils.html)
- jax.experimental.mesh_utils.create_hybrid_device_mesh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.mesh_utils.create_hybrid_device_mesh.rst "Download source file")
-  .pdf

# jax.experimental.mesh_utils.create_hybrid_device_mesh

## Contents

- [`create_hybrid_device_mesh()`](#jax.experimental.mesh_utils.create_hybrid_device_mesh)

# jax.experimental.mesh_utils.create_hybrid_device_mesh[\#](#jax-experimental-mesh-utils-create-hybrid-device-mesh "Link to this heading")

jax.experimental.mesh_utils.create_hybrid_device_mesh(*mesh_shape*, *dcn_mesh_shape*, *devices=None*, *\**, *process_is_granule=False*, *should_sort_granules_by_key=True*, *allow_split_physical_axes=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/mesh_utils.py#L861-L935)[\#](#jax.experimental.mesh_utils.create_hybrid_device_mesh "Link to this definition")  
Creates a device mesh for hybrid (e.g., ICI and DCN) parallelism.

Parameters:  
- **mesh_shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – shape of the logical mesh for the faster/inner network, ordered by increasing network intensity, e.g. \[replica, data, mdl\] where mdl has the most network communication requirements.

- **dcn_mesh_shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – shape of the logical mesh for the slower/outer network, in the same order as mesh_shape.

- **devices** (*Sequence\[Any\]* *\|* *None*) – optionally, the devices to construct a mesh for. Defaults to jax.devices().

- **process_is_granule** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, this function will treat processes as the units of the slower/outer network. Otherwise it will look for slice_index attributes on devices and use slices as the units. Enabling this is meant as a fallback for platforms that don’t set slice_index.

- **should_sort_granules_by_key** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether device granules should be sorted by the granule key, either slice or process index, depending on process_is_granule.

- **allow_split_physical_axes** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, we will split physical axes if necessary to produce the desired device mesh.

Raises:  
[**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – if the number of slices to which the devices belong doesn’t equal the product of dcn_mesh_shape, or if the number of devices belonging to any single slice does not equal the product of mesh_shape.

Returns:  
A np.ndarray of JAX devices with mesh_shape \* dcn_mesh_shape as its shape that can be fed into jax.sharding.Mesh for hybrid parallelism.

Return type:  
np.ndarray

[](jax.experimental.mesh_utils.create_device_mesh.html "previous page")

previous

jax.experimental.mesh_utils.create_device_mesh

[](../jax.experimental.multihost_utils.html "next page")

next

`jax.experimental.multihost_utils` module

Contents

- [`create_hybrid_device_mesh()`](#jax.experimental.mesh_utils.create_hybrid_device_mesh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
