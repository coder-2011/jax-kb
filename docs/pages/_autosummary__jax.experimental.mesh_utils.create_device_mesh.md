- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.mesh_utils` module](../jax.experimental.mesh_utils.html)
- jax.experimental.mesh_utils.create_device_mesh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.mesh_utils.create_device_mesh.rst "Download source file")
-  .pdf

# jax.experimental.mesh_utils.create_device_mesh

## Contents

- [`create_device_mesh()`](#jax.experimental.mesh_utils.create_device_mesh)

# jax.experimental.mesh_utils.create_device_mesh[\#](#jax-experimental-mesh-utils-create-device-mesh "Link to this heading")

jax.experimental.mesh_utils.create_device_mesh(*mesh_shape*, *devices=None*, *\**, *contiguous_submeshes=False*, *allow_split_physical_axes=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/mesh_utils.py#L782-L859)[\#](#jax.experimental.mesh_utils.create_device_mesh "Link to this definition")  
Creates a performant device mesh for jax.sharding.Mesh.

Parameters:  
- **mesh_shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – shape of logical mesh, ordered by increasing network-intensity e.g. \[replica, data, mdl\] where mdl has the most network communication requirements.

- **devices** (*Sequence\[Any\]* *\|* *None*) – optionally, the devices to construct a mesh for. Defaults to jax.devices().

- **contiguous_submeshes** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, this function will attempt to create a mesh where each process’s local devices form a contiguous submesh. A ValueError will be raised if this function can’t produce a suitable mesh. This setting was sometimes necessary before the introduction of jax.Array to ensure non-ragged local arrays; if using jax.Arrays, it’s better to keep this set to False.

- **allow_split_physical_axes** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, we will split physical axes if necessary to produce the desired device mesh.

Raises:  
[**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – if the number of devices doesn’t equal the product of mesh_shape.

Returns:  
A np.ndarray of JAX devices with mesh_shape as its shape that can be fed into jax.sharding.Mesh with good collective performance.

Return type:  
np.ndarray

[](../jax.experimental.mesh_utils.html "previous page")

previous

`jax.experimental.mesh_utils` module

[](jax.experimental.mesh_utils.create_hybrid_device_mesh.html "next page")

next

jax.experimental.mesh_utils.create_hybrid_device_mesh

Contents

- [`create_device_mesh()`](#jax.experimental.mesh_utils.create_device_mesh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
