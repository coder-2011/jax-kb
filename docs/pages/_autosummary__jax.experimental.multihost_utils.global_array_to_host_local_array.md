- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.multihost_utils` module](../jax.experimental.multihost_utils.html)
- jax.experimental.multihost_utils.global_array_to_host_local_array

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.multihost_utils.global_array_to_host_local_array.rst "Download source file")
-  .pdf

# jax.experimental.multihost_utils.global_array_to_host_local_array

## Contents

- [`global_array_to_host_local_array()`](#jax.experimental.multihost_utils.global_array_to_host_local_array)

# jax.experimental.multihost_utils.global_array_to_host_local_array[\#](#jax-experimental-multihost-utils-global-array-to-host-local-array "Link to this heading")

jax.experimental.multihost_utils.global_array_to_host_local_array(*global_inputs*, *global_mesh*, *pspecs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/multihost_utils.py#L460-L502)[\#](#jax.experimental.multihost_utils.global_array_to_host_local_array "Link to this definition")  
Converts a global jax.Array to a host local jax.Array.

You can use this function to transition to jax.Array. Using jax.Array with pjit has the same semantics of using GDA with pjit i.e. all jax.Array inputs to pjit should be globally shaped and the output from pjit will also be globally shaped jax.Array’s

You can use this function to convert the globally shaped jax.Array output from pjit to host local values again so that the transition to jax.Array can be a mechanical change.

Example usage:

    >>> from jax.experimental import multihost_utils 
    >>>
    >>> global_inputs = multihost_utils.host_local_array_to_global_array(host_local_inputs, global_mesh, in_pspecs) 
    >>>
    >>> with mesh: 
    ...   global_out = pjitted_fun(global_inputs) 
    >>>
    >>> host_local_output = multihost_utils.global_array_to_host_local_array(global_out, mesh, out_pspecs) 

Parameters:  
- **global_inputs** (*Any*) – A Pytree of global jax.Array’s.

- **global_mesh** ([*jax.sharding.Mesh*](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh")) – A [`jax.sharding.Mesh`](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") object. The mesh must be contiguous meaning all local devices of the host must form a subcube.

- **pspecs** (*Any*) – A Pytree of [`jax.sharding.PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") objects.

Returns:  
A Pytree of host local arrays.

[](jax.experimental.multihost_utils.host_local_array_to_global_array.html "previous page")

previous

jax.experimental.multihost_utils.host_local_array_to_global_array

[](../jax.experimental.pallas.html "next page")

next

`jax.experimental.pallas` module

Contents

- [`global_array_to_host_local_array()`](#jax.experimental.multihost_utils.global_array_to_host_local_array)

By The JAX authors

© Copyright 2024, The JAX Authors.\
