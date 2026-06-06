- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.multihost_utils` module](../jax.experimental.multihost_utils.html)
- jax.experimental.multihost_utils.host_local_array_to_global_array

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.multihost_utils.host_local_array_to_global_array.rst "Download source file")
-  .pdf

# jax.experimental.multihost_utils.host_local_array_to_global_array

## Contents

- [`host_local_array_to_global_array()`](#jax.experimental.multihost_utils.host_local_array_to_global_array)

# jax.experimental.multihost_utils.host_local_array_to_global_array[\#](#jax-experimental-multihost-utils-host-local-array-to-global-array "Link to this heading")

jax.experimental.multihost_utils.host_local_array_to_global_array(*local_inputs*, *global_mesh*, *pspecs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/multihost_utils.py#L296-L384)[\#](#jax.experimental.multihost_utils.host_local_array_to_global_array "Link to this definition")  
Converts a host local value to a globally sharded jax.Array.

This function takes host-local data (which might be different across hosts), and populates a global array with this data, where each device on each host, get the appropriate slice of the data according to sharding defined by the global_mesh/pspects.

For example:

    >>> global_mesh = jax.sharding.Mesh(jax.devices(), 'x')
    >>> pspecs = jax.sharding.PartitionSpec('x')
    >>> host_id = jax.process_index()
    >>> arr = host_local_array_to_global_array(np.arange(4) * host_id, mesh, pspecs)  # NB: assumes jax.local_device_count() divides 4.   

The resulting array will have the shape (4 \* num_processes) and will have distributed value of: (0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9, … ), where each slice np.arange(4) \* host_id will be partitioned across the corresponding host’s devices.

Similarly:

    >>> mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()), ['host', 'dev'])
    >>> pspecs = jax.sharding.PartitionSpec('host')
    >>> host_id = jax.process_index()
    >>> arr = host_local_array_to_global_array(np.arange(4) * host_id, mesh, pspecs)  

will create the same distributed value (0, 1, 2, 3, 0, 2, 4, 6, …), however each slice np.arange(4) \* i will be *replicated* across corresponding host devices.

On the other hand, if pspecs = PartitionSpec(), which means replication across all axes, then this snippet:

    >>> pspecs = jax.sharding.PartitionSpec()
    >>> arr = host_local_array_to_global_array(np.arange(4), mesh, pspecs)  

will have the shape (4,) and the value (0, 1, 2, 3) will be replicated across all hosts and devices.

It is an undefined behavior to have not identical local_inputs with pspec indicating data replication.

You can use this function to transition to jax.Array. Using jax.Array with pjit has the same semantics of using GDA with pjit i.e. all jax.Array inputs to pjit should be globally shaped.

If you are currently passing host local values to pjit, you can use this function to convert your host local values to global Arrays and then pass that to pjit.

Example usage.

    >>> from jax.experimental import multihost_utils 
    >>>
    >>> global_inputs = multihost_utils.host_local_array_to_global_array(host_local_inputs, global_mesh, in_pspecs) 
    >>>
    >>> with mesh: 
    >>>   global_out = pjitted_fun(global_inputs) 
    >>>
    >>> host_local_output = multihost_utils.global_array_to_host_local_array(global_out, mesh, out_pspecs) 

Please note this function requires global mesh to be a continuous mesh, meaning that devices that belong to each host should form a subcube in this mesh. To move local data to global array with non-continuous mesh use jax.make_array_from_callback or jax.make_array_from_single_device_arrays instead.

Parameters:  
- **local_inputs** (*Any*) – A Pytree of host local values.

- **global_mesh** ([*jax.sharding.Mesh*](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh")) – A jax.sharding.Mesh object. The mesh must be a contiguous mesh,

- **mesh.** (*that is all hosts' devices must form a subcube in this*)

- **pspecs** (*Any*) – A Pytree of jax.sharding.PartitionSpec’s.

Returns:  
A pytree of global arrays.

[](jax.experimental.multihost_utils.assert_equal.html "previous page")

previous

jax.experimental.multihost_utils.assert_equal

[](jax.experimental.multihost_utils.global_array_to_host_local_array.html "next page")

next

jax.experimental.multihost_utils.global_array_to_host_local_array

Contents

- [`host_local_array_to_global_array()`](#jax.experimental.multihost_utils.host_local_array_to_global_array)

By The JAX authors

© Copyright 2024, The JAX Authors.\
