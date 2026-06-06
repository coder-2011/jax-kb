- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.multihost_utils` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.multihost_utils.rst "Download source file")
-  .pdf

# jax.experimental.multihost_utils module

## Contents

- [Multihost Utils API Reference](#multihost-utils-api-reference)

# `jax.experimental.multihost_utils` module[\#](#module-jax.experimental.multihost_utils "Link to this heading")

Utilities for synchronizing and communication across multiple hosts.

## Multihost Utils API Reference[\#](#multihost-utils-api-reference "Link to this heading")

|  |  |
|----|----|
| [`broadcast_one_to_all`](_autosummary/jax.experimental.multihost_utils.broadcast_one_to_all.html#jax.experimental.multihost_utils.broadcast_one_to_all "jax.experimental.multihost_utils.broadcast_one_to_all")(in_tree\[, is_source\]) | Broadcast data from a source host (host 0 by default) to all other hosts. |
| [`sync_global_devices`](_autosummary/jax.experimental.multihost_utils.sync_global_devices.html#jax.experimental.multihost_utils.sync_global_devices "jax.experimental.multihost_utils.sync_global_devices")(name) | Creates a barrier across all hosts/devices. |
| [`process_allgather`](_autosummary/jax.experimental.multihost_utils.process_allgather.html#jax.experimental.multihost_utils.process_allgather "jax.experimental.multihost_utils.process_allgather")(in_tree\[, tiled\]) | Gather data from across processes. |
| [`assert_equal`](_autosummary/jax.experimental.multihost_utils.assert_equal.html#jax.experimental.multihost_utils.assert_equal "jax.experimental.multihost_utils.assert_equal")(in_tree\[, fail_message\]) | Verifies that all the hosts have the same tree of values. |
| [`host_local_array_to_global_array`](_autosummary/jax.experimental.multihost_utils.host_local_array_to_global_array.html#jax.experimental.multihost_utils.host_local_array_to_global_array "jax.experimental.multihost_utils.host_local_array_to_global_array")(...) | Converts a host local value to a globally sharded jax.Array. |
| [`global_array_to_host_local_array`](_autosummary/jax.experimental.multihost_utils.global_array_to_host_local_array.html#jax.experimental.multihost_utils.global_array_to_host_local_array "jax.experimental.multihost_utils.global_array_to_host_local_array")(...) | Converts a global jax.Array to a host local jax.Array. |

[](_autosummary/jax.experimental.mesh_utils.create_hybrid_device_mesh.html "previous page")

previous

jax.experimental.mesh_utils.create_hybrid_device_mesh

[](_autosummary/jax.experimental.multihost_utils.broadcast_one_to_all.html "next page")

next

jax.experimental.multihost_utils.broadcast_one_to_all

Contents

- [Multihost Utils API Reference](#multihost-utils-api-reference)

By The JAX authors

© Copyright 2024, The JAX Authors.\
