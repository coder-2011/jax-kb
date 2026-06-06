- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.multihost_utils` module](../jax.experimental.multihost_utils.html)
- jax.experimental.multihost_utils.broadcast_one_to_all

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.multihost_utils.broadcast_one_to_all.rst "Download source file")
-  .pdf

# jax.experimental.multihost_utils.broadcast_one_to_all

## Contents

- [`broadcast_one_to_all()`](#jax.experimental.multihost_utils.broadcast_one_to_all)

# jax.experimental.multihost_utils.broadcast_one_to_all[\#](#jax-experimental-multihost-utils-broadcast-one-to-all "Link to this heading")

jax.experimental.multihost_utils.broadcast_one_to_all(*in_tree*, *is_source=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/multihost_utils.py#L48-L89)[\#](#jax.experimental.multihost_utils.broadcast_one_to_all "Link to this definition")  
Broadcast data from a source host (host 0 by default) to all other hosts.

Parameters:  
- **in_tree** (*Any*) – pytree of arrays - each array *must* have the same shape across the hosts.

- **is_source** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – optional bool denoting whether the caller is the source. Only ‘source host’ will contribute the data for the broadcast. If None, then host 0 is used.

Returns:  
A pytree matching in_tree where the leaves now all contain the data from the first host.

Return type:  
Any

[](../jax.experimental.multihost_utils.html "previous page")

previous

`jax.experimental.multihost_utils` module

[](jax.experimental.multihost_utils.sync_global_devices.html "next page")

next

jax.experimental.multihost_utils.sync_global_devices

Contents

- [`broadcast_one_to_all()`](#jax.experimental.multihost_utils.broadcast_one_to_all)

By The JAX authors

© Copyright 2024, The JAX Authors.\
