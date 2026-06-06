- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.multihost_utils` module](../jax.experimental.multihost_utils.html)
- jax.experimental.multihost_utils.process_allgather

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.multihost_utils.process_allgather.rst "Download source file")
-  .pdf

# jax.experimental.multihost_utils.process_allgather

## Contents

- [`process_allgather()`](#jax.experimental.multihost_utils.process_allgather)

# jax.experimental.multihost_utils.process_allgather[\#](#jax-experimental-multihost-utils-process-allgather "Link to this heading")

jax.experimental.multihost_utils.process_allgather(*in_tree*, *tiled=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/multihost_utils.py#L138-L160)[\#](#jax.experimental.multihost_utils.process_allgather "Link to this definition")  
Gather data from across processes.

Parameters:  
- **in_tree** (*Any*) – pytree of arrays - each array \_must\_ have the same shape across the hosts.

- **tiled** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to stack or concat the output. Defaults to False i.e. stack into a new positional axis at index 0.

Returns:  
Pytrees of numpy arrays.  
- If the input is a non-fully addressable jax.Array, then the data is fully replicated.

- If the input is numpy array or fully addressable jax.Array, then the output shape is dependent on the tiled argument. If its False, then the output will be stacked else concatenated.

- If the input is a scalar, then the output will be stacked.

Return type:  
Any

[](jax.experimental.multihost_utils.sync_global_devices.html "previous page")

previous

jax.experimental.multihost_utils.sync_global_devices

[](jax.experimental.multihost_utils.assert_equal.html "next page")

next

jax.experimental.multihost_utils.assert_equal

Contents

- [`process_allgather()`](#jax.experimental.multihost_utils.process_allgather)

By The JAX authors

© Copyright 2024, The JAX Authors.\
