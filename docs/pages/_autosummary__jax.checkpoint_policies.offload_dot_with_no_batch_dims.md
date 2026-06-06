- [](../index.html)
- [API Reference](../jax.html)
- jax.checkpoint_policies.offload_dot_with_no_batch_dims

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.checkpoint_policies.offload_dot_with_no_batch_dims.rst "Download source file")
-  .pdf

# jax.checkpoint_policies.offload_dot_with_no_batch_dims

## Contents

- [`checkpoint_policies.offload_dot_with_no_batch_dims()`](#jax.checkpoint_policies.offload_dot_with_no_batch_dims)

# jax.checkpoint_policies.offload_dot_with_no_batch_dims[\#](#jax-checkpoint-policies-offload-dot-with-no-batch-dims "Link to this heading")

checkpoint_policies.offload_dot_with_no_batch_dims(*offload_dst*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ad_checkpoint.py#L101-L113)[\#](#jax.checkpoint_policies.offload_dot_with_no_batch_dims "Link to this definition")  
Same as `dots_with_no_batch_dims_saveable`, but offload to CPU memory instead of recomputing.

This is a useful heuristic for transformers.

[](jax.checkpoint_policies.save_only_these_names.html "previous page")

previous

jax.checkpoint_policies.save_only_these_names

[](jax.checkpoint_policies.save_and_offload_only_these_names.html "next page")

next

jax.checkpoint_policies.save_and_offload_only_these_names

Contents

- [`checkpoint_policies.offload_dot_with_no_batch_dims()`](#jax.checkpoint_policies.offload_dot_with_no_batch_dims)

By The JAX authors

© Copyright 2024, The JAX Authors.\
