- [](../index.html)
- [API Reference](../jax.html)
- jax.checkpoint_policies.save_from_both_policies

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.checkpoint_policies.save_from_both_policies.rst "Download source file")
-  .pdf

# jax.checkpoint_policies.save_from_both_policies

## Contents

- [`checkpoint_policies.save_from_both_policies()`](#jax.checkpoint_policies.save_from_both_policies)

# jax.checkpoint_policies.save_from_both_policies[\#](#jax-checkpoint-policies-save-from-both-policies "Link to this heading")

checkpoint_policies.save_from_both_policies(*policy_2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ad_checkpoint.py#L169-L183)[\#](#jax.checkpoint_policies.save_from_both_policies "Link to this definition")  
Logical OR of the given policies.

A residual is saveable iff it is saveable according to either policy.

[](jax.checkpoint_policies.save_and_offload_only_these_names.html "previous page")

previous

jax.checkpoint_policies.save_and_offload_only_these_names

[](../contributor_guide.html "next page")

next

Developer notes

Contents

- [`checkpoint_policies.save_from_both_policies()`](#jax.checkpoint_policies.save_from_both_policies)

By The JAX authors

© Copyright 2024, The JAX Authors.\
