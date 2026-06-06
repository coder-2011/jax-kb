- [](../index.html)
- [API Reference](../jax.html)
- jax.checkpoint_policies.nothing_saveable

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.checkpoint_policies.nothing_saveable.rst "Download source file")
-  .pdf

# jax.checkpoint_policies.nothing_saveable

## Contents

- [`checkpoint_policies.nothing_saveable()`](#jax.checkpoint_policies.nothing_saveable)

# jax.checkpoint_policies.nothing_saveable[\#](#jax-checkpoint-policies-nothing-saveable "Link to this heading")

checkpoint_policies.nothing_saveable(*\*\*\_\_*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ad_checkpoint.py#L70-L75)[\#](#jax.checkpoint_policies.nothing_saveable "Link to this definition")  
Rematerialize everything, as if a custom policy were not being used at all.

This is the effective policy when using jax.remat without explicit policy.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

[](jax.checkpoint_policies.everything_saveable.html "previous page")

previous

jax.checkpoint_policies.everything_saveable

[](jax.checkpoint_policies.dots_saveable.html "next page")

next

jax.checkpoint_policies.dots_saveable

Contents

- [`checkpoint_policies.nothing_saveable()`](#jax.checkpoint_policies.nothing_saveable)

By The JAX authors

© Copyright 2024, The JAX Authors.\
