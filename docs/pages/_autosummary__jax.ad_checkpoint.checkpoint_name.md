- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ad_checkpoint` module](../jax.ad_checkpoint.html)
- jax.ad_checkpoint.checkpoint_name

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ad_checkpoint.checkpoint_name.rst "Download source file")
-  .pdf

# jax.ad_checkpoint.checkpoint_name

## Contents

- [`checkpoint_name()`](#jax.ad_checkpoint.checkpoint_name)

# jax.ad_checkpoint.checkpoint_name[\#](#jax-ad-checkpoint-checkpoint-name "Link to this heading")

jax.ad_checkpoint.checkpoint_name(*x*, *name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ad_checkpoint.py#L898-L940)[\#](#jax.ad_checkpoint.checkpoint_name "Link to this definition")  
Identifies a value with a name within [`jax.checkpoint()`](jax.checkpoint.html#jax.checkpoint "jax.checkpoint").

This function acts as an identity function at runtime (returning `x` unchanged) but attaches a string name to the value in the JAX trace. These names can be targeted by specific checkpointing policies (see [Checkpoint policies](../jax.html#checkpoint-policies)) to control which intermediate values are saved during the forward pass and which are recomputed during the backward pass.

Parameters:  
- **x** – array or PyTree of arrays to be named.

- **name** – A string name to associate with the value `x`.

Returns:  
The input `x`, unchanged.

See also

- [`jax.checkpoint()`](jax.checkpoint.html#jax.checkpoint "jax.checkpoint") (alias: `jax.remat()`): decorator to enable checkpointing.

- `jax.checkpoint_policies`: a namespace containing policies that use names marked via `checkpoint_name` to determine behavior.

Example

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jax.ad_checkpoint import checkpoint_name

    >>> # Define a function where we explicitly name an intermediate value
    >>> def f(x):
    ...   y = jnp.sin(x)
    ...   z = checkpoint_name(y, "my_intermediate")
    ...   return jnp.cos(z)

    >>> # Use a policy that saves only the named value
    >>> policy = jax.checkpoint_policies.save_only_these_names("my_intermediate")
    >>> f_checkpointed = jax.checkpoint(f, policy=policy)

For further examples, see the [remat example notebook](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html).

[](../jax.ad_checkpoint.html "previous page")

previous

`jax.ad_checkpoint` module

[](../jax.debug.html "next page")

next

`jax.debug` module

Contents

- [`checkpoint_name()`](#jax.ad_checkpoint.checkpoint_name)

By The JAX authors

© Copyright 2024, The JAX Authors.\
