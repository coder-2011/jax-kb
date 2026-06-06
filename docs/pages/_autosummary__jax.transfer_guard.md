- [](../index.html)
- [API Reference](../jax.html)
- jax.transfer_guard

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.transfer_guard.rst "Download source file")
-  .pdf

# jax.transfer_guard

## Contents

- [`transfer_guard()`](#jax.transfer_guard)

# jax.transfer_guard[\#](#jax-transfer-guard "Link to this heading")

jax.transfer_guard(*new_val*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/config.py#L1989-L2008)[\#](#jax.transfer_guard "Link to this definition")  
A contextmanager to control the transfer guard level for all transfers.

For more information, see [https://docs.jax.dev/en/latest/transfer_guard.html](https://docs.jax.dev/en/latest/transfer_guard.html)

Parameters:  
**new_val** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The new thread-local transfer guard level for all transfers.

Yields:  
None.

Return type:  
Generator\[None, None, None\]

[](jax.numpy_rank_promotion.html "previous page")

previous

jax.numpy_rank_promotion

[](jax.jit.html "next page")

next

jax.jit

Contents

- [`transfer_guard()`](#jax.transfer_guard)

By The JAX authors

© Copyright 2024, The JAX Authors.\
