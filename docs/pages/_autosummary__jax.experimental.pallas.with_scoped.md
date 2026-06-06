- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.with_scoped

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.with_scoped.rst "Download source file")
-  .pdf

# jax.experimental.pallas.with_scoped

## Contents

- [`with_scoped()`](#jax.experimental.pallas.with_scoped)

# jax.experimental.pallas.with_scoped[\#](#jax-experimental-pallas-with-scoped "Link to this heading")

jax.experimental.pallas.with_scoped(*\*types*, *collective_axes=()*, *\*\*kw_types*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/helpers.py#L280-L320)[\#](#jax.experimental.pallas.with_scoped "Link to this definition")  
Returns a function decorator that runs a function with provided allocations.

Example:

    @pl.with_scoped(pltpu.VMEM((8, 128), jnp.float32),
                    sem_ref=pltpu.SemaphoreType.DMA)
    def f(vmem_ref, sem_ref):
      ...

    f()

The arguments to f will be forwarded to the decorated function as the initial arguments.

Example:

    @pl.with_scoped(pltpu.VMEM((8, 128), jnp.float32),
                    sem_ref=pltpu.SemaphoreType.DMA)
    def f(outer_ref, vmem_ref, sem_ref):
      ...

    outer_ref = ...
    f(outer_ref)

Parameters:  
- **types** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)"))

- **collective_axes** ([*Hashable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Hashable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable "(in Python v3.14)")*,* *...\]*)

- **kw_types** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)"))

[](jax.experimental.pallas.when.html "previous page")

previous

jax.experimental.pallas.when

[](jax.experimental.pallas.semaphore_read.html "next page")

next

jax.experimental.pallas.semaphore_read

Contents

- [`with_scoped()`](#jax.experimental.pallas.with_scoped)

By The JAX authors

© Copyright 2024, The JAX Authors.\
