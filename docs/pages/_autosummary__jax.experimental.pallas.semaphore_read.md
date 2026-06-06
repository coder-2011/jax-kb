- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.semaphore_read

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.semaphore_read.rst "Download source file")
-  .pdf

# jax.experimental.pallas.semaphore_read

## Contents

- [`semaphore_read()`](#jax.experimental.pallas.semaphore_read)

# jax.experimental.pallas.semaphore_read[\#](#jax-experimental-pallas-semaphore-read "Link to this heading")

jax.experimental.pallas.semaphore_read(*sem_or_view*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L911-L924)[\#](#jax.experimental.pallas.semaphore_read "Link to this definition")  
Reads the value of a semaphore.

Parameters:  
**sem_or_view** – A Ref (or view) representing a semaphore.

Returns:  
A scalar Array containing the value of the semaphore.

Return type:  
jax_typing.Array

[](jax.experimental.pallas.with_scoped.html "previous page")

previous

jax.experimental.pallas.with_scoped

[](jax.experimental.pallas.semaphore_signal.html "next page")

next

jax.experimental.pallas.semaphore_signal

Contents

- [`semaphore_read()`](#jax.experimental.pallas.semaphore_read)

By The JAX authors

© Copyright 2024, The JAX Authors.\
