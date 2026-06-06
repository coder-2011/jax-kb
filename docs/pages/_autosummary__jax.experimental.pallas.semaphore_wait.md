- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.semaphore_wait

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.semaphore_wait.rst "Download source file")
-  .pdf

# jax.experimental.pallas.semaphore_wait

## Contents

- [`semaphore_wait()`](#jax.experimental.pallas.semaphore_wait)

# jax.experimental.pallas.semaphore_wait[\#](#jax-experimental-pallas-semaphore-wait "Link to this heading")

jax.experimental.pallas.semaphore_wait(*sem_or_view*, *value=1*, *\**, *decrement=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L1120-L1136)[\#](#jax.experimental.pallas.semaphore_wait "Link to this definition")  
Blocks execution of the current thread until a semaphore reaches a value.

Parameters:  
- **sem_or_view** – A Ref (or view) representing a semaphore.

- **value** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *jax_typing.Array*) – The target value that the semaphore should reach before unblocking.

- **decrement** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to decrement the value of the semaphore after a successful wait.

[](jax.experimental.pallas.semaphore_signal.html "previous page")

previous

jax.experimental.pallas.semaphore_signal

[](../jax.experimental.random.html "next page")

next

`jax.experimental.random` module

Contents

- [`semaphore_wait()`](#jax.experimental.pallas.semaphore_wait)

By The JAX authors

© Copyright 2024, The JAX Authors.\
