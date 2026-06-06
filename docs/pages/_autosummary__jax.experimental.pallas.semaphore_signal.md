- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.semaphore_signal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.semaphore_signal.rst "Download source file")
-  .pdf

# jax.experimental.pallas.semaphore_signal

## Contents

- [`semaphore_signal()`](#jax.experimental.pallas.semaphore_signal)

# jax.experimental.pallas.semaphore_signal[\#](#jax-experimental-pallas-semaphore-signal "Link to this heading")

jax.experimental.pallas.semaphore_signal(*sem_or_view*, *inc=1*, *\**, *device_id=None*, *device_id_type=DeviceIdType.MESH*, *core_index=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L967-L1002)[\#](#jax.experimental.pallas.semaphore_signal "Link to this definition")  
Increments the value of a semaphore.

This operation can also be performed remotely if `device_id` is specified, in which `sem_or_view` refers to a Ref located on another device. Note that it is assumed that `sem_or_view` is already allocated (e.g. through the proper use of barriers), or else this operation could result in undefined behavior.

Parameters:  
- **sem_or_view** – A Ref (or view) representing a semaphore.

- **inc** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *jax_typing.Array*) – The value to increment by.

- **device_id** (*optional*) – Specifies which device to signal. If not specified, `sem_or_view` is assumed to be local.

- **device_id_type** (*optional*) – The format in which `device_id` should be specified.

- **core_index** (*optional*) – If on a multi-core device, specifies which core to signal.

[](jax.experimental.pallas.semaphore_read.html "previous page")

previous

jax.experimental.pallas.semaphore_read

[](jax.experimental.pallas.semaphore_wait.html "next page")

next

jax.experimental.pallas.semaphore_wait

Contents

- [`semaphore_signal()`](#jax.experimental.pallas.semaphore_signal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
