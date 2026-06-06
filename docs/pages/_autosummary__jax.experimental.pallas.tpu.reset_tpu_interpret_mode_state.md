- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state

## Contents

- [`reset_tpu_interpret_mode_state()`](#jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state)

# jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state[\#](#jax-experimental-pallas-tpu-reset-tpu-interpret-mode-state "Link to this heading")

jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py#L148-L165)[\#](#jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state "Link to this definition")  
Resets all global, shared state used by TPU interpret mode.

TPU interpret mode uses global, shared state for simulating memory buffers and semaphores, for race detection, etc., when interpreting a kernel. Normally, this shared state is cleaned up after a kernel is interpreted.

But if an exception is thrown while interpreting a kernel, the shared state is not cleaned up, allowing the simulated TPU state to be examined for debugging purposes. In this case, the shared state must be reset before any further kernels are interpreted.

[](jax.experimental.pallas.tpu.InterpretParams.html "previous page")

previous

jax.experimental.pallas.tpu.InterpretParams

[](jax.experimental.pallas.tpu.set_tpu_interpret_mode.html "next page")

next

jax.experimental.pallas.tpu.set_tpu_interpret_mode

Contents

- [`reset_tpu_interpret_mode_state()`](#jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state)

By The JAX authors

© Copyright 2024, The JAX Authors.\
