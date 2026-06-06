- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- Buffer donation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/buffer_donation.md "Download source file")
-  .pdf

# Buffer donation

# Buffer donation[\#](#buffer-donation "Link to this heading")

When JAX executes a computation it uses buffers on the device for all inputs and outputs. If you know that one of the inputs is not needed after the computation, and if it matches the shape and element type of one of the outputs, you can specify that you want the corresponding input buffer to be donated to hold an output. This will reduce the memory required for the execution by the size of the donated buffer.

If you have something like the following pattern, you can use buffer donation:

    params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params, state)

You can think of this as a way to do a memory-efficient functional update on your immutable JAX arrays. Within the boundaries of a computation XLA can make this optimization for you, but at the jit/pmap boundary you need to guarantee to XLA that you will not use the donated input buffer after calling the donating function.

You achieve this by using the `donate_argnums` parameter to the functions [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), `jax.pjit()`, and [`jax.pmap()`](_autosummary/jax.pmap.html#jax.pmap "jax.pmap"). This parameter is a sequence of indices (0 based) into the positional argument list:

    def add(x, y):
      return x + y

    x = jax.device_put(np.ones((2, 3)))
    y = jax.device_put(np.ones((2, 3)))
    # Execute `add` with donation of the buffer for `y`. The result has
    # the same shape and type as `y`, so it will share its buffer.
    z = jax.jit(add, donate_argnums=(1,))(x, y)

Note that this currently does not work when calling your function with key-word arguments! The following code will not donate any buffers:

    params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params=params, state=state)

If an argument whose buffer is donated is a pytree, then all the buffers for its components are donated:

    def add_ones(xs: List[Array]):
      return [x + 1 for x in xs]

    xs = [jax.device_put(np.ones((2, 3))), jax.device_put(np.ones((3, 4)))]
    # Execute `add_ones` with donation of all the buffers for `xs`.
    # The outputs have the same shape and type as the elements of `xs`,
    # so they will share those buffers.
    z = jax.jit(add_ones, donate_argnums=0)(xs)

It is not allowed to donate a buffer that is used subsequently in the computation, and JAX will give an error because the buffer for `y` has become invalid after it was donated:

    # Donate the buffer for `y`
    z = jax.jit(add, donate_argnums=(1,))(x, y)
    w = y + 1  # Reuses `y` whose buffer was donated above
    # >> RuntimeError: Invalid argument: CopyToHostAsync() called on invalid buffer

You will get a warning if the donated buffer is not used, e.g., because there are more donated buffers than can be used for the outputs:

    # Execute `add` with donation of the buffers for both `x` and `y`.
    # One of those buffers will be used for the result, but the other will
    # not be used.
    z = jax.jit(add, donate_argnums=(0, 1))(x, y)
    # >> UserWarning: Some donated buffers were not usable: f32[2,3]{1,0}

The donation may also be unused if there is no output whose shape matches the donation:

    y = jax.device_put(np.ones((1, 3)))  # `y` has different shape than the output
    # Execute `add` with donation of the buffer for `y`.
    z = jax.jit(add, donate_argnums=(1,))(x, y)
    # >> UserWarning: Some donated buffers were not usable: f32[1,3]{1,0}

[](persistent_compilation_cache.html "previous page")

previous

Persistent compilation cache

[](gpu_performance_tips.html "next page")

next

GPU performance tips

By The JAX authors

© Copyright 2024, The JAX Authors.\
