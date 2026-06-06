- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.random` module](../jax.experimental.random.html)
- jax.experimental.random.stateful_rng

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.random.stateful_rng.rst "Download source file")
-  .pdf

# jax.experimental.random.stateful_rng

## Contents

- [`stateful_rng()`](#jax.experimental.random.stateful_rng)

# jax.experimental.random.stateful_rng[\#](#jax-experimental-random-stateful-rng "Link to this heading")

jax.experimental.random.stateful_rng(*seed=None*, *\**, *impl=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/stateful_rng.py#L217-L303)[\#](#jax.experimental.random.stateful_rng "Link to this definition")  
Experimental stateful RNG with implicitly-updated state.

This implements a stateful PRNG API similar to [`numpy.random.default_rng()`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng "(in NumPy v2.4)"). It is compatible with JAX transformations like [`jit()`](jax.jit.html#jax.jit "jax.jit") and others, with a few exceptions mentioned in the Notes below.

Note

This stateful PRNG API is a convenience wrapper around JAX’s classic stateless, explicitly updated PRNG, described in [`jax.random`](../jax.random.html#module-jax.random "jax.random"). For performance-critical applications, it is recommended to use [`jax.random.key()`](jax.random.key.html#jax.random.key "jax.random.key") with explicit random state semantics.

For a discussion of design considerations for this API, refer to [JEP 28845: Stateful Randomness in JAX](../jep/28845-stateful-rng.html#stateful-randomness-jep).

Parameters:  
- **seed** ([*ArrayLike*](jax.typing.ArrayLike.html#jax.typing.ArrayLike "jax.typing.ArrayLike") *\|* *None*) – an optional 64- or 32-bit integer used as the value of the key. This must be specified if the generator is instantiated within transformed code; when used at the top level of the program, it may be omitted in which case the RNG will be seeded using the default NumPy seeding.

- **impl** (*random.PRNGSpecDesc* *\|* *None*) – optional string specifying the PRNG implementation (e.g. `'threefry2x32'`)

Returns:  
A [`StatefulPRNG`](jax.experimental.random.StatefulPRNG.html#jax.experimental.random.StatefulPRNG "jax.experimental.random.StatefulPRNG") object, with methods for generating random values.

Return type:  
[StatefulPRNG](jax.experimental.random.StatefulPRNG.html#jax.experimental.random.StatefulPRNG "jax.experimental.random.StatefulPRNG")

Notes

The [`StatefulPRNG`](jax.experimental.random.StatefulPRNG.html#jax.experimental.random.StatefulPRNG "jax.experimental.random.StatefulPRNG") object created by this method uses `Ref()` objects to allow implicit updates of state, and thus inherits some of its limitiations. For example:

- [`StatefulPRNG`](jax.experimental.random.StatefulPRNG.html#jax.experimental.random.StatefulPRNG "jax.experimental.random.StatefulPRNG") objects cannot be among the return values of functions wrapped in JIT or other JAX transformations. This means in particular they cannot be used as carry values for [`jax.lax.scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan"), [`jax.lax.while_loop()`](jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop"), and other JAX control flow.

- [`StatefulPRNG`](jax.experimental.random.StatefulPRNG.html#jax.experimental.random.StatefulPRNG "jax.experimental.random.StatefulPRNG") objects cannot be used together with [`jax.checkpoint()`](jax.checkpoint.html#jax.checkpoint "jax.checkpoint") or `jax.remat()`; in these cases it’s best to use the `StatefulPRNG.key()` method to produce a standard JAX PRNG key.

Examples

    >>> from jax.experimental import random
    >>> rng = random.stateful_rng(42)

Repeated draws implicitly update the key:

    >>> rng.uniform()
    Array(0.5302608, dtype=float32)
    >>> rng.uniform()
    Array(0.72766423, dtype=float32)

This also works under transformations like [`jax.jit()`](jax.jit.html#jax.jit "jax.jit"):

    >>> import jax
    >>> jit_uniform = jax.jit(rng.uniform)
    >>> jit_uniform()
    Array(0.6672406, dtype=float32)
    >>> jit_uniform()
    Array(0.3890121, dtype=float32)

Keys can be generated directly if desired:

    >>> rng.key()
    Array((), dtype=key<fry>) overlaying:
    [2954079971 3276725750]
    >>> rng.key()
    Array((), dtype=key<fry>) overlaying:
    [2765691542  824333390]

[](../jax.experimental.random.html "previous page")

previous

`jax.experimental.random` module

[](jax.experimental.random.StatefulPRNG.html "next page")

next

jax.experimental.random.StatefulPRNG

Contents

- [`stateful_rng()`](#jax.experimental.random.stateful_rng)

By The JAX authors

© Copyright 2024, The JAX Authors.\
