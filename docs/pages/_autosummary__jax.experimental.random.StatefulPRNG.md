- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.random` module](../jax.experimental.random.html)
- jax.experimental.random.StatefulPRNG

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.random.StatefulPRNG.rst "Download source file")
-  .pdf

# jax.experimental.random.StatefulPRNG

## Contents

- [`StatefulPRNG`](#jax.experimental.random.StatefulPRNG)
  - [`StatefulPRNG._base_key`](#jax.experimental.random.StatefulPRNG._base_key)
  - [`StatefulPRNG._counter`](#jax.experimental.random.StatefulPRNG._counter)
  - [`StatefulPRNG.__init__()`](#jax.experimental.random.StatefulPRNG.__init__)

# jax.experimental.random.StatefulPRNG[\#](#jax-experimental-random-statefulprng "Link to this heading")

*class* jax.experimental.random.StatefulPRNG(*\_base_key*, *\_counter*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/stateful_rng.py#L48-L215)[\#](#jax.experimental.random.StatefulPRNG "Link to this definition")  
Stateful JAX random generator.

This should be instantiated using the [`jax.experimental.random.stateful_rng()`](jax.experimental.random.stateful_rng.html#jax.experimental.random.stateful_rng "jax.experimental.random.stateful_rng") function.

Parameters:  
- **\_base_key** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **\_counter** ([*core.Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref"))

\_base_key[\#](#jax.experimental.random.StatefulPRNG._base_key "Link to this definition")  
a typed JAX PRNG key object (see [`jax.random.key()`](jax.random.key.html#jax.random.key "jax.random.key")).

Type:  
[Array](jax.Array.html#jax.Array "jax.Array")

\_counter[\#](#jax.experimental.random.StatefulPRNG._counter "Link to this definition")  
a scalar integer wrapped in a `jax.Ref`.

Type:  
[core.Ref](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref")

Examples:

    >>> from jax.experimental import random
    >>> rng = random.stateful_rng(42)
    >>> rng
    StatefulPRNG(_base_key=Array((), dtype=key<fry>) overlaying:
    [ 0 42], _counter=Ref(0, dtype=int32, weak_type=True))

\_\_init\_\_(*\_base_key*, *\_counter*)[\#](#jax.experimental.random.StatefulPRNG.__init__ "Link to this definition")  
Parameters:  
- **\_base_key** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **\_counter** ([*core.Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref"))

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.random.StatefulPRNG.__init__ "jax.experimental.random.StatefulPRNG.__init__")(\_base_key, \_counter) |  |
| `integers`(low\[, high, size, dtype\]) | Draw pseudorandom integers. |
| `key`(\[shape\]) | Generate a new JAX PRNGKey, updating the internal state. |
| `normal`(\[loc, scale, size, dtype\]) | Draw normally-distributed pseudorandom values. |
| `random`(\[size, dtype\]) | Return random floats in the half-open interval \[0.0, 1.0). |
| `spawn`(n_children) | Create a list of independent child generators. |
| `split`(num) | Create independent child generators suitable for use in [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"). |
| `uniform`(\[low, high, size, dtype\]) | Draw uniformly distributed pseudorandom values. |

[](jax.experimental.random.stateful_rng.html "previous page")

previous

jax.experimental.random.stateful_rng

[](../jax.experimental.serialize_executable.html "next page")

next

`jax.experimental.serialize_executable` module

Contents

- [`StatefulPRNG`](#jax.experimental.random.StatefulPRNG)
  - [`StatefulPRNG._base_key`](#jax.experimental.random.StatefulPRNG._base_key)
  - [`StatefulPRNG._counter`](#jax.experimental.random.StatefulPRNG._counter)
  - [`StatefulPRNG.__init__()`](#jax.experimental.random.StatefulPRNG.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
