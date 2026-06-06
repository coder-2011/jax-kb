- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.loop

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.loop.rst "Download source file")
-  .pdf

# jax.experimental.pallas.loop

## Contents

- [`loop()`](#jax.experimental.pallas.loop)

# jax.experimental.pallas.loop[\#](#jax-experimental-pallas-loop "Link to this heading")

jax.experimental.pallas.loop(*lower: jax_typing.ArrayLike*, *upper: jax_typing.ArrayLike*, *\**, *init_carry: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *step: jax_typing.ArrayLike = 1*, *unroll: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[jax_typing.Array\], [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")\]\], [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/helpers.py#L117-L154)[\#](#jax.experimental.pallas.loop "Link to this definition")\
jax.experimental.pallas.loop(*lower: jax_typing.ArrayLike*, *upper: jax_typing.ArrayLike*, *\**, *init_carry: \_T = None*, *step: jax_typing.ArrayLike = 1*, *unroll: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[jax_typing.Array, \_T\], \_T\]\], \_T\]  
Returns a decorator that calls the decorated function in a loop.

[](jax.experimental.pallas.get_global.html "previous page")

previous

jax.experimental.pallas.get_global

[](jax.experimental.pallas.multiple_of.html "next page")

next

jax.experimental.pallas.multiple_of

Contents

- [`loop()`](#jax.experimental.pallas.loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\
