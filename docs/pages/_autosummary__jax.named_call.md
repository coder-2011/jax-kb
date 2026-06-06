- [](../index.html)
- [API Reference](../jax.html)
- jax.named_call

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.named_call.rst "Download source file")
-  .pdf

# jax.named_call

## Contents

- [`named_call()`](#jax.named_call)

# jax.named_call[\#](#jax-named-call "Link to this heading")

jax.named_call(*fun*, *\**, *name=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2382-L2415)[\#](#jax.named_call "Link to this definition")  
Adds a user specified name to a function when staging out JAX computations.

When staging out computations for just-in-time compilation to XLA (or other backends such as TensorFlow) JAX runs your Python program but by default does not preserve any of the function names or other metadata associated with it. This can make debugging the staged out (and/or compiled) representation of your program complicated because there is limited context information for each operation being executed.

named_call tells JAX to stage the given function out as a subcomputation with a specific name. When the staged out program is compiled with XLA these named subcomputations are preserved and show up in debugging utilities like the TensorFlow Profiler in TensorBoard. Names are also preserved when staging out JAX programs to TensorFlow using `experimental.jax2tf.convert()`.

Parameters:  
- **fun** (*F*) – Function to be wrapped. This can be any Callable.

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Optional. The prefix to use to name all sub computations created within the name scope. Use the fun.\_\_name\_\_ if not specified.

Returns:  
A version of `fun` that is wrapped in a `named_scope`.

Return type:  
F

[](jax.default_backend.html "previous page")

previous

jax.default_backend

[](jax.named_scope.html "next page")

next

jax.named_scope

Contents

- [`named_call()`](#jax.named_call)

By The JAX authors

© Copyright 2024, The JAX Authors.\
