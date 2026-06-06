- [](../index.html)
- [API Reference](../jax.html)
- jax.named_scope

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.named_scope.rst "Download source file")
-  .pdf

# jax.named_scope

## Contents

- [`named_scope()`](#jax.named_scope)

# jax.named_scope[\#](#jax-named-scope "Link to this heading")

jax.named_scope(*name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2417-L2467)[\#](#jax.named_scope "Link to this definition")  
A context manager that adds a user specified name to the JAX name stack.

When staging out computations for just-in-time compilation to XLA (or other backends such as TensorFlow) JAX does not, by default, preserve the names (or other source metadata) of Python functions it encounters. This can make debugging the staged out (and/or compiled) representation of your program complicated because there is limited context information for each operation being executed.

`named_scope` tells JAX to stage the given function with additional annotations on the underlying operations. JAX internally keeps track of these annotations in a name stack. When the staged out program is compiled with XLA these annotations are preserved and show up in debugging utilities like the TensorFlow Profiler in TensorBoard. Names are also preserved when staging out JAX programs to TensorFlow using `experimental.jax2tf.convert()`.

Parameters:  
**name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The prefix to use to name all operations created within the name scope.

Yields:  
Yields `None`, but enters a context in which name will be appended to the active name stack.

Return type:  
source_info_util.ExtendNameStackContextManager

Examples

`named_scope` can be used as a context manager inside compiled functions:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def layer(w, x):
    ...   with jax.named_scope("dot_product"):
    ...     logits = w.dot(x)
    ...   with jax.named_scope("activation"):
    ...     return jax.nn.relu(logits)

It can also be used as a decorator:

    >>> @jax.jit
    ... @jax.named_scope("layer")
    ... def layer(w, x):
    ...   logits = w.dot(x)
    ...   return jax.nn.relu(logits)

[](jax.named_call.html "previous page")

previous

jax.named_call

[](jax.block_until_ready.html "next page")

next

jax.block_until_ready

Contents

- [`named_scope()`](#jax.named_scope)

By The JAX authors

© Copyright 2024, The JAX Authors.\
