- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.map

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.map.rst "Download source file")
-  .pdf

# jax.tree.map

## Contents

- [`map()`](#jax.tree.map)

# jax.tree.map[\#](#jax-tree-map "Link to this heading")

jax.tree.map(*f*, *tree*, *\*rest*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L116-L157)[\#](#jax.tree.map "Link to this definition")  
Maps a multi-input function over pytree args to produce a new pytree.

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*) – function that takes `1`` ``+`` ``len(rest)` arguments, to be applied at the corresponding leaves of the pytrees.

- **tree** (*Any*) – a pytree to be mapped over, with each leaf providing the first positional argument to `f`.

- **rest** (*Any*) – a tuple of pytrees, each of which has the same structure as `tree` or has `tree` as a prefix.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, which indicates whether the flattening should traverse the current object, or if it should be stopped immediately, with the whole subtree being treated as a leaf.

Returns:  
A new pytree with the same structure as `tree` but with the value at each leaf given by `f(x,`` ``*xs)` where `x` is the value at the corresponding leaf in `tree` and `xs` is the tuple of values at corresponding nodes in `rest`.

Return type:  
Any

Examples

    >>> import jax
    >>> jax.tree.map(lambda x: x + 1, {"x": 7, "y": 42})
    {'x': 8, 'y': 43}

If multiple inputs are passed, the structure of the tree is taken from the first input; subsequent inputs need only have `tree` as a prefix:

    >>> jax.tree.map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

See also

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.reduce()`](jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce")

[](jax.tree.leaves_with_path.html "previous page")

previous

jax.tree.leaves_with_path

[](jax.tree.map_with_path.html "next page")

next

jax.tree.map_with_path

Contents

- [`map()`](#jax.tree.map)

By The JAX authors

© Copyright 2024, The JAX Authors.\
