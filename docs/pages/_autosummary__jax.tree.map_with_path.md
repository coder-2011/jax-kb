- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.map_with_path

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.map_with_path.rst "Download source file")
-  .pdf

# jax.tree.map_with_path

## Contents

- [`map_with_path()`](#jax.tree.map_with_path)

# jax.tree.map_with_path[\#](#jax-tree-map-with-path "Link to this heading")

jax.tree.map_with_path(*f*, *tree*, *\*rest*, *is_leaf=None*, *is_leaf_takes_path=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L388-L428)[\#](#jax.tree.map_with_path "Link to this definition")  
Maps a multi-input function over pytree key path and args to produce a new pytree.

This is a more powerful alternative of `tree_map` that can take the key path of each leaf as input argument as well.

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*) – function that takes `2`` ``+`` ``len(rest)` arguments, aka. the key path and each corresponding leaves of the pytrees.

- **tree** (*Any*) – a pytree to be mapped over, with each leaf’s key path as the first positional argument and the leaf itself as the second argument to `f`.

- **\*rest** (*Any*) – a tuple of pytrees, each of which has the same structure as `tree` or has `tree` as a prefix.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

- **is_leaf_takes_path** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Returns:  
A new pytree with the same structure as `tree` but with the value at each leaf given by `f(kp,`` ``x,`` ``*xs)` where `kp` is the key path of the leaf at the corresponding leaf in `tree`, `x` is the leaf value and `xs` is the tuple of values at corresponding nodes in `rest`.

Return type:  
Any

Examples

    >>> import jax
    >>> jax.tree.map_with_path(lambda path, x: x + path[0].idx, [1, 2, 3])
    [1, 3, 5]

See also

- [`jax.tree.map()`](jax.tree.map.html#jax.tree.map "jax.tree.map")

- [`jax.tree.flatten_with_path()`](jax.tree.flatten_with_path.html#jax.tree.flatten_with_path "jax.tree.flatten_with_path")

- [`jax.tree.leaves_with_path()`](jax.tree.leaves_with_path.html#jax.tree.leaves_with_path "jax.tree.leaves_with_path")

- [`jax.tree_util.register_pytree_with_keys()`](jax.tree_util.register_pytree_with_keys.html#jax.tree_util.register_pytree_with_keys "jax.tree_util.register_pytree_with_keys")

[](jax.tree.map.html "previous page")

previous

jax.tree.map

[](jax.tree.reduce.html "next page")

next

jax.tree.reduce

Contents

- [`map_with_path()`](#jax.tree.map_with_path)

By The JAX authors

© Copyright 2024, The JAX Authors.\
