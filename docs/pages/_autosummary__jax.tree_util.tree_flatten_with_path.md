- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.tree_flatten_with_path

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.tree_flatten_with_path.rst "Download source file")
-  .pdf

# jax.tree_util.tree_flatten_with_path

## Contents

- [`tree_flatten_with_path()`](#jax.tree_util.tree_flatten_with_path)

# jax.tree_util.tree_flatten_with_path[\#](#jax-tree-util-tree-flatten-with-path "Link to this heading")

jax.tree_util.tree_flatten_with_path(*tree*, *is_leaf=None*, *is_leaf_takes_path=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L1228-L1238)[\#](#jax.tree_util.tree_flatten_with_path "Link to this definition")  
Alias of [`jax.tree.flatten_with_path()`](jax.tree.flatten_with_path.html#jax.tree.flatten_with_path "jax.tree.flatten_with_path").

Parameters:  
- **tree** (*Any*)

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

- **is_leaf_takes_path** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[KeyPath, Any\]\], PyTreeDef\]

[](jax.tree_util.register_static.html "previous page")

previous

jax.tree_util.register_static

[](jax.tree_util.tree_leaves_with_path.html "next page")

next

jax.tree_util.tree_leaves_with_path

Contents

- [`tree_flatten_with_path()`](#jax.tree_util.tree_flatten_with_path)

By The JAX authors

© Copyright 2024, The JAX Authors.\
