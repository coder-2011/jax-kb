- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.tree_map_with_path

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.tree_map_with_path.rst "Download source file")
-  .pdf

# jax.tree_util.tree_map_with_path

## Contents

- [`tree_map_with_path()`](#jax.tree_util.tree_map_with_path)

# jax.tree_util.tree_map_with_path[\#](#jax-tree-util-tree-map-with-path "Link to this heading")

jax.tree_util.tree_map_with_path(*f*, *tree*, *\*rest*, *is_leaf=None*, *is_leaf_takes_path=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L1250-L1269)[\#](#jax.tree_util.tree_map_with_path "Link to this definition")  
Alias of [`jax.tree.map_with_path()`](jax.tree.map_with_path.html#jax.tree.map_with_path "jax.tree.map_with_path").

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*)

- **tree** (*Any*)

- **rest** (*Any*)

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

- **is_leaf_takes_path** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
Any

[](jax.tree_util.tree_leaves_with_path.html "previous page")

previous

jax.tree_util.tree_leaves_with_path

[](jax.tree_util.treedef_children.html "next page")

next

jax.tree_util.treedef_children

Contents

- [`tree_map_with_path()`](#jax.tree_util.tree_map_with_path)

By The JAX authors

© Copyright 2024, The JAX Authors.\
