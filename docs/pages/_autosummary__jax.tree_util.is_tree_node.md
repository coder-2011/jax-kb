- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.is_tree_node

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.is_tree_node.rst "Download source file")
-  .pdf

# jax.tree_util.is_tree_node

## Contents

- [`is_tree_node()`](#jax.tree_util.is_tree_node)

# jax.tree_util.is_tree_node[\#](#jax-tree-util-is-tree-node "Link to this heading")

jax.tree_util.is_tree_node(*typ*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L233-L245)[\#](#jax.tree_util.is_tree_node "Link to this definition")  
Returns True if the type is a registered PyTree node type.

Parameters:  
**typ** ([*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")) – The type to check.

Returns:  
True if the type is a registered PyTree node type (built-in or custom) or a namedtuple type.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

[](jax.tree_util.flatten_one_level_with_keys.html "previous page")

previous

jax.tree_util.flatten_one_level_with_keys

[](jax.tree_util.register_dataclass.html "next page")

next

jax.tree_util.register_dataclass

Contents

- [`is_tree_node()`](#jax.tree_util.is_tree_node)

By The JAX authors

© Copyright 2024, The JAX Authors.\
