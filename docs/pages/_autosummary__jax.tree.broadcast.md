- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.broadcast

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.broadcast.rst "Download source file")
-  .pdf

# jax.tree.broadcast

## Contents

- [`broadcast()`](#jax.tree.broadcast)

# jax.tree.broadcast[\#](#jax-tree-broadcast "Link to this heading")

jax.tree.broadcast(*prefix_tree*, *full_tree*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L430-L459)[\#](#jax.tree.broadcast "Link to this definition")  
Broadcasts a tree prefix into the full structure of a given tree.

Parameters:  
- **prefix_tree** (*Any*) – a pytree that is a tree prefix of full_tree.

- **full_tree** (*Any*) – a pytree with the structure to broadcast the prefix leaves into.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, with true stopping the traversal and the whole subtree being treated as a leaf, and false indicating the flattening should traverse the current object.

Returns:  
A pytree matching the structure of full_tree where the leaves of prefix_tree have been broadcasted into the leaves of each corresponding subtree.

Return type:  
Any

Examples

    >>> import jax
    >>> prefix = (1, 2, 3)
    >>> full = (0, {'a': 0, 'b': 0}, (0, 0))
    >>> jax.tree.broadcast(prefix, full)
    (1, {'a': 2, 'b': 2}, (3, 3))

See also

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.structure()`](jax.tree.structure.html#jax.tree.structure "jax.tree.structure")

[](jax.tree.all.html "previous page")

previous

jax.tree.all

[](jax.tree.flatten.html "next page")

next

jax.tree.flatten

Contents

- [`broadcast()`](#jax.tree.broadcast)

By The JAX authors

© Copyright 2024, The JAX Authors.\
