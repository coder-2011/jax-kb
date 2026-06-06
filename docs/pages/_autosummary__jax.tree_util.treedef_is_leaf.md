- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.treedef_is_leaf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.treedef_is_leaf.rst "Download source file")
-  .pdf

# jax.tree_util.treedef_is_leaf

## Contents

- [`treedef_is_leaf()`](#jax.tree_util.treedef_is_leaf)

# jax.tree_util.treedef_is_leaf[\#](#jax-tree-util-treedef-is-leaf "Link to this heading")

jax.tree_util.treedef_is_leaf(*treedef*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L175-L195)[\#](#jax.tree_util.treedef_is_leaf "Link to this definition")  
Return True if the treedef represents a leaf.

Parameters:  
**treedef** (*PyTreeDef*) – tree to check

Returns:  
True if treedef is a leaf (i.e. has a single node); False otherwise.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

Examples

    >>> import jax
    >>> tree1 = jax.tree.structure(1)
    >>> jax.tree_util.treedef_is_leaf(tree1)
    True
    >>> tree2 = jax.tree.structure([1, 2])
    >>> jax.tree_util.treedef_is_leaf(tree2)
    False

[](jax.tree_util.treedef_children.html "previous page")

previous

jax.tree_util.treedef_children

[](jax.tree_util.treedef_tuple.html "next page")

next

jax.tree_util.treedef_tuple

Contents

- [`treedef_is_leaf()`](#jax.tree_util.treedef_is_leaf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
