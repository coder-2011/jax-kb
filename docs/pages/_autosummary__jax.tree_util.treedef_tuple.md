- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.treedef_tuple

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.treedef_tuple.rst "Download source file")
-  .pdf

# jax.tree_util.treedef_tuple

## Contents

- [`treedef_tuple()`](#jax.tree_util.treedef_tuple)

# jax.tree_util.treedef_tuple[\#](#jax-tree-util-treedef-tuple "Link to this heading")

jax.tree_util.treedef_tuple(*treedefs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L124-L148)[\#](#jax.tree_util.treedef_tuple "Link to this definition")  
Makes a tuple treedef from an iterable of child treedefs.

Parameters:  
**treedefs** (*Iterable\[PyTreeDef\]*) – iterable of PyTree structures

Returns:  
a single treedef representing a tuple of the structures

Return type:  
PyTreeDef

Examples

    >>> import jax
    >>> x = [1, 2, 3]
    >>> y = {'a': 4, 'b': 5}
    >>> x_tree = jax.tree.structure(x)
    >>> y_tree = jax.tree.structure(y)
    >>> xy_tree = jax.tree_util.treedef_tuple([x_tree, y_tree])
    >>> xy_tree == jax.tree.structure((x, y))
    True

See also

- [`jax.tree_util.treedef_children()`](jax.tree_util.treedef_children.html#jax.tree_util.treedef_children "jax.tree_util.treedef_children")

[](jax.tree_util.treedef_is_leaf.html "previous page")

previous

jax.tree_util.treedef_is_leaf

[](jax.tree_util.KeyEntry.html "next page")

next

jax.tree_util.KeyEntry

Contents

- [`treedef_tuple()`](#jax.tree_util.treedef_tuple)

By The JAX authors

© Copyright 2024, The JAX Authors.\
