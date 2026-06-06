- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.treedef_children

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.treedef_children.rst "Download source file")
-  .pdf

# jax.tree_util.treedef_children

## Contents

- [`treedef_children()`](#jax.tree_util.treedef_children)

# jax.tree_util.treedef_children[\#](#jax-tree-util-treedef-children "Link to this heading")

jax.tree_util.treedef_children(*treedef*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L150-L173)[\#](#jax.tree_util.treedef_children "Link to this definition")  
Return a list of treedefs for immediate children

Parameters:  
**treedef** (*PyTreeDef*) – a single PyTreeDef

Returns:  
a list of PyTreeDefs representing the children of treedef.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[PyTreeDef\]

Examples

    >>> import jax
    >>> x = [(1, 2), 3, {'a': 4}]
    >>> treedef = jax.tree.structure(x)
    >>> jax.tree_util.treedef_children(treedef)
    [PyTreeDef((*, *)), PyTreeDef(*), PyTreeDef({'a': *})]
    >>> _ == [jax.tree.structure(vals) for vals in x]
    True

See also

- [`jax.tree_util.treedef_tuple()`](jax.tree_util.treedef_tuple.html#jax.tree_util.treedef_tuple "jax.tree_util.treedef_tuple")

[](jax.tree_util.tree_map_with_path.html "previous page")

previous

jax.tree_util.tree_map_with_path

[](jax.tree_util.treedef_is_leaf.html "next page")

next

jax.tree_util.treedef_is_leaf

Contents

- [`treedef_children()`](#jax.tree_util.treedef_children)

By The JAX authors

© Copyright 2024, The JAX Authors.\
