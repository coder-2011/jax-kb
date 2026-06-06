- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.flatten

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.flatten.rst "Download source file")
-  .pdf

# jax.tree.flatten

## Contents

- [`flatten()`](#jax.tree.flatten)

# jax.tree.flatten[\#](#jax-tree-flatten "Link to this heading")

jax.tree.flatten(*tree*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L52-L86)[\#](#jax.tree.flatten "Link to this definition")  
Flattens a pytree.

The flattening order (i.e. the order of elements in the output list) is deterministic, corresponding to a left-to-right depth-first tree traversal.

Parameters:  
- **tree** (*Any*) – a pytree to flatten.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, with true stopping the traversal and the whole subtree being treated as a leaf, and false indicating the flattening should traverse the current object.

Returns:  
A pair where the first element is a list of leaf values and the second element is a treedef representing the structure of the flattened tree.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[tree_util.Leaf\], tree_util.PyTreeDef\]

Examples

    >>> import jax
    >>> vals, treedef = jax.tree.flatten([1, (2, 3), [4, 5]])
    >>> vals
    [1, 2, 3, 4, 5]
    >>> treedef
    PyTreeDef([*, (*, *), [*, *]])

See also

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.structure()`](jax.tree.structure.html#jax.tree.structure "jax.tree.structure")

- [`jax.tree.unflatten()`](jax.tree.unflatten.html#jax.tree.unflatten "jax.tree.unflatten")

[](jax.tree.broadcast.html "previous page")

previous

jax.tree.broadcast

[](jax.tree.flatten_with_path.html "next page")

next

jax.tree.flatten_with_path

Contents

- [`flatten()`](#jax.tree.flatten)

By The JAX authors

© Copyright 2024, The JAX Authors.\
