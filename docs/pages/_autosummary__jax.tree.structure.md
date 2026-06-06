- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.structure

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.structure.rst "Download source file")
-  .pdf

# jax.tree.structure

## Contents

- [`structure()`](#jax.tree.structure)

# jax.tree.structure[\#](#jax-tree-structure "Link to this heading")

jax.tree.structure(*tree*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L243-L268)[\#](#jax.tree.structure "Link to this definition")  
Gets the treedef for a pytree.

Parameters:  
- **tree** (*Any*) – the pytree for which to get the leaves

- **is_leaf** (*None* *\|* [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]*) – an optionally specified function that will be called at each flattening step. It should return a boolean, which indicates whether the flattening should traverse the current object, or if it should be stopped immediately, with the whole subtree being treated as a leaf.

Returns:  
a PyTreeDef representing the structure of the tree.

Return type:  
pytreedef

Examples

    >>> import jax
    >>> jax.tree.structure([1, (2, 3), [4, 5]])
    PyTreeDef([*, (*, *), [*, *]])

See also

- [`jax.tree.flatten()`](jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten")

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.unflatten()`](jax.tree.unflatten.html#jax.tree.unflatten "jax.tree.unflatten")

[](jax.tree.static.html "previous page")

previous

jax.tree.static

[](jax.tree.transpose.html "next page")

next

jax.tree.transpose

Contents

- [`structure()`](#jax.tree.structure)

By The JAX authors

© Copyright 2024, The JAX Authors.\
