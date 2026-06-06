- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.leaves

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.leaves.rst "Download source file")
-  .pdf

# jax.tree.leaves

## Contents

- [`leaves()`](#jax.tree.leaves)

# jax.tree.leaves[\#](#jax-tree-leaves "Link to this heading")

jax.tree.leaves(*tree*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L88-L114)[\#](#jax.tree.leaves "Link to this definition")  
Gets the leaves of a pytree.

Parameters:  
- **tree** (*Any*) – the pytree for which to get the leaves

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, which indicates whether the flattening should traverse the current object, or if it should be stopped immediately, with the whole subtree being treated as a leaf.

Returns:  
a list of tree leaves.

Return type:  
leaves

Examples

    >>> import jax
    >>> jax.tree.leaves([1, (2, 3), [4, 5]])
    [1, 2, 3, 4, 5]

See also

- [`jax.tree.flatten()`](jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten")

- [`jax.tree.structure()`](jax.tree.structure.html#jax.tree.structure "jax.tree.structure")

- [`jax.tree.unflatten()`](jax.tree.unflatten.html#jax.tree.unflatten "jax.tree.unflatten")

[](jax.tree.flatten_with_path.html "previous page")

previous

jax.tree.flatten_with_path

[](jax.tree.leaves_with_path.html "next page")

next

jax.tree.leaves_with_path

Contents

- [`leaves()`](#jax.tree.leaves)

By The JAX authors

© Copyright 2024, The JAX Authors.\
