- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.all_leaves

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.all_leaves.rst "Download source file")
-  .pdf

# jax.tree_util.all_leaves

## Contents

- [`all_leaves()`](#jax.tree_util.all_leaves)

# jax.tree_util.all_leaves[\#](#jax-tree-util-all-leaves "Link to this heading")

jax.tree_util.all_leaves(*iterable*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L202-L231)[\#](#jax.tree_util.all_leaves "Link to this definition")  
Tests whether all elements in the given iterable are all leaves.

This function is useful in advanced cases, for example if a library allows arbitrary map operations on a flat iterable of leaves it may want to check if the result is still a flat iterable of leaves.

Parameters:  
- **iterable** (*Iterable\[Any\]*) – Iterable of leaves.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

Returns:  
A boolean indicating if all elements in the input are leaves.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

Examples

    >>> import jax
    >>> tree = {"a": [1, 2, 3]}
    >>> assert all_leaves(jax.tree_util.tree_leaves(tree))
    >>> assert not all_leaves([tree])

[](jax.tree_util.Partial.html "previous page")

previous

jax.tree_util.Partial

[](jax.tree_util.flatten_one_level.html "next page")

next

jax.tree_util.flatten_one_level

Contents

- [`all_leaves()`](#jax.tree_util.all_leaves)

By The JAX authors

© Copyright 2024, The JAX Authors.\
