- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.reduce

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.reduce.rst "Download source file")
-  .pdf

# jax.tree.reduce

## Contents

- [`reduce()`](#jax.tree.reduce)

# jax.tree.reduce[\#](#jax-tree-reduce "Link to this heading")

jax.tree.reduce(*function*, *tree*, *initializer=\<jax.\_src.tree_util.Unspecified object\>*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L159-L194)[\#](#jax.tree.reduce "Link to this definition")  
Call reduce() over the leaves of a tree.

Parameters:  
- **function** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T,* *Any\],* *T\]*) – the reduction function

- **tree** (*Any*) – the pytree to reduce over

- **initializer** (*T* *\|* *tree_util.Unspecified*) – the optional initial value

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, which indicates whether the flattening should traverse the current object, or if it should be stopped immediately, with the whole subtree being treated as a leaf.

Returns:  
the reduced value.

Return type:  
result

Examples

    >>> import jax
    >>> import operator
    >>> jax.tree.reduce(operator.add, [1, (2, 3), [4, 5, 6]])
    21

Notes

**Tip**: You can exclude leaves from the reduction by first mapping them to `None` using [`jax.tree.map()`](jax.tree.map.html#jax.tree.map "jax.tree.map"). This causes them to not be counted as leaves after that.

See also

- [`jax.tree.reduce_associative()`](jax.tree.reduce_associative.html#jax.tree.reduce_associative "jax.tree.reduce_associative")

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.map()`](jax.tree.map.html#jax.tree.map "jax.tree.map")

[](jax.tree.map_with_path.html "previous page")

previous

jax.tree.map_with_path

[](jax.tree.reduce_associative.html "next page")

next

jax.tree.reduce_associative

Contents

- [`reduce()`](#jax.tree.reduce)

By The JAX authors

© Copyright 2024, The JAX Authors.\
