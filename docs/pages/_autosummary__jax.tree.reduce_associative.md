- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.reduce_associative

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.reduce_associative.rst "Download source file")
-  .pdf

# jax.tree.reduce_associative

## Contents

- [`reduce_associative()`](#jax.tree.reduce_associative)

# jax.tree.reduce_associative[\#](#jax-tree-reduce-associative "Link to this heading")

jax.tree.reduce_associative(*operation*, *tree*, *\**, *identity=\<jax.\_src.tree_util.Unspecified object\>*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L196-L241)[\#](#jax.tree.reduce_associative "Link to this definition")  
Perform a reduction over a pytree with an associative binary operation.

This function exploits the fact that the operation is associative to perform the reduction in parallel (logarithmic depth).

Parameters:  
- **operation** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T,* *T\],* *T\]*) – the associative binary operation

- **tree** (*Any*) – the pytree to reduce

- **identity** (*T* *\|* *tree_util.Unspecified*) – the identity element of the associative binary operation. This is used only when the tree is empty. It is optional otherwise.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, which indicates whether the flattening should traverse the current object, or if it should be stopped immediately, with the whole subtree being treated as a leaf.

Returns:  
the reduced value

Return type:  
result

Examples

    >>> import jax
    >>> import operator
    >>> jax.tree.reduce_associative(operator.add, [1, (2, 3), [4, 5, 6]])
    21

Notes

**Tip**: You can exclude leaves from the reduction by first mapping them to `None` using [`jax.tree.map()`](jax.tree.map.html#jax.tree.map "jax.tree.map"). This causes them to not be counted as leaves after that.

See also

- [`jax.tree.reduce()`](jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce")

[](jax.tree.reduce.html "previous page")

previous

jax.tree.reduce

[](jax.tree.static.html "next page")

next

jax.tree.static

Contents

- [`reduce_associative()`](#jax.tree.reduce_associative)

By The JAX authors

© Copyright 2024, The JAX Authors.\
