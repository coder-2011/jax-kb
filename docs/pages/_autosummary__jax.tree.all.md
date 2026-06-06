- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.all

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.all.rst "Download source file")
-  .pdf

# jax.tree.all

## Contents

- [`all()`](#jax.tree.all)

# jax.tree.all[\#](#jax-tree-all "Link to this heading")

jax.tree.all(*tree*, *\**, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L25-L50)[\#](#jax.tree.all "Link to this definition")  
Call all() over the leaves of a tree.

Parameters:  
- **tree** (*Any*) – the pytree to evaluate

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*) – an optionally specified function that will be called at each flattening step. It should return a boolean, which indicates whether the flattening should traverse the current object, or if it should be stopped immediately, with the whole subtree being treated as a leaf.

Returns:  
boolean True or False

Return type:  
result

Examples

    >>> import jax
    >>> jax.tree.all([True, {'a': True, 'b': (True, True)}])
    True
    >>> jax.tree.all([False, (True, False)])
    False

See also

- [`jax.tree.reduce()`](jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce")

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

[](../jax.tree.html "previous page")

previous

`jax.tree` module

[](jax.tree.broadcast.html "next page")

next

jax.tree.broadcast

Contents

- [`all()`](#jax.tree.all)

By The JAX authors

© Copyright 2024, The JAX Authors.\
