- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.leaves_with_path

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.leaves_with_path.rst "Download source file")
-  .pdf

# jax.tree.leaves_with_path

## Contents

- [`leaves_with_path()`](#jax.tree.leaves_with_path)

# jax.tree.leaves_with_path[\#](#jax-tree-leaves-with-path "Link to this heading")

jax.tree.leaves_with_path(*tree*, *is_leaf=None*, *is_leaf_takes_path=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L362-L386)[\#](#jax.tree.leaves_with_path "Link to this definition")  
Gets the leaves of a pytree like `tree_leaves` and returns each leaf’s key path.

Parameters:  
- **tree** (*Any*) – a pytree. If it contains a custom type, it is recommended to be registered with `register_pytree_with_keys`.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

- **is_leaf_takes_path** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Returns:  
A list of key-leaf pairs, each of which contains a leaf and its key path.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[tree_util.KeyPath, Any\]\]

Examples

    >>> import jax
    >>> jax.tree.leaves_with_path([1, {'x': 3}])
    [((SequenceKey(idx=0),), 1), ((SequenceKey(idx=1), DictKey(key='x')), 3)]

See also

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.flatten_with_path()`](jax.tree.flatten_with_path.html#jax.tree.flatten_with_path "jax.tree.flatten_with_path")

- [`jax.tree_util.register_pytree_with_keys()`](jax.tree_util.register_pytree_with_keys.html#jax.tree_util.register_pytree_with_keys "jax.tree_util.register_pytree_with_keys")

[](jax.tree.leaves.html "previous page")

previous

jax.tree.leaves

[](jax.tree.map.html "next page")

next

jax.tree.map

Contents

- [`leaves_with_path()`](#jax.tree.leaves_with_path)

By The JAX authors

© Copyright 2024, The JAX Authors.\
