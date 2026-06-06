- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.flatten_with_path

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.flatten_with_path.rst "Download source file")
-  .pdf

# jax.tree.flatten_with_path

## Contents

- [`flatten_with_path()`](#jax.tree.flatten_with_path)

# jax.tree.flatten_with_path[\#](#jax-tree-flatten-with-path "Link to this heading")

jax.tree.flatten_with_path(*tree*, *is_leaf=None*, *is_leaf_takes_path=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L331-L360)[\#](#jax.tree.flatten_with_path "Link to this definition")  
Flattens a pytree like `tree_flatten`, but also returns each leaf’s key path.

Parameters:  
- **tree** (*Any*) – a pytree to flatten. If it contains a custom type, it is recommended to be registered with `register_pytree_with_keys`.

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

- **is_leaf_takes_path** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Returns:  
A pair which the first element is a list of key-leaf pairs, each of which contains a leaf and its key path. The second element is a treedef representing the structure of the flattened tree.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[tree_util.KeyPath, Any\]\], tree_util.PyTreeDef\]

Examples

    >>> import jax
    >>> path_vals, treedef = jax.tree.flatten_with_path([1, {'x': 3}])
    >>> path_vals
    [((SequenceKey(idx=0),), 1), ((SequenceKey(idx=1), DictKey(key='x')), 3)]
    >>> treedef
    PyTreeDef([*, {'x': *}])

See also

- [`jax.tree.flatten()`](jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten")

- [`jax.tree.map_with_path()`](jax.tree.map_with_path.html#jax.tree.map_with_path "jax.tree.map_with_path")

- [`jax.tree_util.register_pytree_with_keys()`](jax.tree_util.register_pytree_with_keys.html#jax.tree_util.register_pytree_with_keys "jax.tree_util.register_pytree_with_keys")

[](jax.tree.flatten.html "previous page")

previous

jax.tree.flatten

[](jax.tree.leaves.html "next page")

next

jax.tree.leaves

Contents

- [`flatten_with_path()`](#jax.tree.flatten_with_path)

By The JAX authors

© Copyright 2024, The JAX Authors.\
