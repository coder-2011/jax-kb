- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.flatten_one_level

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.flatten_one_level.rst "Download source file")
-  .pdf

# jax.tree_util.flatten_one_level

## Contents

- [`flatten_one_level()`](#jax.tree_util.flatten_one_level)

# jax.tree_util.flatten_one_level[\#](#jax-tree-util-flatten-one-level "Link to this heading")

jax.tree_util.flatten_one_level(*tree*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L687-L716)[\#](#jax.tree_util.flatten_one_level "Link to this definition")  
Flatten the given pytree node by one level.

Parameters:  
**tree** (*Any*) – A valid pytree node, either built-in or registered via [`register_pytree_node()`](jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node") or related functions.

Returns:  
A pair of the pytrees flattened children and its hashable metadata.

Raises:  
- [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – If the given pytree is not a built-in or registered container

- **via register_pytree_node** **or** **register_pytree_with_keys.** –

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Iterable\[Any\], Hashable\]

Examples

    >>> import jax
    >>> from jax._src.tree_util import flatten_one_level
    >>> flattened, meta = flatten_one_level({'a': [1, 2], 'b': {'c': 3}})
    >>> flattened
    ([1, 2], {'c': 3})
    >>> meta
    ('a', 'b')

[](jax.tree_util.all_leaves.html "previous page")

previous

jax.tree_util.all_leaves

[](jax.tree_util.flatten_one_level_with_keys.html "next page")

next

jax.tree_util.flatten_one_level_with_keys

Contents

- [`flatten_one_level()`](#jax.tree_util.flatten_one_level)

By The JAX authors

© Copyright 2024, The JAX Authors.\
