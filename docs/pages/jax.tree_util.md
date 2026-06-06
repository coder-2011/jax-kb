- [](index.html)
- [API Reference](jax.html)
- `jax.tree_util` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.tree_util.rst "Download source file")
-  .pdf

# jax.tree_util module

## Contents

- [List of Functions](#list-of-functions)
- [Legacy APIs](#legacy-apis)

# `jax.tree_util` module[\#](#module-jax.tree_util "Link to this heading")

Utilities for working with tree-like container data structures.

This module provides a small set of utility functions for working with tree-like data structures, such as nested tuples, lists, and dicts. We call these structures pytrees. They are trees in that they are defined recursively (any non-pytree is a pytree, i.e. a leaf, and any pytree of pytrees is a pytree) and can be operated on recursively (object identity equivalence is not preserved by mapping operations, and the structures cannot contain reference cycles).

The set of Python types that are considered pytree nodes (e.g. that can be mapped over, rather than treated as leaves) is extensible. There is a single module-level registry of types, and class hierarchy is ignored. By registering a new pytree node type, that type in effect becomes transparent to the utility functions in this file.

The primary purpose of this module is to enable the interoperability between user defined data structures and JAX transformations (e.g. jit). This is not meant to be a general purpose tree-like data structure handling library.

See the [JAX pytrees note](pytrees.html) for examples.

## List of Functions[\#](#list-of-functions "Link to this heading")

|  |  |
|----|----|
| [`Partial`](_autosummary/jax.tree_util.Partial.html#jax.tree_util.Partial "jax.tree_util.Partial")(func, \*args, \*\*kw) | A version of functools.partial that works in pytrees. |
| [`all_leaves`](_autosummary/jax.tree_util.all_leaves.html#jax.tree_util.all_leaves "jax.tree_util.all_leaves")(iterable\[, is_leaf\]) | Tests whether all elements in the given iterable are all leaves. |
| [`flatten_one_level`](_autosummary/jax.tree_util.flatten_one_level.html#jax.tree_util.flatten_one_level "jax.tree_util.flatten_one_level")(tree) | Flatten the given pytree node by one level. |
| [`flatten_one_level_with_keys`](_autosummary/jax.tree_util.flatten_one_level_with_keys.html#jax.tree_util.flatten_one_level_with_keys "jax.tree_util.flatten_one_level_with_keys")(tree) | Flatten the given pytree node by one level, with keys. |
| [`is_tree_node`](_autosummary/jax.tree_util.is_tree_node.html#jax.tree_util.is_tree_node "jax.tree_util.is_tree_node")(typ) | Returns True if the type is a registered PyTree node type. |
| [`register_dataclass`](_autosummary/jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass")(nodetype\[, data_fields, ...\]) | Extends the set of types that are considered internal nodes in pytrees. |
| [`register_pytree_node`](_autosummary/jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node")(nodetype, flatten_func, ...) | Extends the set of types that are considered internal nodes in pytrees. |
| [`register_pytree_node_class`](_autosummary/jax.tree_util.register_pytree_node_class.html#jax.tree_util.register_pytree_node_class "jax.tree_util.register_pytree_node_class")(cls) | Extends the set of types that are considered internal nodes in pytrees. |
| [`register_pytree_with_keys`](_autosummary/jax.tree_util.register_pytree_with_keys.html#jax.tree_util.register_pytree_with_keys "jax.tree_util.register_pytree_with_keys")(nodetype, ...\[, ...\]) | Extends the set of types that are considered internal nodes in pytrees. |
| [`register_pytree_with_keys_class`](_autosummary/jax.tree_util.register_pytree_with_keys_class.html#jax.tree_util.register_pytree_with_keys_class "jax.tree_util.register_pytree_with_keys_class")(cls) | Extends the set of types that are considered internal nodes in pytrees. |
| [`register_static`](_autosummary/jax.tree_util.register_static.html#jax.tree_util.register_static "jax.tree_util.register_static")(cls) | Registers cls as a pytree with no leaves. |
| [`tree_flatten_with_path`](_autosummary/jax.tree_util.tree_flatten_with_path.html#jax.tree_util.tree_flatten_with_path "jax.tree_util.tree_flatten_with_path")(tree\[, is_leaf, ...\]) | Alias of [`jax.tree.flatten_with_path()`](_autosummary/jax.tree.flatten_with_path.html#jax.tree.flatten_with_path "jax.tree.flatten_with_path"). |
| [`tree_leaves_with_path`](_autosummary/jax.tree_util.tree_leaves_with_path.html#jax.tree_util.tree_leaves_with_path "jax.tree_util.tree_leaves_with_path")(tree\[, is_leaf, ...\]) | Alias of [`jax.tree.leaves_with_path()`](_autosummary/jax.tree.leaves_with_path.html#jax.tree.leaves_with_path "jax.tree.leaves_with_path"). |
| [`tree_map_with_path`](_autosummary/jax.tree_util.tree_map_with_path.html#jax.tree_util.tree_map_with_path "jax.tree_util.tree_map_with_path")(f, tree, \*rest\[, ...\]) | Alias of [`jax.tree.map_with_path()`](_autosummary/jax.tree.map_with_path.html#jax.tree.map_with_path "jax.tree.map_with_path"). |
| [`treedef_children`](_autosummary/jax.tree_util.treedef_children.html#jax.tree_util.treedef_children "jax.tree_util.treedef_children")(treedef) | Return a list of treedefs for immediate children |
| [`treedef_is_leaf`](_autosummary/jax.tree_util.treedef_is_leaf.html#jax.tree_util.treedef_is_leaf "jax.tree_util.treedef_is_leaf")(treedef) | Return True if the treedef represents a leaf. |
| [`treedef_tuple`](_autosummary/jax.tree_util.treedef_tuple.html#jax.tree_util.treedef_tuple "jax.tree_util.treedef_tuple")(treedefs) | Makes a tuple treedef from an iterable of child treedefs. |
| [`KeyEntry`](_autosummary/jax.tree_util.KeyEntry.html#jax.tree_util.KeyEntry "jax.tree_util.KeyEntry") | Type variable. |
| [`KeyPath`](_autosummary/jax.tree_util.KeyPath.html#jax.tree_util.KeyPath "jax.tree_util.KeyPath") | Built-in immutable sequence. |
| [`keystr`](_autosummary/jax.tree_util.keystr.html#jax.tree_util.keystr "jax.tree_util.keystr")(keys, \*\[, simple, separator\]) | Helper to pretty-print a tuple of keys. |

## Legacy APIs[\#](#legacy-apis "Link to this heading")

These APIs are now accessed via [`jax.tree`](jax.tree.html#module-jax.tree "jax.tree").

|  |  |
|----|----|
| [`tree_all`](_autosummary/jax.tree_util.tree_all.html#jax.tree_util.tree_all "jax.tree_util.tree_all")(tree, \*\[, is_leaf\]) | Alias of [`jax.tree.all()`](_autosummary/jax.tree.all.html#jax.tree.all "jax.tree.all"). |
| [`tree_broadcast`](_autosummary/jax.tree_util.tree_broadcast.html#jax.tree_util.tree_broadcast "jax.tree_util.tree_broadcast")(prefix_tree, full_tree\[, is_leaf\]) | Alias of [`jax.tree.broadcast()`](_autosummary/jax.tree.broadcast.html#jax.tree.broadcast "jax.tree.broadcast"). |
| [`tree_flatten`](_autosummary/jax.tree_util.tree_flatten.html#jax.tree_util.tree_flatten "jax.tree_util.tree_flatten")(tree\[, is_leaf\]) | Alias of [`jax.tree.flatten()`](_autosummary/jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten"). |
| [`tree_leaves`](_autosummary/jax.tree_util.tree_leaves.html#jax.tree_util.tree_leaves "jax.tree_util.tree_leaves")(tree\[, is_leaf\]) | Alias of [`jax.tree.leaves()`](_autosummary/jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves"). |
| [`tree_map`](_autosummary/jax.tree_util.tree_map.html#jax.tree_util.tree_map "jax.tree_util.tree_map")(f, tree, \*rest\[, is_leaf\]) | Alias of [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map"). |
| [`tree_reduce`](_autosummary/jax.tree_util.tree_reduce.html#jax.tree_util.tree_reduce "jax.tree_util.tree_reduce")(function, tree\[, initializer, ...\]) | Alias of [`jax.tree.reduce()`](_autosummary/jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce"). |
| [`tree_reduce_associative`](_autosummary/jax.tree_util.tree_reduce_associative.html#jax.tree_util.tree_reduce_associative "jax.tree_util.tree_reduce_associative")(operation, tree, \*) | Alias of [`jax.tree.reduce_associative()`](_autosummary/jax.tree.reduce_associative.html#jax.tree.reduce_associative "jax.tree.reduce_associative"). |
| [`tree_structure`](_autosummary/jax.tree_util.tree_structure.html#jax.tree_util.tree_structure "jax.tree_util.tree_structure")(tree\[, is_leaf\]) | Alias of [`jax.tree.structure()`](_autosummary/jax.tree.structure.html#jax.tree.structure "jax.tree.structure"). |
| [`tree_transpose`](_autosummary/jax.tree_util.tree_transpose.html#jax.tree_util.tree_transpose "jax.tree_util.tree_transpose")(outer_treedef, inner_treedef, ...) | Alias of [`jax.tree.transpose()`](_autosummary/jax.tree.transpose.html#jax.tree.transpose "jax.tree.transpose"). |
| [`tree_unflatten`](_autosummary/jax.tree_util.tree_unflatten.html#jax.tree_util.tree_unflatten "jax.tree_util.tree_unflatten")(treedef, leaves) | Alias of [`jax.tree.unflatten()`](_autosummary/jax.tree.unflatten.html#jax.tree.unflatten "jax.tree.unflatten"). |

[](_autosummary/jax.tree.unflatten.html "previous page")

previous

jax.tree.unflatten

[](_autosummary/jax.tree_util.Partial.html "next page")

next

jax.tree_util.Partial

Contents

- [List of Functions](#list-of-functions)
- [Legacy APIs](#legacy-apis)

By The JAX authors

© Copyright 2024, The JAX Authors.\
