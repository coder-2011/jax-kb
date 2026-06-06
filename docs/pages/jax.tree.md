- [](index.html)
- [API Reference](jax.html)
- `jax.tree` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.tree.rst "Download source file")
-  .pdf

# jax.tree module

## Contents

- [List of Functions](#list-of-functions)

# `jax.tree` module[\#](#module-jax.tree "Link to this heading")

Utilities for working with tree-like container data structures.

The [`jax.tree`](#module-jax.tree "jax.tree") namespace contains aliases of utilities from [`jax.tree_util`](jax.tree_util.html#module-jax.tree_util "jax.tree_util").

## List of Functions[\#](#list-of-functions "Link to this heading")

|  |  |
|----|----|
| [`all`](_autosummary/jax.tree.all.html#jax.tree.all "jax.tree.all")(tree, \*\[, is_leaf\]) | Call all() over the leaves of a tree. |
| [`broadcast`](_autosummary/jax.tree.broadcast.html#jax.tree.broadcast "jax.tree.broadcast")(prefix_tree, full_tree\[, is_leaf\]) | Broadcasts a tree prefix into the full structure of a given tree. |
| [`flatten`](_autosummary/jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten")(tree\[, is_leaf\]) | Flattens a pytree. |
| [`flatten_with_path`](_autosummary/jax.tree.flatten_with_path.html#jax.tree.flatten_with_path "jax.tree.flatten_with_path")(tree\[, is_leaf, ...\]) | Flattens a pytree like `tree_flatten`, but also returns each leaf's key path. |
| [`leaves`](_autosummary/jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")(tree\[, is_leaf\]) | Gets the leaves of a pytree. |
| [`leaves_with_path`](_autosummary/jax.tree.leaves_with_path.html#jax.tree.leaves_with_path "jax.tree.leaves_with_path")(tree\[, is_leaf, ...\]) | Gets the leaves of a pytree like `tree_leaves` and returns each leaf's key path. |
| [`map`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map")(f, tree, \*rest\[, is_leaf\]) | Maps a multi-input function over pytree args to produce a new pytree. |
| [`map_with_path`](_autosummary/jax.tree.map_with_path.html#jax.tree.map_with_path "jax.tree.map_with_path")(f, tree, \*rest\[, is_leaf, ...\]) | Maps a multi-input function over pytree key path and args to produce a new pytree. |
| [`reduce`](_autosummary/jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce")(function, tree\[, initializer, is_leaf\]) | Call reduce() over the leaves of a tree. |
| [`reduce_associative`](_autosummary/jax.tree.reduce_associative.html#jax.tree.reduce_associative "jax.tree.reduce_associative")(operation, tree, \*\[, ...\]) | Perform a reduction over a pytree with an associative binary operation. |
| [`static`](_autosummary/jax.tree.static.html#jax.tree.static "jax.tree.static")(\*\*kwargs) | Convenience wrapper to declare a static pytree attribute. |
| [`structure`](_autosummary/jax.tree.structure.html#jax.tree.structure "jax.tree.structure")(tree\[, is_leaf\]) | Gets the treedef for a pytree. |
| [`transpose`](_autosummary/jax.tree.transpose.html#jax.tree.transpose "jax.tree.transpose")(outer_treedef, inner_treedef, ...) | Transform a tree having tree structure (outer, inner) into one having structure (inner, outer). |
| [`unflatten`](_autosummary/jax.tree.unflatten.html#jax.tree.unflatten "jax.tree.unflatten")(treedef, leaves) | Reconstructs a pytree from the treedef and the leaves. |

[](_autosummary/jax.test_util.check_vjp.html "previous page")

previous

jax.test_util.check_vjp

[](_autosummary/jax.tree.all.html "next page")

next

jax.tree.all

Contents

- [List of Functions](#list-of-functions)

By The JAX authors

© Copyright 2024, The JAX Authors.\
