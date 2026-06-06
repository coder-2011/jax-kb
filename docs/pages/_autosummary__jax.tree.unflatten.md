- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.unflatten

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.unflatten.rst "Download source file")
-  .pdf

# jax.tree.unflatten

## Contents

- [`unflatten()`](#jax.tree.unflatten)

# jax.tree.unflatten[\#](#jax-tree-unflatten "Link to this heading")

jax.tree.unflatten(*treedef*, *leaves*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L301-L329)[\#](#jax.tree.unflatten "Link to this definition")  
Reconstructs a pytree from the treedef and the leaves.

The inverse of `tree_flatten()`.

Parameters:  
- **treedef** (*tree_util.PyTreeDef*) – the treedef to reconstruct

- **leaves** (*Iterable\[tree_util.Leaf\]*) – the iterable of leaves to use for reconstruction. The iterable must match the leaves of the treedef.

Returns:  
The reconstructed pytree, containing the `leaves` placed in the structure described by `treedef`.

Return type:  
Any

Examples

    >>> import jax
    >>> vals, treedef = jax.tree.flatten([1, (2, 3), [4, 5]])
    >>> newvals = [100, 200, 300, 400, 500]
    >>> jax.tree.unflatten(treedef, newvals)
    [100, (200, 300), [400, 500]]

See also

- [`jax.tree.flatten()`](jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten")

- [`jax.tree.leaves()`](jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves")

- [`jax.tree.structure()`](jax.tree.structure.html#jax.tree.structure "jax.tree.structure")

[](jax.tree.transpose.html "previous page")

previous

jax.tree.transpose

[](../jax.tree_util.html "next page")

next

`jax.tree_util` module

Contents

- [`unflatten()`](#jax.tree.unflatten)

By The JAX authors

© Copyright 2024, The JAX Authors.\
