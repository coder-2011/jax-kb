- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.transpose.rst "Download source file")
-  .pdf

# jax.tree.transpose

## Contents

- [`transpose()`](#jax.tree.transpose)

# jax.tree.transpose[\#](#jax-tree-transpose "Link to this heading")

jax.tree.transpose(*outer_treedef*, *inner_treedef*, *pytree_to_transpose*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L270-L299)[\#](#jax.tree.transpose "Link to this definition")  
Transform a tree having tree structure (outer, inner) into one having structure (inner, outer).

Parameters:  
- **outer_treedef** (*tree_util.PyTreeDef*) – PyTreeDef representing the outer tree.

- **inner_treedef** (*tree_util.PyTreeDef* *\|* *None*) – PyTreeDef representing the inner tree. If None, then it will be inferred from outer_treedef and the structure of pytree_to_transpose.

- **pytree_to_transpose** (*Any*) – the pytree to be transposed.

Returns:  
the transposed pytree.

Return type:  
transposed_pytree

Examples

    >>> import jax
    >>> tree = [(1, 2, 3), (4, 5, 6)]
    >>> inner_structure = jax.tree.structure(('*', '*', '*'))
    >>> outer_structure = jax.tree.structure(['*', '*'])
    >>> jax.tree.transpose(outer_structure, inner_structure, tree)
    ([1, 4], [2, 5], [3, 6])

Inferring the inner structure:

    >>> jax.tree.transpose(outer_structure, None, tree)
    ([1, 4], [2, 5], [3, 6])

[](jax.tree.structure.html "previous page")

previous

jax.tree.structure

[](jax.tree.unflatten.html "next page")

next

jax.tree.unflatten

Contents

- [`transpose()`](#jax.tree.transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
