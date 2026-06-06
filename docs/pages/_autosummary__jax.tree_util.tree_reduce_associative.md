- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.tree_reduce_associative

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.tree_reduce_associative.rst "Download source file")
-  .pdf

# jax.tree_util.tree_reduce_associative

## Contents

- [`tree_reduce_associative()`](#jax.tree_util.tree_reduce_associative)

# jax.tree_util.tree_reduce_associative[\#](#jax-tree-util-tree-reduce-associative "Link to this heading")

jax.tree_util.tree_reduce_associative(*operation*, *tree*, *\**, *identity=\<jax.\_src.tree_util.Unspecified object\>*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L471-L482)[\#](#jax.tree_util.tree_reduce_associative "Link to this definition")  
Alias of [`jax.tree.reduce_associative()`](jax.tree.reduce_associative.html#jax.tree.reduce_associative "jax.tree.reduce_associative").

Parameters:  
- **operation** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T,* *T\],* *T\]*)

- **tree** (*Any*)

- **identity** (*T* *\|* *Unspecified*)

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

Return type:  
T

[](jax.tree_util.tree_reduce.html "previous page")

previous

jax.tree_util.tree_reduce

[](jax.tree_util.tree_structure.html "next page")

next

jax.tree_util.tree_structure

Contents

- [`tree_reduce_associative()`](#jax.tree_util.tree_reduce_associative)

By The JAX authors

© Copyright 2024, The JAX Authors.\
