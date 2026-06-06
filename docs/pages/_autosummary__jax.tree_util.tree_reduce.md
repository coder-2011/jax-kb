- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.tree_reduce

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.tree_reduce.rst "Download source file")
-  .pdf

# jax.tree_util.tree_reduce

## Contents

- [`tree_reduce()`](#jax.tree_util.tree_reduce)

# jax.tree_util.tree_reduce[\#](#jax-tree-util-tree-reduce "Link to this heading")

jax.tree_util.tree_reduce(*function*, *tree*, *initializer=\<jax.\_src.tree_util.Unspecified object\>*, *is_leaf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L440-L450)[\#](#jax.tree_util.tree_reduce "Link to this definition")  
Alias of [`jax.tree.reduce()`](jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce").

Parameters:  
- **function** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T,* *Any\],* *T\]*)

- **tree** (*Any*)

- **initializer** (*T* *\|* *Unspecified*)

- **is_leaf** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any\],* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]* *\|* *None*)

Return type:  
T

[](jax.tree_util.tree_map.html "previous page")

previous

jax.tree_util.tree_map

[](jax.tree_util.tree_reduce_associative.html "next page")

next

jax.tree_util.tree_reduce_associative

Contents

- [`tree_reduce()`](#jax.tree_util.tree_reduce)

By The JAX authors

© Copyright 2024, The JAX Authors.\
