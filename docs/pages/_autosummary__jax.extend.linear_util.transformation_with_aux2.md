- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.linear_util` module](../jax.extend.linear_util.html)
- jax.extend.linear_util.transformation_with_aux2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.linear_util.transformation_with_aux2.rst "Download source file")
-  .pdf

# jax.extend.linear_util.transformation_with_aux2

## Contents

- [`transformation_with_aux2`](#jax.extend.linear_util.transformation_with_aux2)

# jax.extend.linear_util.transformation_with_aux2[\#](#jax-extend-linear-util-transformation-with-aux2 "Link to this heading")

jax.extend.linear_util.transformation_with_aux2 *= functools.partial(\<class 'functools.partial'\>, \<function transformation_with_aux2\>)*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/linear_util.py#L269-L279)[\#](#jax.extend.linear_util.transformation_with_aux2 "Link to this definition")  
Adds one more transformation with auxiliary output to a WrappedFun.

Parameters:  
- **fun** ([*WrappedFun*](jax.extend.linear_util.WrappedFun.html#jax.extend.linear_util.WrappedFun "jax.extend.linear_util.WrappedFun"))

- **use_eq_store** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **unk_names** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[WrappedFun](jax.extend.linear_util.WrappedFun.html#jax.extend.linear_util.WrappedFun "jax.extend.linear_util.WrappedFun"), [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[\], Any\]\]

[](jax.extend.linear_util.transformation_with_aux.html "previous page")

previous

jax.extend.linear_util.transformation_with_aux

[](jax.extend.linear_util.wrap_init.html "next page")

next

jax.extend.linear_util.wrap_init

Contents

- [`transformation_with_aux2`](#jax.extend.linear_util.transformation_with_aux2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
