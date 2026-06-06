- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.value_and_grad

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.value_and_grad.rst "Download source file")
-  .pdf

# jax.experimental.sparse.value_and_grad

## Contents

- [`value_and_grad()`](#jax.experimental.sparse.value_and_grad)

# jax.experimental.sparse.value_and_grad[\#](#jax-experimental-sparse-value-and-grad "Link to this heading")

jax.experimental.sparse.value_and_grad(*fun*, *argnums=0*, *has_aux=False*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/ad.py#L80-L106)[\#](#jax.experimental.sparse.value_and_grad "Link to this definition")  
Sparse-aware version of [`jax.value_and_grad()`](jax.value_and_grad.html#jax.value_and_grad "jax.value_and_grad")

Arguments and return values are the same as [`jax.value_and_grad()`](jax.value_and_grad.html#jax.value_and_grad "jax.value_and_grad"), but when taking the gradient with respect to a [`jax.experimental.sparse`](../jax.experimental.sparse.html#module-jax.experimental.sparse "jax.experimental.sparse") array, the gradient is computed in the subspace defined by the array’s sparsity pattern.

Examples

    >>> from jax.experimental import sparse
    >>> X = sparse.BCOO.fromdense(jnp.arange(6.))
    >>> y = jnp.ones(6)
    >>> sparse.value_and_grad(lambda X, y: X @ y)(X, y)
    (Array(15., dtype=float32), BCOO(float32[6], nse=5))

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"))

- **argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, Any\]\]

[](jax.experimental.sparse.grad.html "previous page")

previous

jax.experimental.sparse.grad

[](jax.experimental.sparse.empty.html "next page")

next

jax.experimental.sparse.empty

Contents

- [`value_and_grad()`](#jax.experimental.sparse.value_and_grad)

By The JAX authors

© Copyright 2024, The JAX Authors.\
