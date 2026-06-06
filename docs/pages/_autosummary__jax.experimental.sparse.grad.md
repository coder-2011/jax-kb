- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.grad

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.grad.rst "Download source file")
-  .pdf

# jax.experimental.sparse.grad

## Contents

- [`grad()`](#jax.experimental.sparse.grad)

# jax.experimental.sparse.grad[\#](#jax-experimental-sparse-grad "Link to this heading")

jax.experimental.sparse.grad(*fun*, *argnums=0*, *has_aux=False*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/ad.py#L108-L136)[\#](#jax.experimental.sparse.grad "Link to this definition")  
Sparse-aware version of [`jax.grad()`](jax.grad.html#jax.grad "jax.grad")

Arguments and return values are the same as [`jax.grad()`](jax.grad.html#jax.grad "jax.grad"), but when taking the gradient with respect to a [`jax.experimental.sparse`](../jax.experimental.sparse.html#module-jax.experimental.sparse "jax.experimental.sparse") array, the gradient is computed in the subspace defined by the array’s sparsity pattern.

Examples

    >>> from jax.experimental import sparse
    >>> X = sparse.BCOO.fromdense(jnp.arange(6.))
    >>> y = jnp.ones(6)
    >>> sparse.grad(lambda X, y: X @ y)(X, y)
    BCOO(float32[6], nse=5)

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"))

- **argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")

[](jax.experimental.sparse.sparsify.html "previous page")

previous

jax.experimental.sparse.sparsify

[](jax.experimental.sparse.value_and_grad.html "next page")

next

jax.experimental.sparse.value_and_grad

Contents

- [`grad()`](#jax.experimental.sparse.grad)

By The JAX authors

© Copyright 2024, The JAX Authors.\
