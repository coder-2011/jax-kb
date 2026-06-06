- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.approx_max_k

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.approx_max_k.rst "Download source file")
-  .pdf

# jax.lax.approx_max_k

## Contents

- [`approx_max_k()`](#jax.lax.approx_max_k)

# jax.lax.approx_max_k[\#](#jax-lax-approx-max-k "Link to this heading")

jax.lax.approx_max_k(*operand*, *k*, *reduction_dimension=-1*, *recall_target=0.95*, *reduction_input_size_override=-1*, *aggregate_to_topk=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/ann.py#L92-L149)[\#](#jax.lax.approx_max_k "Link to this definition")  
Returns max `k` values and their indices of the `operand` in an approximate manner.

See [https://arxiv.org/abs/2206.14286](https://arxiv.org/abs/2206.14286) for the algorithm details.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Array to search for max-k. Must be a floating number type.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Specifies the number of max-k.

- **reduction_dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Integer dimension along which to search. Default: -1.

- **recall_target** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – Recall target for the approximation.

- **reduction_input_size_override** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – When set to a positive value, it overrides the size determined by `operand[reduction_dim]` for evaluating the recall. This option is useful when the given `operand` is only a subset of the overall computation in SPMD or distributed pipelines, where the true input size cannot be deferred by the operand shape.

- **aggregate_to_topk** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – When true, aggregates approximate results to the top-k in sorted order. When false, returns the approximate results unsorted. In this case, the number of the approximate results is implementation defined and is greater or equal to the specified `k`.

Returns:  
Tuple of two arrays. The arrays are the max `k` values and the corresponding indices along the `reduction_dimension` of the input `operand`. The arrays’ dimensions are the same as the input `operand` except for the `reduction_dimension`: when `aggregate_to_topk` is true, the reduction dimension is `k`; otherwise, it is greater equals to `k` where the size is implementation-defined.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[*Array*](jax.Array.html#jax.Array "jax.Array"), [*Array*](jax.Array.html#jax.Array "jax.Array")\]

We encourage users to wrap `approx_max_k` with jit. See the following example for maximal inner production search (MIPS):

    >>> import functools
    >>> import jax
    >>> import numpy as np
    >>> @jax.jit(static_argnames=["k", "recall_target"])
    ... def mips(qy, db, k=10, recall_target=0.95):
    ...   dists = jax.lax.dot(qy, db.transpose())
    ...   # returns (f32[qy_size, k], i32[qy_size, k])
    ...   return jax.lax.approx_max_k(dists, k=k, recall_target=recall_target)
    >>>
    >>> qy = jax.numpy.array(np.random.rand(50, 64))
    >>> db = jax.numpy.array(np.random.rand(1024, 64))
    >>> dot_products, neighbors = mips(qy, db, k=10)

[](jax.lax.after_all.html "previous page")

previous

jax.lax.after_all

[](jax.lax.approx_min_k.html "next page")

next

jax.lax.approx_min_k

Contents

- [`approx_max_k()`](#jax.lax.approx_max_k)

By The JAX authors

© Copyright 2024, The JAX Authors.\
