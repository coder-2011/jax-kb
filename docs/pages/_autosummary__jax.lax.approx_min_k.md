- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.approx_min_k

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.approx_min_k.rst "Download source file")
-  .pdf

# jax.lax.approx_min_k

## Contents

- [`approx_min_k()`](#jax.lax.approx_min_k)

# jax.lax.approx_min_k[\#](#jax-lax-approx-min-k "Link to this heading")

jax.lax.approx_min_k(*operand*, *k*, *reduction_dimension=-1*, *recall_target=0.95*, *reduction_input_size_override=-1*, *aggregate_to_topk=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/ann.py#L151-L212)[\#](#jax.lax.approx_min_k "Link to this definition")  
Returns min `k` values and their indices of the `operand` in an approximate manner.

See [https://arxiv.org/abs/2206.14286](https://arxiv.org/abs/2206.14286) for the algorithm details.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Array to search for min-k. Must be a floating number type.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Specifies the number of min-k.

- **reduction_dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Integer dimension along which to search. Default: -1.

- **recall_target** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – Recall target for the approximation.

- **reduction_input_size_override** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – When set to a positive value, it overrides the size determined by `operand[reduction_dim]` for evaluating the recall. This option is useful when the given operand is only a subset of the overall computation in SPMD or distributed pipelines, where the true input size cannot be deferred by the `operand` shape.

- **aggregate_to_topk** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – When true, aggregates approximate results to the top-k in sorted order. When false, returns the approximate results unsorted. In this case, the number of the approximate results is implementation defined and is greater or equal to the specified `k`.

Returns:  
Tuple of two arrays. The arrays are the least `k` values and the corresponding indices along the `reduction_dimension` of the input `operand`. The arrays’ dimensions are the same as the input `operand` except for the `reduction_dimension`: when `aggregate_to_topk` is true, the reduction dimension is `k`; otherwise, it is greater equals to `k` where the size is implementation-defined.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[*Array*](jax.Array.html#jax.Array "jax.Array"), [*Array*](jax.Array.html#jax.Array "jax.Array")\]

We encourage users to wrap `approx_min_k` with jit. See the following example for nearest neighbor search over the squared l2 distance:

    >>> import functools
    >>> import jax
    >>> import numpy as np
    >>> @jax.jit(static_argnames=["k", "recall_target"])
    ... def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
    ...   dists = half_db_norms - jax.lax.dot(qy, db.transpose())
    ...   return jax.lax.approx_min_k(dists, k=k, recall_target=recall_target)
    >>>
    >>> qy = jax.numpy.array(np.random.rand(50, 64))
    >>> db = jax.numpy.array(np.random.rand(1024, 64))
    >>> half_db_norm_sq = jax.numpy.linalg.norm(db, axis=1)**2 / 2
    >>> dists, neighbors = l2_ann(qy, db, half_db_norm_sq, k=10)

In the example above, we compute `db^2/2`` ``-`` ``dot(qy,`` ``db^T)` instead of `qy^2`` ``-`` ``2`` ``dot(qy,`` ``db^T)`` ``+`` ``db^2` for performance reason. The former uses less arithmetic and produces the same set of neighbors.

[](jax.lax.approx_max_k.html "previous page")

previous

jax.lax.approx_max_k

[](jax.lax.argmax.html "next page")

next

jax.lax.argmax

Contents

- [`approx_min_k()`](#jax.lax.approx_min_k)

By The JAX authors

© Copyright 2024, The JAX Authors.\
