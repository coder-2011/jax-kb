- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.dot_product_attention

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.dot_product_attention.rst "Download source file")
-  .pdf

# jax.nn.dot_product_attention

## Contents

- [`dot_product_attention()`](#jax.nn.dot_product_attention)

# jax.nn.dot_product_attention[\#](#jax-nn-dot-product-attention "Link to this heading")

jax.nn.dot_product_attention(*query: ArrayLike*, *key: ArrayLike*, *value: ArrayLike*, *bias: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *mask: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *\**, *scale: [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *is_causal: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *query_seq_lengths: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *key_value_seq_lengths: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *local_window_size: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *implementation: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['xla', 'cudnn'\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *return_residual: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*) → [Array](jax.Array.html#jax.Array "jax.Array")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L1066-L1267)[\#](#jax.nn.dot_product_attention "Link to this definition")\
jax.nn.dot_product_attention(*query: ArrayLike*, *key: ArrayLike*, *value: ArrayLike*, *bias: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *mask: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *\**, *scale: [float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *is_causal: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *query_seq_lengths: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *key_value_seq_lengths: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *local_window_size: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *implementation: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['xla', 'cudnn'\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *return_residual: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = False*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Scaled dot product attention function.

Computes the following for each head:

\\\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{QK^T}{\sqrt{d}} + B \right) V\\

where \\Q\\ is the query matrix, \\K\\ is the key matrix, \\V\\ is the value matrix, \\d\\ is the dimension of each individual query and key, and \\B\\ is the bias matrix (optional).

Throughout this function, we utilize the following uppercase letters to represent the shape of array:

    B = batch size
    S = length of the key/value (source)
    T = length of the query (target)
    N = number of attention heads
    H = dimensions of each attention head
    K = number of key/value heads
    G = number of groups, which equals to N // K

Parameters:  
- **query** – query array; shape `(BTNH|TNH)`

- **key** – key array: shape `(BSKH|SKH)`. When K equals N, multi-headed attention (MHA [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)) is performed. Otherwise, grouped query attention (GQA [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)) is performed if N is a multiple of K, and multi-query attention (MQA [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)) is performed if K == 1 (a special case of GQA).

- **value** – value array, should have the same shape as the key array.

- **bias** – optional, bias array to be added to logits; The shape must be 4D and be broadcastable to `(BNTS|NTS)`.

- **mask** – optional, mask array used to filter out logits. It is a boolean mask where True indicates the element should take part in attention. For an additive mask, users should pass it to bias. The shape must be 4D and be broadcastable to `(BNTS|NTS)`.

- **scale** – scale for the logits. If None, the scale will be set to 1 divided by the square root of query’s head dimension (i.e. H).

- **is_causal** – If true, causal attention will be applied. Note, some implementations like xla will generate a mask tensor and apply it to the logits to mask out the non-causal parts of the attention matrix, but other implementations like cudnn will avoid computing the non-causal regions, providing speedups.

- **query_seq_lengths** – int32 array of sequence lengths for query; shape `(B)`

- **key_value_seq_lengths** – int32 array of sequence lengths for key and value; shape `(B)`

- **local_window_size** – Window sizes to make self attention to attend to each token’s local window. If set, this specifies the (left_window_size, right_window_size) for each token. E.g., if local_window_size == (3, 2) and the sequence is \[0, 1, 2, 3, 4, 5, c, 7, 8, 9\], token c can attend to \[3, 4, 5, c, 7, 8\]. If a single int is given, it will be interpreted as a symmetric window (window_size, window_size).

- **return_residual** – Whether to return the logsumexp tensor of shape BTN or BNT to users. See section 3.1.1 in the FlashAttention-2 paper: [https://arxiv.org/pdf/2307.08691](https://arxiv.org/pdf/2307.08691) to find the definition of logsumexp.

- **implementation** – A string to control which implementation backend to use. Supported strings are xla, cudnn (cuDNN flash attention). It defaults to None, which currently falls back to xla. Note, cudnn supports only a subset of shapes/dtypes, and an exception will be thrown if its not supported.

Returns:  
If return_residual is False, returns an array of the attention output with the same shape as `query`. If return_residual is True, returns a tuple of (output, residual). The residual is the shape of BTN\|TN.

[](jax.nn.one_hot.html "previous page")

previous

jax.nn.one_hot

[](jax.nn.scaled_matmul.html "next page")

next

jax.nn.scaled_matmul

Contents

- [`dot_product_attention()`](#jax.nn.dot_product_attention)

By The JAX authors

© Copyright 2024, The JAX Authors.\
