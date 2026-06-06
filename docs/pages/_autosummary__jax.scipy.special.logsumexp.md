- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.logsumexp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.logsumexp.rst "Download source file")
-  .pdf

# jax.scipy.special.logsumexp

## Contents

- [`logsumexp()`](#jax.scipy.special.logsumexp)

# jax.scipy.special.logsumexp[\#](#jax-scipy-special-logsumexp "Link to this heading")

jax.scipy.special.logsumexp(*a: ArrayLike*, *axis: Axis = None*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *keepdims: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *return_sign: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *where: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Array](jax.Array.html#jax.Array "jax.Array")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ops/special.py#L46-L105)[\#](#jax.scipy.special.logsumexp "Link to this definition")\
jax.scipy.special.logsumexp(*a: ArrayLike*, *axis: Axis = None*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *keepdims: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *\**, *return_sign: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *where: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.special.logsumexp(*a: ArrayLike*, *axis: Axis = None*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *keepdims: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *return_sign: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *where: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Log-sum-exp reduction.

JAX implementation of [`scipy.special.logsumexp()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.logsumexp.html#scipy.special.logsumexp "(in SciPy v1.19.0.dev)").

\\\operatorname{logsumexp} a = \log \sum_i b_i \exp a_i\\

where the \\i\\ indices range over one or more dimensions to be reduced.

Parameters:  
- **a** – the input array

- **axis** – int or sequence of ints, default=None. Axis along which the sum to be computed. If None, the sum is computed along all the axes.

- **b** – scaling factors for the exponentials. Must be broadcastable to the shape of a.

- **keepdims** – If `True`, the axes that are reduced are left in the output as dimensions of size 1.

- **return_sign** – If `True`, the output will be a `(result,`` ``sign)` pair, where `sign` is the sign of the sums and `result` contains the logarithms of their absolute values. If `False` only `result` is returned and it will contain NaN values if the sums are negative.

- **where** – Elements to include in the reduction.

Returns:  
Either an array `result` or a pair of arrays `(result,`` ``sign)`, depending on the value of the `return_sign` argument.

See also

[`jax.nn.logmeanexp()`](jax.nn.logmeanexp.html#jax.nn.logmeanexp "jax.nn.logmeanexp")

[](jax.scipy.special.loggamma.html "previous page")

previous

jax.scipy.special.loggamma

[](jax.scipy.special.lpmn.html "next page")

next

jax.scipy.special.lpmn

Contents

- [`logsumexp()`](#jax.scipy.special.logsumexp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
