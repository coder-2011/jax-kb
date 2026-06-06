- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.orthogonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.orthogonal.rst "Download source file")
-  .pdf

# jax.random.orthogonal

## Contents

- [`orthogonal()`](#jax.random.orthogonal)

# jax.random.orthogonal[\#](#jax-random-orthogonal "Link to this heading")

jax.random.orthogonal(*key*, *n*, *shape=()*, *dtype=None*, *m=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2641-L2697)[\#](#jax.random.orthogonal "Link to this definition")  
Sample uniformly from the orthogonal group O(n).

If the dtype is complex, sample uniformly from the unitary group U(n).

For unequal rows and columns, this samples a semi-orthogonal matrix instead. That is, if \\A\\ is the resulting matrix and \\A^\*\\ is its conjugate transpose, then:

- If \\n \leq m\\, the rows are mutually orthonormal: \\A A^\* = I_n\\.

- If \\m \leq n\\, the columns are mutually orthonormal: \\A^\* A = I_m\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – an integer indicating the number of rows.

- **shape** (*Shape*) – optional, the batch dimensions of the result. Default ().

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **m** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – an integer indicating the number of columns. Defaults to n.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array of shape (\*shape, n, m) and specified dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

References

\[1\]

Mezzadri, Francesco. (2007). “How to generate random matrices from the classical compact groups”. Notices of the American Mathematical Society, 54(5), 592-604. [https://arxiv.org/abs/math-ph/0609050](https://arxiv.org/abs/math-ph/0609050).

[](jax.random.normal.html "previous page")

previous

jax.random.normal

[](jax.random.pareto.html "next page")

next

jax.random.pareto

Contents

- [`orthogonal()`](#jax.random.orthogonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
