- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.random_bcoo

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.random_bcoo.rst "Download source file")
-  .pdf

# jax.experimental.sparse.random_bcoo

## Contents

- [`random_bcoo()`](#jax.experimental.sparse.random_bcoo)

# jax.experimental.sparse.random_bcoo[\#](#jax-experimental-sparse-random-bcoo "Link to this heading")

jax.experimental.sparse.random_bcoo(*key*, *shape*, *\**, *dtype=\<class 'jax.numpy.float64'\>*, *indices_dtype=None*, *nse=0.2*, *n_batch=0*, *n_dense=0*, *unique_indices=True*, *sorted_indices=False*, *generator=\<function uniform\>*, *\*\*kwds*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/random.py#L30-L103)[\#](#jax.experimental.sparse.random_bcoo "Link to this definition")  
Generate a random BCOO matrix.

Parameters:  
- **key** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – PRNG key to be passed to `generator` function.

- **shape** ([*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – tuple specifying the shape of the array to be generated.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType*) – dtype of the array to be generated.

- **indices_dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – dtype of the BCOO indices.

- **nse** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of specified elements in the matrix, or if 0 \< nse \< 1, a fraction of sparse dimensions to be specified (default: 0.2).

- **n_batch** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of batch dimensions. must satisfy `n_batch`` ``>=`` ``0` and `n_batch`` ``+`` ``n_dense`` ``<=`` ``len(shape)`.

- **n_dense** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of batch dimensions. must satisfy `n_dense`` ``>=`` ``0` and `n_batch`` ``+`` ``n_dense`` ``<=`` ``len(shape)`.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether indices should be unique (default: True).

- **sorted_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether indices should be row-sorted in lexicographical order (default: False).

- **generator** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[...\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – function for generating random values accepting a key, shape, and dtype. It defaults to [`jax.random.uniform()`](jax.random.uniform.html#jax.random.uniform "jax.random.uniform"), and may be any function with a similar signature.

- **\*\*kwds** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – additional keyword arguments to pass to `generator`.

Returns:  
a sparse.BCOO array with the specified properties.

Return type:  
arr

[](jax.experimental.sparse.todense.html "previous page")

previous

jax.experimental.sparse.todense

[](jax.experimental.sparse.JAXSparse.html "next page")

next

jax.experimental.sparse.JAXSparse

Contents

- [`random_bcoo()`](#jax.experimental.sparse.random_bcoo)

By The JAX authors

© Copyright 2024, The JAX Authors.\
