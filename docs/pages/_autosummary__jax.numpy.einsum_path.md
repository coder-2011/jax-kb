- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.einsum_path

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.einsum_path.rst "Download source file")
-  .pdf

# jax.numpy.einsum_path

## Contents

- [`einsum_path()`](#jax.numpy.einsum_path)

# jax.numpy.einsum_path[\#](#jax-numpy-einsum-path "Link to this heading")

jax.numpy.einsum_path(*subscripts*, */*, *\*operands*, *optimize='auto'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/einsum.py#L370-L429)[\#](#jax.numpy.einsum_path "Link to this definition")  
Evaluates the optimal contraction path without evaluating the einsum.

JAX implementation of [`numpy.einsum_path()`](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html#numpy.einsum_path "(in NumPy v2.4)"). This function calls into the [opt_einsum](https://github.com/dgasmith/opt_einsum) package, and makes use of its optimization routines.

Parameters:  
- **subscripts** – string containing axes names separated by commas.

- **\*operands** – sequence of one or more arrays corresponding to the subscripts.

- **optimize** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]\]*) – specify how to optimize the order of computation. In JAX this defaults to `"auto"`. Other options are `True` (same as `"optimize"`), `False` (unoptimized), or any string supported by `opt_einsum`, which includes `"optimize"`,, `"greedy"`, `"eager"`, and others.

Returns:  
A tuple containing the path that may be passed to [`einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"), and a printable object representing this optimal path.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), …\]\], [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")\]

Examples

    >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    >>> x = jax.random.randint(key1, minval=-5, maxval=5, shape=(2, 3))
    >>> y = jax.random.randint(key2, minval=-5, maxval=5, shape=(3, 100))
    >>> z = jax.random.randint(key3, minval=-5, maxval=5, shape=(100, 5))
    >>> path, path_info = jnp.einsum_path("ij,jk,kl", x, y, z, optimize="optimal")
    >>> print(path)
    [(1, 2), (0, 1)]
    >>> print(path_info)
          Complete contraction:  ij,jk,kl->il
                Naive scaling:  4
            Optimized scaling:  3
              Naive FLOP count:  9.000e+3
          Optimized FLOP count:  3.060e+3
          Theoretical speedup:  2.941e+0
          Largest intermediate:  1.500e+1 elements
        --------------------------------------------------------------------------------
        scaling        BLAS                current                             remaining
        --------------------------------------------------------------------------------
          3           GEMM              kl,jk->lj                             ij,lj->il
          3           GEMM              lj,ij->il                                il->il

Use the computed path in [`einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"):

    >>> jnp.einsum("ij,jk,kl", x, y, z, optimize=path)
    Array([[-754,  324, -142,   82,   50],
           [ 408,  -50,   87,  -29,    7]], dtype=int32)

[](jax.numpy.einsum.html "previous page")

previous

jax.numpy.einsum

[](jax.numpy.empty.html "next page")

next

jax.numpy.empty

Contents

- [`einsum_path()`](#jax.numpy.einsum_path)

By The JAX authors

© Copyright 2024, The JAX Authors.\
