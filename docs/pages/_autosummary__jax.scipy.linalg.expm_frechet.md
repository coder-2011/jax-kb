- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.expm_frechet

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.expm_frechet.rst "Download source file")
-  .pdf

# jax.scipy.linalg.expm_frechet

## Contents

- [`expm_frechet()`](#jax.scipy.linalg.expm_frechet)

# jax.scipy.linalg.expm_frechet[\#](#jax-scipy-linalg-expm-frechet "Link to this heading")

jax.scipy.linalg.expm_frechet(*A: ArrayLike*, *E: ArrayLike*, *\**, *method: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *compute_expm: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1571-L1626)[\#](#jax.scipy.linalg.expm_frechet "Link to this definition")\
jax.scipy.linalg.expm_frechet(*A: ArrayLike*, *E: ArrayLike*, *\**, *method: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *compute_expm: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\]*) → [Array](jax.Array.html#jax.Array "jax.Array")\
jax.scipy.linalg.expm_frechet(*A: ArrayLike*, *E: ArrayLike*, *\**, *method: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *compute_expm: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Compute the Frechet derivative of the matrix exponential.

JAX implementation of [`scipy.linalg.expm_frechet()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.expm_frechet.html#scipy.linalg.expm_frechet "(in SciPy v1.19.0.dev)")

Parameters:  
- **A** – array of shape `(...,`` ``N,`` ``N)`

- **E** – array of shape `(...,`` ``N,`` ``N)`; specifies the direction of the derivative.

- **compute_expm** – if True (default) then compute and return `expm(A)`.

- **method** – ignored by JAX

Returns:  
A tuple `(expm_A,`` ``expm_frechet_AE)` if `compute_expm` is True, else the array `expm_frechet_AE`. Both returned arrays have shape `(...,`` ``N,`` ``N)`.

See also

[`jax.scipy.linalg.expm()`](jax.scipy.linalg.expm.html#jax.scipy.linalg.expm "jax.scipy.linalg.expm")

Examples

We can use this API to compute the matrix exponential of `A`, as well as its derivative in the direction `E`:

    >>> key1, key2 = jax.random.split(jax.random.key(3372))
    >>> A = jax.random.normal(key1, (3, 3))
    >>> E = jax.random.normal(key2, (3, 3))
    >>> expmA, expm_frechet_AE = jax.scipy.linalg.expm_frechet(A, E)

This can be equivalently computed using JAX’s automatic differentiation methods; here we’ll compute the derivative of [`expm()`](jax.scipy.linalg.expm.html#jax.scipy.linalg.expm "jax.scipy.linalg.expm") in the direction of `E` using [`jax.jvp()`](jax.jvp.html#jax.jvp "jax.jvp"), and find the same results:

    >>> expmA2, expm_frechet_AE2 = jax.jvp(jax.scipy.linalg.expm, (A,), (E,))
    >>> jnp.allclose(expmA, expmA2)
    Array(True, dtype=bool)
    >>> jnp.allclose(expm_frechet_AE, expm_frechet_AE2)
    Array(True, dtype=bool)

[](jax.scipy.linalg.expm.html "previous page")

previous

jax.scipy.linalg.expm

[](jax.scipy.linalg.fiedler.html "next page")

next

jax.scipy.linalg.fiedler

Contents

- [`expm_frechet()`](#jax.scipy.linalg.expm_frechet)

By The JAX authors

© Copyright 2024, The JAX Authors.\
