- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.funm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.funm.rst "Download source file")
-  .pdf

# jax.scipy.linalg.funm

## Contents

- [`funm()`](#jax.scipy.linalg.funm)

# jax.scipy.linalg.funm[\#](#jax-scipy-linalg-funm "Link to this heading")

jax.scipy.linalg.funm(*A*, *func*, *disp=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/third_party/scipy/linalg.py#L45-L117)[\#](#jax.scipy.linalg.funm "Link to this definition")  
Evaluate a matrix-valued function

JAX implementation of [`scipy.linalg.funm()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.funm.html#scipy.linalg.funm "(in SciPy v1.19.0.dev)").

Parameters:  
- **A** (*ArrayLike*) – array of shape `(N,`` ``N)` for which the function is to be computed.

- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – Callable object that takes a scalar argument and returns a scalar result. Represents the function to be evaluated over the eigenvalues of A.

- **disp** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If true (default), error information is not returned. Unlike scipy’s version JAX does not attempt to display information at runtime.

- **compute_expm** – (N, N) array_like or None, optional. If provided, the matrix exponential of A. This is used for improving efficiency when func is the exponential function. If not provided, it is computed internally. Defaults to None.

Returns:  
Array of same shape as `A`, containing the result of `func` evaluated on the eigenvalues of `A`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Notes

The returned dtype of JAX’s implementation may differ from that of scipy; specifically, in cases where all imaginary parts of the array values are close to zero, the SciPy function may return a real-valued array, whereas the JAX implementation will return a complex-valued array.

Examples

Applying an arbitrary matrix function:

    >>> A = jnp.array([[1., 2.], [3., 4.]])
    >>> def func(x):
    ...   return jnp.sin(x) + 2 * jnp.cos(x)
    >>> jax.scipy.linalg.funm(A, func)  
    Array([[ 1.2452652 +0.j, -0.3701772 +0.j],
           [-0.55526584+0.j,  0.6899995 +0.j]], dtype=complex64)

Comparing two ways of computing the matrix exponent:

    >>> expA_1 = jax.scipy.linalg.funm(A, jnp.exp)
    >>> expA_2 = jax.scipy.linalg.expm(A)
    >>> jnp.allclose(expA_1, expA_2, rtol=1E-4)
    Array(True, dtype=bool)

[](jax.scipy.linalg.fiedler_companion.html "previous page")

previous

jax.scipy.linalg.fiedler_companion

[](jax.scipy.linalg.hadamard.html "next page")

next

jax.scipy.linalg.hadamard

Contents

- [`funm()`](#jax.scipy.linalg.funm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
