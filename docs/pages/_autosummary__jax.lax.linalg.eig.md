- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.eig

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.eig.rst "Download source file")
-  .pdf

# jax.lax.linalg.eig

## Contents

- [`eig()`](#jax.lax.linalg.eig)

# jax.lax.linalg.eig[\#](#jax-lax-linalg-eig "Link to this heading")

jax.lax.linalg.eig(*x*, *\**, *compute_left_eigenvectors=True*, *compute_right_eigenvectors=True*, *enable_eigvec_derivs=False*, *implementation=None*, *use_magma=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L137-L227)[\#](#jax.lax.linalg.eig "Link to this definition")  
Eigendecomposition of a general matrix.

Nonsymmetric eigendecomposition is only implemented on CPU and GPU. On GPU, the default implementation calls LAPACK directly on the host CPU, but an experimental GPU implementation using [MAGMA](https://icl.utk.edu/magma/) is also available. The MAGMA implementation is typically slower than the equivalent LAPACK implementation for small matrices (less than about 2048), but it may perform better for larger matrices.

To enable the MAGMA implementation, you must install MAGMA yourself (there are Debian and conda-forge packages, or you can build from source). Then set the `use_magma` argument to `True`, or set the `jax_use_magma` configuration variable to `"on"` or `"auto"`:

    jax.config.update('jax_use_magma', 'on')

JAX will try to `dlopen` the installed MAGMA shared library, raising an error if it is not found. To explicitly specify the path to the MAGMA library, set the environment variable JAX_GPU_MAGMA_PATH to the full installation path.

If `jax_use_magma` is set to `"auto"`, the MAGMA implementation will be used if the library can be found, and the input matrix is sufficiently large (\>= 2048x2048).

Parameters:  
- **x** (*ArrayLike*) – A batch of square matrices with shape `[...,`` ``n,`` ``n]`.

- **compute_left_eigenvectors** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If true, the left eigenvectors will be computed.

- **compute_right_eigenvectors** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If true, the right eigenvectors will be computed.

- **enable_eigvec_derivs** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If true, enable autodiff of the returned eigenvectors. The eigenvector derivative is taken under the LAPACK `geev` normalisation (each eigenvector has unit 2-norm and its largest-magnitude component is real). It is only valid when (i) all eigenvalues are distinct and (ii) no eigenvector has two components tied for largest magnitude. Defaults to `False` because these conditions cannot be checked statically; see [jax-ml/jax#2748](https://github.com/jax-ml/jax/issues/2748) for discussion.

- **use_magma** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – Deprecated, please use `implementation` instead. Locally override the `jax_use_magma` flag. If `True`, the eigendecomposition is computed using MAGMA. If `False`, the computation is done using LAPACK on to the host CPU. If `None` (default), the behavior is controlled by the `jax_use_magma` flag. This argument is only used on GPU. Will be removed in JAX 0.9.

- **implementation** ([*EigImplementation*](../jax.lax.html#jax.lax.linalg.EigImplementation "jax.lax.linalg.EigImplementation") *\|* *None*) – Controls the choice of eigendecomposition algorithm. If

- **LAPACK** – If `MAGMA`, the computation will be performed using the MAGMA library on the GPU. If `CUSOLVER`, the computation will be performed using the Cusolver library on the GPU. The `CUSOLVER` implementation requires Cusolver 11.7.1 (from CUDA 12.6 update 2) to be installed, and does not support computing left eigenvectors. If `None` (default), an automatic choice will be made, depending on the Cusolver version, whether left eigenvectors were requested, and the `jax_use_magma` configuration variable.

- **CPU.** (*the computation will be performed using LAPACK on the host*) – If `MAGMA`, the computation will be performed using the MAGMA library on the GPU. If `CUSOLVER`, the computation will be performed using the Cusolver library on the GPU. The `CUSOLVER` implementation requires Cusolver 11.7.1 (from CUDA 12.6 update 2) to be installed, and does not support computing left eigenvectors. If `None` (default), an automatic choice will be made, depending on the Cusolver version, whether left eigenvectors were requested, and the `jax_use_magma` configuration variable.

Returns:  
The eigendecomposition of `x`, which is a tuple of the form `(w,`` ``vl,`` ``vr)` where `w` are the eigenvalues, `vl` are the left eigenvectors, and `vr` are the right eigenvectors. `vl` and `vr` are optional and will only be included if `compute_left_eigenvectors` or `compute_right_eigenvectors` respectively are `True`.

If the eigendecomposition fails, then arrays full of NaNs will be returned for that batch element.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.linalg.cholesky_update.html "previous page")

previous

jax.lax.linalg.cholesky_update

[](jax.lax.linalg.eigh.html "next page")

next

jax.lax.linalg.eigh

Contents

- [`eig()`](#jax.lax.linalg.eig)

By The JAX authors

© Copyright 2024, The JAX Authors.\
