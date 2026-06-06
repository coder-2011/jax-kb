- [](index.html)
- [API Reference](jax.html)
- `jax.scipy` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.scipy.rst "Download source file")
-  .pdf

# jax.scipy module

## Contents

- [jax.scipy.cluster](#module-jax.scipy.cluster.vq)
- [jax.scipy.fft](#module-jax.scipy.fft)
- [jax.scipy.integrate](#module-jax.scipy.integrate)
- [jax.scipy.interpolate](#module-jax.scipy.interpolate)
- [jax.scipy.linalg](#module-jax.scipy.linalg)
- [jax.scipy.ndimage](#module-jax.scipy.ndimage)
- [jax.scipy.optimize](#module-jax.scipy.optimize)
- [jax.scipy.signal](#module-jax.scipy.signal)
- [jax.scipy.spatial.transform](#module-jax.scipy.spatial.transform)
- [jax.scipy.sparse.linalg](#module-jax.scipy.sparse.linalg)
- [jax.scipy.special](#module-jax.scipy.special)
- [jax.scipy.stats](#module-jax.scipy.stats)
  - [jax.scipy.stats.bernoulli](#module-jax.scipy.stats.bernoulli)
  - [jax.scipy.stats.beta](#module-jax.scipy.stats.beta)
  - [jax.scipy.stats.betabinom](#module-jax.scipy.stats.betabinom)
  - [jax.scipy.stats.binom](#module-jax.scipy.stats.binom)
  - [jax.scipy.stats.cauchy](#module-jax.scipy.stats.cauchy)
  - [jax.scipy.stats.chi2](#module-jax.scipy.stats.chi2)
  - [jax.scipy.stats.dirichlet](#module-jax.scipy.stats.dirichlet)
  - [jax.scipy.stats.expon](#module-jax.scipy.stats.expon)
  - [jax.scipy.stats.gamma](#module-jax.scipy.stats.gamma)
  - [jax.scipy.stats.gumbel_l](#module-jax.scipy.stats.gumbel_l)
  - [jax.scipy.stats.gumbel_r](#module-jax.scipy.stats.gumbel_r)
  - [jax.scipy.stats.gennorm](#module-jax.scipy.stats.gennorm)
  - [jax.scipy.stats.geom](#module-jax.scipy.stats.geom)
  - [jax.scipy.stats.laplace](#module-jax.scipy.stats.laplace)
  - [jax.scipy.stats.logistic](#module-jax.scipy.stats.logistic)
  - [jax.scipy.stats.multinomial](#module-jax.scipy.stats.multinomial)
  - [jax.scipy.stats.multivariate_normal](#module-jax.scipy.stats.multivariate_normal)
  - [jax.scipy.stats.nbinom](#module-jax.scipy.stats.nbinom)
  - [jax.scipy.stats.norm](#module-jax.scipy.stats.norm)
  - [jax.scipy.stats.pareto](#module-jax.scipy.stats.pareto)
  - [jax.scipy.stats.poisson](#module-jax.scipy.stats.poisson)
  - [jax.scipy.stats.t](#module-jax.scipy.stats.t)
  - [jax.scipy.stats.truncnorm](#module-jax.scipy.stats.truncnorm)
  - [jax.scipy.stats.uniform](#module-jax.scipy.stats.uniform)
  - [jax.scipy.stats.gaussian_kde](#jax-scipy-stats-gaussian-kde)
  - [jax.scipy.stats.vonmises](#module-jax.scipy.stats.vonmises)
  - [jax.scipy.stats.wrapcauchy](#module-jax.scipy.stats.wrapcauchy)

# `jax.scipy` module[\#](#jax-scipy-module "Link to this heading")

## jax.scipy.cluster[\#](#module-jax.scipy.cluster.vq "Link to this heading")

|  |  |
|----|----|
| [`vq`](_autosummary/jax.scipy.cluster.vq.vq.html#jax.scipy.cluster.vq.vq "jax.scipy.cluster.vq.vq")(obs, code_book\[, check_finite\]) | Assign codes from a code book to a set of observations. |

## jax.scipy.fft[\#](#module-jax.scipy.fft "Link to this heading")

|  |  |
|----|----|
| [`dct`](_autosummary/jax.scipy.fft.dct.html#jax.scipy.fft.dct "jax.scipy.fft.dct")(x\[, type, n, axis, norm\]) | Computes the discrete cosine transform of the input |
| [`dctn`](_autosummary/jax.scipy.fft.dctn.html#jax.scipy.fft.dctn "jax.scipy.fft.dctn")(x\[, type, s, axes, norm\]) | Computes the multidimensional discrete cosine transform of the input |
| [`idct`](_autosummary/jax.scipy.fft.idct.html#jax.scipy.fft.idct "jax.scipy.fft.idct")(x\[, type, n, axis, norm\]) | Computes the inverse discrete cosine transform of the input |
| [`idctn`](_autosummary/jax.scipy.fft.idctn.html#jax.scipy.fft.idctn "jax.scipy.fft.idctn")(x\[, type, s, axes, norm\]) | Computes the multidimensional inverse discrete cosine transform of the input |

## jax.scipy.integrate[\#](#module-jax.scipy.integrate "Link to this heading")

|  |  |
|----|----|
| [`trapezoid`](_autosummary/jax.scipy.integrate.trapezoid.html#jax.scipy.integrate.trapezoid "jax.scipy.integrate.trapezoid")(y\[, x, dx, axis\]) | Integrate along the given axis using the composite trapezoidal rule. |

## jax.scipy.interpolate[\#](#module-jax.scipy.interpolate "Link to this heading")

|  |  |
|----|----|
| [`RegularGridInterpolator`](_autosummary/jax.scipy.interpolate.RegularGridInterpolator.html#jax.scipy.interpolate.RegularGridInterpolator "jax.scipy.interpolate.RegularGridInterpolator")(points, values\[, ...\]) | Interpolate points on a regular rectangular grid. |

## jax.scipy.linalg[\#](#module-jax.scipy.linalg "Link to this heading")

|  |  |
|----|----|
| [`block_diag`](_autosummary/jax.scipy.linalg.block_diag.html#jax.scipy.linalg.block_diag "jax.scipy.linalg.block_diag")(\*arrs) | Create a block diagonal matrix from input arrays. |
| [`cho_factor`](_autosummary/jax.scipy.linalg.cho_factor.html#jax.scipy.linalg.cho_factor "jax.scipy.linalg.cho_factor")(a\[, lower, overwrite_a, check_finite\]) | Factorization for Cholesky-based linear solves |
| [`cho_solve`](_autosummary/jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve")(c_and_lower, b\[, overwrite_b, ...\]) | Solve a linear system using a Cholesky factorization |
| [`cholesky`](_autosummary/jax.scipy.linalg.cholesky.html#jax.scipy.linalg.cholesky "jax.scipy.linalg.cholesky")(a\[, lower, overwrite_a, check_finite\]) | Compute the Cholesky decomposition of a matrix. |
| [`circulant`](_autosummary/jax.scipy.linalg.circulant.html#jax.scipy.linalg.circulant "jax.scipy.linalg.circulant")(c) | Construct a circulant matrix. |
| [`companion`](_autosummary/jax.scipy.linalg.companion.html#jax.scipy.linalg.companion "jax.scipy.linalg.companion")(a) | Construct a companion matrix. |
| [`convolution_matrix`](_autosummary/jax.scipy.linalg.convolution_matrix.html#jax.scipy.linalg.convolution_matrix "jax.scipy.linalg.convolution_matrix")(a, n\[, mode\]) | Construct a convolution matrix. |
| [`det`](_autosummary/jax.scipy.linalg.det.html#jax.scipy.linalg.det "jax.scipy.linalg.det")(a\[, overwrite_a, check_finite\]) | Compute the determinant of a matrix |
| [`dft`](_autosummary/jax.scipy.linalg.dft.html#jax.scipy.linalg.dft "jax.scipy.linalg.dft")(n\[, scale, dtype\]) | Construct an n-by-n discrete Fourier transform matrix. |
| [`eigh`](_autosummary/jax.scipy.linalg.eigh.html#jax.scipy.linalg.eigh "jax.scipy.linalg.eigh")() | Compute eigenvalues and eigenvectors for a Hermitian matrix |
| [`eigh_tridiagonal`](_autosummary/jax.scipy.linalg.eigh_tridiagonal.html#jax.scipy.linalg.eigh_tridiagonal "jax.scipy.linalg.eigh_tridiagonal")(d, e, \*\[, eigvals_only, ...\]) | Solve the eigenvalue problem for a symmetric real tridiagonal matrix |
| [`expm`](_autosummary/jax.scipy.linalg.expm.html#jax.scipy.linalg.expm "jax.scipy.linalg.expm")(A, \*\[, upper_triangular, max_squarings\]) | Compute the matrix exponential |
| [`expm_frechet`](_autosummary/jax.scipy.linalg.expm_frechet.html#jax.scipy.linalg.expm_frechet "jax.scipy.linalg.expm_frechet")() | Compute the Frechet derivative of the matrix exponential. |
| [`fiedler`](_autosummary/jax.scipy.linalg.fiedler.html#jax.scipy.linalg.fiedler "jax.scipy.linalg.fiedler")(a) | Construct a symmetric Fiedler matrix. |
| [`fiedler_companion`](_autosummary/jax.scipy.linalg.fiedler_companion.html#jax.scipy.linalg.fiedler_companion "jax.scipy.linalg.fiedler_companion")(a) | Construct a Fiedler companion matrix. |
| [`funm`](_autosummary/jax.scipy.linalg.funm.html#jax.scipy.linalg.funm "jax.scipy.linalg.funm")(A, func\[, disp\]) | Evaluate a matrix-valued function |
| [`hadamard`](_autosummary/jax.scipy.linalg.hadamard.html#jax.scipy.linalg.hadamard "jax.scipy.linalg.hadamard")(n\[, dtype\]) | Construct an n-by-n Hadamard matrix. |
| [`helmert`](_autosummary/jax.scipy.linalg.helmert.html#jax.scipy.linalg.helmert "jax.scipy.linalg.helmert")(n\[, full\]) | Construct a Helmert matrix. |
| [`hessenberg`](_autosummary/jax.scipy.linalg.hessenberg.html#jax.scipy.linalg.hessenberg "jax.scipy.linalg.hessenberg")() | Compute the Hessenberg form of the matrix |
| [`hankel`](_autosummary/jax.scipy.linalg.hankel.html#jax.scipy.linalg.hankel "jax.scipy.linalg.hankel")(c\[, r\]) | Construct a Hankel matrix. |
| [`hilbert`](_autosummary/jax.scipy.linalg.hilbert.html#jax.scipy.linalg.hilbert "jax.scipy.linalg.hilbert")(n) | Create a Hilbert matrix of order n. |
| [`inv`](_autosummary/jax.scipy.linalg.inv.html#jax.scipy.linalg.inv "jax.scipy.linalg.inv")(a\[, overwrite_a, check_finite\]) | Return the inverse of a square matrix |
| [`invhilbert`](_autosummary/jax.scipy.linalg.invhilbert.html#jax.scipy.linalg.invhilbert "jax.scipy.linalg.invhilbert")(n) | Compute the inverse of the Hilbert matrix of order n. |
| [`invpascal`](_autosummary/jax.scipy.linalg.invpascal.html#jax.scipy.linalg.invpascal "jax.scipy.linalg.invpascal")(n\[, kind\]) | Compute the inverse of the Pascal matrix of order n. |
| [`leslie`](_autosummary/jax.scipy.linalg.leslie.html#jax.scipy.linalg.leslie "jax.scipy.linalg.leslie")(f, s) | Construct a Leslie matrix. |
| [`lu`](_autosummary/jax.scipy.linalg.lu.html#jax.scipy.linalg.lu "jax.scipy.linalg.lu")() | Compute the LU decomposition |
| [`lu_factor`](_autosummary/jax.scipy.linalg.lu_factor.html#jax.scipy.linalg.lu_factor "jax.scipy.linalg.lu_factor")(a\[, overwrite_a, check_finite\]) | Factorization for LU-based linear solves |
| [`lu_solve`](_autosummary/jax.scipy.linalg.lu_solve.html#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve")(lu_and_piv, b\[, trans, ...\]) | Solve a linear system using an LU factorization |
| [`pascal`](_autosummary/jax.scipy.linalg.pascal.html#jax.scipy.linalg.pascal "jax.scipy.linalg.pascal")(n\[, kind\]) | Create a Pascal matrix approximation of order n. |
| [`polar`](_autosummary/jax.scipy.linalg.polar.html#jax.scipy.linalg.polar "jax.scipy.linalg.polar")(a\[, side, method, eps, max_iterations\]) | Computes the polar decomposition. |
| [`qr`](_autosummary/jax.scipy.linalg.qr.html#jax.scipy.linalg.qr "jax.scipy.linalg.qr")() | Compute the QR decomposition of an array |
| [`qr_multiply`](_autosummary/jax.scipy.linalg.qr_multiply.html#jax.scipy.linalg.qr_multiply "jax.scipy.linalg.qr_multiply")() | Calculate the QR decomposition and multiply Q with a matrix. |
| [`rsf2csf`](_autosummary/jax.scipy.linalg.rsf2csf.html#jax.scipy.linalg.rsf2csf "jax.scipy.linalg.rsf2csf")(T, Z\[, check_finite\]) | Convert real Schur form to complex Schur form. |
| [`schur`](_autosummary/jax.scipy.linalg.schur.html#jax.scipy.linalg.schur "jax.scipy.linalg.schur")(a\[, output\]) | Compute the Schur decomposition |
| [`solve`](_autosummary/jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve")(a, b\[, lower, overwrite_a, ...\]) | Solve a linear system of equations. |
| [`solve_sylvester`](_autosummary/jax.scipy.linalg.solve_sylvester.html#jax.scipy.linalg.solve_sylvester "jax.scipy.linalg.solve_sylvester")(A, B, C, \*\[, method, tol\]) | Solves the Sylvester equation .. math::. |
| [`solve_triangular`](_autosummary/jax.scipy.linalg.solve_triangular.html#jax.scipy.linalg.solve_triangular "jax.scipy.linalg.solve_triangular")(a, b\[, trans, lower, ...\]) | Solve a triangular linear system of equations |
| [`sqrtm`](_autosummary/jax.scipy.linalg.sqrtm.html#jax.scipy.linalg.sqrtm "jax.scipy.linalg.sqrtm")(A\[, blocksize\]) | Compute the matrix square root |
| [`svd`](_autosummary/jax.scipy.linalg.svd.html#jax.scipy.linalg.svd "jax.scipy.linalg.svd")() | Compute the singular value decomposition. |
| [`toeplitz`](_autosummary/jax.scipy.linalg.toeplitz.html#jax.scipy.linalg.toeplitz "jax.scipy.linalg.toeplitz")(c\[, r\]) | Construct a Toeplitz matrix. |

## jax.scipy.ndimage[\#](#module-jax.scipy.ndimage "Link to this heading")

|  |  |
|----|----|
| [`map_coordinates`](_autosummary/jax.scipy.ndimage.map_coordinates.html#jax.scipy.ndimage.map_coordinates "jax.scipy.ndimage.map_coordinates")(input, coordinates, order\[, ...\]) | Map the input array to new coordinates using interpolation. |

## jax.scipy.optimize[\#](#module-jax.scipy.optimize "Link to this heading")

|  |  |
|----|----|
| [`minimize`](_autosummary/jax.scipy.optimize.minimize.html#jax.scipy.optimize.minimize "jax.scipy.optimize.minimize")(fun, x0\[, args, tol, options\]) | Minimization of scalar function of one or more variables. |
| [`OptimizeResults`](_autosummary/jax.scipy.optimize.OptimizeResults.html#jax.scipy.optimize.OptimizeResults "jax.scipy.optimize.OptimizeResults")(x, success, status, fun, ...) | Object holding optimization results. |

## jax.scipy.signal[\#](#module-jax.scipy.signal "Link to this heading")

|  |  |
|----|----|
| [`fftconvolve`](_autosummary/jax.scipy.signal.fftconvolve.html#jax.scipy.signal.fftconvolve "jax.scipy.signal.fftconvolve")(in1, in2\[, mode, axes\]) | Convolve two N-dimensional arrays using Fast Fourier Transform (FFT). |
| [`convolve`](_autosummary/jax.scipy.signal.convolve.html#jax.scipy.signal.convolve "jax.scipy.signal.convolve")(in1, in2\[, mode, method, precision\]) | Convolution of two N-dimensional arrays. |
| [`convolve2d`](_autosummary/jax.scipy.signal.convolve2d.html#jax.scipy.signal.convolve2d "jax.scipy.signal.convolve2d")(in1, in2\[, mode, boundary, ...\]) | Convolution of two 2-dimensional arrays. |
| [`correlate`](_autosummary/jax.scipy.signal.correlate.html#jax.scipy.signal.correlate "jax.scipy.signal.correlate")(in1, in2\[, mode, method, precision\]) | Cross-correlation of two N-dimensional arrays. |
| [`correlate2d`](_autosummary/jax.scipy.signal.correlate2d.html#jax.scipy.signal.correlate2d "jax.scipy.signal.correlate2d")(in1, in2\[, mode, boundary, ...\]) | Cross-correlation of two 2-dimensional arrays. |
| [`csd`](_autosummary/jax.scipy.signal.csd.html#jax.scipy.signal.csd "jax.scipy.signal.csd")(x, y\[, fs, window, nperseg, noverlap, ...\]) | Estimate cross power spectral density (CSD) using Welch's method. |
| [`detrend`](_autosummary/jax.scipy.signal.detrend.html#jax.scipy.signal.detrend "jax.scipy.signal.detrend")(data\[, axis, type, bp, overwrite_data\]) | Remove linear or piecewise linear trends from data. |
| [`istft`](_autosummary/jax.scipy.signal.istft.html#jax.scipy.signal.istft "jax.scipy.signal.istft")(Zxx\[, fs, window, nperseg, noverlap, ...\]) | Perform the inverse short-time Fourier transform (ISTFT). |
| [`stft`](_autosummary/jax.scipy.signal.stft.html#jax.scipy.signal.stft "jax.scipy.signal.stft")(x\[, fs, window, nperseg, noverlap, ...\]) | Compute the short-time Fourier transform (STFT). |
| [`welch`](_autosummary/jax.scipy.signal.welch.html#jax.scipy.signal.welch "jax.scipy.signal.welch")(x\[, fs, window, nperseg, noverlap, ...\]) | Estimate power spectral density (PSD) using Welch's method. |

## jax.scipy.spatial.transform[\#](#module-jax.scipy.spatial.transform "Link to this heading")

|  |  |
|----|----|
| [`Rotation`](_autosummary/jax.scipy.spatial.transform.Rotation.html#jax.scipy.spatial.transform.Rotation "jax.scipy.spatial.transform.Rotation")(quat) | Rotation in 3 dimensions. |
| [`Slerp`](_autosummary/jax.scipy.spatial.transform.Slerp.html#jax.scipy.spatial.transform.Slerp "jax.scipy.spatial.transform.Slerp")(times, timedelta, rotations, rotvecs) | Spherical Linear Interpolation of Rotations. |

## jax.scipy.sparse.linalg[\#](#module-jax.scipy.sparse.linalg "Link to this heading")

|  |  |
|----|----|
| [`bicgstab`](_autosummary/jax.scipy.sparse.linalg.bicgstab.html#jax.scipy.sparse.linalg.bicgstab "jax.scipy.sparse.linalg.bicgstab")(A, b\[, x0, tol, atol, maxiter, M\]) | Use Bi-Conjugate Gradient Stable iteration to solve `Ax`` ``=`` ``b`. |
| [`cg`](_autosummary/jax.scipy.sparse.linalg.cg.html#jax.scipy.sparse.linalg.cg "jax.scipy.sparse.linalg.cg")(A, b\[, x0, tol, atol, maxiter, M\]) | Use Conjugate Gradient iteration to solve `Ax`` ``=`` ``b`. |
| [`gmres`](_autosummary/jax.scipy.sparse.linalg.gmres.html#jax.scipy.sparse.linalg.gmres "jax.scipy.sparse.linalg.gmres")(A, b\[, x0, tol, atol, restart, ...\]) | GMRES solves the linear system A x = b for x, given A and b. |

## jax.scipy.special[\#](#module-jax.scipy.special "Link to this heading")

|  |  |
|----|----|
| [`bernoulli`](_autosummary/jax.scipy.special.bernoulli.html#jax.scipy.special.bernoulli "jax.scipy.special.bernoulli")(n) | Generate the first N Bernoulli numbers. |
| [`beta`](_autosummary/jax.scipy.special.beta.html#jax.scipy.special.beta "jax.scipy.special.beta")(a, b) | The beta function |
| [`betainc`](_autosummary/jax.scipy.special.betainc.html#jax.scipy.special.betainc "jax.scipy.special.betainc")(a, b, x) | The regularized incomplete beta function. |
| [`betaln`](_autosummary/jax.scipy.special.betaln.html#jax.scipy.special.betaln "jax.scipy.special.betaln")(a, b) | Natural log of the absolute value of the beta function |
| [`boxcox`](_autosummary/jax.scipy.special.boxcox.html#jax.scipy.special.boxcox "jax.scipy.special.boxcox")(x, lmbda) | Box-Cox power transformation. |
| [`boxcox1p`](_autosummary/jax.scipy.special.boxcox1p.html#jax.scipy.special.boxcox1p "jax.scipy.special.boxcox1p")(x, lmbda) | Box-Cox transformation of `1`` ``+`` ``x`. |
| [`comb`](_autosummary/jax.scipy.special.comb.html#jax.scipy.special.comb "jax.scipy.special.comb")(N, k, \*\[, repetition\]) | The number of combinations of N things taken k at a time ("N choose k"). |
| [`dawsn`](_autosummary/jax.scipy.special.dawsn.html#jax.scipy.special.dawsn "jax.scipy.special.dawsn")(x) | Dawson's integral. |
| [`digamma`](_autosummary/jax.scipy.special.digamma.html#jax.scipy.special.digamma "jax.scipy.special.digamma")(x) | The digamma function |
| [`entr`](_autosummary/jax.scipy.special.entr.html#jax.scipy.special.entr "jax.scipy.special.entr")(x) | The entropy function |
| [`erf`](_autosummary/jax.scipy.special.erf.html#jax.scipy.special.erf "jax.scipy.special.erf")(x) | The error function |
| [`erfc`](_autosummary/jax.scipy.special.erfc.html#jax.scipy.special.erfc "jax.scipy.special.erfc")(x) | The complement of the error function |
| [`erfcx`](_autosummary/jax.scipy.special.erfcx.html#jax.scipy.special.erfcx "jax.scipy.special.erfcx")(x) | Scaled complementary error function. |
| [`erfinv`](_autosummary/jax.scipy.special.erfinv.html#jax.scipy.special.erfinv "jax.scipy.special.erfinv")(x) | The inverse of the error function |
| [`exp1`](_autosummary/jax.scipy.special.exp1.html#jax.scipy.special.exp1 "jax.scipy.special.exp1")(x) | Exponential integral function. |
| [`expi`](_autosummary/jax.scipy.special.expi.html#jax.scipy.special.expi "jax.scipy.special.expi") | Exponential integral function. |
| [`expit`](_autosummary/jax.scipy.special.expit.html#jax.scipy.special.expit "jax.scipy.special.expit")(x) | The logistic sigmoid (expit) function |
| [`expn`](_autosummary/jax.scipy.special.expn.html#jax.scipy.special.expn "jax.scipy.special.expn") | Generalized exponential integral function. |
| [`factorial`](_autosummary/jax.scipy.special.factorial.html#jax.scipy.special.factorial "jax.scipy.special.factorial")(n\[, exact\]) | Factorial function |
| [`fresnel`](_autosummary/jax.scipy.special.fresnel.html#jax.scipy.special.fresnel "jax.scipy.special.fresnel") | The Fresnel integrals |
| [`gamma`](_autosummary/jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")(x) | The gamma function. |
| [`gammainc`](_autosummary/jax.scipy.special.gammainc.html#jax.scipy.special.gammainc "jax.scipy.special.gammainc")(a, x) | The regularized lower incomplete gamma function. |
| [`gammaincc`](_autosummary/jax.scipy.special.gammaincc.html#jax.scipy.special.gammaincc "jax.scipy.special.gammaincc")(a, x) | The regularized upper incomplete gamma function. |
| [`gammaln`](_autosummary/jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln")(x) | Natural log of the absolute value of the gamma function. |
| [`gammasgn`](_autosummary/jax.scipy.special.gammasgn.html#jax.scipy.special.gammasgn "jax.scipy.special.gammasgn")(x) | Sign of the gamma function. |
| [`hyp1f1`](_autosummary/jax.scipy.special.hyp1f1.html#jax.scipy.special.hyp1f1 "jax.scipy.special.hyp1f1") | The 1F1 hypergeometric function. |
| [`hyp2f1`](_autosummary/jax.scipy.special.hyp2f1.html#jax.scipy.special.hyp2f1 "jax.scipy.special.hyp2f1") | The 2F1 hypergeometric function. |
| [`i0`](_autosummary/jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0")(x) | Modified bessel function of zeroth order. |
| [`i0e`](_autosummary/jax.scipy.special.i0e.html#jax.scipy.special.i0e "jax.scipy.special.i0e")(x) | Exponentially scaled modified bessel function of zeroth order. |
| [`i1`](_autosummary/jax.scipy.special.i1.html#jax.scipy.special.i1 "jax.scipy.special.i1")(x) | Modified bessel function of first order. |
| [`i1e`](_autosummary/jax.scipy.special.i1e.html#jax.scipy.special.i1e "jax.scipy.special.i1e")(x) | Exponentially scaled modified bessel function of first order. |
| [`kl_div`](_autosummary/jax.scipy.special.kl_div.html#jax.scipy.special.kl_div "jax.scipy.special.kl_div")(p, q) | The Kullback-Leibler divergence. |
| [`log_ndtr`](_autosummary/jax.scipy.special.log_ndtr.html#jax.scipy.special.log_ndtr "jax.scipy.special.log_ndtr") | Log Normal distribution function. |
| [`log_softmax`](_autosummary/jax.scipy.special.log_softmax.html#jax.scipy.special.log_softmax "jax.scipy.special.log_softmax")(x, /, \*\[, axis\]) | Log-Softmax function. |
| [`logit`](_autosummary/jax.scipy.special.logit.html#jax.scipy.special.logit "jax.scipy.special.logit") | The logit function |
| [`loggamma`](_autosummary/jax.scipy.special.loggamma.html#jax.scipy.special.loggamma "jax.scipy.special.loggamma")(x) | Principal branch of the logarithm of the gamma function. |
| [`logsumexp`](_autosummary/jax.scipy.special.logsumexp.html#jax.scipy.special.logsumexp "jax.scipy.special.logsumexp")() | Log-sum-exp reduction. |
| [`lpmn`](_autosummary/jax.scipy.special.lpmn.html#jax.scipy.special.lpmn "jax.scipy.special.lpmn")(m, n, z) | The associated Legendre functions (ALFs) of the first kind. |
| [`lpmn_values`](_autosummary/jax.scipy.special.lpmn_values.html#jax.scipy.special.lpmn_values "jax.scipy.special.lpmn_values")(m, n, z, is_normalized) | The associated Legendre functions (ALFs) of the first kind. |
| [`multigammaln`](_autosummary/jax.scipy.special.multigammaln.html#jax.scipy.special.multigammaln "jax.scipy.special.multigammaln")(a, d) | The natural log of the multivariate gamma function. |
| [`ndtr`](_autosummary/jax.scipy.special.ndtr.html#jax.scipy.special.ndtr "jax.scipy.special.ndtr")(x) | Normal distribution function. |
| [`ndtri`](_autosummary/jax.scipy.special.ndtri.html#jax.scipy.special.ndtri "jax.scipy.special.ndtri")(p) | The inverse of the CDF of the Normal distribution function. |
| [`owens_t`](_autosummary/jax.scipy.special.owens_t.html#jax.scipy.special.owens_t "jax.scipy.special.owens_t")(h, a) | Owen's T function. |
| [`poch`](_autosummary/jax.scipy.special.poch.html#jax.scipy.special.poch "jax.scipy.special.poch") | The Pochammer symbol. |
| [`polygamma`](_autosummary/jax.scipy.special.polygamma.html#jax.scipy.special.polygamma "jax.scipy.special.polygamma")(n, x) | The polygamma function. |
| [`rel_entr`](_autosummary/jax.scipy.special.rel_entr.html#jax.scipy.special.rel_entr "jax.scipy.special.rel_entr")(p, q) | The relative entropy function. |
| [`sici`](_autosummary/jax.scipy.special.sici.html#jax.scipy.special.sici "jax.scipy.special.sici") | Sine and cosine integrals. |
| [`softmax`](_autosummary/jax.scipy.special.softmax.html#jax.scipy.special.softmax "jax.scipy.special.softmax")(x, /, \*\[, axis\]) | Softmax function. |
| [`spence`](_autosummary/jax.scipy.special.spence.html#jax.scipy.special.spence "jax.scipy.special.spence")(x) | Spence's function, also known as the dilogarithm for real values. |
| [`sph_harm_y`](_autosummary/jax.scipy.special.sph_harm_y.html#jax.scipy.special.sph_harm_y "jax.scipy.special.sph_harm_y")(n, m, theta, phi\[, diff_n, n_max\]) | Computes the spherical harmonics. |
| [`wofz`](_autosummary/jax.scipy.special.wofz.html#jax.scipy.special.wofz "jax.scipy.special.wofz")(z) | Faddeeva function. |
| [`xlog1py`](_autosummary/jax.scipy.special.xlog1py.html#jax.scipy.special.xlog1py "jax.scipy.special.xlog1py") | Compute x\*log(1 + y), returning 0 for x=0. |
| [`xlogy`](_autosummary/jax.scipy.special.xlogy.html#jax.scipy.special.xlogy "jax.scipy.special.xlogy") | Compute x\*log(y), returning 0 for x=0. |
| [`zeta`](_autosummary/jax.scipy.special.zeta.html#jax.scipy.special.zeta "jax.scipy.special.zeta") | The Hurwitz zeta function. |

## jax.scipy.stats[\#](#module-jax.scipy.stats "Link to this heading")

|  |  |
|----|----|
| [`mode`](_autosummary/jax.scipy.stats.mode.html#jax.scipy.stats.mode "jax.scipy.stats.mode")(a\[, axis, nan_policy, keepdims\]) | Compute the mode (most common value) along an axis of an array. |
| [`rankdata`](_autosummary/jax.scipy.stats.rankdata.html#jax.scipy.stats.rankdata "jax.scipy.stats.rankdata")(a\[, method, axis, nan_policy\]) | Compute the rank of data along an array axis. |
| [`sem`](_autosummary/jax.scipy.stats.sem.html#jax.scipy.stats.sem "jax.scipy.stats.sem")(a\[, axis, ddof, nan_policy, keepdims\]) | Compute the standard error of the mean. |

### jax.scipy.stats.bernoulli[\#](#module-jax.scipy.stats.bernoulli "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.bernoulli.logpmf.html#jax.scipy.stats.bernoulli.logpmf "jax.scipy.stats.bernoulli.logpmf")(k, p\[, loc\]) | Bernoulli log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.bernoulli.pmf.html#jax.scipy.stats.bernoulli.pmf "jax.scipy.stats.bernoulli.pmf")(k, p\[, loc\]) | Bernoulli probability mass function. |
| [`cdf`](_autosummary/jax.scipy.stats.bernoulli.cdf.html#jax.scipy.stats.bernoulli.cdf "jax.scipy.stats.bernoulli.cdf")(k, p) | Bernoulli cumulative distribution function. |
| [`ppf`](_autosummary/jax.scipy.stats.bernoulli.ppf.html#jax.scipy.stats.bernoulli.ppf "jax.scipy.stats.bernoulli.ppf")(q, p) | Bernoulli percent point function. |

### jax.scipy.stats.beta[\#](#module-jax.scipy.stats.beta "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.beta.logpdf.html#jax.scipy.stats.beta.logpdf "jax.scipy.stats.beta.logpdf")(x, a, b\[, loc, scale\]) | Beta log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.beta.pdf.html#jax.scipy.stats.beta.pdf "jax.scipy.stats.beta.pdf")(x, a, b\[, loc, scale\]) | Beta probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.beta.cdf.html#jax.scipy.stats.beta.cdf "jax.scipy.stats.beta.cdf")(x, a, b\[, loc, scale\]) | Beta cumulative distribution function |
| [`logcdf`](_autosummary/jax.scipy.stats.beta.logcdf.html#jax.scipy.stats.beta.logcdf "jax.scipy.stats.beta.logcdf")(x, a, b\[, loc, scale\]) | Beta log cumulative distribution function. |
| [`sf`](_autosummary/jax.scipy.stats.beta.sf.html#jax.scipy.stats.beta.sf "jax.scipy.stats.beta.sf")(x, a, b\[, loc, scale\]) | Beta distribution survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.beta.logsf.html#jax.scipy.stats.beta.logsf "jax.scipy.stats.beta.logsf")(x, a, b\[, loc, scale\]) | Beta distribution log survival function. |

### jax.scipy.stats.betabinom[\#](#module-jax.scipy.stats.betabinom "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.betabinom.logpmf.html#jax.scipy.stats.betabinom.logpmf "jax.scipy.stats.betabinom.logpmf")(k, n, a, b\[, loc\]) | Beta-binomial log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.betabinom.pmf.html#jax.scipy.stats.betabinom.pmf "jax.scipy.stats.betabinom.pmf")(k, n, a, b\[, loc\]) | Beta-binomial probability mass function. |

### jax.scipy.stats.binom[\#](#module-jax.scipy.stats.binom "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.binom.logpmf.html#jax.scipy.stats.binom.logpmf "jax.scipy.stats.binom.logpmf")(k, n, p\[, loc\]) | Binomial log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.binom.pmf.html#jax.scipy.stats.binom.pmf "jax.scipy.stats.binom.pmf")(k, n, p\[, loc\]) | Binomial probability mass function. |

### jax.scipy.stats.cauchy[\#](#module-jax.scipy.stats.cauchy "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.cauchy.logpdf.html#jax.scipy.stats.cauchy.logpdf "jax.scipy.stats.cauchy.logpdf")(x\[, loc, scale\]) | Cauchy log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.cauchy.pdf.html#jax.scipy.stats.cauchy.pdf "jax.scipy.stats.cauchy.pdf")(x\[, loc, scale\]) | Cauchy probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.cauchy.cdf.html#jax.scipy.stats.cauchy.cdf "jax.scipy.stats.cauchy.cdf")(x\[, loc, scale\]) | Cauchy cumulative distribution function. |
| [`logcdf`](_autosummary/jax.scipy.stats.cauchy.logcdf.html#jax.scipy.stats.cauchy.logcdf "jax.scipy.stats.cauchy.logcdf")(x\[, loc, scale\]) | Cauchy log cumulative distribution function. |
| [`sf`](_autosummary/jax.scipy.stats.cauchy.sf.html#jax.scipy.stats.cauchy.sf "jax.scipy.stats.cauchy.sf")(x\[, loc, scale\]) | Cauchy distribution log survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.cauchy.logsf.html#jax.scipy.stats.cauchy.logsf "jax.scipy.stats.cauchy.logsf")(x\[, loc, scale\]) | Cauchy distribution log survival function. |
| [`isf`](_autosummary/jax.scipy.stats.cauchy.isf.html#jax.scipy.stats.cauchy.isf "jax.scipy.stats.cauchy.isf")(q\[, loc, scale\]) | Cauchy distribution inverse survival function. |
| [`ppf`](_autosummary/jax.scipy.stats.cauchy.ppf.html#jax.scipy.stats.cauchy.ppf "jax.scipy.stats.cauchy.ppf")(q\[, loc, scale\]) | Cauchy distribution percent point function. |

### jax.scipy.stats.chi2[\#](#module-jax.scipy.stats.chi2 "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.chi2.logpdf.html#jax.scipy.stats.chi2.logpdf "jax.scipy.stats.chi2.logpdf")(x, df\[, loc, scale\]) | Chi-square log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.chi2.pdf.html#jax.scipy.stats.chi2.pdf "jax.scipy.stats.chi2.pdf")(x, df\[, loc, scale\]) | Chi-square probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.chi2.cdf.html#jax.scipy.stats.chi2.cdf "jax.scipy.stats.chi2.cdf")(x, df\[, loc, scale\]) | Chi-square cumulative distribution function. |
| [`logcdf`](_autosummary/jax.scipy.stats.chi2.logcdf.html#jax.scipy.stats.chi2.logcdf "jax.scipy.stats.chi2.logcdf")(x, df\[, loc, scale\]) | Chi-square log cumulative distribution function. |
| [`sf`](_autosummary/jax.scipy.stats.chi2.sf.html#jax.scipy.stats.chi2.sf "jax.scipy.stats.chi2.sf")(x, df\[, loc, scale\]) | Chi-square survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.chi2.logsf.html#jax.scipy.stats.chi2.logsf "jax.scipy.stats.chi2.logsf")(x, df\[, loc, scale\]) | Chi-square log survival function. |

### jax.scipy.stats.dirichlet[\#](#module-jax.scipy.stats.dirichlet "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.dirichlet.logpdf.html#jax.scipy.stats.dirichlet.logpdf "jax.scipy.stats.dirichlet.logpdf")(x, alpha) | Dirichlet log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.dirichlet.pdf.html#jax.scipy.stats.dirichlet.pdf "jax.scipy.stats.dirichlet.pdf")(x, alpha) | Dirichlet probability distribution function. |

### jax.scipy.stats.expon[\#](#module-jax.scipy.stats.expon "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.expon.logpdf.html#jax.scipy.stats.expon.logpdf "jax.scipy.stats.expon.logpdf")(x\[, loc, scale\]) | Exponential log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.expon.pdf.html#jax.scipy.stats.expon.pdf "jax.scipy.stats.expon.pdf")(x\[, loc, scale\]) | Exponential probability distribution function. |
| [`logcdf`](_autosummary/jax.scipy.stats.expon.logcdf.html#jax.scipy.stats.expon.logcdf "jax.scipy.stats.expon.logcdf")(x\[, loc, scale\]) | Exponential log cumulative density function. |
| [`cdf`](_autosummary/jax.scipy.stats.expon.cdf.html#jax.scipy.stats.expon.cdf "jax.scipy.stats.expon.cdf")(x\[, loc, scale\]) | Exponential cumulative density function. |
| [`logsf`](_autosummary/jax.scipy.stats.expon.logsf.html#jax.scipy.stats.expon.logsf "jax.scipy.stats.expon.logsf")(x\[, loc, scale\]) | Exponential log survival function. |
| [`sf`](_autosummary/jax.scipy.stats.expon.sf.html#jax.scipy.stats.expon.sf "jax.scipy.stats.expon.sf")(x\[, loc, scale\]) | Exponential survival function. |
| [`ppf`](_autosummary/jax.scipy.stats.expon.ppf.html#jax.scipy.stats.expon.ppf "jax.scipy.stats.expon.ppf")(q\[, loc, scale\]) | Exponential percent point function. |

### jax.scipy.stats.gamma[\#](#module-jax.scipy.stats.gamma "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.gamma.logpdf.html#jax.scipy.stats.gamma.logpdf "jax.scipy.stats.gamma.logpdf")(x, a\[, loc, scale\]) | Gamma log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.gamma.pdf.html#jax.scipy.stats.gamma.pdf "jax.scipy.stats.gamma.pdf")(x, a\[, loc, scale\]) | Gamma probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.gamma.cdf.html#jax.scipy.stats.gamma.cdf "jax.scipy.stats.gamma.cdf")(x, a\[, loc, scale\]) | Gamma cumulative distribution function. |
| [`logcdf`](_autosummary/jax.scipy.stats.gamma.logcdf.html#jax.scipy.stats.gamma.logcdf "jax.scipy.stats.gamma.logcdf")(x, a\[, loc, scale\]) | Gamma log cumulative distribution function. |
| [`sf`](_autosummary/jax.scipy.stats.gamma.sf.html#jax.scipy.stats.gamma.sf "jax.scipy.stats.gamma.sf")(x, a\[, loc, scale\]) | Gamma survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.gamma.logsf.html#jax.scipy.stats.gamma.logsf "jax.scipy.stats.gamma.logsf")(x, a\[, loc, scale\]) | Gamma log survival function. |

### jax.scipy.stats.gumbel_l[\#](#module-jax.scipy.stats.gumbel_l "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.gumbel_l.logpdf.html#jax.scipy.stats.gumbel_l.logpdf "jax.scipy.stats.gumbel_l.logpdf")(x\[, loc, scale\]) | Gumbel Distribution (Left Skewed) log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.gumbel_l.pdf.html#jax.scipy.stats.gumbel_l.pdf "jax.scipy.stats.gumbel_l.pdf")(x\[, loc, scale\]) | Gumbel Distribution (Left Skewed) probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.gumbel_l.cdf.html#jax.scipy.stats.gumbel_l.cdf "jax.scipy.stats.gumbel_l.cdf")(x\[, loc, scale\]) | Gumbel Distribution (Left Skewed) cumulative density function. |
| [`logcdf`](_autosummary/jax.scipy.stats.gumbel_l.logcdf.html#jax.scipy.stats.gumbel_l.logcdf "jax.scipy.stats.gumbel_l.logcdf")(x\[, loc, scale\]) | Gumbel Distribution (Left Skewed) log cumulative density function. |
| [`sf`](_autosummary/jax.scipy.stats.gumbel_l.sf.html#jax.scipy.stats.gumbel_l.sf "jax.scipy.stats.gumbel_l.sf")(x\[, loc, scale\]) | Gumbel Distribution (Left Skewed) survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.gumbel_l.logsf.html#jax.scipy.stats.gumbel_l.logsf "jax.scipy.stats.gumbel_l.logsf")(x\[, loc, scale\]) | Gumbel Distribution (Left Skewed) log survival function. |
| [`ppf`](_autosummary/jax.scipy.stats.gumbel_l.ppf.html#jax.scipy.stats.gumbel_l.ppf "jax.scipy.stats.gumbel_l.ppf")(p\[, loc, scale\]) | Gumbel Distribution (Left Skewed) percent point function (inverse of CDF) |

### jax.scipy.stats.gumbel_r[\#](#module-jax.scipy.stats.gumbel_r "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.gumbel_r.logpdf.html#jax.scipy.stats.gumbel_r.logpdf "jax.scipy.stats.gumbel_r.logpdf")(x\[, loc, scale\]) | Gumbel Distribution (Right Skewed) log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.gumbel_r.pdf.html#jax.scipy.stats.gumbel_r.pdf "jax.scipy.stats.gumbel_r.pdf")(x\[, loc, scale\]) | Gumbel Distribution (Right Skewed) probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.gumbel_r.cdf.html#jax.scipy.stats.gumbel_r.cdf "jax.scipy.stats.gumbel_r.cdf")(x\[, loc, scale\]) | Gumbel Distribution (Right Skewed) cumulative density function. |
| [`logcdf`](_autosummary/jax.scipy.stats.gumbel_r.logcdf.html#jax.scipy.stats.gumbel_r.logcdf "jax.scipy.stats.gumbel_r.logcdf")(x\[, loc, scale\]) | Gumbel Distribution (Right Skewed) log cumulative density function. |
| [`sf`](_autosummary/jax.scipy.stats.gumbel_r.sf.html#jax.scipy.stats.gumbel_r.sf "jax.scipy.stats.gumbel_r.sf")(x\[, loc, scale\]) | Gumbel Distribution (Right Skewed) survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.gumbel_r.logsf.html#jax.scipy.stats.gumbel_r.logsf "jax.scipy.stats.gumbel_r.logsf")(x\[, loc, scale\]) | Gumbel Distribution (Right Skewed) log survival function. |
| [`ppf`](_autosummary/jax.scipy.stats.gumbel_r.ppf.html#jax.scipy.stats.gumbel_r.ppf "jax.scipy.stats.gumbel_r.ppf")(p\[, loc, scale\]) | Gumbel Distribution (Right Skewed) percent point function. |

### jax.scipy.stats.gennorm[\#](#module-jax.scipy.stats.gennorm "Link to this heading")

|  |  |
|----|----|
| [`cdf`](_autosummary/jax.scipy.stats.gennorm.cdf.html#jax.scipy.stats.gennorm.cdf "jax.scipy.stats.gennorm.cdf")(x, beta) | Generalized normal cumulative distribution function. |
| [`logpdf`](_autosummary/jax.scipy.stats.gennorm.logpdf.html#jax.scipy.stats.gennorm.logpdf "jax.scipy.stats.gennorm.logpdf")(x, beta) | Generalized normal log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.gennorm.pdf.html#jax.scipy.stats.gennorm.pdf "jax.scipy.stats.gennorm.pdf")(x, beta) | Generalized normal probability distribution function. |

### jax.scipy.stats.geom[\#](#module-jax.scipy.stats.geom "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.geom.logpmf.html#jax.scipy.stats.geom.logpmf "jax.scipy.stats.geom.logpmf")(k, p\[, loc\]) | Geometric log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.geom.pmf.html#jax.scipy.stats.geom.pmf "jax.scipy.stats.geom.pmf")(k, p\[, loc\]) | Geometric probability mass function. |

### jax.scipy.stats.laplace[\#](#module-jax.scipy.stats.laplace "Link to this heading")

|  |  |
|----|----|
| [`cdf`](_autosummary/jax.scipy.stats.laplace.cdf.html#jax.scipy.stats.laplace.cdf "jax.scipy.stats.laplace.cdf")(x\[, loc, scale\]) | Laplace cumulative distribution function. |
| [`logpdf`](_autosummary/jax.scipy.stats.laplace.logpdf.html#jax.scipy.stats.laplace.logpdf "jax.scipy.stats.laplace.logpdf")(x\[, loc, scale\]) | Laplace log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.laplace.pdf.html#jax.scipy.stats.laplace.pdf "jax.scipy.stats.laplace.pdf")(x\[, loc, scale\]) | Laplace probability distribution function. |

### jax.scipy.stats.logistic[\#](#module-jax.scipy.stats.logistic "Link to this heading")

|  |  |
|----|----|
| [`cdf`](_autosummary/jax.scipy.stats.logistic.cdf.html#jax.scipy.stats.logistic.cdf "jax.scipy.stats.logistic.cdf")(x\[, loc, scale\]) | Logistic cumulative distribution function. |
| [`isf`](_autosummary/jax.scipy.stats.logistic.isf.html#jax.scipy.stats.logistic.isf "jax.scipy.stats.logistic.isf")(x\[, loc, scale\]) | Logistic distribution inverse survival function. |
| [`logpdf`](_autosummary/jax.scipy.stats.logistic.logpdf.html#jax.scipy.stats.logistic.logpdf "jax.scipy.stats.logistic.logpdf")(x\[, loc, scale\]) | Logistic log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.logistic.pdf.html#jax.scipy.stats.logistic.pdf "jax.scipy.stats.logistic.pdf")(x\[, loc, scale\]) | Logistic probability distribution function. |
| [`ppf`](_autosummary/jax.scipy.stats.logistic.ppf.html#jax.scipy.stats.logistic.ppf "jax.scipy.stats.logistic.ppf")(x\[, loc, scale\]) | Logistic distribution percent point function. |
| [`sf`](_autosummary/jax.scipy.stats.logistic.sf.html#jax.scipy.stats.logistic.sf "jax.scipy.stats.logistic.sf")(x\[, loc, scale\]) | Logistic distribution survival function. |

### jax.scipy.stats.multinomial[\#](#module-jax.scipy.stats.multinomial "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.multinomial.logpmf.html#jax.scipy.stats.multinomial.logpmf "jax.scipy.stats.multinomial.logpmf")(x, n, p) | Multinomial log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.multinomial.pmf.html#jax.scipy.stats.multinomial.pmf "jax.scipy.stats.multinomial.pmf")(x, n, p) | Multinomial probability mass function. |

### jax.scipy.stats.multivariate_normal[\#](#module-jax.scipy.stats.multivariate_normal "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.multivariate_normal.logpdf.html#jax.scipy.stats.multivariate_normal.logpdf "jax.scipy.stats.multivariate_normal.logpdf")(x, mean, cov\[, allow_singular\]) | Multivariate normal log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.multivariate_normal.pdf.html#jax.scipy.stats.multivariate_normal.pdf "jax.scipy.stats.multivariate_normal.pdf")(x, mean, cov) | Multivariate normal probability distribution function. |

### jax.scipy.stats.nbinom[\#](#module-jax.scipy.stats.nbinom "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.nbinom.logpmf.html#jax.scipy.stats.nbinom.logpmf "jax.scipy.stats.nbinom.logpmf")(k, n, p\[, loc\]) | Negative-binomial log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.nbinom.pmf.html#jax.scipy.stats.nbinom.pmf "jax.scipy.stats.nbinom.pmf")(k, n, p\[, loc\]) | Negative-binomial probability mass function. |

### jax.scipy.stats.norm[\#](#module-jax.scipy.stats.norm "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.norm.logpdf.html#jax.scipy.stats.norm.logpdf "jax.scipy.stats.norm.logpdf")(x\[, loc, scale\]) | Normal log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.norm.pdf.html#jax.scipy.stats.norm.pdf "jax.scipy.stats.norm.pdf")(x\[, loc, scale\]) | Normal probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.norm.cdf.html#jax.scipy.stats.norm.cdf "jax.scipy.stats.norm.cdf")(x\[, loc, scale\]) | Normal cumulative distribution function. |
| [`logcdf`](_autosummary/jax.scipy.stats.norm.logcdf.html#jax.scipy.stats.norm.logcdf "jax.scipy.stats.norm.logcdf")(x\[, loc, scale\]) | Normal log cumulative distribution function. |
| [`ppf`](_autosummary/jax.scipy.stats.norm.ppf.html#jax.scipy.stats.norm.ppf "jax.scipy.stats.norm.ppf")(q\[, loc, scale\]) | Normal distribution percent point function. |
| [`sf`](_autosummary/jax.scipy.stats.norm.sf.html#jax.scipy.stats.norm.sf "jax.scipy.stats.norm.sf")(x\[, loc, scale\]) | Normal distribution survival function. |
| [`logsf`](_autosummary/jax.scipy.stats.norm.logsf.html#jax.scipy.stats.norm.logsf "jax.scipy.stats.norm.logsf")(x\[, loc, scale\]) | Normal distribution log survival function. |
| [`isf`](_autosummary/jax.scipy.stats.norm.isf.html#jax.scipy.stats.norm.isf "jax.scipy.stats.norm.isf")(q\[, loc, scale\]) | Normal distribution inverse survival function. |

### jax.scipy.stats.pareto[\#](#module-jax.scipy.stats.pareto "Link to this heading")

|  |  |
|----|----|
| [`logcdf`](_autosummary/jax.scipy.stats.pareto.logcdf.html#jax.scipy.stats.pareto.logcdf "jax.scipy.stats.pareto.logcdf")(x, b\[, loc, scale\]) | Pareto log cumulative distribution function. |
| [`logpdf`](_autosummary/jax.scipy.stats.pareto.logpdf.html#jax.scipy.stats.pareto.logpdf "jax.scipy.stats.pareto.logpdf")(x, b\[, loc, scale\]) | Pareto log probability distribution function. |
| [`logsf`](_autosummary/jax.scipy.stats.pareto.logsf.html#jax.scipy.stats.pareto.logsf "jax.scipy.stats.pareto.logsf")(x, b\[, loc, scale\]) | Pareto log survival function. |
| [`cdf`](_autosummary/jax.scipy.stats.pareto.cdf.html#jax.scipy.stats.pareto.cdf "jax.scipy.stats.pareto.cdf")(x, b\[, loc, scale\]) | Pareto cumulative distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.pareto.pdf.html#jax.scipy.stats.pareto.pdf "jax.scipy.stats.pareto.pdf")(x, b\[, loc, scale\]) | Pareto probability distribution function. |
| [`ppf`](_autosummary/jax.scipy.stats.pareto.ppf.html#jax.scipy.stats.pareto.ppf "jax.scipy.stats.pareto.ppf")(q, b\[, loc, scale\]) | Pareto percent point function (inverse CDF). |
| [`sf`](_autosummary/jax.scipy.stats.pareto.sf.html#jax.scipy.stats.pareto.sf "jax.scipy.stats.pareto.sf")(x, b\[, loc, scale\]) | Pareto survival function. |

### jax.scipy.stats.poisson[\#](#module-jax.scipy.stats.poisson "Link to this heading")

|  |  |
|----|----|
| [`logpmf`](_autosummary/jax.scipy.stats.poisson.logpmf.html#jax.scipy.stats.poisson.logpmf "jax.scipy.stats.poisson.logpmf")(k, mu\[, loc\]) | Poisson log probability mass function. |
| [`pmf`](_autosummary/jax.scipy.stats.poisson.pmf.html#jax.scipy.stats.poisson.pmf "jax.scipy.stats.poisson.pmf")(k, mu\[, loc\]) | Poisson probability mass function. |
| [`cdf`](_autosummary/jax.scipy.stats.poisson.cdf.html#jax.scipy.stats.poisson.cdf "jax.scipy.stats.poisson.cdf")(k, mu\[, loc\]) | Poisson cumulative distribution function. |
| [`entropy`](_autosummary/jax.scipy.stats.poisson.entropy.html#jax.scipy.stats.poisson.entropy "jax.scipy.stats.poisson.entropy")(mu\[, loc\]) | Shannon entropy of the Poisson distribution. |

### jax.scipy.stats.t[\#](#module-jax.scipy.stats.t "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.t.logpdf.html#jax.scipy.stats.t.logpdf "jax.scipy.stats.t.logpdf")(x, df\[, loc, scale\]) | Student's T log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.t.pdf.html#jax.scipy.stats.t.pdf "jax.scipy.stats.t.pdf")(x, df\[, loc, scale\]) | Student's T probability distribution function. |

### jax.scipy.stats.truncnorm[\#](#module-jax.scipy.stats.truncnorm "Link to this heading")

|  |  |
|----|----|
| [`cdf`](_autosummary/jax.scipy.stats.truncnorm.cdf.html#jax.scipy.stats.truncnorm.cdf "jax.scipy.stats.truncnorm.cdf")(x, a, b\[, loc, scale\]) | Truncated normal cumulative distribution function. |
| [`logcdf`](_autosummary/jax.scipy.stats.truncnorm.logcdf.html#jax.scipy.stats.truncnorm.logcdf "jax.scipy.stats.truncnorm.logcdf")(x, a, b\[, loc, scale\]) | Truncated normal log cumulative distribution function. |
| [`logpdf`](_autosummary/jax.scipy.stats.truncnorm.logpdf.html#jax.scipy.stats.truncnorm.logpdf "jax.scipy.stats.truncnorm.logpdf")(x, a, b\[, loc, scale\]) | Truncated normal log probability distribution function. |
| [`logsf`](_autosummary/jax.scipy.stats.truncnorm.logsf.html#jax.scipy.stats.truncnorm.logsf "jax.scipy.stats.truncnorm.logsf")(x, a, b\[, loc, scale\]) | Truncated normal distribution log survival function. |
| [`pdf`](_autosummary/jax.scipy.stats.truncnorm.pdf.html#jax.scipy.stats.truncnorm.pdf "jax.scipy.stats.truncnorm.pdf")(x, a, b\[, loc, scale\]) | Truncated normal probability distribution function. |
| [`sf`](_autosummary/jax.scipy.stats.truncnorm.sf.html#jax.scipy.stats.truncnorm.sf "jax.scipy.stats.truncnorm.sf")(x, a, b\[, loc, scale\]) | Truncated normal distribution survival function. |

### jax.scipy.stats.uniform[\#](#module-jax.scipy.stats.uniform "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.uniform.logpdf.html#jax.scipy.stats.uniform.logpdf "jax.scipy.stats.uniform.logpdf")(x\[, loc, scale\]) | Uniform log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.uniform.pdf.html#jax.scipy.stats.uniform.pdf "jax.scipy.stats.uniform.pdf")(x\[, loc, scale\]) | Uniform probability distribution function. |
| [`cdf`](_autosummary/jax.scipy.stats.uniform.cdf.html#jax.scipy.stats.uniform.cdf "jax.scipy.stats.uniform.cdf")(x\[, loc, scale\]) | Uniform cumulative distribution function. |
| [`ppf`](_autosummary/jax.scipy.stats.uniform.ppf.html#jax.scipy.stats.uniform.ppf "jax.scipy.stats.uniform.ppf")(q\[, loc, scale\]) | Uniform distribution percent point function. |

### jax.scipy.stats.gaussian_kde[\#](#jax-scipy-stats-gaussian-kde "Link to this heading")

|  |  |
|----|----|
| [`gaussian_kde`](_autosummary/jax.scipy.stats.gaussian_kde.html#jax.scipy.stats.gaussian_kde "jax.scipy.stats.gaussian_kde")(dataset\[, bw_method, weights\]) | Gaussian Kernel Density Estimator |
| [`gaussian_kde.evaluate`](_autosummary/jax.scipy.stats.gaussian_kde.evaluate.html#jax.scipy.stats.gaussian_kde.evaluate "jax.scipy.stats.gaussian_kde.evaluate")(points) | Evaluate the Gaussian KDE on the given points. |
| [`gaussian_kde.integrate_gaussian`](_autosummary/jax.scipy.stats.gaussian_kde.integrate_gaussian.html#jax.scipy.stats.gaussian_kde.integrate_gaussian "jax.scipy.stats.gaussian_kde.integrate_gaussian")(mean, cov) | Integrate the distribution weighted by a Gaussian. |
| [`gaussian_kde.integrate_box_1d`](_autosummary/jax.scipy.stats.gaussian_kde.integrate_box_1d.html#jax.scipy.stats.gaussian_kde.integrate_box_1d "jax.scipy.stats.gaussian_kde.integrate_box_1d")(low, high) | Integrate the distribution over the given limits. |
| [`gaussian_kde.integrate_kde`](_autosummary/jax.scipy.stats.gaussian_kde.integrate_kde.html#jax.scipy.stats.gaussian_kde.integrate_kde "jax.scipy.stats.gaussian_kde.integrate_kde")(other) | Integrate the product of two Gaussian KDE distributions. |
| [`gaussian_kde.resample`](_autosummary/jax.scipy.stats.gaussian_kde.resample.html#jax.scipy.stats.gaussian_kde.resample "jax.scipy.stats.gaussian_kde.resample")(key\[, shape\]) | Randomly sample a dataset from the estimated pdf |
| [`gaussian_kde.pdf`](_autosummary/jax.scipy.stats.gaussian_kde.pdf.html#jax.scipy.stats.gaussian_kde.pdf "jax.scipy.stats.gaussian_kde.pdf")(x) | Probability density function |
| [`gaussian_kde.logpdf`](_autosummary/jax.scipy.stats.gaussian_kde.logpdf.html#jax.scipy.stats.gaussian_kde.logpdf "jax.scipy.stats.gaussian_kde.logpdf")(x) | Log probability density function |

### jax.scipy.stats.vonmises[\#](#module-jax.scipy.stats.vonmises "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.vonmises.logpdf.html#jax.scipy.stats.vonmises.logpdf "jax.scipy.stats.vonmises.logpdf")(x, kappa) | von Mises log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.vonmises.pdf.html#jax.scipy.stats.vonmises.pdf "jax.scipy.stats.vonmises.pdf")(x, kappa) | von Mises probability distribution function. |

### jax.scipy.stats.wrapcauchy[\#](#module-jax.scipy.stats.wrapcauchy "Link to this heading")

|  |  |
|----|----|
| [`logpdf`](_autosummary/jax.scipy.stats.wrapcauchy.logpdf.html#jax.scipy.stats.wrapcauchy.logpdf "jax.scipy.stats.wrapcauchy.logpdf")(x, c) | Wrapped Cauchy log probability distribution function. |
| [`pdf`](_autosummary/jax.scipy.stats.wrapcauchy.pdf.html#jax.scipy.stats.wrapcauchy.pdf "jax.scipy.stats.wrapcauchy.pdf")(x, c) | Wrapped Cauchy probability distribution function. |

[](_autosummary/jax.numpy.linalg.vecdot.html "previous page")

previous

jax.numpy.linalg.vecdot

[](_autosummary/jax.scipy.cluster.vq.vq.html "next page")

next

jax.scipy.cluster.vq.vq

Contents

- [jax.scipy.cluster](#module-jax.scipy.cluster.vq)
- [jax.scipy.fft](#module-jax.scipy.fft)
- [jax.scipy.integrate](#module-jax.scipy.integrate)
- [jax.scipy.interpolate](#module-jax.scipy.interpolate)
- [jax.scipy.linalg](#module-jax.scipy.linalg)
- [jax.scipy.ndimage](#module-jax.scipy.ndimage)
- [jax.scipy.optimize](#module-jax.scipy.optimize)
- [jax.scipy.signal](#module-jax.scipy.signal)
- [jax.scipy.spatial.transform](#module-jax.scipy.spatial.transform)
- [jax.scipy.sparse.linalg](#module-jax.scipy.sparse.linalg)
- [jax.scipy.special](#module-jax.scipy.special)
- [jax.scipy.stats](#module-jax.scipy.stats)
  - [jax.scipy.stats.bernoulli](#module-jax.scipy.stats.bernoulli)
  - [jax.scipy.stats.beta](#module-jax.scipy.stats.beta)
  - [jax.scipy.stats.betabinom](#module-jax.scipy.stats.betabinom)
  - [jax.scipy.stats.binom](#module-jax.scipy.stats.binom)
  - [jax.scipy.stats.cauchy](#module-jax.scipy.stats.cauchy)
  - [jax.scipy.stats.chi2](#module-jax.scipy.stats.chi2)
  - [jax.scipy.stats.dirichlet](#module-jax.scipy.stats.dirichlet)
  - [jax.scipy.stats.expon](#module-jax.scipy.stats.expon)
  - [jax.scipy.stats.gamma](#module-jax.scipy.stats.gamma)
  - [jax.scipy.stats.gumbel_l](#module-jax.scipy.stats.gumbel_l)
  - [jax.scipy.stats.gumbel_r](#module-jax.scipy.stats.gumbel_r)
  - [jax.scipy.stats.gennorm](#module-jax.scipy.stats.gennorm)
  - [jax.scipy.stats.geom](#module-jax.scipy.stats.geom)
  - [jax.scipy.stats.laplace](#module-jax.scipy.stats.laplace)
  - [jax.scipy.stats.logistic](#module-jax.scipy.stats.logistic)
  - [jax.scipy.stats.multinomial](#module-jax.scipy.stats.multinomial)
  - [jax.scipy.stats.multivariate_normal](#module-jax.scipy.stats.multivariate_normal)
  - [jax.scipy.stats.nbinom](#module-jax.scipy.stats.nbinom)
  - [jax.scipy.stats.norm](#module-jax.scipy.stats.norm)
  - [jax.scipy.stats.pareto](#module-jax.scipy.stats.pareto)
  - [jax.scipy.stats.poisson](#module-jax.scipy.stats.poisson)
  - [jax.scipy.stats.t](#module-jax.scipy.stats.t)
  - [jax.scipy.stats.truncnorm](#module-jax.scipy.stats.truncnorm)
  - [jax.scipy.stats.uniform](#module-jax.scipy.stats.uniform)
  - [jax.scipy.stats.gaussian_kde](#jax-scipy-stats-gaussian-kde)
  - [jax.scipy.stats.vonmises](#module-jax.scipy.stats.vonmises)
  - [jax.scipy.stats.wrapcauchy](#module-jax.scipy.stats.wrapcauchy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
