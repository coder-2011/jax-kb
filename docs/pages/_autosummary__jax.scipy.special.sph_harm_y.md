- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.sph_harm_y

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.sph_harm_y.rst "Download source file")
-  .pdf

# jax.scipy.special.sph_harm_y

## Contents

- [`sph_harm_y()`](#jax.scipy.special.sph_harm_y)

# jax.scipy.special.sph_harm_y[\#](#jax-scipy-special-sph-harm-y "Link to this heading")

jax.scipy.special.sph_harm_y(*n*, *m*, *theta*, *phi*, *diff_n=None*, *n_max=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L2322-L2372)[\#](#jax.scipy.special.sph_harm_y "Link to this definition")  
Computes the spherical harmonics.

The JAX version has one extra argument n_max, the maximum value in n.

The spherical harmonic of degree n and order m can be written as \\Y_n^m(\theta, \phi) = N_n^m \* P_n^m(\cos \theta) \* \exp(i m \phi)\\, where \\N_n^m = \sqrt{\frac{\left(2n+1\right) \left(n-m\right)!} {4 \pi \left(n+m\right)!}}\\ is the normalization factor and \\\theta\\ and \\\phi\\ are the colatitude and longitude, respectively. \\N_n^m\\ is chosen in the way that the spherical harmonics form a set of orthonormal basis functions of \\L^2(S^2)\\.

Parameters:  
- **n** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The degree of the harmonic; must have n \>= 0. The standard notation for degree in descriptions of spherical harmonics is l (lower case L). We use n here to be consistent with scipy.special.sph_harm_y. Return values for n \< 0 are undefined.

- **m** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The order of the harmonic; must have \|m\| \<= n. Return values for \|m\| \> n are undefined.

- **theta** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The polar (colatitudinal) coordinate; must be in \[0, pi\].

- **phi** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The azimuthal (longitudinal) coordinate; must be in \[0, 2\*pi\].

- **diff_n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Unsupported by JAX.

- **n_max** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – The maximum degree max(n). If the supplied n_max is not the true maximum value of n, the results are clipped to n_max. For example, sph_harm(m=jnp.array(\[2\]), n=jnp.array(\[10\]), theta, phi, n_max=6) actually returns sph_harm(m=jnp.array(\[2\]), n=jnp.array(\[6\]), theta, phi, n_max=6)

Returns:  
A 1D array containing the spherical harmonics at (m, n, theta, phi).

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.spence.html "previous page")

previous

jax.scipy.special.spence

[](jax.scipy.special.wofz.html "next page")

next

jax.scipy.special.wofz

Contents

- [`sph_harm_y()`](#jax.scipy.special.sph_harm_y)

By The JAX authors

© Copyright 2024, The JAX Authors.\
