- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.jet` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.jet.rst "Download source file")
-  .pdf

# jax.experimental.jet module

## Contents

- [API](#api)
  - [`jet()`](#jax.experimental.jet.jet)

# `jax.experimental.jet` module[\#](#module-jax.experimental.jet "Link to this heading")

Jet is an experimental module for higher-order automatic differentiation that does not rely on repeated first-order automatic differentiation.

How? Through the propagation of truncated Taylor polynomials. Consider a function \\f = g \circ h\\, some point \\x\\ and some offset \\v\\. First-order automatic differentiation (such as [`jax.jvp()`](_autosummary/jax.jvp.html#jax.jvp "jax.jvp")) computes the pair \\(f(x), \partial f(x)\[v\])\\ from the pair \\(h(x), \partial h(x)\[v\])\\.

[`jet()`](#jax.experimental.jet.jet "jax.experimental.jet.jet") implements the higher-order analogue: Given the tuple

\\(h_0, ... h_K) := (h(x), \partial h(x)\[v\], \partial^2 h(x)\[v, v\], ..., \partial^K h(x)\[v,...,v\]),\\

which represents a \\K\\-th order Taylor approximation of \\h\\ at \\x\\, [`jet()`](#jax.experimental.jet.jet "jax.experimental.jet.jet") returns a \\K\\-th order Taylor approximation of \\f\\ at \\x\\,

\\(f_0, ..., f_K) := (f(x), \partial f(x)\[v\], \partial^2 f(x)\[v, v\], ..., \partial^K f(x)\[v,...,v\]).\\

More specifically, [`jet()`](#jax.experimental.jet.jet "jax.experimental.jet.jet") computes

\\f_0, (f_1, . . . , f_K) = \texttt{jet} (f, h_0, (h_1, . . . , h_K))\\

and can thus be used for high-order automatic differentiation of \\f\\. Details are explained in [these notes](https://github.com/jax-ml/jax/files/6717197/jet.pdf).

Note

Help improve [`jet()`](#jax.experimental.jet.jet "jax.experimental.jet.jet") by contributing [outstanding primitive rules](https://github.com/jax-ml/jax/issues/2431).

## API[\#](#api "Link to this heading")

jax.experimental.jet.jet(*fun*, *primals*, *series*, *factorial_scaled=True*, *\*\*\_*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/jet.py#L79-L169)[\#](#jax.experimental.jet.jet "Link to this definition")  
Taylor-mode higher-order automatic differentiation.

Parameters:  
- **fun** – Function to be differentiated. Its arguments should be arrays, scalars, or standard Python containers of arrays or scalars. It should return an array, scalar, or standard Python container of arrays or scalars.

- **primals** – The primal values at which the Taylor approximation of `fun` should be evaluated. Should be either a tuple or a list of arguments, and its length should be equal to the number of positional parameters of `fun`.

- **series** – Higher order Taylor-series-coefficients. Together, primals and series make up a truncated Taylor polynomial. Should be either a tuple or a list of tuples or lists, and its length dictates the degree of the truncated Taylor polynomial.

- **factorial_scaled** – If True, each term in both the input and output series is scaled by the factorial of its order, so that the input and output series is a Taylor series. This is the default behavior so that the n-th order term in the input and output series is the n-th order derivative of the function. If False, the input and output series are the non-factorial scaled Taylor coefficients (i.e., the constant coefficients for each term in the Taylor series).

Returns:  
A `(primals_out,`` ``series_out)` pair, where `primals_out` is `fun(*primals)`, and together, `primals_out` and `series_out` are a truncated Taylor polynomial of \\f(h(\cdot))\\. The `primals_out` value has the same Python tree structure as `primals`, and the `series_out` value the same Python tree structure as `series`.

For example:

    >>> import jax
    >>> import jax.numpy as np

Consider the function \\h(z) = z^3\\, \\x = 0.5\\, and the first few Taylor coefficients \\h_0=x^3\\, \\h_1=3x^2\\, and \\h_2=6x\\. Let \\f(y) = \sin(y)\\.

    >>> h0, h1, h2 = 0.5**3., 3.*0.5**2., 6.*0.5
    >>> f, df, ddf = np.sin, np.cos, lambda *args: -np.sin(*args)

[`jet()`](#jax.experimental.jet.jet "jax.experimental.jet.jet") returns the Taylor coefficients of \\f(h(z)) = \sin(z^3)\\ according to Faà di Bruno’s formula:

    >>> f0, (f1, f2) =  jet(f, (h0,), ((h1, h2),))
    >>> print(f0,  f(h0))
    0.12467473 0.12467473

    >>> print(f1, df(h0) * h1)
    0.74414825 0.74414825

    >>> print(f2, ddf(h0) * h1 ** 2 + df(h0) * h2)
    2.9064636 2.9064634

[](jax.experimental.custom_partitioning.html "previous page")

previous

`jax.experimental.custom_partitioning` module

[](jax.experimental.key_reuse.html "next page")

next

`jax.experimental.key_reuse` module

Contents

- [API](#api)
  - [`jet()`](#jax.experimental.jet.jet)

By The JAX authors

© Copyright 2024, The JAX Authors.\
