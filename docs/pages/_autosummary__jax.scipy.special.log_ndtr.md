- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.log_ndtr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.log_ndtr.rst "Download source file")
-  .pdf

# jax.scipy.special.log_ndtr

## Contents

- [`log_ndtr`](#jax.scipy.special.log_ndtr)

# jax.scipy.special.log_ndtr[\#](#jax-scipy-special-log-ndtr "Link to this heading")

jax.scipy.special.log_ndtr *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1614-L1722)[\#](#jax.scipy.special.log_ndtr "Link to this definition")  
Log Normal distribution function.

JAX implementation of [`scipy.special.log_ndtr`](https://scipy.github.io/devdocs/reference/generated/scipy.special.log_ndtr.html#scipy.special.log_ndtr "(in SciPy v1.19.0.dev)").

For details of the Normal distribution function see ndtr.

This function calculates \\\log(\mathrm{ndtr}(x))\\ by either calling \\\log(\mathrm{ndtr}(x))\\ or using an asymptotic series. Specifically:

- For x \> upper_segment, use the approximation -ndtr(-x) based on \\\log(1-x) \approx -x, x \ll 1\\.

- For lower_segment \< x \<= upper_segment, use the existing ndtr technique and take a log.

- For x \<= lower_segment, we use the series approximation of erf to compute the log CDF directly.

The lower_segment is set based on the precision of the input:

\\\begin{split}\begin{align} \mathit{lower\\segment} =& \\ \begin{cases} -20 & x.\mathrm{dtype}=\mathit{float64} \\ -10 & x.\mathrm{dtype}=\mathit{float32} \\ \end{cases} \\ \mathit{upper\\segment} =& \\ \begin{cases} 8& x.\mathrm{dtype}=\mathit{float64} \\ 5& x.\mathrm{dtype}=\mathit{float32} \\ \end{cases} \end{align}\end{split}\\

When x \< lower_segment, the ndtr asymptotic series approximation is:

\\\begin{split}\begin{align} \mathrm{ndtr}(x) =&\\ \mathit{scale} \* (1 + \mathit{sum}) + R_N \\ \mathit{scale} =&\\ \frac{e^{-0.5 x^2}}{-x \sqrt{2 \pi}} \\ \mathit{sum} =&\\ \sum\_{n=1}^N {-1}^n (2n-1)!! / (x^2)^n \\ R_N =&\\ O(e^{-0.5 x^2} (2N+1)!! / \|x\|^{2N+3}) \end{align}\end{split}\\

where \\(2n-1)!! = (2n-1) (2n-3) (2n-5) ... (3) (1)\\ is a [double-factorial](https://en.wikipedia.org/wiki/Double_factorial) operator.

Parameters:  
- **x** (*ArrayLike*) – an array of type float32, float64.

- **series_order** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Positive Python integer. Maximum depth to evaluate the asymptotic expansion. This is the N above.

Returns:  
an array with dtype=x.dtype.

Raises:  
- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – if x.dtype is not handled.

- [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – if series_order is a not Python integer.

- [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – if series_order is not in \[0, 30\].

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.kl_div.html "previous page")

previous

jax.scipy.special.kl_div

[](jax.scipy.special.log_softmax.html "next page")

next

jax.scipy.special.log_softmax

Contents

- [`log_ndtr`](#jax.scipy.special.log_ndtr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
