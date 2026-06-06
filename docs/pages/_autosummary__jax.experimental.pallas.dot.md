- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.dot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.dot.rst "Download source file")
-  .pdf

# jax.experimental.pallas.dot

## Contents

- [`dot()`](#jax.experimental.pallas.dot)

# jax.experimental.pallas.dot[\#](#jax-experimental-pallas-dot "Link to this heading")

jax.experimental.pallas.dot(*a*, *b*, *trans_a=False*, *trans_b=False*, *allow_tf32=None*, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L503-L546)[\#](#jax.experimental.pallas.dot "Link to this definition")  
Computes the dot product of two arrays.

The inputs can optionally be transposed before computing the product. Depending on the hardware, this can be cheaper than computing the transpose beforehand.

Parameters:  
- **a** – The left-hand size of the dot product, of shape `(...,`` ``N)`.

- **b** – The right-hand size of the dot product, of shape `(...N,`` ``M)`.

- **trans_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to transpose `a` before the product.

- **trans_b** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to transpose `b` before the product.

- **allow_tf32** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – Whether to use tf32 precision. Mutually exclusive with `precision`.

- **precision** – Specifies the precision of the dot product.

See also

[`jax.numpy.dot()`](jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot")

[](jax.experimental.pallas.debug_print.html "previous page")

previous

jax.experimental.pallas.debug_print

[](jax.experimental.pallas.get_global.html "next page")

next

jax.experimental.pallas.get_global

Contents

- [`dot()`](#jax.experimental.pallas.dot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
