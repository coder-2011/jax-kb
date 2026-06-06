- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_gather

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_gather.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_gather

## Contents

- [`bcoo_gather()`](#jax.experimental.sparse.bcoo_gather)

# jax.experimental.sparse.bcoo_gather[\#](#jax-experimental-sparse-bcoo-gather "Link to this heading")

jax.experimental.sparse.bcoo_gather(*operand*, *start_indices*, *dimension_numbers*, *slice_sizes*, *\**, *unique_indices=False*, *indices_are_sorted=False*, *mode=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2298-L2360)[\#](#jax.experimental.sparse.bcoo_gather "Link to this definition")  
BCOO version of lax.gather.

Parameters:  
- **operand** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO"))

- **start_indices** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **dimension_numbers** ([*GatherDimensionNumbers*](../jax.lax.html#jax.lax.GatherDimensionNumbers "jax.lax.GatherDimensionNumbers"))

- **slice_sizes** (*Shape*)

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") *\|* *None*)

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")

[](jax.experimental.sparse.bcoo_fromdense.html "previous page")

previous

jax.experimental.sparse.bcoo_fromdense

[](jax.experimental.sparse.bcoo_multiply_dense.html "next page")

next

jax.experimental.sparse.bcoo_multiply_dense

Contents

- [`bcoo_gather()`](#jax.experimental.sparse.bcoo_gather)

By The JAX authors

© Copyright 2024, The JAX Authors.\
