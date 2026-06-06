- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dot.rst "Download source file")
-  .pdf

# jax.lax.dot

## Contents

- [`dot()`](#jax.lax.dot)

# jax.lax.dot[\#](#jax-lax-dot "Link to this heading")

jax.lax.dot(*lhs*, *rhs*, *\**, *dimension_numbers=None*, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2538-L2622)[\#](#jax.lax.dot "Link to this definition")  
General dot product/contraction operator.

This operation lowers directly to the [stablehlo.dot_general](https://openxla.org/stablehlo/spec#dot_general) operation.

The semantics of `dot_general` are complicated, but most users should not have to use it directly. Instead, you can use higher-level functions like [`jax.numpy.dot()`](jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot"), [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"), [`jax.numpy.tensordot()`](jax.numpy.tensordot.html#jax.numpy.tensordot "jax.numpy.tensordot"), [`jax.numpy.einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"), and others which will construct appropriate calls to `dot_general` under the hood. If you really want to understand `dot_general` itself, we recommend reading XLA’s [DotGeneral](https://www.openxla.org/xla/operation_semantics#dotgeneral) operator documentation.

Parameters:  
- **lhs** (*ArrayLike*) – an array

- **rhs** (*ArrayLike*) – an array

- **dimension_numbers** (*DotDimensionNumbers* *\|* *None*) – an optional tuple of tuples of sequences of ints of the form `((lhs_contracting_dims,`` ``rhs_contracting_dims),`` ``(lhs_batch_dims,`` ``rhs_batch_dims))`. This may be left unspecified in the common case of un-batched matrix-matrix, matrix-vector, or vector-vector dot products, as determined by the shape of `lhs` and `rhs`.

- **precision** (*PrecisionLike*) –

  Optional. This parameter controls the numerics of the computation, and it can be one of the following:

  - `None`, which means the default precision for the current backend,

  - a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value or a tuple of two [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enums indicating precision of `` lhs` `` and `rhs`, or

  - a [`DotAlgorithm`](../jax.lax.html#jax.lax.DotAlgorithm "jax.lax.DotAlgorithm") or a [`DotAlgorithmPreset`](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax.lax.DotAlgorithmPreset") indicating the algorithm that must be used to accumulate the dot product.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. This parameter controls the data type output by the dot product. By default, the output element type of this operation will match the `lhs` and `rhs` input element types under the usual type promotion rules. Setting `preferred_element_type` to a specific `dtype` will mean that the operation returns that element type. When `precision` is not a [`DotAlgorithm`](../jax.lax.html#jax.lax.DotAlgorithm "jax.lax.DotAlgorithm") or [`DotAlgorithmPreset`](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax.lax.DotAlgorithmPreset"), `preferred_element_type` provides a hint to the compiler to accumulate the dot product using this data type.

- **out_sharding** – an optional sharding specification for the output. If not specified, it will be determined automatically by the compiler.

Returns:  
An array whose first dimensions are the (shared) batch dimensions, followed by the `lhs` non-contracting/non-batch dimensions, and finally the `rhs` non-contracting/non-batch dimensions.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.div.html "previous page")

previous

jax.lax.div

[](jax.lax.dot_general.html "next page")

next

jax.lax.dot_general

Contents

- [`dot()`](#jax.lax.dot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
