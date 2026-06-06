- [](index.html)
- [API Reference](jax.html)
- `jax.lax` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.lax.rst "Download source file")
-  .pdf

# jax.lax module

## Contents

- [Operators](#operators)
- [Control flow operators](#control-flow-operators)
- [Custom gradient operators](#custom-gradient-operators)
- [Parallel operators](#parallel-operators)
- [Sharding-related operators](#sharding-related-operators)
- [Linear algebra operators (jax.lax.linalg)](#module-jax.lax.linalg)
  - [`EigImplementation`](#jax.lax.linalg.EigImplementation)
    - [`EigImplementation.CUSOLVER`](#jax.lax.linalg.EigImplementation.CUSOLVER)
    - [`EigImplementation.LAPACK`](#jax.lax.linalg.EigImplementation.LAPACK)
    - [`EigImplementation.MAGMA`](#jax.lax.linalg.EigImplementation.MAGMA)
  - [`EighImplementation`](#jax.lax.linalg.EighImplementation)
    - [`EighImplementation.JACOBI`](#jax.lax.linalg.EighImplementation.JACOBI)
    - [`EighImplementation.QDWH`](#jax.lax.linalg.EighImplementation.QDWH)
    - [`EighImplementation.QR`](#jax.lax.linalg.EighImplementation.QR)
- [Argument classes](#argument-classes)
  - [`AccuracyMode`](#jax.lax.AccuracyMode)
    - [`AccuracyMode.DEFAULT`](#jax.lax.AccuracyMode.DEFAULT)
    - [`AccuracyMode.HIGHEST`](#jax.lax.AccuracyMode.HIGHEST)
  - [`ConvDimensionNumbers`](#jax.lax.ConvDimensionNumbers)
  - [`ConvGeneralDilatedDimensionNumbers`](#jax.lax.ConvGeneralDilatedDimensionNumbers)
  - [`DotAlgorithm`](#jax.lax.DotAlgorithm)
  - [`DotAlgorithmPreset`](#jax.lax.DotAlgorithmPreset)
    - [`DotAlgorithmPreset.DEFAULT`](#jax.lax.DotAlgorithmPreset.DEFAULT)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_F32`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_ANY`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_ANY)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM)
    - [`DotAlgorithmPreset.F16_F16_F16`](#jax.lax.DotAlgorithmPreset.F16_F16_F16)
    - [`DotAlgorithmPreset.F16_F16_F32`](#jax.lax.DotAlgorithmPreset.F16_F16_F32)
    - [`DotAlgorithmPreset.BF16_BF16_BF16`](#jax.lax.DotAlgorithmPreset.BF16_BF16_BF16)
    - [`DotAlgorithmPreset.BF16_BF16_F32`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32)
    - [`DotAlgorithmPreset.BF16_BF16_F32_X3`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3)
    - [`DotAlgorithmPreset.BF16_BF16_F32_X6`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6)
    - [`DotAlgorithmPreset.BF16_BF16_F32_X9`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X9)
    - [`DotAlgorithmPreset.TF32_TF32_F32`](#jax.lax.DotAlgorithmPreset.TF32_TF32_F32)
    - [`DotAlgorithmPreset.TF32_TF32_F32_X3`](#jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3)
    - [`DotAlgorithmPreset.F32_F32_F32`](#jax.lax.DotAlgorithmPreset.F32_F32_F32)
    - [`DotAlgorithmPreset.F64_F64_F64`](#jax.lax.DotAlgorithmPreset.F64_F64_F64)
    - [`DotAlgorithmPreset.supported_lhs_types`](#jax.lax.DotAlgorithmPreset.supported_lhs_types)
    - [`DotAlgorithmPreset.supported_rhs_types`](#jax.lax.DotAlgorithmPreset.supported_rhs_types)
    - [`DotAlgorithmPreset.accumulation_type`](#jax.lax.DotAlgorithmPreset.accumulation_type)
    - [`DotAlgorithmPreset.supported_output_types()`](#jax.lax.DotAlgorithmPreset.supported_output_types)
  - [`DotDimensionNumbers`](#jax.lax.DotDimensionNumbers)
  - [`FftType`](#jax.lax.FftType)
    - [`FftType.FFT`](#jax.lax.FftType.FFT)
    - [`FftType.IFFT`](#jax.lax.FftType.IFFT)
    - [`FftType.IRFFT`](#jax.lax.FftType.IRFFT)
    - [`FftType.RFFT`](#jax.lax.FftType.RFFT)
  - [`GatherDimensionNumbers`](#jax.lax.GatherDimensionNumbers)
  - [`GatherScatterMode`](#jax.lax.GatherScatterMode)
  - [`Precision`](#jax.lax.Precision)
  - [`PrecisionLike`](#jax.lax.PrecisionLike)
  - [`RaggedDotDimensionNumbers`](#jax.lax.RaggedDotDimensionNumbers)
  - [`RandomAlgorithm`](#jax.lax.RandomAlgorithm)
    - [`RandomAlgorithm.RNG_DEFAULT`](#jax.lax.RandomAlgorithm.RNG_DEFAULT)
    - [`RandomAlgorithm.RNG_THREE_FRY`](#jax.lax.RandomAlgorithm.RNG_THREE_FRY)
    - [`RandomAlgorithm.RNG_PHILOX`](#jax.lax.RandomAlgorithm.RNG_PHILOX)
  - [`RoundingMethod`](#jax.lax.RoundingMethod)
    - [`RoundingMethod.AWAY_FROM_ZERO`](#jax.lax.RoundingMethod.AWAY_FROM_ZERO)
    - [`RoundingMethod.TO_NEAREST_EVEN`](#jax.lax.RoundingMethod.TO_NEAREST_EVEN)
  - [`ScatterDimensionNumbers`](#jax.lax.ScatterDimensionNumbers)
  - [`Tolerance`](#jax.lax.Tolerance)

# `jax.lax` module[\#](#module-jax.lax "Link to this heading")

[`jax.lax`](#module-jax.lax "jax.lax") is a library of primitives operations that underpins libraries such as [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy"). Transformation rules, such as JVP and batching rules, are typically defined as transformations on [`jax.lax`](#module-jax.lax "jax.lax") primitives.

Many of the primitives are thin wrappers around equivalent XLA operations, described by the [XLA operation semantics](https://www.openxla.org/xla/operation_semantics) documentation. In a few cases JAX diverges from XLA, usually to ensure that the set of operations is closed under the operation of JVP and transpose rules.

Where possible, prefer to use libraries such as [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") instead of using [`jax.lax`](#module-jax.lax "jax.lax") directly. The [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") API follows NumPy, and is therefore more stable and less likely to change than the [`jax.lax`](#module-jax.lax "jax.lax") API.

## Operators[\#](#operators "Link to this heading")

|  |  |
|----|----|
| [`abs`](_autosummary/jax.lax.abs.html#jax.lax.abs "jax.lax.abs")(x) | Elementwise absolute value: \\\|x\|\\. |
| [`acos`](_autosummary/jax.lax.acos.html#jax.lax.acos "jax.lax.acos")(x) | Elementwise arc cosine: \\\mathrm{acos}(x)\\. |
| [`acosh`](_autosummary/jax.lax.acosh.html#jax.lax.acosh "jax.lax.acosh")(x) | Elementwise inverse hyperbolic cosine: \\\mathrm{acosh}(x)\\. |
| [`add`](_autosummary/jax.lax.add.html#jax.lax.add "jax.lax.add")(x, y) | Elementwise addition: \\x + y\\. |
| [`after_all`](_autosummary/jax.lax.after_all.html#jax.lax.after_all "jax.lax.after_all")(\*operands) | Merges one or more XLA token values. |
| [`approx_max_k`](_autosummary/jax.lax.approx_max_k.html#jax.lax.approx_max_k "jax.lax.approx_max_k")(operand, k\[, ...\]) | Returns max `k` values and their indices of the `operand` in an approximate manner. |
| [`approx_min_k`](_autosummary/jax.lax.approx_min_k.html#jax.lax.approx_min_k "jax.lax.approx_min_k")(operand, k\[, ...\]) | Returns min `k` values and their indices of the `operand` in an approximate manner. |
| [`argmax`](_autosummary/jax.lax.argmax.html#jax.lax.argmax "jax.lax.argmax")(operand, axis, index_dtype) | Computes the index of the maximum element along `axis`. |
| [`argmin`](_autosummary/jax.lax.argmin.html#jax.lax.argmin "jax.lax.argmin")(operand, axis, index_dtype) | Computes the index of the minimum element along `axis`. |
| [`asin`](_autosummary/jax.lax.asin.html#jax.lax.asin "jax.lax.asin")(x) | Elementwise arc sine: \\\mathrm{asin}(x)\\. |
| [`asinh`](_autosummary/jax.lax.asinh.html#jax.lax.asinh "jax.lax.asinh")(x) | Elementwise inverse hyperbolic sine: \\\mathrm{asinh}(x)\\. |
| [`atan`](_autosummary/jax.lax.atan.html#jax.lax.atan "jax.lax.atan")(x) | Elementwise arc tangent: \\\mathrm{atan}(x)\\. |
| [`atan2`](_autosummary/jax.lax.atan2.html#jax.lax.atan2 "jax.lax.atan2")(x, y) | Elementwise two-term arc tangent: \\\mathrm{atan}({x \over y})\\. |
| [`atanh`](_autosummary/jax.lax.atanh.html#jax.lax.atanh "jax.lax.atanh")(x) | Elementwise inverse hyperbolic tangent: \\\mathrm{atanh}(x)\\. |
| [`batch_matmul`](_autosummary/jax.lax.batch_matmul.html#jax.lax.batch_matmul "jax.lax.batch_matmul")(lhs, rhs\[, precision\]) | Batch matrix multiplication. |
| [`bessel_i0e`](_autosummary/jax.lax.bessel_i0e.html#jax.lax.bessel_i0e "jax.lax.bessel_i0e")(x) | Exponentially scaled modified Bessel function of order 0: \\\mathrm{i0e}(x) = e^{-\|x\|} \mathrm{i0}(x)\\ |
| [`bessel_i1e`](_autosummary/jax.lax.bessel_i1e.html#jax.lax.bessel_i1e "jax.lax.bessel_i1e")(x) | Exponentially scaled modified Bessel function of order 1: \\\mathrm{i1e}(x) = e^{-\|x\|} \mathrm{i1}(x)\\ |
| [`betainc`](_autosummary/jax.lax.betainc.html#jax.lax.betainc "jax.lax.betainc")(a, b, x) | Elementwise regularized incomplete beta integral. |
| [`bitcast_convert_type`](_autosummary/jax.lax.bitcast_convert_type.html#jax.lax.bitcast_convert_type "jax.lax.bitcast_convert_type")(operand, new_dtype) | Elementwise bitcast. |
| [`bitwise_and`](_autosummary/jax.lax.bitwise_and.html#jax.lax.bitwise_and "jax.lax.bitwise_and")(x, y) | Elementwise AND: \\x \wedge y\\. |
| [`bitwise_not`](_autosummary/jax.lax.bitwise_not.html#jax.lax.bitwise_not "jax.lax.bitwise_not")(x) | Elementwise NOT: \\\neg x\\. |
| [`bitwise_or`](_autosummary/jax.lax.bitwise_or.html#jax.lax.bitwise_or "jax.lax.bitwise_or")(x, y) | Elementwise OR: \\x \vee y\\. |
| [`bitwise_xor`](_autosummary/jax.lax.bitwise_xor.html#jax.lax.bitwise_xor "jax.lax.bitwise_xor")(x, y) | Elementwise exclusive OR: \\x \oplus y\\. |
| [`population_count`](_autosummary/jax.lax.population_count.html#jax.lax.population_count "jax.lax.population_count")(x) | Elementwise popcount, count the number of set bits in each element. |
| [`broadcast`](_autosummary/jax.lax.broadcast.html#jax.lax.broadcast "jax.lax.broadcast")(operand, sizes, \*\[, out_sharding\]) | Broadcasts an array, adding new leading dimensions only. |
| [`broadcast_in_dim`](_autosummary/jax.lax.broadcast_in_dim.html#jax.lax.broadcast_in_dim "jax.lax.broadcast_in_dim")(operand, shape, ...\[, ...\]) | General broadcasting operation. |
| [`broadcast_like`](_autosummary/jax.lax.broadcast_like.html#jax.lax.broadcast_like "jax.lax.broadcast_like")(arr, like_arr) | Broadcasts an array to match the shape and sharding of another array. |
| [`broadcast_shapes`](_autosummary/jax.lax.broadcast_shapes.html#jax.lax.broadcast_shapes "jax.lax.broadcast_shapes")(\*shapes) | Returns the shape that results from NumPy broadcasting of shapes. |
| [`broadcast_to_rank`](_autosummary/jax.lax.broadcast_to_rank.html#jax.lax.broadcast_to_rank "jax.lax.broadcast_to_rank")(x, rank) | Adds leading dimensions of `1` to give `x` rank `rank`. |
| [`broadcasted_iota`](_autosummary/jax.lax.broadcasted_iota.html#jax.lax.broadcasted_iota "jax.lax.broadcasted_iota")(dtype, shape, dimension, \*) | Convenience wrapper around `iota`. |
| [`cbrt`](_autosummary/jax.lax.cbrt.html#jax.lax.cbrt "jax.lax.cbrt")(x, \*\[, accuracy\]) | Elementwise cube root: \\\sqrt\[3\]{x}\\. |
| [`ceil`](_autosummary/jax.lax.ceil.html#jax.lax.ceil "jax.lax.ceil")(x) | Elementwise ceiling: \\\left\lceil x \right\rceil\\. |
| [`clamp`](_autosummary/jax.lax.clamp.html#jax.lax.clamp "jax.lax.clamp")(min, x, max) | Elementwise clamp. |
| [`clz`](_autosummary/jax.lax.clz.html#jax.lax.clz "jax.lax.clz")(x) | Elementwise count-leading-zeros. |
| [`collapse`](_autosummary/jax.lax.collapse.html#jax.lax.collapse "jax.lax.collapse")(operand, start_dimension\[, ...\]) | Collapses dimensions of an array into a single dimension. |
| [`complex`](_autosummary/jax.lax.complex.html#jax.lax.complex "jax.lax.complex")(x, y) | Elementwise make complex number: \\x + jy\\. |
| [`composite`](_autosummary/jax.lax.composite.html#jax.lax.composite "jax.lax.composite")(decomposition, name\[, version\]) | Composite with semantics defined by the decomposition function. |
| [`concatenate`](_autosummary/jax.lax.concatenate.html#jax.lax.concatenate "jax.lax.concatenate")(operands, dimension) | Concatenates a sequence of arrays along dimension. |
| [`conj`](_autosummary/jax.lax.conj.html#jax.lax.conj "jax.lax.conj")(x) | Elementwise complex conjugate function: \\\overline{x}\\. |
| [`conv`](_autosummary/jax.lax.conv.html#jax.lax.conv "jax.lax.conv")(lhs, rhs, window_strides, padding\[, ...\]) | Convenience wrapper around conv_general_dilated. |
| [`convert_element_type`](_autosummary/jax.lax.convert_element_type.html#jax.lax.convert_element_type "jax.lax.convert_element_type")(operand, new_dtype) | Elementwise cast. |
| [`conv_dimension_numbers`](_autosummary/jax.lax.conv_dimension_numbers.html#jax.lax.conv_dimension_numbers "jax.lax.conv_dimension_numbers")(lhs_shape, rhs_shape, ...) | Converts convolution dimension_numbers to a ConvDimensionNumbers. |
| [`conv_general_dilated`](_autosummary/jax.lax.conv_general_dilated.html#jax.lax.conv_general_dilated "jax.lax.conv_general_dilated")(lhs, rhs, ...\[, ...\]) | General n-dimensional convolution operator, with optional dilation. |
| [`conv_general_dilated_local`](_autosummary/jax.lax.conv_general_dilated_local.html#jax.lax.conv_general_dilated_local "jax.lax.conv_general_dilated_local")(lhs, rhs, ...\[, ...\]) | General n-dimensional unshared convolution operator with optional dilation. |
| [`conv_general_dilated_patches`](_autosummary/jax.lax.conv_general_dilated_patches.html#jax.lax.conv_general_dilated_patches "jax.lax.conv_general_dilated_patches")(lhs, ...\[, ...\]) | Extract patches subject to the receptive field of conv_general_dilated. |
| [`conv_transpose`](_autosummary/jax.lax.conv_transpose.html#jax.lax.conv_transpose "jax.lax.conv_transpose")(lhs, rhs, strides, padding\[, ...\]) | Convenience wrapper for calculating the N-d convolution "transpose". |
| [`conv_with_general_padding`](_autosummary/jax.lax.conv_with_general_padding.html#jax.lax.conv_with_general_padding "jax.lax.conv_with_general_padding")(lhs, rhs, ...\[, ...\]) | Convenience wrapper around conv_general_dilated. |
| [`cos`](_autosummary/jax.lax.cos.html#jax.lax.cos "jax.lax.cos")(x, \*\[, accuracy\]) | Elementwise cosine: \\\mathrm{cos}(x)\\. |
| [`cosh`](_autosummary/jax.lax.cosh.html#jax.lax.cosh "jax.lax.cosh")(x) | Elementwise hyperbolic cosine: \\\mathrm{cosh}(x)\\. |
| [`cumlogsumexp`](_autosummary/jax.lax.cumlogsumexp.html#jax.lax.cumlogsumexp "jax.lax.cumlogsumexp")(operand\[, axis, reverse\]) | Computes a cumulative logsumexp along axis. |
| [`cummax`](_autosummary/jax.lax.cummax.html#jax.lax.cummax "jax.lax.cummax")(operand\[, axis, reverse\]) | Computes a cumulative maximum along axis. |
| [`cummin`](_autosummary/jax.lax.cummin.html#jax.lax.cummin "jax.lax.cummin")(operand\[, axis, reverse\]) | Computes a cumulative minimum along axis. |
| [`cumprod`](_autosummary/jax.lax.cumprod.html#jax.lax.cumprod "jax.lax.cumprod")(operand\[, axis, reverse\]) | Computes a cumulative product along axis. |
| [`cumsum`](_autosummary/jax.lax.cumsum.html#jax.lax.cumsum "jax.lax.cumsum")(operand\[, axis, reverse\]) | Computes a cumulative sum along axis. |
| [`digamma`](_autosummary/jax.lax.digamma.html#jax.lax.digamma "jax.lax.digamma")(x) | Elementwise digamma: \\\psi(x)\\. |
| [`div`](_autosummary/jax.lax.div.html#jax.lax.div "jax.lax.div")(x, y) | Elementwise division: \\x \over y\\. |
| [`dot`](_autosummary/jax.lax.dot.html#jax.lax.dot "jax.lax.dot")(lhs, rhs, \*\[, dimension_numbers, ...\]) | General dot product/contraction operator. |
| [`dot_general`](_autosummary/jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general")(lhs, rhs, dimension_numbers\[, ...\]) | Alias of [`jax.lax.dot()`](_autosummary/jax.lax.dot.html#jax.lax.dot "jax.lax.dot"). |
| [`dynamic_index_in_dim`](_autosummary/jax.lax.dynamic_index_in_dim.html#jax.lax.dynamic_index_in_dim "jax.lax.dynamic_index_in_dim")(operand, index\[, axis, ...\]) | Convenience wrapper around dynamic_slice to perform int indexing. |
| [`dynamic_slice`](_autosummary/jax.lax.dynamic_slice.html#jax.lax.dynamic_slice "jax.lax.dynamic_slice")(operand, start_indices, ...\[, ...\]) | Wraps XLA's [DynamicSlice](https://www.openxla.org/xla/operation_semantics#dynamicslice) operator. |
| [`dynamic_slice_in_dim`](_autosummary/jax.lax.dynamic_slice_in_dim.html#jax.lax.dynamic_slice_in_dim "jax.lax.dynamic_slice_in_dim")(operand, start_index, ...) | Convenience wrapper around `lax.dynamic_slice()` applied to one dimension. |
| [`dynamic_update_index_in_dim`](_autosummary/jax.lax.dynamic_update_index_in_dim.html#jax.lax.dynamic_update_index_in_dim "jax.lax.dynamic_update_index_in_dim")(operand, update, ...) | Convenience wrapper around [`dynamic_update_slice()`](_autosummary/jax.lax.dynamic_update_slice.html#jax.lax.dynamic_update_slice "jax.lax.dynamic_update_slice") to update a slice of size 1 in a single `axis`. |
| [`dynamic_update_slice`](_autosummary/jax.lax.dynamic_update_slice.html#jax.lax.dynamic_update_slice "jax.lax.dynamic_update_slice")(operand, update, ...\[, ...\]) | Wraps XLA's [DynamicUpdateSlice](https://www.openxla.org/xla/operation_semantics#dynamicupdateslice) operator. |
| [`dynamic_update_slice_in_dim`](_autosummary/jax.lax.dynamic_update_slice_in_dim.html#jax.lax.dynamic_update_slice_in_dim "jax.lax.dynamic_update_slice_in_dim")(operand, update, ...) | Convenience wrapper around [`dynamic_update_slice()`](_autosummary/jax.lax.dynamic_update_slice.html#jax.lax.dynamic_update_slice "jax.lax.dynamic_update_slice") to update a slice in a single `axis`. |
| [`empty`](_autosummary/jax.lax.empty.html#jax.lax.empty "jax.lax.empty")(shape, dtype, \*\[, out_sharding\]) | Create an empty array of possibly uninitialized values. |
| [`eq`](_autosummary/jax.lax.eq.html#jax.lax.eq "jax.lax.eq")(x, y) | Elementwise equals: \\x = y\\. |
| [`erf`](_autosummary/jax.lax.erf.html#jax.lax.erf "jax.lax.erf")(x) | Elementwise error function: \\\mathrm{erf}(x)\\. |
| [`erfc`](_autosummary/jax.lax.erfc.html#jax.lax.erfc "jax.lax.erfc")(x) | Elementwise complementary error function: \\\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)\\. |
| [`erf_inv`](_autosummary/jax.lax.erf_inv.html#jax.lax.erf_inv "jax.lax.erf_inv")(x) | Elementwise inverse error function: \\\mathrm{erf}^{-1}(x)\\. |
| [`exp`](_autosummary/jax.lax.exp.html#jax.lax.exp "jax.lax.exp")(x, \*\[, accuracy\]) | Elementwise exponential: \\e^x\\. |
| [`exp2`](_autosummary/jax.lax.exp2.html#jax.lax.exp2 "jax.lax.exp2")(x, \*\[, accuracy\]) | Elementwise base-2 exponential: \\2^x\\. |
| [`expand_dims`](_autosummary/jax.lax.expand_dims.html#jax.lax.expand_dims "jax.lax.expand_dims")(array, dimensions) | Insert any number of size 1 dimensions into an array. |
| [`expm1`](_autosummary/jax.lax.expm1.html#jax.lax.expm1 "jax.lax.expm1")(x, \*\[, accuracy\]) | Elementwise \\e^{x} - 1\\. |
| [`fft`](_autosummary/jax.lax.fft.html#jax.lax.fft "jax.lax.fft")(x, fft_type, fft_lengths) |  |
| [`floor`](_autosummary/jax.lax.floor.html#jax.lax.floor "jax.lax.floor")(x) | Elementwise floor: \\\left\lfloor x \right\rfloor\\. |
| [`full`](_autosummary/jax.lax.full.html#jax.lax.full "jax.lax.full")(shape, fill_value\[, dtype, sharding\]) | Returns an array of shape filled with fill_value. |
| [`full_like`](_autosummary/jax.lax.full_like.html#jax.lax.full_like "jax.lax.full_like")(x, fill_value\[, dtype, shape, ...\]) | Create a full array like np.full based on the example array x. |
| [`gather`](_autosummary/jax.lax.gather.html#jax.lax.gather "jax.lax.gather")(operand, start_indices, ...\[, ...\]) | Gather operator. |
| [`ge`](_autosummary/jax.lax.ge.html#jax.lax.ge "jax.lax.ge")(x, y) | Elementwise greater-than-or-equals: \\x \geq y\\. |
| [`gt`](_autosummary/jax.lax.gt.html#jax.lax.gt "jax.lax.gt")(x, y) | Elementwise greater-than: \\x \> y\\. |
| [`igamma`](_autosummary/jax.lax.igamma.html#jax.lax.igamma "jax.lax.igamma")(a, x) | Elementwise regularized incomplete gamma function. |
| [`igamma_grad_a`](_autosummary/jax.lax.igamma_grad_a.html#jax.lax.igamma_grad_a "jax.lax.igamma_grad_a")(a, x) | Elementwise derivative of the regularized incomplete gamma function. |
| [`igammac`](_autosummary/jax.lax.igammac.html#jax.lax.igammac "jax.lax.igammac")(a, x) | Elementwise complementary regularized incomplete gamma function. |
| [`imag`](_autosummary/jax.lax.imag.html#jax.lax.imag "jax.lax.imag")(x) | Elementwise extract imaginary part: \\\mathrm{Im}(x)\\. |
| [`index_in_dim`](_autosummary/jax.lax.index_in_dim.html#jax.lax.index_in_dim "jax.lax.index_in_dim")(operand, index\[, axis, keepdims\]) | Convenience wrapper around `lax.slice()` to perform int indexing. |
| [`index_take`](_autosummary/jax.lax.index_take.html#jax.lax.index_take "jax.lax.index_take")(src, idxs, axes) |  |
| [`integer_pow`](_autosummary/jax.lax.integer_pow.html#jax.lax.integer_pow "jax.lax.integer_pow")(x, y) | Elementwise power: \\x^y\\, where \\y\\ is a static integer. |
| [`iota`](_autosummary/jax.lax.iota.html#jax.lax.iota "jax.lax.iota")(dtype, size) | Wraps XLA's [Iota](https://www.openxla.org/xla/operation_semantics#iota) operator. |
| [`is_finite`](_autosummary/jax.lax.is_finite.html#jax.lax.is_finite "jax.lax.is_finite")(x) | Elementwise \\\mathrm{isfinite}\\. |
| [`le`](_autosummary/jax.lax.le.html#jax.lax.le "jax.lax.le")(x, y) | Elementwise less-than-or-equals: \\x \leq y\\. |
| [`lgamma`](_autosummary/jax.lax.lgamma.html#jax.lax.lgamma "jax.lax.lgamma")(x) | Elementwise log gamma: \\\mathrm{log}(\Gamma(x))\\. |
| [`log`](_autosummary/jax.lax.log.html#jax.lax.log "jax.lax.log")(x, \*\[, accuracy\]) | Elementwise natural logarithm: \\\mathrm{log}(x)\\. |
| [`log1p`](_autosummary/jax.lax.log1p.html#jax.lax.log1p "jax.lax.log1p")(x, \*\[, accuracy\]) | Elementwise \\\mathrm{log}(1 + x)\\. |
| [`logistic`](_autosummary/jax.lax.logistic.html#jax.lax.logistic "jax.lax.logistic")(x, \*\[, accuracy\]) | Elementwise logistic (sigmoid) function: \\\frac{1}{1 + e^{-x}}\\. |
| [`lt`](_autosummary/jax.lax.lt.html#jax.lax.lt "jax.lax.lt")(x, y) | Elementwise less-than: \\x \< y\\. |
| [`max`](_autosummary/jax.lax.max.html#jax.lax.max "jax.lax.max")(x, y) | Elementwise maximum: \\\mathrm{max}(x, y)\\. |
| [`min`](_autosummary/jax.lax.min.html#jax.lax.min "jax.lax.min")(x, y) | Elementwise minimum: \\\mathrm{min}(x, y)\\ |
| [`mul`](_autosummary/jax.lax.mul.html#jax.lax.mul "jax.lax.mul")(x, y, \*\[, out_dtype\]) | Elementwise multiplication: \\x \times y\\. |
| [`mulhi`](_autosummary/jax.lax.mulhi.html#jax.lax.mulhi "jax.lax.mulhi")(x, y, /) | Elementwise multiply-high: high bits of \\x \times y\\. |
| [`ne`](_autosummary/jax.lax.ne.html#jax.lax.ne "jax.lax.ne")(x, y) | Elementwise not-equals: \\x \neq y\\. |
| [`neg`](_autosummary/jax.lax.neg.html#jax.lax.neg "jax.lax.neg")(x) | Elementwise negation: \\-x\\. |
| [`nextafter`](_autosummary/jax.lax.nextafter.html#jax.lax.nextafter "jax.lax.nextafter")(x1, x2) | Returns the next representable value after `x1` in the direction of `x2`. |
| [`optimization_barrier`](_autosummary/jax.lax.optimization_barrier.html#jax.lax.optimization_barrier "jax.lax.optimization_barrier")(operand, /) | Prevents the compiler from moving operations across the barrier. |
| [`pad`](_autosummary/jax.lax.pad.html#jax.lax.pad "jax.lax.pad")(operand, padding_value, padding_config) | Applies low, high, and/or interior padding to an array. |
| [`platform_dependent`](_autosummary/jax.lax.platform_dependent.html#jax.lax.platform_dependent "jax.lax.platform_dependent")(\*args\[, default\]) | Stages out platform-specific code. |
| [`polygamma`](_autosummary/jax.lax.polygamma.html#jax.lax.polygamma "jax.lax.polygamma")(m, x) | Elementwise polygamma: \\\psi^{(m)}(x)\\. |
| [`population_count`](_autosummary/jax.lax.population_count.html#jax.lax.population_count "jax.lax.population_count")(x) | Elementwise popcount, count the number of set bits in each element. |
| [`pow`](_autosummary/jax.lax.pow.html#jax.lax.pow "jax.lax.pow")(x, y) | Elementwise power: \\x^y\\. |
| [`random_gamma_grad`](_autosummary/jax.lax.random_gamma_grad.html#jax.lax.random_gamma_grad "jax.lax.random_gamma_grad")(\*args) |  |
| [`ragged_all_to_all`](_autosummary/jax.lax.ragged_all_to_all.html#jax.lax.ragged_all_to_all "jax.lax.ragged_all_to_all")(operand, output, ...\[, ...\]) | Ragged version of [`all_to_all()`](_autosummary/jax.lax.all_to_all.html#jax.lax.all_to_all "jax.lax.all_to_all") collective. |
| [`ragged_dot`](_autosummary/jax.lax.ragged_dot.html#jax.lax.ragged_dot "jax.lax.ragged_dot")(lhs, rhs, group_sizes\[, ...\]) | Ragged matrix multiplication. |
| [`ragged_dot_general`](_autosummary/jax.lax.ragged_dot_general.html#jax.lax.ragged_dot_general "jax.lax.ragged_dot_general")(lhs, rhs, group_sizes, ...) | Ragged matrix multiplication. |
| [`real`](_autosummary/jax.lax.real.html#jax.lax.real "jax.lax.real")(x) | Elementwise extract real part: \\\mathrm{Re}(x)\\. |
| [`reciprocal`](_autosummary/jax.lax.reciprocal.html#jax.lax.reciprocal "jax.lax.reciprocal")(x) | Elementwise reciprocal: \\1 \over x\\. |
| [`reduce`](_autosummary/jax.lax.reduce.html#jax.lax.reduce "jax.lax.reduce")(operands, init_values, computation, ...) | Wraps XLA's [Reduce](https://www.openxla.org/xla/operation_semantics#reduce) operator. |
| [`reduce_and`](_autosummary/jax.lax.reduce_and.html#jax.lax.reduce_and "jax.lax.reduce_and")(operand, axes) | Compute the bitwise AND of elements over one or more array axes. |
| [`reduce_max`](_autosummary/jax.lax.reduce_max.html#jax.lax.reduce_max "jax.lax.reduce_max")(operand, axes) | Compute the maximum of elements over one or more array axes. |
| [`reduce_min`](_autosummary/jax.lax.reduce_min.html#jax.lax.reduce_min "jax.lax.reduce_min")(operand, axes) | Compute the minimum of elements over one or more array axes. |
| [`reduce_or`](_autosummary/jax.lax.reduce_or.html#jax.lax.reduce_or "jax.lax.reduce_or")(operand, axes) | Compute the bitwise OR of elements over one or more array axes. |
| [`reduce_precision`](_autosummary/jax.lax.reduce_precision.html#jax.lax.reduce_precision "jax.lax.reduce_precision")(operand, exponent_bits, ...) | Wraps XLA's [ReducePrecision](https://www.openxla.org/xla/operation_semantics#reduceprecision) operator. |
| [`reduce_prod`](_autosummary/jax.lax.reduce_prod.html#jax.lax.reduce_prod "jax.lax.reduce_prod")(operand, axes) | Compute the product of elements over one or more array axes. |
| [`reduce_sum`](_autosummary/jax.lax.reduce_sum.html#jax.lax.reduce_sum "jax.lax.reduce_sum")(operand, axes, \*\[, out_sharding\]) | Compute the sum of elements over one or more array axes. |
| [`reduce_window`](_autosummary/jax.lax.reduce_window.html#jax.lax.reduce_window "jax.lax.reduce_window")(operand, init_value, ...\[, ...\]) | Reduction over padded windows. |
| [`reduce_xor`](_autosummary/jax.lax.reduce_xor.html#jax.lax.reduce_xor "jax.lax.reduce_xor")(operand, axes) | Compute the bitwise XOR of elements over one or more array axes. |
| [`rem`](_autosummary/jax.lax.rem.html#jax.lax.rem "jax.lax.rem")(x, y) | Elementwise remainder: \\x \bmod y\\. |
| [`reshape`](_autosummary/jax.lax.reshape.html#jax.lax.reshape "jax.lax.reshape")(operand, new_sizes\[, dimensions, ...\]) | Wraps XLA's [Reshape](https://www.openxla.org/xla/operation_semantics#reshape) operator. |
| [`rev`](_autosummary/jax.lax.rev.html#jax.lax.rev "jax.lax.rev")(operand, dimensions) | Wraps XLA's [Rev](https://www.openxla.org/xla/operation_semantics#rev_reverse) operator. |
| [`rng_bit_generator`](_autosummary/jax.lax.rng_bit_generator.html#jax.lax.rng_bit_generator "jax.lax.rng_bit_generator")(key, shape\[, dtype, ...\]) | Stateless PRNG bit generator. |
| [`rng_uniform`](_autosummary/jax.lax.rng_uniform.html#jax.lax.rng_uniform "jax.lax.rng_uniform")(a, b, shape) | Stateful PRNG generator. |
| [`round`](_autosummary/jax.lax.round.html#jax.lax.round "jax.lax.round")(x\[, rounding_method\]) | Elementwise round. |
| [`rsqrt`](_autosummary/jax.lax.rsqrt.html#jax.lax.rsqrt "jax.lax.rsqrt")(x, \*\[, accuracy\]) | Elementwise reciprocal square root: \\1 \over \sqrt{x}\\. |
| [`scaled_dot`](_autosummary/jax.lax.scaled_dot.html#jax.lax.scaled_dot "jax.lax.scaled_dot")(lhs, rhs, \*\[, lhs_scale, ...\]) | Computes a scaled dot product. |
| [`scatter`](_autosummary/jax.lax.scatter.html#jax.lax.scatter "jax.lax.scatter")(operand, scatter_indices, updates, ...) | Scatter-update operator. |
| [`scatter_add`](_autosummary/jax.lax.scatter_add.html#jax.lax.scatter_add "jax.lax.scatter_add")(operand, scatter_indices, ...\[, ...\]) | Scatter-add operator. |
| [`scatter_apply`](_autosummary/jax.lax.scatter_apply.html#jax.lax.scatter_apply "jax.lax.scatter_apply")(operand, scatter_indices, ...) | Scatter-apply operator. |
| [`scatter_max`](_autosummary/jax.lax.scatter_max.html#jax.lax.scatter_max "jax.lax.scatter_max")(operand, scatter_indices, ...\[, ...\]) | Scatter-max operator. |
| [`scatter_min`](_autosummary/jax.lax.scatter_min.html#jax.lax.scatter_min "jax.lax.scatter_min")(operand, scatter_indices, ...\[, ...\]) | Scatter-min operator. |
| [`scatter_mul`](_autosummary/jax.lax.scatter_mul.html#jax.lax.scatter_mul "jax.lax.scatter_mul")(operand, scatter_indices, ...\[, ...\]) | Scatter-multiply operator. |
| [`scatter_sub`](_autosummary/jax.lax.scatter_sub.html#jax.lax.scatter_sub "jax.lax.scatter_sub")(operand, scatter_indices, ...\[, ...\]) | Scatter-sub operator. |
| [`shift_left`](_autosummary/jax.lax.shift_left.html#jax.lax.shift_left "jax.lax.shift_left")(x, y) | Elementwise left shift: \\x \ll y\\. |
| [`shift_right_arithmetic`](_autosummary/jax.lax.shift_right_arithmetic.html#jax.lax.shift_right_arithmetic "jax.lax.shift_right_arithmetic")(x, y) | Elementwise arithmetic right shift: \\x \gg y\\. |
| [`shift_right_logical`](_autosummary/jax.lax.shift_right_logical.html#jax.lax.shift_right_logical "jax.lax.shift_right_logical")(x, y) | Elementwise logical right shift: \\x \gg y\\. |
| [`sign`](_autosummary/jax.lax.sign.html#jax.lax.sign "jax.lax.sign")(x) | Elementwise sign. |
| [`sin`](_autosummary/jax.lax.sin.html#jax.lax.sin "jax.lax.sin")(x, \*\[, accuracy\]) | Elementwise sine: \\\mathrm{sin}(x)\\. |
| [`sinh`](_autosummary/jax.lax.sinh.html#jax.lax.sinh "jax.lax.sinh")(x) | Elementwise hyperbolic sine: \\\mathrm{sinh}(x)\\. |
| [`slice`](_autosummary/jax.lax.slice.html#jax.lax.slice "jax.lax.slice")(operand, start_indices, limit_indices) | Wraps XLA's [Slice](https://www.openxla.org/xla/operation_semantics#slice) operator. |
| [`slice_in_dim`](_autosummary/jax.lax.slice_in_dim.html#jax.lax.slice_in_dim "jax.lax.slice_in_dim")(operand, start_index, limit_index) | Convenience wrapper around `lax.slice()` applying to only one dimension. |
| [`sort`](_autosummary/jax.lax.sort.html#jax.lax.sort "jax.lax.sort")() | Wraps XLA's [Sort](https://www.openxla.org/xla/operation_semantics#sort) operator. |
| [`sort_key_val`](_autosummary/jax.lax.sort_key_val.html#jax.lax.sort_key_val "jax.lax.sort_key_val")(keys, values\[, dimension, ...\]) | Sorts `keys` along `dimension` and applies the same permutation to `values`. |
| [`split`](_autosummary/jax.lax.split.html#jax.lax.split "jax.lax.split")(operand, sizes\[, axis\]) | Splits an array along `axis`. |
| [`sqrt`](_autosummary/jax.lax.sqrt.html#jax.lax.sqrt "jax.lax.sqrt")(x, \*\[, accuracy\]) | Elementwise square root: \\\sqrt{x}\\. |
| [`square`](_autosummary/jax.lax.square.html#jax.lax.square "jax.lax.square")(x) | Elementwise square: \\x^2\\. |
| [`squeeze`](_autosummary/jax.lax.squeeze.html#jax.lax.squeeze "jax.lax.squeeze")(array, dimensions) | Squeeze any number of size 1 dimensions from an array. |
| [`stack`](_autosummary/jax.lax.stack.html#jax.lax.stack "jax.lax.stack")(operands\[, axis\]) | Joins a sequence of arrays along a new axis. |
| [`stage`](_autosummary/jax.lax.stage.html#jax.lax.stage "jax.lax.stage")(x, /) | Lifts a value into a trace. |
| [`sub`](_autosummary/jax.lax.sub.html#jax.lax.sub "jax.lax.sub")(x, y) | Elementwise subtraction: \\x - y\\. |
| [`tan`](_autosummary/jax.lax.tan.html#jax.lax.tan "jax.lax.tan")(x, \*\[, accuracy\]) | Elementwise tangent: \\\mathrm{tan}(x)\\. |
| [`tanh`](_autosummary/jax.lax.tanh.html#jax.lax.tanh "jax.lax.tanh")(x, \*\[, accuracy\]) | Elementwise hyperbolic tangent: \\\mathrm{tanh}(x)\\. |
| [`tile`](_autosummary/jax.lax.tile.html#jax.lax.tile "jax.lax.tile")(operand, reps) | Tiles an array by repeating it along each dimension. |
| [`top_k`](_autosummary/jax.lax.top_k.html#jax.lax.top_k "jax.lax.top_k")(operand, k, \*\[, axis\]) | Returns top `k` values and their indices along the specified axis of `operand`. |
| [`transpose`](_autosummary/jax.lax.transpose.html#jax.lax.transpose "jax.lax.transpose")(operand, permutation) | Wraps XLA's [Transpose](https://www.openxla.org/xla/operation_semantics#transpose) operator. |
| [`unstack`](_autosummary/jax.lax.unstack.html#jax.lax.unstack "jax.lax.unstack")(x\[, axis\]) | Unstacks an array along an axis. |
| [`zeta`](_autosummary/jax.lax.zeta.html#jax.lax.zeta "jax.lax.zeta")(x, q) | Elementwise Hurwitz zeta function: \\\zeta(x, q)\\ |

## Control flow operators[\#](#control-flow-operators "Link to this heading")

|  |  |
|----|----|
| [`associative_scan`](_autosummary/jax.lax.associative_scan.html#jax.lax.associative_scan "jax.lax.associative_scan")(fn, elems\[, reverse, axis\]) | Performs a scan with an associative binary operation, in parallel. |
| [`cond`](_autosummary/jax.lax.cond.html#jax.lax.cond "jax.lax.cond")(pred, true_fun, false_fun, \*operands\[, ...\]) | Conditionally apply `true_fun` or `false_fun`. |
| [`fori_loop`](_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop "jax.lax.fori_loop")(lower, upper, body_fun, init_val, \*) | Loop from `lower` to `upper` by reduction to [`jax.lax.while_loop()`](_autosummary/jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop"). |
| [`map`](_autosummary/jax.lax.map.html#jax.lax.map "jax.lax.map")(f, xs, \*\[, batch_size\]) | Map a function over leading array axes. |
| [`scan`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan")(f, init\[, xs, length, reverse, unroll, ...\]) | Scan a function over leading array axes while carrying along state. |
| [`select`](_autosummary/jax.lax.select.html#jax.lax.select "jax.lax.select")(pred, on_true, on_false) | Selects between two branches based on a boolean predicate. |
| [`select_n`](_autosummary/jax.lax.select_n.html#jax.lax.select_n "jax.lax.select_n")(which, \*cases) | Selects array values from multiple cases. |
| [`switch`](_autosummary/jax.lax.switch.html#jax.lax.switch "jax.lax.switch")(index, branches, \*operands\[, operand\]) | Apply exactly one of the `branches` given by `index`. |
| [`while_loop`](_autosummary/jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop")(cond_fun, body_fun, init_val) | Call `body_fun` repeatedly in a loop while `cond_fun` is True. |

## Custom gradient operators[\#](#custom-gradient-operators "Link to this heading")

|  |  |
|----|----|
| [`stop_gradient`](_autosummary/jax.lax.stop_gradient.html#jax.lax.stop_gradient "jax.lax.stop_gradient")(x) | Stops gradient computation. |
| [`custom_linear_solve`](_autosummary/jax.lax.custom_linear_solve.html#jax.lax.custom_linear_solve "jax.lax.custom_linear_solve")(matvec, b, solve\[, ...\]) | Perform a matrix-free linear solve with implicitly defined gradients. |
| [`custom_root`](_autosummary/jax.lax.custom_root.html#jax.lax.custom_root "jax.lax.custom_root")(f, initial_guess, solve, ...\[, ...\]) | Differentiably solve for the roots of a function. |

## Parallel operators[\#](#parallel-operators "Link to this heading")

|  |  |
|----|----|
| [`all_gather`](_autosummary/jax.lax.all_gather.html#jax.lax.all_gather "jax.lax.all_gather")(x, axis_name, \*\[, ...\]) | Gather values of x across all replicas. |
| [`all_to_all`](_autosummary/jax.lax.all_to_all.html#jax.lax.all_to_all "jax.lax.all_to_all")(x, axis_name, split_axis, ...\[, ...\]) | Materialize the mapped axis and map a different axis. |
| [`psum`](_autosummary/jax.lax.psum.html#jax.lax.psum "jax.lax.psum")(x, axis_name, \*\[, axis_index_groups\]) | Compute an all-reduce sum on `x` over the pmapped axis `axis_name`. |
| [`psum_scatter`](_autosummary/jax.lax.psum_scatter.html#jax.lax.psum_scatter "jax.lax.psum_scatter")(x, axis_name, \*\[, ...\]) | Like `psum(x,`` ``axis_name)` but each device retains only part of the result. |
| [`pmax`](_autosummary/jax.lax.pmax.html#jax.lax.pmax "jax.lax.pmax")(x, axis_name, \*\[, axis_index_groups\]) | Compute an all-reduce max on `x` over the pmapped axis `axis_name`. |
| [`pmin`](_autosummary/jax.lax.pmin.html#jax.lax.pmin "jax.lax.pmin")(x, axis_name, \*\[, axis_index_groups\]) | Compute an all-reduce min on `x` over the pmapped axis `axis_name`. |
| [`pmean`](_autosummary/jax.lax.pmean.html#jax.lax.pmean "jax.lax.pmean")(x, axis_name, \*\[, axis_index_groups\]) | Compute an all-reduce mean on `x` over the pmapped axis `axis_name`. |
| [`ppermute`](_autosummary/jax.lax.ppermute.html#jax.lax.ppermute "jax.lax.ppermute")(x, axis_name, perm) | Perform a collective permutation according to the permutation `perm`. |
| [`pshuffle`](_autosummary/jax.lax.pshuffle.html#jax.lax.pshuffle "jax.lax.pshuffle")(x, axis_name, perm) | Convenience wrapper of jax.lax.ppermute with alternate permutation encoding |
| [`pswapaxes`](_autosummary/jax.lax.pswapaxes.html#jax.lax.pswapaxes "jax.lax.pswapaxes")(x, axis_name, axis, \*\[, ...\]) | Swap the pmapped axis `axis_name` with the unmapped axis `axis`. |
| [`axis_index`](_autosummary/jax.lax.axis_index.html#jax.lax.axis_index "jax.lax.axis_index")(axis_name) | Return the index along the mapped axis `axis_name`. |
| [`axis_size`](_autosummary/jax.lax.axis_size.html#jax.lax.axis_size "jax.lax.axis_size")(axis_name) | Return the size of the mapped axis `axis_name`. |
| [`psend`](_autosummary/jax.lax.psend.html#jax.lax.psend "jax.lax.psend")(x, axis_name, perm) | Perform a collective send according to the permutation `perm`. |
| [`precv`](_autosummary/jax.lax.precv.html#jax.lax.precv "jax.lax.precv")(token, out_shape, axis_name, perm) | Perform a collective recv according to the permutation `perm`. |
| [`pbroadcast`](_autosummary/jax.lax.pbroadcast.html#jax.lax.pbroadcast "jax.lax.pbroadcast")(x, axis_name, source) | Perform a collective broadcast and replicate from `source`. |

## Sharding-related operators[\#](#sharding-related-operators "Link to this heading")

|  |  |
|----|----|
| [`with_sharding_constraint`](_autosummary/jax.lax.with_sharding_constraint.html#jax.lax.with_sharding_constraint "jax.lax.with_sharding_constraint")(x, shardings) | Mechanism to constrain the sharding of an Array inside a jitted computation |

## Linear algebra operators (jax.lax.linalg)[\#](#module-jax.lax.linalg "Link to this heading")

|  |  |
|----|----|
| [`cholesky`](_autosummary/jax.lax.linalg.cholesky.html#jax.lax.linalg.cholesky "jax.lax.linalg.cholesky")(x, \*\[, symmetrize_input\]) | Cholesky decomposition. |
| [`cholesky_update`](_autosummary/jax.lax.linalg.cholesky_update.html#jax.lax.linalg.cholesky_update "jax.lax.linalg.cholesky_update")(r_matrix, w_vector) | Cholesky rank-1 update. |
| [`eig`](_autosummary/jax.lax.linalg.eig.html#jax.lax.linalg.eig "jax.lax.linalg.eig")(x, \*\[, compute_left_eigenvectors, ...\]) | Eigendecomposition of a general matrix. |
| [`eigh`](_autosummary/jax.lax.linalg.eigh.html#jax.lax.linalg.eigh "jax.lax.linalg.eigh")(x, \*\[, lower, symmetrize_input, ...\]) | Eigendecomposition of a Hermitian matrix. |
| [`hessenberg`](_autosummary/jax.lax.linalg.hessenberg.html#jax.lax.linalg.hessenberg "jax.lax.linalg.hessenberg")(a) | Reduces a square matrix to upper Hessenberg form. |
| [`householder_product`](_autosummary/jax.lax.linalg.householder_product.html#jax.lax.linalg.householder_product "jax.lax.linalg.householder_product")(a, taus) | Product of elementary Householder reflectors. |
| [`lu`](_autosummary/jax.lax.linalg.lu.html#jax.lax.linalg.lu "jax.lax.linalg.lu")(x) | LU decomposition with partial pivoting. |
| [`lu_pivots_to_permutation`](_autosummary/jax.lax.linalg.lu_pivots_to_permutation.html#jax.lax.linalg.lu_pivots_to_permutation "jax.lax.linalg.lu_pivots_to_permutation")(pivots, ...) | Converts the pivots (row swaps) returned by LU to a permutation. |
| [`ormqr`](_autosummary/jax.lax.linalg.ormqr.html#jax.lax.linalg.ormqr "jax.lax.linalg.ormqr")(a, taus, c, \*\[, left, transpose\]) | Multiplies a matrix by Q from a QR factorization without materializing Q. |
| [`qdwh`](_autosummary/jax.lax.linalg.qdwh.html#jax.lax.linalg.qdwh "jax.lax.linalg.qdwh")(x, \*\[, is_hermitian, max_iterations, ...\]) | QR-based dynamically weighted Halley iteration for polar decomposition. |
| [`qr`](_autosummary/jax.lax.linalg.qr.html#jax.lax.linalg.qr "jax.lax.linalg.qr")() | QR decomposition. |
| [`schur`](_autosummary/jax.lax.linalg.schur.html#jax.lax.linalg.schur "jax.lax.linalg.schur")(x, \*\[, compute_schur_vectors, ...\]) | Schur decomposition. |
| [`svd`](_autosummary/jax.lax.linalg.svd.html#jax.lax.linalg.svd "jax.lax.linalg.svd")() | Singular value decomposition. |
| [`SvdAlgorithm`](_autosummary/jax.lax.linalg.SvdAlgorithm.html#jax.lax.linalg.SvdAlgorithm "jax.lax.linalg.SvdAlgorithm")(value\[, names, module, ...\]) | Enum for SVD algorithm. |
| [`symmetric_product`](_autosummary/jax.lax.linalg.symmetric_product.html#jax.lax.linalg.symmetric_product "jax.lax.linalg.symmetric_product")(a_matrix, c_matrix, \*\[, ...\]) | Symmetric product. |
| [`triangular_solve`](_autosummary/jax.lax.linalg.triangular_solve.html#jax.lax.linalg.triangular_solve "jax.lax.linalg.triangular_solve")(a, b, \*\[, left_side, ...\]) | Triangular solve. |
| [`tridiagonal`](_autosummary/jax.lax.linalg.tridiagonal.html#jax.lax.linalg.tridiagonal "jax.lax.linalg.tridiagonal")(a, \*\[, lower\]) | Reduces a symmetric/Hermitian matrix to tridiagonal form. |
| [`tridiagonal_solve`](_autosummary/jax.lax.linalg.tridiagonal_solve.html#jax.lax.linalg.tridiagonal_solve "jax.lax.linalg.tridiagonal_solve")(dl, d, du, b, \*\[, ...\]) | Computes the solution of a tridiagonal linear system. |

*class* jax.lax.linalg.EigImplementation(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L130-L135)[\#](#jax.lax.linalg.EigImplementation "Link to this definition")  
Enum for eigendecomposition algorithm.

CUSOLVER *= 'cusolver'*[\#](#jax.lax.linalg.EigImplementation.CUSOLVER "Link to this definition")  

LAPACK *= 'lapack'*[\#](#jax.lax.linalg.EigImplementation.LAPACK "Link to this definition")  

MAGMA *= 'magma'*[\#](#jax.lax.linalg.EigImplementation.MAGMA "Link to this definition")  

&nbsp;

*class* jax.lax.linalg.EighImplementation(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L229-L234)[\#](#jax.lax.linalg.EighImplementation "Link to this definition")  
Implementation for symmetric/Hermitian eigendecomposition.

JACOBI *= 'jacobi'*[\#](#jax.lax.linalg.EighImplementation.JACOBI "Link to this definition")  

QDWH *= 'qdwh'*[\#](#jax.lax.linalg.EighImplementation.QDWH "Link to this definition")  

QR *= 'qr'*[\#](#jax.lax.linalg.EighImplementation.QR "Link to this definition")  

## Argument classes[\#](#argument-classes "Link to this heading")

*class* jax.lax.AccuracyMode(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L468-L471)[\#](#jax.lax.AccuracyMode "Link to this definition")  
DEFAULT *= 2*[\#](#jax.lax.AccuracyMode.DEFAULT "Link to this definition")  

HIGHEST *= 1*[\#](#jax.lax.AccuracyMode.HIGHEST "Link to this definition")  

&nbsp;

*class* jax.lax.ConvDimensionNumbers(*lhs_spec*, *rhs_spec*, *out_spec*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L38-L52)[\#](#jax.lax.ConvDimensionNumbers "Link to this definition")  
Describes batch, spatial, and feature dimensions of a convolution.

Parameters:  
- **lhs_spec** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a tuple of nonnegative integer dimension numbers containing (batch dimension, feature dimension, spatial dimensions…).

- **rhs_spec** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a tuple of nonnegative integer dimension numbers containing (out feature dimension, in feature dimension, spatial dimensions…).

- **out_spec** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a tuple of nonnegative integer dimension numbers containing (batch dimension, feature dimension, spatial dimensions…).

&nbsp;

jax.lax.ConvGeneralDilatedDimensionNumbers[\#](#jax.lax.ConvGeneralDilatedDimensionNumbers "Link to this definition")  
alias of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\] \| [`ConvDimensionNumbers`](#jax.lax.ConvDimensionNumbers "jax._src.lax.convolution.ConvDimensionNumbers") \| [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")

&nbsp;

*class* jax.lax.DotAlgorithm(*lhs_precision_type*, *rhs_precision_type*, *accumulation_type*, *lhs_component_count=1*, *rhs_component_count=1*, *num_primitive_operations=1*, *allow_imprecise_accumulation=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2178-L2266)[\#](#jax.lax.DotAlgorithm "Link to this definition")  
Specify the algorithm used for computing dot products.

When used to specify the `precision` input to [`dot()`](_autosummary/jax.lax.dot.html#jax.lax.dot "jax.lax.dot"), [`dot_general()`](_autosummary/jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"), and other dot product functions, this data structure is used for controlling the properties of the algorithm used for computing the dot product. This API controls the precision used for the computation, and allows users to access hardware-specific accelerations.

Support for these algorithms is platform dependent, and using an unsupported algorithm will raise a Python exception when the computation is compiled. The algorithms that are known to be supported on at least some platforms are listed in the [`DotAlgorithmPreset`](#jax.lax.DotAlgorithmPreset "jax.lax.DotAlgorithmPreset") enum, and these are a good starting point for experimenting with this API.

A “dot algorithm” is specified by the following parameters:

- `lhs_precision_type` and `rhs_precision_type`, the data types that the LHS and RHS of the operation are rounded to.

- `accumulation_type` the data type used for accumulation.

- `lhs_component_count`, `rhs_component_count`, and `num_primitive_operations` apply to algorithms that decompose the LHS and/or RHS into multiple components and execute multiple operations on those values, usually to emulate a higher precision. For algorithms with no decomposition, these values should be set to `1`.

- `allow_imprecise_accumulation` to specify if accumulation in lower precision is permitted for some steps (e.g. `CUBLASLT_MATMUL_DESC_FAST_ACCUM`).

The [StableHLO spec](https://openxla.org/stablehlo/spec#dot_general) for the dot operation doesn’t require that the precision types be the same as the storage types for the inputs or outputs, but some platforms may require that these types match. Furthermore, the return type of [`dot_general()`](_autosummary/jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general") is always defined by the `accumulation_type` parameter of the input algorithm, if specified.

Examples

Accumulate two 16-bit floats using a 32-bit float accumulator:

    >>> algorithm = DotAlgorithm(
    ...     lhs_precision_type=np.float16,
    ...     rhs_precision_type=np.float16,
    ...     accumulation_type=np.float32,
    ... )
    >>> lhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    >>> rhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    >>> dot(lhs, rhs, precision=algorithm)  
    array([ 1.,  4.,  9., 16.], dtype=float16)

Or, equivalently, using a preset:

    >>> algorithm = DotAlgorithmPreset.F16_F16_F32
    >>> dot(lhs, rhs, precision=algorithm)  
    array([ 1.,  4.,  9., 16.], dtype=float16)

Presets can also be specified by name:

    >>> dot(lhs, rhs, precision="F16_F16_F32")  
    array([ 1.,  4.,  9., 16.], dtype=float16)

The `preferred_element_type` parameter can be used to return the output without downcasting the accumulation type:

    >>> dot(lhs, rhs, precision="F16_F16_F32", preferred_element_type=np.float32)  
    array([ 1.,  4.,  9., 16.], dtype=float32)

Parameters:  
- **lhs_precision_type** (*DTypeLike*)

- **rhs_precision_type** (*DTypeLike*)

- **accumulation_type** (*DTypeLike*)

- **lhs_component_count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **rhs_component_count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_primitive_operations** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **allow_imprecise_accumulation** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

&nbsp;

*class* jax.lax.DotAlgorithmPreset(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2268-L2498)[\#](#jax.lax.DotAlgorithmPreset "Link to this definition")  
An enum of known algorithms for computing dot products.

This `Enum` provides a named set of [`DotAlgorithm`](#jax.lax.DotAlgorithm "jax.lax.DotAlgorithm") objects that are known to be supported on at least platform. See the [`DotAlgorithm`](#jax.lax.DotAlgorithm "jax.lax.DotAlgorithm") documentation for more details about the behavior of these algorithms.

An algorithm can be selected from this list when calling [`dot()`](_autosummary/jax.lax.dot.html#jax.lax.dot "jax.lax.dot"), [`dot_general()`](_autosummary/jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"), or most other JAX dot product functions, by passing either a member of this `Enum` or it’s name as a string using the `precision` argument.

For example, users can specify the preset using this `Enum` directly:

    >>> lhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    >>> rhs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    >>> algorithm = DotAlgorithmPreset.F16_F16_F32
    >>> dot(lhs, rhs, precision=algorithm)  
    array([ 1.,  4.,  9., 16.], dtype=float16)

or, equivalently, they can be specified by name:

    >>> dot(lhs, rhs, precision="F16_F16_F32")  
    array([ 1.,  4.,  9., 16.], dtype=float16)

The names of the presets are typically `LHS_RHS_ACCUM` where `LHS` and `RHS` are the element types of the `lhs` and `rhs` inputs respectively, and `ACCUM` is the element type of the accumulator. Some presets have an extra suffix, and the meaning of each of these is documented below. The supported presets are:

DEFAULT *= 1*[\#](#jax.lax.DotAlgorithmPreset.DEFAULT "Link to this definition")  
An algorithm will be selected based on input and output types.

ANY_F8_ANY_F8_F32 *= 2*[\#](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32 "Link to this definition")  
Accepts any float8 input types and accumulates into float32.

ANY_F8_ANY_F8_F32_FAST_ACCUM *= 3*[\#](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM "Link to this definition")  
Like `ANY_F8_ANY_F8_F32`, but using faster accumulation with the cost of lower accuracy.

ANY_F8_ANY_F8_ANY *= 4*[\#](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_ANY "Link to this definition")  
Like `ANY_F8_ANY_F8_F32`, but the accumulation type is controlled by `preferred_element_type`.

ANY_F8_ANY_F8_ANY_FAST_ACCUM *= 5*[\#](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM "Link to this definition")  
Like `ANY_F8_ANY_F8_F32_FAST_ACCUM`, but the accumulation type is controlled by `preferred_element_type`.

F16_F16_F16 *= 6*[\#](#jax.lax.DotAlgorithmPreset.F16_F16_F16 "Link to this definition")  

F16_F16_F32 *= 7*[\#](#jax.lax.DotAlgorithmPreset.F16_F16_F32 "Link to this definition")  

BF16_BF16_BF16 *= 8*[\#](#jax.lax.DotAlgorithmPreset.BF16_BF16_BF16 "Link to this definition")  

BF16_BF16_F32 *= 9*[\#](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32 "Link to this definition")  

BF16_BF16_F32_X3 *= 10*[\#](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3 "Link to this definition")  
The `_X3` suffix indicates that the algorithm uses 3 operations to emulate higher precision.

BF16_BF16_F32_X6 *= 11*[\#](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6 "Link to this definition")  
Like `BF16_BF16_F32_X3`, but using 6 operations instead of 3.

BF16_BF16_F32_X9 *= 12*[\#](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X9 "Link to this definition")  
Like `BF16_BF16_F32_X3`, but using 9 operations instead of 3.

TF32_TF32_F32 *= 13*[\#](#jax.lax.DotAlgorithmPreset.TF32_TF32_F32 "Link to this definition")  

TF32_TF32_F32_X3 *= 14*[\#](#jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3 "Link to this definition")  
The `_X3` suffix indicates that the algorithm uses 3 operations to emulate higher precision.

F32_F32_F32 *= 15*[\#](#jax.lax.DotAlgorithmPreset.F32_F32_F32 "Link to this definition")  

F64_F64_F64 *= 16*[\#](#jax.lax.DotAlgorithmPreset.F64_F64_F64 "Link to this definition")  

*property* supported_lhs_types*: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[DTypeLike, ...\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2351-L2376)[\#](#jax.lax.DotAlgorithmPreset.supported_lhs_types "Link to this definition")  

*property* supported_rhs_types*: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[DTypeLike, ...\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2377-L2380)[\#](#jax.lax.DotAlgorithmPreset.supported_rhs_types "Link to this definition")  

*property* accumulation_type*: DTypeLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2381-L2398)[\#](#jax.lax.DotAlgorithmPreset.accumulation_type "Link to this definition")  

supported_output_types(*lhs_dtype*, *rhs_dtype*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2399-L2432)[\#](#jax.lax.DotAlgorithmPreset.supported_output_types "Link to this definition")  
Parameters:  
- **lhs_dtype** (*DTypeLike*)

- **rhs_dtype** (*DTypeLike*)

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[DTypeLike, …\] \| None

&nbsp;

jax.lax.DotDimensionNumbers[\#](#jax.lax.DotDimensionNumbers "Link to this definition")  
alias of [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]\], [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")\[[`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]\]\]

&nbsp;

*class* jax.lax.FftType(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/fft.py#L39-L53)[\#](#jax.lax.FftType "Link to this definition")  
Describes which FFT operation to perform.

FFT *= 0*[\#](#jax.lax.FftType.FFT "Link to this definition")  
Forward complex-to-complex FFT.

IFFT *= 1*[\#](#jax.lax.FftType.IFFT "Link to this definition")  
Inverse complex-to-complex FFT.

IRFFT *= 3*[\#](#jax.lax.FftType.IRFFT "Link to this definition")  
Inverse real-to-complex FFT.

RFFT *= 2*[\#](#jax.lax.FftType.RFFT "Link to this definition")  
Forward real-to-complex FFT.

&nbsp;

*class* jax.lax.GatherDimensionNumbers(*offset_dims*, *collapsed_slice_dims*, *start_index_map*, *operand_batching_dims=()*, *start_indices_batching_dims=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L243-L280)[\#](#jax.lax.GatherDimensionNumbers "Link to this definition")  
Describes the dimension number arguments to an [XLA’s Gather operator](https://www.openxla.org/xla/operation_semantics#gather). See the XLA documentation for more details of what the dimension numbers mean.

Parameters:  
- **offset_dims** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – the set of dimensions in the gather output that offset into an array sliced from operand. Must be a tuple of integers in ascending order, each representing a dimension number of the output.

- **collapsed_slice_dims** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – the set of dimensions i in operand that have slice_sizes\[i\] == 1 and that should not have a corresponding dimension in the output of the gather. Must be a tuple of integers in ascending order.

- **start_index_map** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – for each dimension in start_indices, gives the corresponding dimension in the operand that is to be sliced. Must be a tuple of integers with size equal to start_indices.shape\[-1\].

- **operand_batching_dims** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – the set of batching dimensions i in operand that have slice_sizes\[i\] == 1 and that should have a corresponding dimension in both the start_indices (at the same index in start_indices_batching_dims) and output of the gather. Must be a tuple of integers in ascending order.

- **start_indices_batching_dims** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – the set of batching dimensions i in start_indices that should have a corresponding dimension in both the operand (at the same index in operand_batching_dims) and output of the gather. Must be a tuple of integers (order is fixed based on correspondence with operand_batching_dims).

Unlike XLA’s GatherDimensionNumbers structure, index_vector_dim is implicit; there is always an index vector dimension and it must always be the last dimension. To gather scalar indices, add a trailing dimension of size 1.

&nbsp;

*class* jax.lax.GatherScatterMode(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L282-L322)[\#](#jax.lax.GatherScatterMode "Link to this definition")  
Describes how to handle out-of-bounds indices in a gather or scatter.

Possible values are:

CLIP:  
Indices will be clamped to the nearest in-range value, i.e., such that the entire window to be gathered is in-range.

FILL_OR_DROP:  
If any part of a gathered window is out of bounds, the entire window that is returned, even those elements that were otherwise in-bounds, will be filled with a constant. If any part of a scattered window is out of bounds, the entire window will be discarded.

PROMISE_IN_BOUNDS:  
The user promises that indices are in bounds. No additional checking will be performed. In practice, with the current XLA implementation this means that out-of-bounds gathers will be clamped but out-of-bounds scatters will be discarded. Gradients will not be correct if indices are out-of-bounds.

&nbsp;

*class* jax.lax.Precision(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2129-L2165)[\#](#jax.lax.Precision "Link to this definition")  
Precision enum for lax matrix multiply related functions.

The device-dependent precision argument to JAX functions generally controls the tradeoff between speed and accuracy for array computations on accelerator backends, (i.e. TPU and GPU). Has no impact on CPU backends. This only has an effect on float32 computations, and does not affect the input/output datatypes. Members are:

DEFAULT:  
Fastest mode, but least accurate. On TPU: performs float32 computations in bfloat16. On GPU: uses tensorfloat32 if available (e.g. on A100 and H100 GPUs), otherwise standard float32 (e.g. on V100 GPUs). Aliases: `'default'`, `'fastest'`.

HIGH:  
Slower but more accurate. On TPU: performs float32 computations in 3 bfloat16 passes. On GPU: uses tensorfloat32 where available, otherwise float32. Aliases: `'high'`..

HIGHEST:  
Slowest but most accurate. On TPU: performs float32 computations in 6 bfloat16. Aliases: `'highest'`. On GPU: uses float32.

&nbsp;

jax.lax.PrecisionLike[\#](#jax.lax.PrecisionLike "Link to this definition")  
alias of [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") \| [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`Precision`](#jax.lax.Precision "jax._src.lax.lax.Precision") \| [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\] \| [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[`Precision`](#jax.lax.Precision "jax._src.lax.lax.Precision"), [`Precision`](#jax.lax.Precision "jax._src.lax.lax.Precision")\] \| [`DotAlgorithm`](#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") \| [`DotAlgorithmPreset`](#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")

&nbsp;

*class* jax.lax.RaggedDotDimensionNumbers(*dot_dimension_numbers*, *lhs_ragged_dimensions*, *rhs_group_dimensions*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2658-L2684)[\#](#jax.lax.RaggedDotDimensionNumbers "Link to this definition")  
Describes ragged, group, and dot dimensions for ragged dot general.

Parameters:  
- **dot_dimension_numbers** (*DotDimensionNumbers*) – a tuple of tuples of sequences of ints of the form ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)).

- **lhs_ragged_dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of ints indicating the ‘lhs’ ragged dimensions.

- **rhs_group_dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of ints indicating the ‘rhs’ group dimensions.

&nbsp;

*class* jax.lax.RandomAlgorithm(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L9076-L9090)[\#](#jax.lax.RandomAlgorithm "Link to this definition")  
Describes which PRNG algorithm to use for rng_bit_generator.

RNG_DEFAULT *= 0*[\#](#jax.lax.RandomAlgorithm.RNG_DEFAULT "Link to this definition")  
The platform’s default algorithm.

RNG_THREE_FRY *= 1*[\#](#jax.lax.RandomAlgorithm.RNG_THREE_FRY "Link to this definition")  
The Threefry-2x32 PRNG algorithm.

RNG_PHILOX *= 2*[\#](#jax.lax.RandomAlgorithm.RNG_PHILOX "Link to this definition")  
The Philox-4x32 PRNG algorithm.

&nbsp;

*class* jax.lax.RoundingMethod(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L381-L393)[\#](#jax.lax.RoundingMethod "Link to this definition")  
Rounding strategies for handling halfway values (e.g., 0.5) in [`jax.lax.round()`](_autosummary/jax.lax.round.html#jax.lax.round "jax.lax.round").

AWAY_FROM_ZERO *= 0*[\#](#jax.lax.RoundingMethod.AWAY_FROM_ZERO "Link to this definition")  
Rounds halfway values away from zero (e.g., 0.5 -\> 1, -0.5 -\> -1).

TO_NEAREST_EVEN *= 1*[\#](#jax.lax.RoundingMethod.TO_NEAREST_EVEN "Link to this definition")  
Rounds halfway values to the nearest even integer. This is also known as “banker’s rounding” (e.g., 0.5 -\> 0, 1.5 -\> 2).

&nbsp;

*class* jax.lax.ScatterDimensionNumbers(*update_window_dims*, *inserted_window_dims*, *scatter_dims_to_operand_dims*, *operand_batching_dims=()*, *scatter_indices_batching_dims=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L434-L472)[\#](#jax.lax.ScatterDimensionNumbers "Link to this definition")  
Describes the dimension number arguments to an [XLA’s Scatter operator](https://www.openxla.org/xla/operation_semantics#scatter). See the XLA documentation for more details of what the dimension numbers mean.

Parameters:  
- **update_window_dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the set of dimensions in the updates that are window dimensions. Must be a tuple of integers in ascending order, each representing a dimension number.

- **inserted_window_dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the set of size 1 window dimensions that must be inserted into the shape of updates. Must be a tuple of integers in ascending order, each representing a dimension number of the output. These are the mirror image of collapsed_slice_dims in the case of gather.

- **scatter_dims_to_operand_dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – for each dimension in scatter_indices, gives the corresponding dimension in operand. Must be a sequence of integers with size equal to scatter_indices.shape\[-1\].

- **operand_batching_dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the set of batching dimensions i in operand that should have a corresponding dimension in both the scatter_indices (at the same index in scatter_indices_batching_dims) and updates. Must be a tuple of integers in ascending order. These are the mirror image of operand_batching_dims in the case of gather.

- **scatter_indices_batching_dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the set of batching dimensions i in scatter_indices that should have a corresponding dimension in both the operand (at the same index in operand_batching_dims) and output of the gather. Must be a tuple of integers (order is fixed based on correspondence with input_batching_dims). These are the mirror image of start_indices_batching_dims in the case of gather.

Unlike XLA’s ScatterDimensionNumbers structure, index_vector_dim is implicit; there is always an index vector dimension and it must always be the last dimension. To scatter scalar indices, add a trailing dimension of size 1.

&nbsp;

*class* jax.lax.Tolerance(*atol=0.0*, *rtol=0.0*, *ulps=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L451-L466)[\#](#jax.lax.Tolerance "Link to this definition")  
Specify the tolerances used for computing unary functions.

Maximum two tolerances can be specified: (atol and rtol) or (atol and ulps).

Parameters:  
- **atol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"))

- **rtol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"))

- **ulps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

[](_autosummary/jax.scipy.stats.wrapcauchy.pdf.html "previous page")

previous

jax.scipy.stats.wrapcauchy.pdf

[](_autosummary/jax.lax.abs.html "next page")

next

jax.lax.abs

Contents

- [Operators](#operators)
- [Control flow operators](#control-flow-operators)
- [Custom gradient operators](#custom-gradient-operators)
- [Parallel operators](#parallel-operators)
- [Sharding-related operators](#sharding-related-operators)
- [Linear algebra operators (jax.lax.linalg)](#module-jax.lax.linalg)
  - [`EigImplementation`](#jax.lax.linalg.EigImplementation)
    - [`EigImplementation.CUSOLVER`](#jax.lax.linalg.EigImplementation.CUSOLVER)
    - [`EigImplementation.LAPACK`](#jax.lax.linalg.EigImplementation.LAPACK)
    - [`EigImplementation.MAGMA`](#jax.lax.linalg.EigImplementation.MAGMA)
  - [`EighImplementation`](#jax.lax.linalg.EighImplementation)
    - [`EighImplementation.JACOBI`](#jax.lax.linalg.EighImplementation.JACOBI)
    - [`EighImplementation.QDWH`](#jax.lax.linalg.EighImplementation.QDWH)
    - [`EighImplementation.QR`](#jax.lax.linalg.EighImplementation.QR)
- [Argument classes](#argument-classes)
  - [`AccuracyMode`](#jax.lax.AccuracyMode)
    - [`AccuracyMode.DEFAULT`](#jax.lax.AccuracyMode.DEFAULT)
    - [`AccuracyMode.HIGHEST`](#jax.lax.AccuracyMode.HIGHEST)
  - [`ConvDimensionNumbers`](#jax.lax.ConvDimensionNumbers)
  - [`ConvGeneralDilatedDimensionNumbers`](#jax.lax.ConvGeneralDilatedDimensionNumbers)
  - [`DotAlgorithm`](#jax.lax.DotAlgorithm)
  - [`DotAlgorithmPreset`](#jax.lax.DotAlgorithmPreset)
    - [`DotAlgorithmPreset.DEFAULT`](#jax.lax.DotAlgorithmPreset.DEFAULT)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_F32`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32_FAST_ACCUM)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_ANY`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_ANY)
    - [`DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM`](#jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_ANY_FAST_ACCUM)
    - [`DotAlgorithmPreset.F16_F16_F16`](#jax.lax.DotAlgorithmPreset.F16_F16_F16)
    - [`DotAlgorithmPreset.F16_F16_F32`](#jax.lax.DotAlgorithmPreset.F16_F16_F32)
    - [`DotAlgorithmPreset.BF16_BF16_BF16`](#jax.lax.DotAlgorithmPreset.BF16_BF16_BF16)
    - [`DotAlgorithmPreset.BF16_BF16_F32`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32)
    - [`DotAlgorithmPreset.BF16_BF16_F32_X3`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3)
    - [`DotAlgorithmPreset.BF16_BF16_F32_X6`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6)
    - [`DotAlgorithmPreset.BF16_BF16_F32_X9`](#jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X9)
    - [`DotAlgorithmPreset.TF32_TF32_F32`](#jax.lax.DotAlgorithmPreset.TF32_TF32_F32)
    - [`DotAlgorithmPreset.TF32_TF32_F32_X3`](#jax.lax.DotAlgorithmPreset.TF32_TF32_F32_X3)
    - [`DotAlgorithmPreset.F32_F32_F32`](#jax.lax.DotAlgorithmPreset.F32_F32_F32)
    - [`DotAlgorithmPreset.F64_F64_F64`](#jax.lax.DotAlgorithmPreset.F64_F64_F64)
    - [`DotAlgorithmPreset.supported_lhs_types`](#jax.lax.DotAlgorithmPreset.supported_lhs_types)
    - [`DotAlgorithmPreset.supported_rhs_types`](#jax.lax.DotAlgorithmPreset.supported_rhs_types)
    - [`DotAlgorithmPreset.accumulation_type`](#jax.lax.DotAlgorithmPreset.accumulation_type)
    - [`DotAlgorithmPreset.supported_output_types()`](#jax.lax.DotAlgorithmPreset.supported_output_types)
  - [`DotDimensionNumbers`](#jax.lax.DotDimensionNumbers)
  - [`FftType`](#jax.lax.FftType)
    - [`FftType.FFT`](#jax.lax.FftType.FFT)
    - [`FftType.IFFT`](#jax.lax.FftType.IFFT)
    - [`FftType.IRFFT`](#jax.lax.FftType.IRFFT)
    - [`FftType.RFFT`](#jax.lax.FftType.RFFT)
  - [`GatherDimensionNumbers`](#jax.lax.GatherDimensionNumbers)
  - [`GatherScatterMode`](#jax.lax.GatherScatterMode)
  - [`Precision`](#jax.lax.Precision)
  - [`PrecisionLike`](#jax.lax.PrecisionLike)
  - [`RaggedDotDimensionNumbers`](#jax.lax.RaggedDotDimensionNumbers)
  - [`RandomAlgorithm`](#jax.lax.RandomAlgorithm)
    - [`RandomAlgorithm.RNG_DEFAULT`](#jax.lax.RandomAlgorithm.RNG_DEFAULT)
    - [`RandomAlgorithm.RNG_THREE_FRY`](#jax.lax.RandomAlgorithm.RNG_THREE_FRY)
    - [`RandomAlgorithm.RNG_PHILOX`](#jax.lax.RandomAlgorithm.RNG_PHILOX)
  - [`RoundingMethod`](#jax.lax.RoundingMethod)
    - [`RoundingMethod.AWAY_FROM_ZERO`](#jax.lax.RoundingMethod.AWAY_FROM_ZERO)
    - [`RoundingMethod.TO_NEAREST_EVEN`](#jax.lax.RoundingMethod.TO_NEAREST_EVEN)
  - [`ScatterDimensionNumbers`](#jax.lax.ScatterDimensionNumbers)
  - [`Tolerance`](#jax.lax.Tolerance)

By The JAX authors

© Copyright 2024, The JAX Authors.\
