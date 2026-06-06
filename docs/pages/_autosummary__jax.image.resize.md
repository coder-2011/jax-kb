- [](../index.html)
- [API Reference](../jax.html)
- [`jax.image` module](../jax.image.html)
- jax.image.resize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.image.resize.rst "Download source file")
-  .pdf

# jax.image.resize

## Contents

- [`resize()`](#jax.image.resize)

# jax.image.resize[\#](#jax-image-resize "Link to this heading")

jax.image.resize(*image*, *shape*, *method*, *antialias=True*, *precision=Precision.HIGHEST*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/image/scale.py#L444-L506)[\#](#jax.image.resize "Link to this definition")  
Image resize.

The `method` argument expects one of the following resize methods:

`ResizeMethod.NEAREST`, `"nearest"`  
[Nearest neighbor interpolation](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation). The values of `antialias` and `precision` are ignored.

`ResizeMethod.LINEAR`, `"linear"`, `"bilinear"`, `"trilinear"`, `"triangle"`  
[Linear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation). If `antialias` is `True`, uses a triangular filter when downsampling.

`ResizeMethod.CUBIC`, `"cubic"`, `"bicubic"`, `"tricubic"`  
[Cubic interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation), using the Keys cubic kernel.

`ResizeMethod.CUBIC_PYTORCH`, `"cubic-pytorch"`, `"bicubic-pytorch"`  
[Cubic interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation), matching PyTorch’s bicubic resizing behavior. Identical to `ResizeMethod.CUBIC` when antialiasing is enabled, but uses a different kernel and enables edge padding when antialiasing is disabled.

`ResizeMethod.LANCZOS3`, `"lanczos3"`  
[Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling), using a kernel of radius 3.

`ResizeMethod.LANCZOS5`, `"lanczos5"`  
[Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling), using a kernel of radius 5.

`ResizeMethod.AREA`, `"area"`  
Area resampling. Computes the average of all pixels that fall within the output pixel’s area. When downscaling, this acts as an anti-aliasing filter. When upscaling, it acts as a box filter, matching TensorFlow’s behavior.

This function does not support an `align_corners` argument like `torch.nn.functional.interpolate`. That behavior can be emulated using [`scale_and_translate()`](jax.image.scale_and_translate.html#jax.image.scale_and_translate "jax.image.scale_and_translate").

Parameters:  
- **image** – a JAX array.

- **shape** (*core.Shape*) – the output shape, as a sequence of integers with length equal to the number of dimensions of image. Note that [`resize()`](#jax.image.resize "jax.image.resize") does not distinguish spatial dimensions from batch or channel dimensions, so this includes all dimensions of the image. To represent a batch or a channel dimension, simply leave that element of the shape unchanged.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*ResizeMethod*](../jax.image.html#jax.image.ResizeMethod "jax.image.ResizeMethod")) – the resizing method to use; either a `ResizeMethod` instance or a string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC, CUBIC_PYTORCH.

- **antialias** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – should an antialiasing filter be used when downsampling? Defaults to `True`. Has no effect when upsampling.

Returns:  
The resized image. The return type may differ from the input type depending on the `method`. For `ResizeMethod.NEAREST`, the return type is the same as the input type. For other methods, the output type will be promoted to a floating point type.

[](../jax.image.html "previous page")

previous

`jax.image` module

[](jax.image.scale_and_translate.html "next page")

next

jax.image.scale_and_translate

Contents

- [`resize()`](#jax.image.resize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
