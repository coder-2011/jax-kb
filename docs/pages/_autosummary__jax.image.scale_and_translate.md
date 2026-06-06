- [](../index.html)
- [API Reference](../jax.html)
- [`jax.image` module](../jax.image.html)
- jax.image.scale_and_translate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.image.scale_and_translate.rst "Download source file")
-  .pdf

# jax.image.scale_and_translate

## Contents

- [`scale_and_translate()`](#jax.image.scale_and_translate)

# jax.image.scale_and_translate[\#](#jax-image-scale-and-translate "Link to this heading")

jax.image.scale_and_translate(*image*, *shape*, *spatial_dims*, *scale*, *translation*, *method*, *antialias=True*, *precision=Precision.HIGHEST*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/image/scale.py#L289-L393)[\#](#jax.image.scale_and_translate "Link to this definition")  
Apply a scale and translation to an image.

Generates a new image of shape ‘shape’ by resampling from the input image using the sampling method corresponding to method. For 2D images, this operation transforms a location in the input images, (x, y), to a location in the output image according to:

    (x * scale[1] + translation[1], y * scale[0] + translation[0])

(Note the *inverse* warp is used to generate the sample locations.) Assumes half-centered pixels, i.e the pixel at integer location `row,`` ``col` has coordinates `y,`` ``x`` ``=`` ``row`` ``+`` ``0.5,`` ``col`` ``+`` ``0.5`, and similarly for other input image dimensions.

If an output location(pixel) maps to an input sample location that is outside the input boundaries then the value for the output location will be set to zero.

This function can be used to imitate the behavior of `torch.nn.functional.interpolate` with `align_corners=True` by setting:

    scale = (n - 1) / (m - 1)
    translation = 0.5 * (1 - scale)

where `m` is the input size and `n` is the output size for a given dimension.

The `method` argument expects one of the following resize methods:

`ResizeMethod.LINEAR`, `"linear"`, `"bilinear"`, `"trilinear"`,  
`"triangle"` [Linear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation). If `antialias` is `True`, uses a triangular filter when downsampling.

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

Parameters:  
- **image** – a JAX array.

- **shape** (*core.Shape*) – the output shape, as a sequence of integers with length equal to the number of dimensions of image.

- **spatial_dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – A length K tuple specifying the spatial dimensions that the passed scale and translation should be applied to.

- **scale** – A \[K\] array with the same number of dimensions as image, containing the scale to apply in each dimension.

- **translation** – A \[K\] array with the same number of dimensions as image, containing the translation to apply in each dimension.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*ResizeMethod*](../jax.image.html#jax.image.ResizeMethod "jax.image.ResizeMethod")) – the resizing method to use; either a `ResizeMethod` instance or a string. Available methods are: `LINEAR`, `LANCZOS3`, `LANCZOS5`, `CUBIC`, `CUBIC_PYTORCH`, `AREA`.

- **antialias** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Should an antialiasing filter be used when downsampling? Defaults to `True`. Has no effect when upsampling.

Returns:  
The scale and translated image.

[](jax.image.resize.html "previous page")

previous

jax.image.resize

[](../jax.nn.html "next page")

next

`jax.nn` module

Contents

- [`scale_and_translate()`](#jax.image.scale_and_translate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
