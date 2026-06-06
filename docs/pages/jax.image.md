- [](index.html)
- [API Reference](jax.html)
- `jax.image` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.image.rst "Download source file")
-  .pdf

# jax.image module

## Contents

- [Image manipulation functions](#image-manipulation-functions)
- [Argument classes](#argument-classes)
  - [`ResizeMethod`](#jax.image.ResizeMethod)

# `jax.image` module[\#](#module-jax.image "Link to this heading")

Image manipulation functions.

More image manipulation functions can be found in libraries built on top of JAX, such as [PIX](https://github.com/deepmind/dm_pix).

## Image manipulation functions[\#](#image-manipulation-functions "Link to this heading")

|  |  |
|----|----|
| [`resize`](_autosummary/jax.image.resize.html#jax.image.resize "jax.image.resize")(image, shape, method\[, antialias, ...\]) | Image resize. |
| [`scale_and_translate`](_autosummary/jax.image.scale_and_translate.html#jax.image.scale_and_translate "jax.image.scale_and_translate")(image, shape, ...\[, ...\]) | Apply a scale and translation to an image. |

## Argument classes[\#](#argument-classes "Link to this heading")

*class* jax.image.ResizeMethod(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/image/scale.py#L215-L276)[\#](#jax.image.ResizeMethod "Link to this definition")  
Image resize method.

Possible values are:

NEAREST:  
Nearest-neighbor interpolation.

LINEAR:  
[Linear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation).

LANCZOS3:  
[Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling), using a kernel of radius 3.

LANCZOS5:  
[Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling), using a kernel of radius 5.

CUBIC:  
[Cubic interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation), using the Keys cubic kernel.

AREA:  
Area resampling. Computes the average of all pixels that fall within the output pixel’s area. When downscaling, this acts as an anti-aliasing filter. When upscaling, it acts as a box filter, matching TensorFlow’s behavior.

[](_autosummary/jax.flatten_util.ravel_pytree.html "previous page")

previous

jax.flatten_util.ravel_pytree

[](_autosummary/jax.image.resize.html "next page")

next

jax.image.resize

Contents

- [Image manipulation functions](#image-manipulation-functions)
- [Argument classes](#argument-classes)
  - [`ResizeMethod`](#jax.image.ResizeMethod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
