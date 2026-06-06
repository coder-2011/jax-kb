- [](index.html)
- [API Reference](jax.html)
- [`jax.example_libraries` module](jax.example_libraries.html)
- `jax.example_libraries.stax` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.example_libraries.stax.rst "Download source file")
-  .pdf

# jax.example_libraries.stax module

## Contents

- [`AvgPool()`](#jax.example_libraries.stax.AvgPool)
- [`BatchNorm()`](#jax.example_libraries.stax.BatchNorm)
- [`Conv()`](#jax.example_libraries.stax.Conv)
- [`Conv1DTranspose()`](#jax.example_libraries.stax.Conv1DTranspose)
- [`ConvTranspose()`](#jax.example_libraries.stax.ConvTranspose)
- [`Dense()`](#jax.example_libraries.stax.Dense)
- [`Dropout()`](#jax.example_libraries.stax.Dropout)
- [`FanInConcat()`](#jax.example_libraries.stax.FanInConcat)
- [`FanOut()`](#jax.example_libraries.stax.FanOut)
- [`GeneralConv()`](#jax.example_libraries.stax.GeneralConv)
- [`GeneralConvTranspose()`](#jax.example_libraries.stax.GeneralConvTranspose)
- [`MaxPool()`](#jax.example_libraries.stax.MaxPool)
- [`SumPool()`](#jax.example_libraries.stax.SumPool)
- [`elementwise()`](#jax.example_libraries.stax.elementwise)
- [`parallel()`](#jax.example_libraries.stax.parallel)
- [`serial()`](#jax.example_libraries.stax.serial)
- [`shape_dependent()`](#jax.example_libraries.stax.shape_dependent)

# `jax.example_libraries.stax` module[\#](#module-jax.example_libraries.stax "Link to this heading")

Stax is a small but flexible neural net specification library from scratch.

You likely do not mean to import this module! Stax is intended as an example library only. There are a number of other much more fully-featured neural network libraries for JAX, including [Flax](https://github.com/google/flax) from Google, and [Haiku](https://github.com/deepmind/dm-haiku) from DeepMind.

jax.example_libraries.stax.AvgPool(*window_shape*, *strides=None*, *padding='VALID'*, *spec=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L164-L190)[\#](#jax.example_libraries.stax.AvgPool "Link to this definition")  
Layer construction function for a pooling layer.

&nbsp;

jax.example_libraries.stax.BatchNorm(*axis=(0*, *1*, *2)*, *epsilon=1e-05*, *center=True*, *scale=True*, *beta_init=\<function zeros\>*, *gamma_init=\<function ones\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L121-L143)[\#](#jax.example_libraries.stax.BatchNorm "Link to this definition")  
Layer construction function for a batch normalization layer.

&nbsp;

jax.example_libraries.stax.Conv(*out_chan*, *filter_shape*, *strides=None*, *padding='VALID'*, *W_init=None*, *b_init=\<function normal.\<locals\>.init\>*)[\#](#jax.example_libraries.stax.Conv "Link to this definition")  
Layer construction function for a general convolution layer.

&nbsp;

jax.example_libraries.stax.Conv1DTranspose(*out_chan*, *filter_shape*, *strides=None*, *padding='VALID'*, *W_init=None*, *b_init=\<function normal.\<locals\>.init\>*)[\#](#jax.example_libraries.stax.Conv1DTranspose "Link to this definition")  
Layer construction function for a general transposed-convolution layer.

&nbsp;

jax.example_libraries.stax.ConvTranspose(*out_chan*, *filter_shape*, *strides=None*, *padding='VALID'*, *W_init=None*, *b_init=\<function normal.\<locals\>.init\>*)[\#](#jax.example_libraries.stax.ConvTranspose "Link to this definition")  
Layer construction function for a general transposed-convolution layer.

&nbsp;

jax.example_libraries.stax.Dense(*out_dim*, *W_init=\<function variance_scaling.\<locals\>.init\>*, *b_init=\<function normal.\<locals\>.init\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L52-L63)[\#](#jax.example_libraries.stax.Dense "Link to this definition")  
Layer constructor function for a dense (fully-connected) layer.

&nbsp;

jax.example_libraries.stax.Dropout(*rate*, *mode='train'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L261-L279)[\#](#jax.example_libraries.stax.Dropout "Link to this definition")  
Layer construction function for a dropout layer with given rate.

&nbsp;

jax.example_libraries.stax.FanInConcat(*axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L249-L259)[\#](#jax.example_libraries.stax.FanInConcat "Link to this definition")  
Layer construction function for a fan-in concatenation layer.

&nbsp;

jax.example_libraries.stax.FanOut(*num*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L234-L239)[\#](#jax.example_libraries.stax.FanOut "Link to this definition")  
Layer construction function for a fan-out layer.

&nbsp;

jax.example_libraries.stax.GeneralConv(*dimension_numbers*, *out_chan*, *filter_shape*, *strides=None*, *padding='VALID'*, *W_init=None*, *b_init=\<function normal.\<locals\>.init\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L65-L89)[\#](#jax.example_libraries.stax.GeneralConv "Link to this definition")  
Layer construction function for a general convolution layer.

&nbsp;

jax.example_libraries.stax.GeneralConvTranspose(*dimension_numbers*, *out_chan*, *filter_shape*, *strides=None*, *padding='VALID'*, *W_init=None*, *b_init=\<function normal.\<locals\>.init\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L92-L116)[\#](#jax.example_libraries.stax.GeneralConvTranspose "Link to this definition")  
Layer construction function for a general transposed-convolution layer.

&nbsp;

jax.example_libraries.stax.MaxPool(*window_shape*, *strides=None*, *padding='VALID'*, *spec=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L164-L190)[\#](#jax.example_libraries.stax.MaxPool "Link to this definition")  
Layer construction function for a pooling layer.

&nbsp;

jax.example_libraries.stax.SumPool(*window_shape*, *strides=None*, *padding='VALID'*, *spec=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L164-L190)[\#](#jax.example_libraries.stax.SumPool "Link to this definition")  
Layer construction function for a pooling layer.

&nbsp;

jax.example_libraries.stax.elementwise(*fun*, *\*\*fun_kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L145-L150)[\#](#jax.example_libraries.stax.elementwise "Link to this definition")  
Layer that applies a scalar function elementwise on its inputs.

&nbsp;

jax.example_libraries.stax.parallel(*\*layers*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L312-L338)[\#](#jax.example_libraries.stax.parallel "Link to this definition")  
Combinator for composing layers in parallel.

The layer resulting from this combinator is often used with the FanOut and FanInSum layers.

Parameters:  
**\*layers** – a sequence of layers, each an (init_fun, apply_fun) pair.

Returns:  
A new layer, meaning an (init_fun, apply_fun) pair, representing the parallel composition of the given sequence of layers. In particular, the returned layer takes a sequence of inputs and returns a sequence of outputs with the same length as the argument layers.

&nbsp;

jax.example_libraries.stax.serial(*\*layers*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L284-L310)[\#](#jax.example_libraries.stax.serial "Link to this definition")  
Combinator for composing layers in serial.

Parameters:  
**\*layers** – a sequence of layers, each an (init_fun, apply_fun) pair.

Returns:  
A new layer, meaning an (init_fun, apply_fun) pair, representing the serial composition of the given sequence of layers.

&nbsp;

jax.example_libraries.stax.shape_dependent(*make_layer*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/stax.py#L340-L357)[\#](#jax.example_libraries.stax.shape_dependent "Link to this definition")  
Combinator to delay layer constructor pair until input shapes are known.

Parameters:  
**make_layer** – a one-argument function that takes an input shape as an argument (a tuple of positive integers) and returns an (init_fun, apply_fun) pair.

Returns:  
A new layer, meaning an (init_fun, apply_fun) pair, representing the same layer as returned by make_layer but with its construction delayed until input shapes are known.

[](jax.example_libraries.optimizers.html "previous page")

previous

`jax.example_libraries.optimizers` module

[](jax.experimental.html "next page")

next

`jax.experimental` module

Contents

- [`AvgPool()`](#jax.example_libraries.stax.AvgPool)
- [`BatchNorm()`](#jax.example_libraries.stax.BatchNorm)
- [`Conv()`](#jax.example_libraries.stax.Conv)
- [`Conv1DTranspose()`](#jax.example_libraries.stax.Conv1DTranspose)
- [`ConvTranspose()`](#jax.example_libraries.stax.ConvTranspose)
- [`Dense()`](#jax.example_libraries.stax.Dense)
- [`Dropout()`](#jax.example_libraries.stax.Dropout)
- [`FanInConcat()`](#jax.example_libraries.stax.FanInConcat)
- [`FanOut()`](#jax.example_libraries.stax.FanOut)
- [`GeneralConv()`](#jax.example_libraries.stax.GeneralConv)
- [`GeneralConvTranspose()`](#jax.example_libraries.stax.GeneralConvTranspose)
- [`MaxPool()`](#jax.example_libraries.stax.MaxPool)
- [`SumPool()`](#jax.example_libraries.stax.SumPool)
- [`elementwise()`](#jax.example_libraries.stax.elementwise)
- [`parallel()`](#jax.example_libraries.stax.parallel)
- [`serial()`](#jax.example_libraries.stax.serial)
- [`shape_dependent()`](#jax.example_libraries.stax.shape_dependent)

By The JAX authors

© Copyright 2024, The JAX Authors.\
